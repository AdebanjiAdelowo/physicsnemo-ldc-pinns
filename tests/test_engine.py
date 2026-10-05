"""Geometry, loss and residuals of the adapted training code against the upstream example."""

import numpy as np
import pytest
import sympy
import torch
from pinn_cavity import engine
from pinn_cavity.config import load_config
from pinn_cavity.equations import NavierStokes

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def cfg():
    return load_config("smoke")


@pytest.fixture(scope="module")
def cavity(cfg):
    return engine.Cavity(cfg, DEVICE)


def test_upstream_model_has_the_expected_size():
    model = engine.build_model(load_config("official"))
    # 2 -> 512, five 512 -> 512 layers, 512 -> 3
    assert engine.count_parameters(model) == (2 * 512 + 512) + 5 * (512 * 512 + 512) + (512 * 3 + 3)


def test_sampling_stays_in_the_cavity(cavity):
    torch.manual_seed(0)
    interior = cavity.sample_interior(5000, DEVICE)
    boundary = cavity.sample_boundary(5000, DEVICE)
    assert interior["x"].abs().max() <= 0.05 and interior["y"].abs().max() <= 0.05
    assert interior["sdf"].min() >= 0 and interior["sdf"].max() <= 0.05
    expected = 0.05 - torch.maximum(interior["x"].abs(), interior["y"].abs())
    assert torch.allclose(interior["sdf"], expected, atol=1e-7)
    on_wall = torch.maximum(boundary["x"].abs(), boundary["y"].abs())
    assert torch.allclose(on_wall, torch.full_like(on_wall, 0.05), atol=1e-7)
    lid = (boundary["y"] >= 0.05 - 1e-7).float().mean().item()
    assert 0.22 < lid < 0.28  # one of four sides


def test_equations_are_steady_incompressible_navier_stokes():
    x, y = sympy.Symbol("x"), sympy.Symbol("y")
    psi = sympy.sin(x) * sympy.cos(2 * y)
    u, v, p = psi.diff(y), -psi.diff(x), x * y
    ns = NavierStokes(nu=0.01)
    names = {"u": sympy.Function("u")(x, y), "v": sympy.Function("v")(x, y), "p": sympy.Function("p")(x, y)}
    assert sympy.simplify(ns.equations["continuity"].subs(names["u"], u).subs(names["v"], v).doit()) == 0
    mom = ns.equations["momentum_x"].subs(names["u"], u).subs(names["v"], v).subs(names["p"], p).doit()
    by_hand = u * u.diff(x) + v * u.diff(y) + p.diff(x) - sympy.Float(0.01) * (u.diff(x, 2) + u.diff(y, 2))
    assert abs(float((mom - by_hand).subs({x: 0.3, y: -0.2}))) < 1e-12


class Exact(torch.nn.Module):
    """A divergence-free field with known residuals, in place of the network."""

    def forward(self, xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        return torch.cat([torch.sin(x) * torch.cos(y), -torch.cos(x) * torch.sin(y), x * y], dim=1)


def test_residuals_match_an_analytic_field(cfg):
    phy = engine.build_physics(cfg, 100, DEVICE)
    nu = 0.1 / 100
    torch.manual_seed(1)
    x, y = torch.rand(64, dtype=torch.float64) - 0.5, torch.rand(64, dtype=torch.float64) - 0.5
    res = engine.residuals(Exact().double(), phy, x, y)
    u, v = torch.sin(x) * torch.cos(y), -torch.cos(x) * torch.sin(y)
    assert res["continuity"].abs().max() < 1e-12
    # u u_x + v u_y + p_x - nu laplace(u), with laplace(u) = -2 u
    expected = u * torch.cos(x) * torch.cos(y) + v * (-torch.sin(x) * torch.sin(y)) + y + 2 * nu * u
    assert torch.allclose(res["momentum_x"].squeeze(1), expected, atol=1e-12)


def test_loss_is_the_upstream_expression(cfg, cavity):
    """Recompute the loss with the lines of the upstream train.py, constants included."""
    torch.manual_seed(3)
    model = engine.build_model(cfg)
    phy = engine.build_physics(cfg, 10, DEVICE)
    bc_data, int_data = cavity.sample_boundary(300, DEVICE), cavity.sample_interior(200, DEVICE)
    terms = engine.loss_terms(model, phy, cavity, bc_data, int_data, 1.0)
    ours = sum(terms[name] for name in engine.LOSS_TERMS)

    height = 0.1
    mask_top_wall = bc_data["y"] >= height / 2 - 1e-7
    mask_no_slip = ~mask_top_wall
    no_slip_xy = torch.stack([bc_data["x"][mask_no_slip], bc_data["y"][mask_no_slip]], dim=-1)
    top_wall_x = bc_data["x"][mask_top_wall].unsqueeze(-1)
    top_wall_xy = torch.stack([bc_data["x"][mask_top_wall], bc_data["y"][mask_top_wall]], dim=-1)
    int_x = int_data["x"].unsqueeze(-1).requires_grad_(True)
    int_y = int_data["y"].unsqueeze(-1).requires_grad_(True)
    int_sdf = int_data["sdf"].unsqueeze(-1)
    coords = torch.cat([int_x, int_y], dim=1)
    no_slip_out, top_wall_out, interior_out = model(no_slip_xy), model(top_wall_xy), model(coords)
    u_no_slip = torch.mean(no_slip_out[:, 0:1] ** 2)
    v_no_slip = torch.mean(no_slip_out[:, 1:2] ** 2)
    u_slip = torch.mean(((top_wall_out[:, 0:1] - 1.0) ** 2) * (1 - 20 * torch.abs(top_wall_x)))
    v_slip = torch.mean(top_wall_out[:, 1:2] ** 2)
    phy_loss_dict = phy.forward({"coordinates": coords, "u": interior_out[:, 0:1], "v": interior_out[:, 1:2], "p": interior_out[:, 2:3]})
    cont = phy_loss_dict["continuity"] * int_sdf
    mom_x = phy_loss_dict["momentum_x"] * int_sdf
    mom_y = phy_loss_dict["momentum_y"] * int_sdf
    upstream = torch.mean(cont**2) + torch.mean(mom_x**2) + torch.mean(mom_y**2) + u_no_slip + v_no_slip + u_slip + v_slip
    assert mask_top_wall.any()
    assert torch.allclose(ours, upstream, rtol=1e-5)


def test_learning_rate_decay_is_the_upstream_constant():
    cfg = load_config("official")
    # 0.95 ** (1 / 4000): the decay of the upstream lambda, per step
    assert cfg.training.lr_decay_per_step == pytest.approx(0.95 ** (1 / 4000), rel=1e-12)


def test_relative_l2():
    a = np.array([[3.0, 4.0]])
    assert engine.relative_l2(a, a) == 0.0
    assert engine.relative_l2(np.zeros_like(a), a) == 1.0
