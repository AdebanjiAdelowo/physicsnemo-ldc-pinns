"""Properties of the loss and of the evaluation established by the audit (docs/SCIENTIFIC_AUDIT.md)."""

import json

import numpy as np
import pytest
import torch
from pinn_cavity import engine
from pinn_cavity import reference as ref
from pinn_cavity.config import REPO_ROOT, load_config
from pinn_cavity.diagnostics import Constant, LidLayer, sampled_loss

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def cfg():
    # the loss of the example: cavity, lid speed and point sampling as upstream
    return load_config("official", ["device=cpu"])


@pytest.fixture(scope="module")
def re100(tmp_path_factory):
    return ref.load_or_solve(100.0, 65, 0.1, 0.1, 1.0, tmp_path_factory.mktemp("ref"))


def test_mean_squared_distance_weight(cfg):
    # the residuals enter the loss multiplied by d, so with weight d^2; for uniform points
    # in a square of side L its mean is (L/2)^2 / 6
    torch.manual_seed(0)
    sdf = engine.Cavity(cfg, DEVICE).sample_interior(400_000, DEVICE)["sdf"].double()
    assert (sdf**2).mean().item() == pytest.approx(0.05**2 / 6, rel=0.01)


def test_constant_field_has_zero_residual_and_loss_one_third_at_its_optimum(cfg):
    # for (u, v) = (c, 0): loss = c^2 + (c - 1)^2 mean(w), mean(w) = 1/2, minimum 1/3 at c = 1/3.
    # This is the state of the collapsed run re0100_seed2 (u = 0.333 everywhere, loss 0.34).
    torch.manual_seed(1)
    x, y = torch.rand(256) * 0.1 - 0.05, torch.rand(256) * 0.1 - 0.05
    res = engine.residuals(Constant(1 / 3), engine.build_physics(cfg, 100, DEVICE), x, y)
    assert all(torch.count_nonzero(res[name]) == 0 for name in engine.RESIDUALS)
    loss = {c: sampled_loss(Constant(c), cfg, 100, 500, 20000, 5, seed=2)["loss"] for c in (0.25, 1 / 3, 0.42)}
    assert loss[1 / 3] == pytest.approx(1 / 3, rel=0.02)
    assert loss[1 / 3] < loss[0.25] and loss[1 / 3] < loss[0.42]


def test_the_collapsed_run_is_a_constant_field():
    # tracked predictions of re0100_seed2 and the 513-node reference, every fourth node
    with np.load(REPO_ROOT / "results" / "local" / "fields.npz") as f:
        u, v, u_ref, v_ref = f["re0100_seed2_u"], f["re0100_seed2_v"], f["re0100_u_reference"], f["re0100_v_reference"]
    assert u.std() < 1e-6 and np.abs(v).max() < 1e-12 and u.mean() == pytest.approx(1 / 3, abs=1e-3)
    truth = np.stack([u_ref.ravel(), v_ref.ravel()], axis=1)
    run = engine.relative_l2(np.stack([u.ravel(), v.ravel()], axis=1), truth)
    constant = engine.relative_l2(np.stack([np.full(u.size, u.mean()), np.zeros(u.size)], axis=1), truth)
    assert run == pytest.approx(constant, rel=1e-5)
    # the stored 513-node value is 1.607; every fourth node gives 1.582
    assert run == pytest.approx(1.607, rel=0.02)


def test_corner_nodes_do_not_enter_the_corner_robust_metrics(re100):
    u, v = re100["u"].copy(), re100["v"].copy()
    clean = engine.corner_robust_metrics(u, v, re100, 0.1, 1.0)
    assert all(value == 0 for value in clean.values())
    # a prediction that is exact except for u = 0 on the two corner nodes
    u[0, -1] = u[-1, -1] = 0.0
    out = engine.corner_robust_metrics(u, v, re100, 0.1, 1.0)
    assert out["lid_u_rms_error_without_corners"] == 0 and out["no_slip_rms_error_without_corners"] == 0
    assert out["velocity_rel_l2_without_lid_corners_r0.02"] == 0
    assert out["velocity_rel_l2_trapezoid"] > 0
    lid_rms_with_corners = float(np.sqrt(np.mean((u[:, -1] - 1.0) ** 2)))
    assert lid_rms_with_corners == pytest.approx(np.sqrt(2 / 65))


def _stored_runs(study):
    return json.loads((REPO_ROOT / "results" / study / "summary.json").read_text())["runs"]


def test_a_field_that_is_not_the_cavity_flow_beats_the_trained_networks_at_re100_and_re400(cfg, tmp_path):
    # u = U g(x) exp(-(L/2 - y)/delta), v = p = 0: a layer under the lid above fluid at rest.
    # Its weighted residual falls like nu^2 / delta while its velocity error stays large, so at
    # Re = 100 and 400 it scores a lower training loss than every trained network of the
    # reduced study; at Re = 10 it does not. See docs/SCIENTIFIC_AUDIT.md, finding F1.
    final_loss = {}
    for run in _stored_runs("local"):
        final_loss.setdefault(run["reynolds"], []).append(run["final_train_loss"])
    loss = {}
    for re, eps in ((10, 0.05), (100, 0.01), (400, 0.002)):
        field = LidLayer(0.1, 1.0, 0.05 * 0.1, eps * 0.1)
        loss[re] = sampled_loss(field, cfg, re, 20000, 10000, 2, seed=7)["loss"]
    assert loss[10] > max(final_loss[10])
    assert loss[100] < min(final_loss[100]) / 10 and loss[400] < min(final_loss[400]) / 100
    reference = ref.load_or_solve(400.0, 65, 0.1, 0.1, 1.0, tmp_path)
    points = engine.grid_points(reference, 1, DEVICE)
    pred = engine.predict(LidLayer(0.1, 1.0, 0.005, 0.0002), points)[:, :2]
    truth = np.stack([reference["u"].ravel(), reference["v"].ravel()], axis=1)
    assert engine.relative_l2(pred, truth) > 0.7
