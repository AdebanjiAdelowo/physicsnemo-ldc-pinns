"""Analytic fields and a low-variance estimate of the training loss, used by the audit.

The fields stand in for the network in :func:`pinn_cavity.engine.loss_terms`, so they are
scored with the loss of the example exactly as a trained network is.
"""

import numpy as np
import torch
from omegaconf import DictConfig

from . import engine


def _zero(xy: torch.Tensor) -> torch.Tensor:
    # identically zero, with second derivatives that stay in the autograd graph
    return 0 * torch.sin(xy[:, 0:1]) * torch.sin(xy[:, 1:2])


class Constant(torch.nn.Module):
    """``u = c``, ``v = p = 0``. All three residuals vanish identically."""

    def __init__(self, u: float):
        super().__init__()
        self.u = u

    def forward(self, xy):
        zero = _zero(xy)
        return torch.cat([zero + self.u, zero, zero], dim=1)


class LidLayer(torch.nn.Module):
    """``u = U g(x) exp(-(L/2 - y) / delta)``, ``v = p = 0``.

    ``g`` vanishes on the side walls and rises to 1 over a length ``eps``. The field meets
    the boundary conditions away from the lid corners and is at rest below a layer of
    thickness ``delta`` under the lid: it is not the cavity flow at any Reynolds number.
    """

    def __init__(self, width: float, lid_velocity: float, delta: float, eps: float):
        super().__init__()
        self.half, self.lid, self.delta, self.eps = width / 2, lid_velocity, delta, eps

    def forward(self, xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        g = (1 - torch.exp(-(self.half - x) / self.eps)) * (1 - torch.exp(-(self.half + x) / self.eps))
        u = self.lid * g * torch.exp(-(self.half - y) / self.delta)
        zero = _zero(xy)
        return torch.cat([u, zero, zero], dim=1)


def sampled_loss(
    model, cfg: DictConfig, reynolds: float, interior: int, boundary: int, batches: int, seed: int, device=torch.device("cpu")
) -> dict:
    """The seven loss terms and their sum, averaged over ``batches`` independent draws of points.

    One draw with the point counts of the example is what the training loop records; the
    average over many draws removes most of the sampling noise of that record.
    """
    torch.manual_seed(seed)
    cavity = engine.Cavity(cfg, device)
    phy = engine.build_physics(cfg, reynolds, device)
    rows = []
    for _ in range(batches):
        terms = engine.loss_terms(
            model, phy, cavity, cavity.sample_boundary(boundary, device), cavity.sample_interior(interior, device), cfg.problem.lid_velocity
        )
        rows.append({k: float(v.detach()) for k, v in terms.items()})
    out = {key: float(np.mean([r[key] for r in rows])) for key in engine.LOSS_TERMS}
    totals = np.array([sum(r.values()) for r in rows])
    out["loss"] = float(totals.mean())
    out["loss_standard_error"] = float(totals.std(ddof=1) / np.sqrt(batches)) if batches > 1 else None
    out["samples"] = {"interior": interior * batches, "boundary": boundary * batches}
    return out


def streamfunction(u: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``psi(x, y) = int u dy`` from the bottom wall, trapezoidal rule, ``u[i, j]`` at ``(x_i, y_j)``.

    For a divergence-free field with no flux through the walls this is the streamfunction
    with ``psi = 0`` on the walls. For a network it is defined by ``u`` alone; its value on
    the lid, which is zero for the exact flow, measures the net horizontal mass flux.
    """
    dy = np.diff(y)
    increments = 0.5 * (u[:, 1:] + u[:, :-1]) * dy
    return np.concatenate([np.zeros((u.shape[0], 1)), np.cumsum(increments, axis=1)], axis=1)


def primary_vortex(psi: np.ndarray, x: np.ndarray, y: np.ndarray, margin: int = 2) -> dict:
    """Minimum of ``psi`` away from the walls: strength and centre of the primary vortex.

    ``margin`` nodes next to each wall are left out, so that a field with no recirculation
    reports a minimum close to zero rather than a wall value.
    """
    inner = psi[margin:-margin, margin:-margin]
    i, j = np.unravel_index(np.argmin(inner), inner.shape)
    return {"psi_min": float(inner[i, j]), "x": float(x[i + margin]), "y": float(y[j + margin])}
