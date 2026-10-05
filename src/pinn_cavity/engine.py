# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ``Cavity``, ``build_model``, ``build_physics``, ``loss_terms`` and the optimisation
# loop of ``train_run`` are adapted from examples/cfd/ldc_pinns/train.py of NVIDIA
# PhysicsNeMo, commit b45a5c810c741e6b41f8515be24c51121f8fc21f: the geometry and the
# point sampling, the network, the PhysicsInformer, the boundary and residual losses
# with their weights, Adam and the per-step learning-rate decay are those of the
# original. Modified by Adebanji Adelowo (2026): the constants of the original are
# read from the configuration, the viscosity is set from a Reynolds number, the run
# is seeded, the device is a configuration key, the loss terms are recorded
# separately, the velocity error against a reference solution is tracked, a
# checkpoint and CSV/JSON records are written, and ``evaluate_run`` is new. The
# in-loop matplotlib figure and the DistributedManager of the original are not used.

import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from physicsnemo.mesh.primitives.planar.structured_grid import (
    load as load_structured_grid,
)
from physicsnemo.mesh.sampling import sample_random_points_on_cells
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.utils import load_checkpoint, save_checkpoint
from torch.optim import Adam, lr_scheduler

from . import reference as ref
from .equations import NavierStokes, viscosity
from .provenance import collect_metadata, write_json

RESIDUALS = ["continuity", "momentum_x", "momentum_y"]
LOSS_TERMS = RESIDUALS + ["u_no_slip", "v_no_slip", "u_lid", "v_lid"]
HISTORY_FIELDS = ["step", "learning_rate", "loss", *LOSS_TERMS, "velocity_rel_l2", "elapsed_seconds"]


def resolve_device(name: str) -> torch.device:
    """``auto`` picks CUDA, then Apple MPS, then CPU. An explicit name must be available."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested but CUDA is not available on this machine.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device=mps was requested but MPS is not available on this machine.")
    return device


def synchronise(device: torch.device) -> None:
    """Wait for queued accelerator work so that a wall-clock reading is meaningful."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run_name(reynolds: int, seed: int) -> str:
    return f"re{reynolds:04d}_seed{seed}"


def study_runs(cfg: DictConfig) -> list[tuple[int, int]]:
    return [(int(r), int(s)) for r in cfg.study.reynolds for s in cfg.study.seeds]


def validate_study(cfg: DictConfig) -> None:
    """Reject settings the pipeline cannot represent."""
    for reynolds in cfg.study.reynolds:
        if int(reynolds) != reynolds or reynolds <= 0:
            raise ValueError(f"Reynolds numbers must be positive integers, got {reynolds!r}.")
    if cfg.problem.width != cfg.problem.height:
        raise ValueError("the reference solver handles a square cavity only.")
    if cfg.reference.grid % 2 == 0:
        raise ValueError("reference.grid must be odd so that the centrelines are grid lines.")
    if cfg.training.steps % cfg.training.log_every or cfg.training.steps % cfg.training.error_every:
        raise ValueError("training.steps must be a multiple of training.log_every and training.error_every.")


class Cavity:
    """Geometry and random point sampling of the example, built with ``physicsnemo.mesh``."""

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.width, self.height = float(cfg.problem.width), float(cfg.problem.height)
        self.x_min, self.x_max = -self.width / 2, self.width / 2
        self.y_min, self.y_max = -self.height / 2, self.height / 2
        self.interior_mesh = load_structured_grid(
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            n_x=cfg.geometry.mesh_points_per_side,
            n_y=cfg.geometry.mesh_points_per_side,
            device=device,
        )
        self.boundary_mesh = self.interior_mesh.get_boundary_mesh()

    def sample_boundary(self, n_points, device):
        """Sample on the rectangle boundary using physicsnemo.mesh."""
        cell_indices = torch.randint(0, self.boundary_mesh.n_cells, (n_points,), device=device)
        pts = sample_random_points_on_cells(self.boundary_mesh, cell_indices)
        return {"x": pts[:, 0], "y": pts[:, 1]}

    def sample_interior(self, n_points, device):
        """Sample inside the rectangle using physicsnemo.mesh, with analytical SDF."""
        cell_indices = torch.randint(0, self.interior_mesh.n_cells, (n_points,), device=device)
        pts = sample_random_points_on_cells(self.interior_mesh, cell_indices)
        x, y = pts[:, 0], pts[:, 1]
        sdf = torch.min(
            torch.stack([x - self.x_min, self.x_max - x, y - self.y_min, self.y_max - y], dim=-1),
            dim=-1,
        ).values
        return {"x": x, "y": y, "sdf": sdf}


def build_model(cfg: DictConfig) -> FullyConnected:
    return FullyConnected(in_features=2, out_features=3, num_layers=cfg.model.num_layers, layer_size=cfg.model.layer_size)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_physics(cfg: DictConfig, reynolds: float, device: torch.device) -> PhysicsInformer:
    nu = viscosity(reynolds, cfg.problem.lid_velocity, cfg.problem.width)
    ns = NavierStokes(nu=nu, rho=cfg.problem.rho, dim=2, time=False)
    return PhysicsInformer(required_outputs=RESIDUALS, equations=ns, grad_method="autodiff", device=device)


def residuals(model, phy_inf: PhysicsInformer, x: torch.Tensor, y: torch.Tensor) -> dict:
    """PDE residuals of the network at the points ``(x, y)``, each of shape ``(n, 1)``."""
    int_x = x.unsqueeze(-1).requires_grad_(True)
    int_y = y.unsqueeze(-1).requires_grad_(True)
    coords = torch.cat([int_x, int_y], dim=1)
    interior_out = model(coords)
    return phy_inf.forward(
        {
            "coordinates": coords,
            "u": interior_out[:, 0:1],
            "v": interior_out[:, 1:2],
            "p": interior_out[:, 2:3],
        }
    )


def loss_terms(model, phy_inf: PhysicsInformer, cavity: Cavity, bc_data: dict, int_data: dict, lid_velocity: float) -> dict:
    """The seven terms of the upstream loss. Their sum is the training loss."""
    y_vals = bc_data["y"]
    mask_top_wall = y_vals >= cavity.height / 2 - 1e-7
    mask_no_slip = ~mask_top_wall

    no_slip_xy = torch.stack([bc_data["x"][mask_no_slip], bc_data["y"][mask_no_slip]], dim=-1)
    top_wall_x = bc_data["x"][mask_top_wall].unsqueeze(-1)
    top_wall_xy = torch.stack([bc_data["x"][mask_top_wall], bc_data["y"][mask_top_wall]], dim=-1)

    no_slip_out = model(no_slip_xy)
    top_wall_out = model(top_wall_xy)

    # the lid weight falls linearly from 1 at the centre to 0 at the corners (1 - 20 |x| upstream)
    lid_weight = 1 - (2 / cavity.width) * torch.abs(top_wall_x)
    terms = {
        "u_no_slip": torch.mean(no_slip_out[:, 0:1] ** 2),
        "v_no_slip": torch.mean(no_slip_out[:, 1:2] ** 2),
        "u_lid": torch.mean(((top_wall_out[:, 0:1] - lid_velocity) ** 2) * lid_weight),
        "v_lid": torch.mean(top_wall_out[:, 1:2] ** 2),
    }

    phy_loss_dict = residuals(model, phy_inf, int_data["x"], int_data["y"])
    int_sdf = int_data["sdf"].unsqueeze(-1)
    for name in RESIDUALS:
        terms[name] = torch.mean((phy_loss_dict[name] * int_sdf) ** 2)
    return terms


def load_reference(cfg: DictConfig, reynolds: int, root: Path) -> dict:
    return ref.load_or_solve(
        float(reynolds), cfg.reference.grid, cfg.problem.width, cfg.problem.height, cfg.problem.lid_velocity, Path(root) / cfg.data_dir
    )


def grid_points(reference: dict, stride: int, device: torch.device) -> torch.Tensor:
    """Nodes of the reference grid, every ``stride``-th per direction, as an ``(n, 2)`` float32 tensor."""
    xx, yy = np.meshgrid(reference["x"][::stride], reference["y"][::stride], indexing="ij")
    return torch.from_numpy(np.stack([xx.ravel(), yy.ravel()], axis=1)).to(torch.float).to(device)


@torch.no_grad()
def predict(model, points: torch.Tensor, batch: int = 65536) -> np.ndarray:
    return torch.cat([model(points[i : i + batch]) for i in range(0, len(points), batch)]).cpu().numpy().astype(np.float64)


def relative_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.sum((pred - truth) ** 2) / np.sum(truth**2)))


def velocity_error(model, reference: dict, stride: int, device: torch.device) -> float:
    """Relative L2 error of the velocity vector on the (sub-sampled) reference grid."""
    out = predict(model, grid_points(reference, stride, device))
    truth = np.stack([reference["u"][::stride, ::stride].ravel(), reference["v"][::stride, ::stride].ravel()], axis=1)
    return relative_l2(out[:, :2], truth)


def train_run(cfg: DictConfig, reynolds: int, seed: int, run_dir: Path, device: torch.device, reference: dict) -> dict:
    """Train one network and write ``history.csv``, ``train_metrics.json`` and a checkpoint."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = cfg.training.steps

    # The model is initialised on CPU from the seed and then moved, so the initial
    # weights do not depend on the device. The collocation points are drawn on the device.
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_model(cfg).to(device)
    cavity = Cavity(cfg, device)
    phy_inf = build_physics(cfg, reynolds, device)

    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cfg.training.lr_decay_per_step**step)

    # error tracking uses at most 129 x 129 nodes of the reference grid
    stride = max(1, (cfg.reference.grid - 1) // 128)
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronise(device)
    start = time.perf_counter()
    tracking_seconds = 0.0

    for i in range(1, steps + 1):
        optimizer.zero_grad()

        bc_data = cavity.sample_boundary(cfg.training.boundary_points, device)
        int_data = cavity.sample_interior(cfg.training.interior_points, device)

        terms = loss_terms(model, phy_inf, cavity, bc_data, int_data, cfg.problem.lid_velocity)
        phy_loss = sum(terms[name] for name in LOSS_TERMS)
        learning_rate = optimizer.param_groups[0]["lr"]
        phy_loss.backward()
        optimizer.step()
        scheduler.step()

        if i % cfg.training.log_every == 0 or i == 1:
            row = {"step": i, "learning_rate": learning_rate, "loss": phy_loss.item()}
            row.update({name: terms[name].item() for name in LOSS_TERMS})
            if not np.isfinite(row["loss"]):
                raise RuntimeError(f"non-finite training loss at step {i}")
            row["velocity_rel_l2"] = None
            if i % cfg.training.error_every == 0:
                synchronise(device)
                tick = time.perf_counter()
                row["velocity_rel_l2"] = velocity_error(model, reference, stride, device)
                tracking_seconds += time.perf_counter() - tick
            synchronise(device)
            row["elapsed_seconds"] = time.perf_counter() - start
            history.append(row)

    synchronise(device)
    train_seconds = time.perf_counter() - start
    save_checkpoint(run_dir / "checkpoints", models=model, optimizer=optimizer, scheduler=scheduler, epoch=steps)

    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)

    metrics = {
        "run": run_dir.name,
        "reynolds": int(reynolds),
        "viscosity": viscosity(reynolds, cfg.problem.lid_velocity, cfg.problem.width),
        "seed": int(seed),
        "parameters": count_parameters(model),
        "optimiser_steps": steps,
        "final_train_loss": history[-1]["loss"],
        "final_loss_terms": {name: history[-1][name] for name in LOSS_TERMS},
        "train_seconds": train_seconds,
        "error_tracking_seconds": tracking_seconds,
        "seconds_per_step": (train_seconds - tracking_seconds) / steps,
        "peak_train_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
    }
    write_json(run_dir / "train_metrics.json", metrics)
    return metrics


def evaluate_run(
    cfg: DictConfig, reynolds: int, run_dir: Path, device: torch.device, reference: dict, write: bool = True
) -> tuple[dict, dict]:
    """Evaluate the checkpoint of ``run_dir`` against the reference solution.

    Returns the metrics and the predicted fields on the reference grid. ``write=False``
    leaves ``eval_metrics.json`` untouched.
    """
    run_dir = Path(run_dir)
    model = build_model(cfg).to(device)
    step = load_checkpoint(run_dir / "checkpoints", models=model, device=device)
    if step != cfg.training.steps:
        raise RuntimeError(f"{run_dir}: checkpoint is at step {step}, expected {cfg.training.steps}.")
    model.eval()

    n = cfg.reference.grid
    width, lid = float(cfg.problem.width), float(cfg.problem.lid_velocity)
    points = grid_points(reference, 1, device)
    pred = predict(model, points)
    u, v, p = (pred[:, k].reshape(n, n) for k in range(3))
    u_ref, v_ref = reference["u"], reference["v"]
    vel, vel_ref = np.stack([u, v]), np.stack([u_ref, v_ref])

    # central part of the cavity: at least a tenth of the width away from every wall
    s = (reference["x"] - reference["x"][0]) / width
    core = np.ix_((s >= 0.1) & (s <= 0.9), (s >= 0.1) & (s <= 0.9))
    mid = n // 2
    out = {
        "checkpoint_step": int(step),
        "reference_file": str(reference["file"]),
        "velocity_rel_l2": relative_l2(vel, vel_ref),
        "u_rel_l2": relative_l2(u, u_ref),
        "v_rel_l2": relative_l2(v, v_ref),
        "velocity_rel_l2_core": relative_l2(np.stack([u[core], v[core]]), np.stack([u_ref[core], v_ref[core]])),
        "velocity_max_abs_error": float(np.sqrt((u - u_ref) ** 2 + (v - v_ref) ** 2).max() / lid),
        "centreline_u_rms_error": float(np.sqrt(np.mean((u[mid, :] - u_ref[mid, :]) ** 2)) / lid),
        "centreline_v_rms_error": float(np.sqrt(np.mean((v[:, mid] - v_ref[:, mid]) ** 2)) / lid),
        "centreline_u_min": float(u[mid, :].min() / lid),
        "centreline_u_min_reference": float(u_ref[mid, :].min() / lid),
    }

    # boundary conditions, unweighted, on the wall nodes of the reference grid
    walls = np.concatenate([np.hypot(u, v)[0, :], np.hypot(u, v)[-1, :], np.hypot(u, v)[:, 0]])
    out["no_slip_rms_error"] = float(np.sqrt(np.mean(walls**2)) / lid)
    out["lid_u_rms_error"] = float(np.sqrt(np.mean((u[:, -1] - lid) ** 2)) / lid)

    if int(reynolds) in ref.GHIA_U:
        # the network is evaluated at the tabulated positions, no interpolation
        gy = torch.tensor([[0.0, (t - 0.5) * width] for t in ref.GHIA_Y], dtype=torch.float, device=device)
        gx = torch.tensor([[(t - 0.5) * width, 0.0] for t in ref.GHIA_X], dtype=torch.float, device=device)
        du = predict(model, gy)[:, 0] / lid - np.array(ref.GHIA_U[int(reynolds)])
        dv = predict(model, gx)[:, 1] / lid - np.array(ref.GHIA_V[int(reynolds)])
        out["ghia_u_rms_difference"] = float(np.sqrt(np.mean(du**2)))
        out["ghia_v_rms_difference"] = float(np.sqrt(np.mean(dv**2)))

    # PDE residuals without the SDF weight, at the centres of a uniform grid of cells, made
    # dimensionless with the cavity width and the lid speed
    phy_inf = build_physics(cfg, reynolds, device)
    m = cfg.evaluation.residual_grid
    centres = (torch.arange(m, dtype=torch.float, device=device) + 0.5) / m * width - width / 2
    cx, cy = torch.meshgrid(centres, centres, indexing="ij")
    res = residuals(model, phy_inf, cx.reshape(-1), cy.reshape(-1))
    scale = {"continuity": width / lid, "momentum_x": width / lid**2, "momentum_y": width / lid**2}
    for name in RESIDUALS:
        out[f"residual_rms_{name}"] = float(torch.sqrt(torch.mean(res[name].detach() ** 2)).item() * scale[name])

    # Inference time of the velocity and pressure on the full reference grid.
    times = []
    for i in range(cfg.evaluation.timing_warmup + cfg.evaluation.timing_repeats):
        synchronise(device)
        tick = time.perf_counter()
        with torch.no_grad():
            model(points)
        synchronise(device)
        if i >= cfg.evaluation.timing_warmup:
            times.append(time.perf_counter() - tick)
    out["inference_points"] = int(len(points))
    out["inference_ms"] = 1e3 * statistics.median(times)

    if write:
        write_json(run_dir / "eval_metrics.json", out)
    return out, {"u": u, "v": v, "p": p}


def prepare_study(cfg: DictConfig, root: Path, write_metadata: bool) -> tuple[Path, torch.device, dict]:
    """Validate the config, resolve the device, load the reference solutions, optionally write ``study_metadata.json``."""
    validate_study(cfg)
    device = resolve_device(cfg.device)
    out_dir = Path(root) / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = collect_metadata(device)  # before anything is computed: the code state at start
    references = {int(r): load_reference(cfg, int(r), root) for r in cfg.study.reynolds}
    metadata["config"] = OmegaConf.to_container(cfg, resolve=True)
    metadata["references"] = {
        str(r): {
            "file": str(s["file"]),
            "grid": int(cfg.reference.grid),
            "solve_seconds": float(s["solve_seconds"]),
            "newton_final_residual": float(s["newton_final_residual"]),
        }
        for r, s in references.items()
    }
    if write_metadata:
        write_json(out_dir / "study_metadata.json", metadata)
    return out_dir, device, references
