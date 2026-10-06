"""The loop over Reynolds numbers and seeds, and the summary tables."""

import csv
import json
import statistics
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from . import engine
from .provenance import write_json

SUMMARY_FIELDS = [
    "run",
    "reynolds",
    "viscosity",
    "seed",
    "parameters",
    "optimiser_steps",
    "final_train_loss",
    "velocity_rel_l2",
    "u_rel_l2",
    "v_rel_l2",
    "velocity_rel_l2_core",
    "velocity_max_abs_error",
    "centreline_u_rms_error",
    "centreline_v_rms_error",
    "centreline_u_min",
    "centreline_u_min_reference",
    "ghia_u_rms_difference",
    "ghia_v_rms_difference",
    "no_slip_rms_error",
    "lid_u_rms_error",
    "residual_rms_continuity",
    "residual_rms_momentum_x",
    "residual_rms_momentum_y",
    "train_seconds",
    "seconds_per_step",
    "inference_ms",
    "peak_train_memory_mb",
]
AGGREGATED = [
    "velocity_rel_l2",
    "velocity_rel_l2_core",
    "centreline_u_rms_error",
    "centreline_v_rms_error",
    "final_train_loss",
    "residual_rms_continuity",
    "residual_rms_momentum_x",
    "residual_rms_momentum_y",
    "no_slip_rms_error",
    "lid_u_rms_error",
    "train_seconds",
    "seconds_per_step",
    "inference_ms",
    "peak_train_memory_mb",
]
PLOT_STRIDE_TARGET = 128  # fields.npz holds at most 129 x 129 nodes


def train_study(cfg: DictConfig, root: Path, resume: bool = False) -> Path:
    """Train every (Reynolds number, seed) run of the study. ``resume`` skips runs that already finished."""
    # a resumed study keeps the record of its first start; a study started with --resume gets one
    started = (Path(root) / cfg.output_dir / "study_metadata.json").exists()
    out_dir, device, references = engine.prepare_study(cfg, root, write_metadata=not (resume and started))
    for reynolds, seed in engine.study_runs(cfg):
        run_dir = out_dir / engine.run_name(reynolds, seed)
        if resume and (run_dir / "train_metrics.json").exists():
            print(f"[skip] {run_dir.name} already trained", flush=True)
            continue
        metrics = engine.train_run(cfg, reynolds, seed, run_dir, device, references[reynolds])
        print(
            f"[train] {run_dir.name}: final loss {metrics['final_train_loss']:.3e}, "
            f"{metrics['train_seconds']:.1f} s on {device} ({1e3 * metrics['seconds_per_step']:.1f} ms per step)",
            flush=True,
        )
    return out_dir


def evaluate_study(cfg: DictConfig, root: Path) -> dict:
    """Evaluate every run from its checkpoint and write the summary files and plotting fields."""
    out_dir, device, references = engine.prepare_study(cfg, root, write_metadata=False)
    stride = max(1, (cfg.reference.grid - 1) // PLOT_STRIDE_TARGET)
    fields = {}
    rows = []
    for reynolds, seed in engine.study_runs(cfg):
        run_dir = out_dir / engine.run_name(reynolds, seed)
        reference = references[reynolds]
        evaluation, prediction = engine.evaluate_run(cfg, reynolds, run_dir, device, reference)
        train = json.loads((run_dir / "train_metrics.json").read_text())
        row = {**train, **evaluation}
        rows.append({k: row.get(k) for k in SUMMARY_FIELDS})
        tag = f"re{reynolds:04d}"
        if f"{tag}_u_reference" not in fields:
            fields[f"{tag}_u_reference"] = reference["u"][::stride, ::stride]
            fields[f"{tag}_v_reference"] = reference["v"][::stride, ::stride]
            fields.setdefault("x", reference["x"][::stride])
            fields.setdefault("y", reference["y"][::stride])
        for name in ("u", "v"):  # every seed: the centreline figure shows all of them
            fields[f"{tag}_seed{seed}_{name}"] = prediction[name][::stride, ::stride].astype(np.float32)
        if seed == cfg.study.seeds[0]:
            fields[f"{tag}_seed{seed}_p"] = prediction["p"][::stride, ::stride].astype(np.float32)
        print(f"[eval] {run_dir.name}: velocity relative L2 {row['velocity_rel_l2']:.4e}", flush=True)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "name": cfg.name,
        "device": str(device),
        "evaluation_device_note": "inference times are those of the evaluating device",
        "reference_grid": int(cfg.reference.grid),
        "by_reynolds": aggregate(rows),
        "runs": rows,
    }
    write_json(out_dir / "summary.json", summary)
    np.savez_compressed(out_dir / "fields.npz", **fields)
    return summary


def verify_study(cfg: DictConfig, root: Path) -> list[dict]:
    """Recompute the velocity error of every run from its checkpoint and compare with the stored value.

    Nothing is written. Used after results were produced on another machine: the stored
    timings and memory figures belong to that machine and must not be replaced. The
    reference solution is recomputed on this machine if it is not cached.
    """
    out_dir, device, references = engine.prepare_study(cfg, root, write_metadata=False)
    rows = []
    for reynolds, seed in engine.study_runs(cfg):
        run_dir = out_dir / engine.run_name(reynolds, seed)
        stored = json.loads((run_dir / "eval_metrics.json").read_text())["velocity_rel_l2"]
        recomputed = engine.evaluate_run(cfg, reynolds, run_dir, device, references[reynolds], write=False)[0]["velocity_rel_l2"]
        rows.append(
            {"run": run_dir.name, "stored": stored, "recomputed": recomputed, "relative_difference": abs(recomputed - stored) / stored}
        )
        print(f"[verify] {run_dir.name}: stored {stored:.6e}, recomputed on {device} {recomputed:.6e}", flush=True)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean, sample standard deviation and range over seeds for every Reynolds number."""
    out = []
    for reynolds in sorted({r["reynolds"] for r in rows}):
        group = [r for r in rows if r["reynolds"] == reynolds]
        entry = {"reynolds": reynolds, "viscosity": group[0]["viscosity"], "n_seeds": len(group)}
        for key in AGGREGATED:
            values = [r[key] for r in group if r[key] is not None]
            entry[f"{key}_mean"] = statistics.fmean(values) if values else None
            entry[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else None
            entry[f"{key}_min"] = min(values) if values else None
            entry[f"{key}_max"] = max(values) if values else None
        out.append(entry)
    return out
