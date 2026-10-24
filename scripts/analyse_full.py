"""Diagnostics of the study at the upstream budget (``configs/full.yaml``), on CPU, from the checkpoints.

    python scripts/analyse_full.py --runs-root runs

``--runs-root`` holds ``full/<run>/checkpoints``; the records are read from ``results/full``.
Nothing under either is written. For every run the script

1. re-evaluates the checkpoint on CPU and compares every metric evaluated on the GPU
   (``eval_metrics_cuda.json``, see ``scripts/results.py merge``) with the recomputed value, and records the SHA-256 of the checkpoint files and of the
   resolved configuration;
2. estimates the loss of the final network on 25 independent batches at the upstream point
   counts: a validation loss on points the network was not trained on;
3. integrates the streamfunction from ``u`` and locates the primary vortex, against the
   reference;
4. records the state of every hidden layer (neurons constant or identically zero over the
   cavity) and scans ``history.csv`` for steps at which the residual terms vanish;
5. finds the analytic lid-layer field (``diagnostics.LidLayer``) closest to the network.

It then scores the zero, best-constant and lid-layer fields with the same objective (the
analytic counterexample of the audit, F1), and aggregates over seeds with a t interval and,
seed by seed, against the reduced study ``results/local``.

Output: ``results/full/analysis.json`` and ``figures/full_streamfunction.png``.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
from _bootstrap import ROOT
from audit import DEVICE, compare_stored, degenerate_section, network_state
from omegaconf import OmegaConf
from pinn_cavity import engine
from pinn_cavity.config import load_config
from pinn_cavity.diagnostics import LidLayer, primary_vortex, sampled_loss, streamfunction
from pinn_cavity.provenance import collect_metadata, write_json

# two-sided 95% quantiles of Student's t for 1 to 4 degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
LID_LAYER_DELTAS = np.geomspace(0.002, 0.3, 16)
LID_LAYER_EPS = np.geomspace(0.002, 0.1, 8)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def config_hash(config: dict) -> str:
    """Hash of the training recipe: everything except names, paths, the device and the runs of the study.

    A session trains a subset of the Reynolds numbers, so ``study`` is left out and recorded separately.
    """
    relevant = {k: v for k, v in config.items() if k not in {"name", "output_dir", "data_dir", "device", "study"}}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def recirculation(u: np.ndarray, reference: dict) -> dict:
    x, y = reference["x"], reference["y"]
    psi, psi_ref = streamfunction(u, y), reference["psi"]
    vortex, vortex_ref = primary_vortex(psi, x, y), primary_vortex(psi_ref, x, y)
    scale = abs(vortex_ref["psi_min"])
    width = float(x[-1] - x[0])
    return {
        "psi_min_over_reference": vortex["psi_min"] / vortex_ref["psi_min"],
        "psi_min": vortex["psi_min"],
        "psi_min_reference": vortex_ref["psi_min"],
        "vortex_centre": [vortex["x"], vortex["y"]],
        "vortex_centre_reference": [vortex_ref["x"], vortex_ref["y"]],
        "vortex_centre_distance_over_width": math.hypot(vortex["x"] - vortex_ref["x"], vortex["y"] - vortex_ref["y"]) / width,
        # psi on the lid from integrating u upwards; zero for a field with no net horizontal flux
        "lid_flux_max_over_psi_min_reference": float(np.abs(psi[1:-1, -1]).max() / scale),
        "psi_rel_l2": float(np.sqrt(np.sum((psi - psi_ref)[1:-1] ** 2) / np.sum(psi_ref[1:-1] ** 2))),
    }, psi


def nearest_lid_layer(u: np.ndarray, v: np.ndarray, reference: dict, width: float, lid: float) -> dict:
    """The lid-layer field closest to the network in relative L2 over the velocity."""
    points = engine.grid_points(reference, 1, DEVICE)
    norm = math.sqrt(float(np.sum(u**2 + v**2)))
    best = None
    for delta in LID_LAYER_DELTAS:
        for eps in LID_LAYER_EPS:
            field = engine.predict(LidLayer(width, lid, delta * width, eps * width), points)
            fu = field[:, 0].reshape(u.shape)
            distance = math.sqrt(float(np.sum((u - fu) ** 2 + v**2))) / norm
            if best is None or distance < best["distance_rel_l2"]:
                best = {"delta_over_width": float(delta), "eps_over_width": float(eps), "distance_rel_l2": distance}
    return best


def history_events(path: Path) -> dict:
    """Records at which every residual term is below 1e-20, and the largest jump of the loss."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    vanished = [int(r["step"]) for r in rows if all(float(r[name]) < 1e-20 for name in engine.RESIDUALS)]
    losses = [float(r["loss"]) for r in rows]
    jumps = [(losses[i] / losses[i - 1], int(rows[i]["step"])) for i in range(1, len(rows))]
    tracked = [(int(r["step"]), float(r["velocity_rel_l2"])) for r in rows if r["velocity_rel_l2"] not in ("", "None")]
    return {
        "first_step_residuals_vanish": vanished[0] if vanished else None,
        "records_with_vanished_residuals": len(vanished),
        "largest_loss_jump": max(jumps)[0],
        "largest_loss_jump_step": max(jumps)[1],
        "tracked_velocity_rel_l2": tracked,
    }


def collapsed(state: dict) -> bool:
    """A hidden layer with more than half of its neurons identically zero over the cavity."""
    return any(layer["neurons_identically_zero"] > layer["neurons"] // 2 for layer in state["hidden_layers"])


def interval(values: list[float]) -> dict:
    n = len(values)
    out = {"n": n, "values": values, "mean": statistics.fmean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}
    if n > 1:
        s = statistics.stdev(values)
        half = T95[n - 1] * s / math.sqrt(n)
        out.update({"std": s, "ci95_t": [out["mean"] - half, out["mean"] + half]})
    return out


def plot_streamfunction(psis: dict, references: dict, seeds: list[int], path: Path) -> None:
    import matplotlib.pyplot as plt

    values = sorted(references)
    fig, axes = plt.subplots(len(values), 1 + len(seeds), figsize=(2.6 * (1 + len(seeds)), 2.6 * len(values)), squeeze=False)
    for row, re in zip(axes, values):
        ref = references[re]
        s = 4
        x, y = ref["x"][::s], ref["y"][::s]
        psi_ref = ref["psi"][::s, ::s]
        lo = psi_ref.min()
        levels = sorted(lo * np.array([0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01]))
        panels = [("reference", psi_ref)] + [(f"seed {seed}", psis[(re, seed)][::s, ::s]) for seed in seeds]
        for ax, (title, psi) in zip(row, panels):
            ax.contourf(x, y, psi.T, levels=np.linspace(lo, -lo * 0.2, 25), cmap="viridis", extend="both")
            ax.contour(x, y, psi.T, levels=levels, colors="white", linewidths=0.6)
            ax.set_title(f"Re = {re}: {title}" if title == "reference" else title, loc="left", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Streamfunction from integrating u, same colour scale per row as the reference; white lines at fixed fractions of the reference minimum", x=0.01, ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--loss-batches", type=int, default=25)
    parser.add_argument("--records", type=Path, default=ROOT / "results" / "full", help="published records of the study")
    parser.add_argument("--out", type=Path, default=None, help="default: <records>/analysis.json")
    parser.add_argument("--figure", type=Path, default=ROOT / "figures" / "full_streamfunction.png")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides used for training (none for the real study)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    overrides = ["device=cpu", "evaluation.timing_warmup=0", "evaluation.timing_repeats=1", *args.overrides]
    cfg = load_config("full", overrides)
    width, lid = float(cfg.problem.width), float(cfg.problem.lid_velocity)
    references = {int(r): engine.load_reference(cfg, int(r), ROOT) for r in cfg.study.reynolds}
    audit = json.loads((ROOT / "results" / "audit" / "audit.json").read_text())
    report = {"metadata": collect_metadata(DEVICE), "runs_root": str(args.runs_root), "runs": {}}

    sessions = {}
    for path in sorted((args.records / "sessions").glob("*/study_metadata.json")):
        meta = json.loads(path.read_text())
        sessions[path.parent.name] = {
            "git_commit": meta["git_commit"],
            "git_dirty": meta["git_dirty"],
            "gpu_name": meta["gpu_name"],
            "cuda_version": meta["cuda_version"],
            "torch_version": meta["torch_version"],
            "physicsnemo": meta["physicsnemo"],
            "study": meta["config"]["study"],
            "config_sha256": config_hash(meta["config"]),
        }
    report["sessions"] = sessions
    report["config_sha256_full_yaml"] = config_hash(OmegaConf.to_container(load_config("full"), resolve=True))

    psis = {}
    for reynolds, seed in engine.study_runs(cfg):
        name = engine.run_name(reynolds, seed)
        run_dir = args.runs_root / "full" / name
        records = args.records / name
        stored = json.loads((records / "eval_metrics_cuda.json").read_text())  # evaluated on the GPU
        train = json.loads((records / "train_metrics.json").read_text())
        metrics, fields = engine.evaluate_run(cfg, reynolds, run_dir, DEVICE, references[reynolds], write=False)
        model = engine.build_model(cfg)
        engine.load_checkpoint(run_dir / "checkpoints", models=model, device=DEVICE)
        model.eval()
        circulation, psis[(reynolds, seed)] = recirculation(fields["u"], references[reynolds])
        state = network_state(model, references[reynolds])
        entry = {
            "check": compare_stored(stored, metrics),
            "checkpoint_sha256": {p.name: sha256(p) for p in sorted((run_dir / "checkpoints").glob("*")) if p.is_file()},
            "train": {k: train[k] for k in ("final_train_loss", "final_loss_terms", "train_seconds", "seconds_per_step", "peak_train_memory_mb")},
            "evaluation": {k: v for k, v in metrics.items() if k not in {"reference_file", "inference_ms", "inference_points"}},
            "validation_loss": sampled_loss(model, cfg, reynolds, 4000, 2000, args.loss_batches, seed=2000 + seed),
            "recirculation": circulation,
            "network_state": state,
            "collapsed": collapsed(state),
            "history": history_events(records / "history.csv"),
            "nearest_lid_layer": nearest_lid_layer(fields["u"], fields["v"], references[reynolds], width, lid),
        }
        report["runs"][name] = entry
        print(
            f"[run] {name}: error {metrics['velocity_rel_l2']:.4f}, stored-vs-CPU {entry['check']['worst_relative_difference']:.1e}, "
            f"validation loss {entry['validation_loss']['loss']:.3e}, psi_min ratio {circulation['psi_min_over_reference']:.3f}, "
            f"collapsed {entry['collapsed']}",
            flush=True,
        )

    # the analytic counterexample, scored with the same objective; not a trained model
    report["analytic_fields_same_objective"] = degenerate_section(load_config("official", overrides), references, 40000, 20000, 10)

    by_re = {}
    local = {r["run"]: r for r in json.loads((ROOT / "results" / "local" / "summary.json").read_text())["runs"]}
    for reynolds in sorted(references):
        names = [engine.run_name(reynolds, s) for s in cfg.study.seeds]
        runs = [report["runs"][n] for n in names]
        lid_layer = report["analytic_fields_same_objective"]["lid_layer"][str(reynolds)]["best"]
        audit_lid = audit["trivial_and_degenerate_fields"]["lid_layer"][str(reynolds)]["best"]
        entry = {
            "velocity_rel_l2": interval([r["evaluation"]["velocity_rel_l2"] for r in runs]),
            "velocity_rel_l2_core": interval([r["evaluation"]["velocity_rel_l2_core"] for r in runs]),
            "validation_loss": interval([r["validation_loss"]["loss"] for r in runs]),
            "psi_min_over_reference": interval([r["recirculation"]["psi_min_over_reference"] for r in runs]),
            "collapsed_runs": [n for n, r in zip(names, runs) if r["collapsed"]],
            "lid_layer_best_loss": lid_layer["loss"]["loss"],
            "lid_layer_best_velocity_rel_l2": lid_layer["velocity_rel_l2"],
            "lid_layer_best_loss_in_audit": audit_lid["loss"]["loss"],
            "validation_loss_over_lid_layer_loss": [r["validation_loss"]["loss"] / lid_layer["loss"]["loss"] for r in runs],
            # same seed, same Reynolds number: upstream budget minus reduced budget
            "paired_difference_to_reduced_budget": {
                n: r["evaluation"]["velocity_rel_l2"] - local[n]["velocity_rel_l2"] for n, r in zip(names, runs) if n in local
            },
        }
        by_re[str(reynolds)] = entry
    report["by_reynolds"] = by_re
    official = json.loads((ROOT / "results" / "official" / "re0010_seed0" / "eval_metrics.json").read_text())
    report["official_re0010_seed0_velocity_rel_l2"] = official["velocity_rel_l2"]

    out = args.out or args.records / "analysis.json"
    write_json(out, report)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    plot_streamfunction(psis, references, list(cfg.study.seeds), args.figure)
    print(out)
    print(args.figure)


if __name__ == "__main__":
    main()
