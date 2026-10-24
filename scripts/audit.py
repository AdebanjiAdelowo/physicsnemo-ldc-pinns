"""Independent recomputation of the reported results, and the diagnostics of the scientific audit.

    python scripts/audit.py --runs-root /path/to/runs

``--runs-root`` is a directory holding ``official/`` and ``local/`` run directories with
their checkpoints (they are not tracked). Nothing there is written. The script

1. recomputes the 513-node reference solutions and, with ``--reference-dir``, compares them
   with previously cached ones;
2. re-evaluates every checkpoint on CPU and compares every stored metric of
   ``results/<study>/<run>/eval_metrics.json`` with the recomputed value;
3. adds metrics that the stored records do not have: the velocity error with trapezoidal
   weights, with the neighbourhoods of the two lid corners removed, the boundary errors
   without the two corner nodes, a low-variance estimate of the training loss, and the
   internal state of a network whose output is constant;
4. scores trivial fields (zero, the best constant) and a family of analytic lid-layer fields
   with the training loss of the example and with the velocity error.

Output: ``results/audit/audit.json``. See docs/SCIENTIFIC_AUDIT.md.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from _bootstrap import ROOT
from pinn_cavity import engine
from pinn_cavity.config import load_config
from pinn_cavity.diagnostics import Constant, LidLayer, sampled_loss
from pinn_cavity.provenance import collect_metadata, write_json

DEVICE = torch.device("cpu")
TIMING_KEYS = {"inference_ms", "reference_file"}


def field_metrics(u, v, reference: dict, width: float, lid: float) -> dict:
    out = engine.corner_robust_metrics(u, v, reference, width, lid)
    out["lid_corner_u"] = [float(u[0, -1] / lid), float(u[-1, -1] / lid)]
    return out


@torch.no_grad()
def network_state(model, reference: dict) -> dict:
    """Spread of every hidden layer over the reference nodes (every fourth per direction)."""
    points = engine.grid_points(reference, 4, DEVICE)
    x = points
    layers = []
    for layer in model.layers:
        x = layer(x)
        spread = x.std(dim=0)
        layers.append(
            {
                "mean_neuron_spread": float(spread.mean()),
                "neurons_with_zero_spread": int((spread == 0).sum()),
                "neurons_identically_zero": int((x.abs().max(dim=0).values == 0).sum()),
                "neurons": int(x.shape[1]),
            }
        )
    out = model(points)
    return {
        "hidden_layers": layers,
        "output_spread": [float(s) for s in out.std(dim=0)],
        "output_mean": [float(m) for m in out.mean(dim=0)],
        "weight_norms": [float(layer.linear.weight.norm()) if hasattr(layer, "linear") else None for layer in model.layers],
    }


def compare_stored(stored: dict, recomputed: dict) -> dict:
    worst, rows = 0.0, {}
    for key, value in stored.items():
        if key in TIMING_KEYS or not isinstance(value, (int, float)):
            continue
        new = recomputed[key]
        diff = abs(new - value) / max(abs(value), 1e-12) if value != 0 else abs(new)
        rows[key] = {"stored": value, "recomputed": new, "relative_difference": diff}
        worst = max(worst, diff)
    return {"worst_relative_difference": worst, "metrics": rows}


def reference_section(cfg, reynolds_list, reference_dir: Path | None) -> tuple[dict, dict]:
    refs, report = {}, {}
    for re in reynolds_list:
        tick = time.perf_counter()
        refs[re] = engine.load_reference(cfg, re, ROOT)
        entry = {"file": refs[re]["file"], "seconds": time.perf_counter() - tick, "newton_final_residual": float(refs[re]["newton_final_residual"])}
        if reference_dir is not None and (reference_dir / refs[re]["file"]).exists():
            with np.load(reference_dir / refs[re]["file"]) as other:
                entry["max_abs_difference_to_cached"] = float(max(np.abs(other[k] - refs[re][k]).max() for k in ("u", "v", "psi")))
        report[str(re)] = entry
        print(f"[reference] Re {re}: {entry}", flush=True)
    return refs, report


def degenerate_section(cfg, refs: dict, interior: int, boundary: int, batches: int) -> dict:
    width, lid = float(cfg.problem.width), float(cfg.problem.lid_velocity)
    points = engine.grid_points(next(iter(refs.values())), 1, DEVICE)
    out = {"constant": {}, "lid_layer": {}}
    for re, reference in refs.items():
        truth = np.stack([reference["u"].ravel(), reference["v"].ravel()], axis=1)
        entries = {}
        for name, model in (("zero", Constant(0.0)), ("best_constant", Constant(1.0 / 3.0))):
            pred = engine.predict(model, points)[:, :2]
            entries[name] = {
                "velocity_rel_l2": engine.relative_l2(pred, truth),
                "loss": sampled_loss(model, cfg, re, interior, boundary, batches, seed=11),
            }
        out["constant"][str(re)] = entries
        grid = []
        for delta in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
            for eps in (0.002, 0.005, 0.01, 0.02, 0.05):
                model = LidLayer(width, lid, delta * width, eps * width)
                pred = engine.predict(model, points)[:, :2]
                loss = sampled_loss(model, cfg, re, interior, boundary, batches, seed=7)
                grid.append({"delta_over_width": delta, "eps_over_width": eps, "loss": loss, "velocity_rel_l2": engine.relative_l2(pred, truth)})
        best = min(grid, key=lambda g: g["loss"]["loss"])
        out["lid_layer"][str(re)] = {"best": best, "grid": grid}
        print(
            f"[lid layer] Re {re}: lowest loss {best['loss']['loss']:.3e} at delta/L={best['delta_over_width']}, "
            f"eps/L={best['eps_over_width']}, velocity error {best['velocity_rel_l2']:.3f}",
            flush=True,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, required=True, help="directory with official/ and local/ run directories")
    parser.add_argument("--reference-dir", type=Path, default=None, help="cached reference solutions to compare with")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--loss-batches", type=int, default=25, help="independent batches at the upstream point counts")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "audit" / "audit.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    overrides = ["device=cpu", "evaluation.timing_warmup=0", "evaluation.timing_repeats=1"]
    upstream = load_config("official", overrides)
    report = {"metadata": collect_metadata(DEVICE), "runs_root": str(args.runs_root), "runs": {}}

    refs, report["references"] = reference_section(upstream, [10, 100, 400], args.reference_dir)

    for study in ("official", "local"):
        cfg = load_config(study, overrides)
        for reynolds, seed in engine.study_runs(cfg):
            name = engine.run_name(reynolds, seed)
            run_dir = args.runs_root / study / name
            stored = json.loads((ROOT / "results" / study / name / "eval_metrics.json").read_text())
            train = json.loads((ROOT / "results" / study / name / "train_metrics.json").read_text())
            metrics, fields = engine.evaluate_run(cfg, reynolds, run_dir, DEVICE, refs[reynolds], write=False)
            model = engine.build_model(cfg)
            engine.load_checkpoint(run_dir / "checkpoints", models=model, device=DEVICE)
            model.eval()
            entry = {
                "check": compare_stored(stored, metrics),
                "additional": field_metrics(fields["u"], fields["v"], refs[reynolds], float(cfg.problem.width), float(cfg.problem.lid_velocity)),
                # upstream point counts for every study, so that the estimates are comparable
                "loss_estimate": sampled_loss(model, upstream, reynolds, 4000, 2000, args.loss_batches, seed=1000 + seed),
                "final_train_loss_single_batch": train["final_train_loss"],
                "network_state": network_state(model, refs[reynolds]),
            }
            report["runs"][f"{study}/{name}"] = entry
            print(
                f"[run] {study}/{name}: worst stored-vs-recomputed {entry['check']['worst_relative_difference']:.1e}, "
                f"loss estimate {entry['loss_estimate']['loss']:.3e} +- {entry['loss_estimate']['loss_standard_error']:.1e}",
                flush=True,
            )

    report["trivial_and_degenerate_fields"] = degenerate_section(upstream, refs, 40000, 20000, 10)
    write_json(args.out, report)
    print(args.out)


if __name__ == "__main__":
    main()
