"""Move study results between a run directory, an archive and the tracked ``results/``.

    python scripts/results.py package runs/full              # on the GPU machine: runs/full.zip
    python scripts/results.py bundle runs/full               # on the GPU machine: runs/full_eval_bundle.zip
    python scripts/results.py publish runs/full.zip          # locally: results/full/
    python scripts/results.py publish runs/local             # from a local run directory
    python scripts/results.py merge runs/full <session>...   # locally: sessions of one study into runs/full

``package`` holds the small records that are tracked in ``results/``; checkpoints are never
included. ``bundle`` holds what a re-evaluation on another machine needs and is not tracked:
the final checkpoint of every run, with paths relative to the repository root, so that
``unzip <bundle> -d .`` restores them. The reference solutions are not in the bundle; they are
recomputed where they are needed. ``publish`` refuses to overwrite an existing
``results/<name>/`` unless ``--force`` is given.

``merge`` combines the outputs of several GPU sessions of one study, each the downloaded
``/kaggle/working`` of a session that trained some of its Reynolds numbers. Run directories,
with their checkpoints, are copied into the target; the evaluation made on the GPU is kept as
``eval_metrics_cuda.json``, because a later evaluation on another machine rewrites
``eval_metrics.json``; a session whose evaluation did not finish is recorded with an empty
``evaluated_on_gpu``. Every session's metadata, summary, ``nvidia-smi`` and ``pip freeze``
output go to ``sessions/<name>/``. The merged ``study_metadata.json`` is the first session's,
with the Reynolds numbers of all sessions and the list of sessions.
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from _bootstrap import ROOT

KEEP_SUFFIXES = {".json", ".csv", ".npz", ".txt"}


def tracked_files(study_dir: Path) -> list[Path]:
    """Summaries, metadata, histories and plotting fields: everything except checkpoints and figures."""
    files = [p for p in sorted(study_dir.rglob("*")) if p.is_file() and p.suffix in KEEP_SUFFIXES and "checkpoints" not in p.parts]
    if not (study_dir / "summary.json").exists():
        raise SystemExit(f"{study_dir} has no summary.json; run scripts/evaluate.py first.")
    return files


def package(study_dir: Path) -> Path:
    archive = study_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in tracked_files(study_dir):
            z.write(path, Path(study_dir.name) / path.relative_to(study_dir))
    return archive


def bundle(study_dir: Path) -> Path:
    meta = json.loads((study_dir / "study_metadata.json").read_text())
    steps = meta["config"]["training"]["steps"]
    files = sorted(p for p in study_dir.glob(f"*/checkpoints/*.{steps}.*") if p.is_file())
    if not any(p.suffix == ".mdlus" for p in files):
        raise SystemExit(f"cannot build the bundle: no checkpoint of step {steps} under {study_dir}")
    archive = study_dir.parent / f"{study_dir.name}_eval_bundle.zip"
    root = study_dir.parents[1]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for path in files:
            z.write(path, path.relative_to(root))
    return archive


def publish(source: Path, force: bool) -> Path:
    name = source.stem if source.suffix == ".zip" else source.name
    target = ROOT / "results" / name
    if target.exists():
        if not force:
            raise SystemExit(f"{target} exists; pass --force to replace it.")
        shutil.rmtree(target)
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as z:
            members = [m for m in z.namelist() if m.startswith(f"{name}/") and Path(m).suffix in KEEP_SUFFIXES]
            z.extractall(ROOT / "results", members)
    else:
        for path in tracked_files(source):
            dest = target / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    return target


def merge(target: Path, sessions: list[Path]) -> Path:
    name = target.name
    merged = None
    for session in sessions:
        study = session / "physicsnemo-ldc-pinns" / "runs" / name
        meta = json.loads((study / "study_metadata.json").read_text())
        reynolds = list(meta["config"]["study"]["reynolds"])
        tag = "re" + "-".join(str(r) for r in reynolds)
        record = target / "sessions" / tag
        record.mkdir(parents=True, exist_ok=True)
        shutil.copy2(study / "study_metadata.json", record / "study_metadata.json")
        # a session whose evaluation did not finish has no summary and no eval_metrics.json
        for path in [study / "summary.json", study / "summary.csv", session / "nvidia-smi.txt", session / "pip-freeze.txt"]:
            if path.exists():
                shutil.copy2(path, record / path.name)
        evaluated = []
        for run in sorted(p for p in study.iterdir() if (p / "train_metrics.json").exists()):
            if (target / run.name).exists():
                raise SystemExit(f"{target / run.name} exists; a run must come from exactly one session.")
            shutil.copytree(run, target / run.name)
            if (run / "eval_metrics.json").exists():
                shutil.copy2(run / "eval_metrics.json", target / run.name / "eval_metrics_cuda.json")
                evaluated.append(run.name)
        if merged is None:
            merged = {**meta, "sessions": []}
            merged["config"]["study"]["reynolds"] = []
        merged["config"]["study"]["reynolds"] = sorted(merged["config"]["study"]["reynolds"] + reynolds)
        merged["sessions"].append(
            {"name": tag, "git_commit": meta["git_commit"], "gpu_name": meta["gpu_name"], "timestamp_utc": meta["timestamp_utc"], "evaluated_on_gpu": evaluated}
        )
    (target / "study_metadata.json").write_text(json.dumps(merged, indent=2) + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("package", help="zip the small result files of a study directory")
    p.add_argument("study", type=Path)
    p = sub.add_parser("bundle", help="zip the final checkpoints")
    p.add_argument("study", type=Path)
    p = sub.add_parser("publish", help="copy a study directory or archive into results/")
    p.add_argument("source", type=Path)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("merge", help="combine the downloaded outputs of several GPU sessions of one study")
    p.add_argument("target", type=Path)
    p.add_argument("sessions", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "package":
        print(package(args.study.resolve()))
    elif args.command == "bundle":
        print(bundle(args.study.resolve()))
    elif args.command == "merge":
        print(merge(args.target.resolve(), [s.resolve() for s in args.sessions]))
    else:
        print(publish(args.source.resolve(), args.force))


if __name__ == "__main__":
    main()
