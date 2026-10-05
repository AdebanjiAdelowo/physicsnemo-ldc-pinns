"""End-to-end: a tiny study is trained, evaluated, verified, packaged and plotted."""

import csv
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
from pinn_cavity import engine, plotting, study
from pinn_cavity.config import REPO_ROOT, load_config

OVERRIDES = ["training.steps=20", "training.log_every=10", "training.error_every=10", "study.reynolds=[10,100]"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    cfg = load_config("smoke", OVERRIDES)
    study.train_study(cfg, root)
    return cfg, root, study.evaluate_study(cfg, root)


def test_records_are_written(trained):
    cfg, root, summary = trained
    out = root / cfg.output_dir
    meta = json.loads((out / "study_metadata.json").read_text())
    assert meta["config"]["training"]["steps"] == 20 and set(meta["references"]) == {"10", "100"}
    assert meta["physicsnemo"]["version"] and meta["torch_version"]
    for run in ("re0010_seed0", "re0100_seed0"):
        rows = list(csv.DictReader(open(out / run / "history.csv")))
        assert [int(r["step"]) for r in rows] == [1, 10, 20]
        assert rows[0]["velocity_rel_l2"] == "" and float(rows[-1]["velocity_rel_l2"]) > 0
        total = sum(float(rows[-1][name]) for name in engine.LOSS_TERMS)
        assert total == pytest.approx(float(rows[-1]["loss"]), rel=1e-5)
        train = json.loads((out / run / "train_metrics.json").read_text())
        assert train["optimiser_steps"] == 20 and train["viscosity"] == pytest.approx(0.1 / train["reynolds"])
    assert [g["reynolds"] for g in summary["by_reynolds"]] == [10, 100]
    assert len(list(csv.DictReader(open(out / "summary.csv")))) == 2


def test_metrics_are_consistent(trained):
    cfg, root, summary = trained
    out = root / cfg.output_dir
    fields = np.load(out / "fields.npz")
    for run in summary["runs"]:
        assert np.isfinite([v for v in run.values() if isinstance(v, float)]).all()
        tag = f"re{run['reynolds']:04d}"
        # the stored error is recovered from the stored fields (the smoke grid is not sub-sampled)
        pred = np.stack([fields[f"{tag}_seed0_u"], fields[f"{tag}_seed0_v"]])
        truth = np.stack([fields[f"{tag}_u_reference"], fields[f"{tag}_v_reference"]])
        assert engine.relative_l2(pred, truth) == pytest.approx(run["velocity_rel_l2"], rel=1e-5)
    # Ghia values exist for Re = 100 and not for Re = 10
    by_re = {r["reynolds"]: r for r in summary["runs"]}
    assert by_re[10]["ghia_u_rms_difference"] is None and by_re[100]["ghia_u_rms_difference"] > 0


def test_last_tracked_error_matches_the_evaluation(trained):
    cfg, root, summary = trained
    out = root / cfg.output_dir
    for run in summary["runs"]:
        rows = list(csv.DictReader(open(out / run["run"] / "history.csv")))
        assert float(rows[-1]["velocity_rel_l2"]) == pytest.approx(run["velocity_rel_l2"], rel=1e-5)


def test_verify_reproduces_the_stored_errors(trained):
    cfg, root, _ = trained
    before = (root / cfg.output_dir / "summary.json").read_text()
    rows = study.verify_study(cfg, root)
    assert max(r["relative_difference"] for r in rows) < 1e-6
    assert (root / cfg.output_dir / "summary.json").read_text() == before


def test_training_is_reproducible_on_cpu(trained, tmp_path):
    cfg, root, summary = trained
    reference = engine.load_reference(cfg, 10, root)
    again = engine.train_run(cfg, 10, 0, tmp_path / "again", engine.resolve_device("cpu"), reference)
    assert again["final_train_loss"] == pytest.approx(summary["runs"][0]["final_train_loss"], rel=1e-5)
    other = engine.train_run(cfg, 10, 1, tmp_path / "other", engine.resolve_device("cpu"), reference)
    assert other["final_train_loss"] != again["final_train_loss"]


def test_evaluation_rejects_a_checkpoint_of_another_length(trained):
    cfg, root, _ = trained
    longer = load_config("smoke", OVERRIDES[1:] + ["training.steps=40"])
    with pytest.raises(RuntimeError, match="checkpoint is at step 20"):
        engine.evaluate_run(longer, 10, root / cfg.output_dir / "re0010_seed0", engine.resolve_device("cpu"), engine.load_reference(cfg, 10, root), write=False)


def test_figures(trained, tmp_path):
    cfg, root, _ = trained
    paths = plotting.plot_all(root / cfg.output_dir, tmp_path, prefix="t")
    assert len(paths) == 5 and all(p.stat().st_size > 5000 for p in paths)


def test_package_bundle_and_publish(trained, tmp_path):
    cfg, root, _ = trained
    work = tmp_path / "repo" / "runs" / "smoke"
    shutil.copytree(root / cfg.output_dir, work)
    script = str(REPO_ROOT / "scripts" / "results.py")
    for command in ("package", "bundle"):
        subprocess.run([sys.executable, script, command, str(work)], check=True, capture_output=True)
    import zipfile

    names = zipfile.ZipFile(work.parent / "smoke.zip").namelist()
    assert "smoke/summary.json" in names and not any("checkpoints" in n for n in names)
    bundled = zipfile.ZipFile(work.parent / "smoke_eval_bundle.zip").namelist()
    assert len(bundled) == 4 and all(n.startswith("runs/smoke/re0") and ".20." in n for n in bundled)
