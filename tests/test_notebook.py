"""Static checks of the Kaggle launcher: it cannot be executed without a CUDA session."""

import ast
import json
import re

from pinn_cavity.config import CONFIG_DIR, REPO_ROOT, load_config

NOTEBOOK = REPO_ROOT / "kaggle" / "run_cuda.ipynb"


def _code():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_cells_parse_and_have_no_stored_output():
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == []


def test_full_study_is_off_by_default_and_guarded():
    code = _code()
    assert re.search(r"^RUN_FULL = False\b", code[0], re.M)
    cells = [c for c in code if "--config full --resume" in c]
    assert len(cells) == 1
    # the step starts only behind `is not True` and a budget comparison, never by reducing the config
    assert "if RUN_FULL is not True:" in cells[0]
    assert "> hours_left()" in cells[0] and "was not reduced" in cells[0]
    assert cells[0].index("if RUN_FULL is not True:") < cells[0].index("--config full --resume")


def test_tests_run_before_any_study():
    text = "\n".join(_code())
    assert text.index("pytest -q") < text.index("scripts/train.py")


def test_archive_names():
    text = "\n".join(_code())
    for name in ("full", "full-eval-bundle"):
        assert f"physicsnemo-pinn-cavity-{name}.zip" in text


def test_ref_must_be_a_full_sha():
    assert 'fullmatch(r"[0-9a-f]{40}", REF)' in _code()[0]


def test_referenced_scripts_and_configs_exist():
    text = "\n".join(_code())
    for script in set(re.findall(r"scripts/\w+\.py", text)):
        assert (REPO_ROOT / script).exists(), script
    for name in set(re.findall(r"--config (\w+)", text)):
        assert (CONFIG_DIR / f"{name}.yaml").exists(), name


def test_projection_matches_the_full_config():
    cfg = load_config("full")
    text = "\n".join(_code())
    assert f"n_runs = {len(cfg.study.reynolds) * len(cfg.study.seeds)}" in text
    assert f"* {cfg.training.steps} * n_runs" in text
    largest = max(cfg.study.reynolds)
    assert f"study.reynolds=[{largest}]" in text and f"full_probe/re{largest:04d}_seed0" in text


def test_official_run_is_part_of_the_full_study():
    official, full = load_config("official"), load_config("full")
    assert official.study.reynolds[0] in full.study.reynolds and official.study.seeds[0] in full.study.seeds
    assert official.training == full.training and official.model == full.model
