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
    # named after this repository, not the sibling repository physicsnemo-pinn-cavity
    text = "\n".join(_code())
    for name in ("full-{TAG}", "full-{TAG}-eval-bundle"):
        assert f"physicsnemo-ldc-pinns-{name}.zip" in text
    assert "physicsnemo-pinn-cavity" not in text


def test_clones_this_repository():
    # a launcher pointed at another repository cannot check out commits that exist only here
    cff = (REPO_ROOT / "CITATION.cff").read_text()
    url = re.search(r'^repository-code: "(.+)"$', cff, re.M).group(1)
    assert f'REPO_URL = "{url}.git"' in _code()[0]


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
    code = _code()
    text = "\n".join(code)
    assert f"REYNOLDS = {list(cfg.study.reynolds)}" in code[0]
    allowed = "{" + ", ".join(str(r) for r in sorted(cfg.study.reynolds)) + "}"
    assert f"set(REYNOLDS) <= {allowed}" in code[0]
    assert f"n_runs = len(REYNOLDS) * {len(cfg.study.seeds)}" in text
    assert f"* {cfg.training.steps} * n_runs" in text
    # the probe runs at the largest Reynolds number of the session
    assert "study.reynolds=[{max(REYNOLDS)}]" in text and "full_probe/re{max(REYNOLDS):04d}_seed0" in text


def test_session_trains_only_its_reynolds_numbers():
    code = _code()
    namespace = {}
    exec(code[0].replace('REF = "SET_TO_FULL_40_CHARACTER_COMMIT_SHA"', 'REF = "' + "0" * 40 + '"').replace("REYNOLDS = [10, 100, 400]", "REYNOLDS = [400, 100]"), namespace)
    assert namespace["RE_OVERRIDE"] == "study.reynolds=[100,400]" and namespace["TAG"] == "re100-400"
    step_b = next(c for c in code if "--config full --resume" in c)
    assert "--config full --resume device=cuda {RE_OVERRIDE}" in step_b


def test_official_run_is_part_of_the_full_study():
    official, full = load_config("official"), load_config("full")
    assert official.study.reynolds[0] in full.study.reynolds and official.study.seeds[0] in full.study.seeds
    assert official.training == full.training and official.model == full.model


def test_environment_is_the_locked_one():
    # the image's packages mixed versions with pip; the launcher installs uv.lock exactly
    text = "\n".join(_code())
    assert "uv sync --frozen" in text and "pip install -q -r requirements.txt" not in text
    for line in text.splitlines():
        assert not line.lstrip().startswith('sh("python '), line
