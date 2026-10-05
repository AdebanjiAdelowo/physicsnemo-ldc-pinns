"""Configurations: upstream values are preserved and the studies differ only where stated."""

import re

import pytest
from omegaconf import OmegaConf
from pinn_cavity import engine
from pinn_cavity.config import CONFIG_DIR, load_config
from pinn_cavity.equations import viscosity

UPSTREAM_KEYS = {
    "arch": {
        "decoder": {"out_features": 3, "layers": 1, "layer_size": 32},
        "fno": {"in_channels": 2, "dimension": 2, "latent_channels": 32, "fno_layers": 4, "fno_modes": 12, "padding": 9},
    },
    "scheduler": {"initial_lr": 1.0e-3, "decay_rate": 0.9995},
}
# constants written in the upstream train.py
UPSTREAM_CONSTANTS = {
    "problem": {"width": 0.1, "height": 0.1, "lid_velocity": 1.0, "rho": 1.0},
    "model": {"num_layers": 6, "layer_size": 512},
    "geometry": {"mesh_points_per_side": 50},
}


def test_official_keeps_upstream_values():
    cfg = OmegaConf.to_container(load_config("official"), resolve=True)
    for key, value in {**UPSTREAM_KEYS, **UPSTREAM_CONSTANTS}.items():
        assert cfg[key] == value
    t = cfg["training"]
    assert (t["steps"], t["interior_points"], t["boundary_points"]) == (10000, 4000, 2000)
    assert t["lr_decay_per_step"] == 0.9999871767586216
    assert cfg["study"] == {"reynolds": [10], "seeds": [0]}


def test_official_file_starts_with_the_upstream_text():
    text = (CONFIG_DIR / "official.yaml").read_text()
    head = text.split("# -----", 1)[0]
    assert head.startswith("# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.")
    assert head.rstrip().endswith("decay_rate: .9995")
    assert "name:" not in head and "study:" not in head


def test_upstream_reynolds_number_is_ten():
    cfg = load_config("official")
    assert viscosity(10, cfg.problem.lid_velocity, cfg.problem.width) == 0.01  # the nu of the upstream train.py
    with pytest.raises(ValueError):
        viscosity(0, 1.0, 0.1)


def test_full_differs_from_official_only_in_the_study():
    official = OmegaConf.to_container(load_config("official"), resolve=True)
    full = OmegaConf.to_container(load_config("full"), resolve=True)
    assert full["study"] == {"reynolds": [10, 100, 400], "seeds": [0, 1, 2]}
    for key in official:
        if key not in ("study", "name", "output_dir"):
            assert full[key] == official[key], key


def test_local_keeps_architecture_and_optimiser():
    official, local = load_config("official"), load_config("local")
    for key in ("problem", "model", "geometry", "scheduler", "reference", "evaluation"):
        assert local[key] == official[key], key
    assert local.training.lr_decay_per_step == official.training.lr_decay_per_step
    assert local.study.reynolds == load_config("full").study.reynolds


@pytest.mark.parametrize("name", ["official", "smoke", "local", "full"])
def test_configs_validate(name):
    cfg = load_config(name)
    engine.validate_study(cfg)
    assert cfg.output_dir == f"runs/{name}"
    runs = [engine.run_name(r, s) for r, s in engine.study_runs(cfg)]
    assert len(set(runs)) == len(runs) and all(re.fullmatch(r"re\d{4}_seed\d+", r) for r in runs)


@pytest.mark.parametrize(
    "override",
    ["study.reynolds=[0]", "study.reynolds=[12.5]", "problem.height=0.2", "reference.grid=64", "training.steps=45"],
)
def test_invalid_settings_are_rejected(override):
    with pytest.raises(ValueError):
        engine.validate_study(load_config("smoke", [override]))
