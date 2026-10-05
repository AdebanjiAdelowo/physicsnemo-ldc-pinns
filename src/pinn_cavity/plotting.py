"""Figures of a study directory (``summary.json``, ``fields.npz`` and the run histories)."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import reference as ref

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
# one hue, light to dark, for the ordered quantity "Reynolds number"
RE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
FIELD_CMAP, ERROR_CMAP = "viridis", "magma"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    }
)


def reynolds_colours(values: list[int]) -> dict:
    """Colour of every Reynolds number, by its position in the sorted list."""
    values = sorted(values)
    if len(values) > len(RE_RAMP):
        raise ValueError(f"the ramp has {len(RE_RAMP)} steps; got {len(values)} Reynolds numbers")
    return dict(zip(values, RE_RAMP[len(RE_RAMP) - len(values) :]))


def load_study(study_dir: Path) -> dict:
    study_dir = Path(study_dir)
    summary = json.loads((study_dir / "summary.json").read_text())
    histories = {}
    for run in summary["runs"]:
        with open(study_dir / run["run"] / "history.csv") as f:
            rows = list(csv.DictReader(f))
        histories[run["run"]] = {key: np.array([float(r[key]) if r[key] != "" else np.nan for r in rows]) for key in rows[0]}
    return {
        "summary": summary,
        "histories": histories,
        "fields": dict(np.load(study_dir / "fields.npz")),
        "metadata": json.loads((study_dir / "study_metadata.json").read_text()),
    }


def _describe(study: dict) -> str:
    cfg = study["metadata"]["config"]
    t = cfg["training"]
    return (
        f"{cfg['model']['num_layers']} x {cfg['model']['layer_size']} network, {t['steps']} steps, "
        f"{t['interior_points']} interior and {t['boundary_points']} boundary points per step, {study['metadata']['device']}"
    )


def _reynolds(study: dict) -> list[int]:
    return [e["reynolds"] for e in study["summary"]["by_reynolds"]]


def plot_error_vs_reynolds(study: dict, path: Path) -> None:
    """Velocity error and PDE residual of every run against the Reynolds number."""
    runs, groups = study["summary"]["runs"], study["summary"]["by_reynolds"]
    values = _reynolds(study)
    colours = reynolds_colours(values)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    panels = [
        ("velocity_rel_l2", "Relative $L^2$ error of the velocity", "Error against the reference solution"),
        ("residual_rms_momentum_x", "RMS residual of the $x$ momentum equation", "PDE residual of the trained network"),
    ]
    for ax, (key, label, title) in zip(axes, panels):
        ax.plot(values, [g[f"{key}_mean"] for g in groups], color=MUTED, linewidth=1.2, zorder=1)
        for r in runs:
            ax.scatter(r["reynolds"], r[key], s=52, color=colours[r["reynolds"]], edgecolor="white", linewidth=1.2, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(values, [str(v) for v in values])
        ax.minorticks_off()
        ax.set_xlabel("Reynolds number")
        ax.set_ylabel(label)
        ax.set_title(title, loc="left")
    fig.suptitle(f"One point per seed, line through the means. {_describe(study)}", x=0.01, ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_fields(study: dict, path: Path) -> None:
    """Reference speed, predicted speed and error of the first seed, one row per Reynolds number."""
    f = study["fields"]
    values = _reynolds(study)
    seed = study["metadata"]["config"]["study"]["seeds"][0]
    lid = study["metadata"]["config"]["problem"]["lid_velocity"]
    extent = [f["x"][0], f["x"][-1], f["y"][0], f["y"][-1]]
    fig, axes = plt.subplots(len(values), 3, figsize=(9.6, 3.0 * len(values)), squeeze=False)
    for row, re in zip(axes, values):
        tag = f"re{re:04d}"
        speed_ref = np.hypot(f[f"{tag}_u_reference"], f[f"{tag}_v_reference"]) / lid
        u, v = f[f"{tag}_seed{seed}_u"].astype(float), f[f"{tag}_seed{seed}_v"].astype(float)
        error = np.hypot(u - f[f"{tag}_u_reference"], v - f[f"{tag}_v_reference"]) / lid
        panels = [
            (speed_ref, FIELD_CMAP, 0.0, 1.0, f"Re = {re}: reference speed"),
            (np.hypot(u, v) / lid, FIELD_CMAP, 0.0, 1.0, "PINN speed"),
            (error, ERROR_CMAP, 0.0, None, "Magnitude of the velocity error"),
        ]
        for ax, (data, cmap, vmin, vmax, title) in zip(row, panels):
            im = ax.imshow(data.T, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            ax.set_title(title, loc="left")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Seed {seed}, speeds in units of the lid speed. {_describe(study)}", x=0.01, ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_centrelines(study: dict, path: Path) -> None:
    """Velocity profiles on the two centrelines: reference, tabulated values and every seed."""
    f = study["fields"]
    cfg = study["metadata"]["config"]
    values = _reynolds(study)
    colours = reynolds_colours(values)
    lid, width = cfg["problem"]["lid_velocity"], cfg["problem"]["width"]
    s = (f["x"] - f["x"][0]) / width
    mid = len(s) // 2
    fig, axes = plt.subplots(2, len(values), figsize=(3.5 * len(values), 6.4), squeeze=False, sharey="row")
    for col, re in enumerate(values):
        tag = f"re{re:04d}"
        for row, (label, pick, table_s, table) in enumerate(
            [
                ("$u / U$ on the vertical centreline", lambda a: a[mid, :], ref.GHIA_Y, ref.GHIA_U),
                ("$v / U$ on the horizontal centreline", lambda a: a[:, mid], ref.GHIA_X, ref.GHIA_V),
            ]
        ):
            ax = axes[row, col]
            for i, seed in enumerate(cfg["study"]["seeds"]):
                ax.plot(s, pick(f[f"{tag}_seed{seed}_{'uv'[row]}"]) / lid, color=colours[re], linewidth=1.6, label="PINN, one line per seed" if i == 0 else None)
            ax.plot(s, pick(f[f"{tag}_{'uv'[row]}_reference"]) / lid, color=INK, linewidth=1.3, linestyle="--", label="finite-difference reference")
            if re in table:
                ax.scatter(table_s, table[re], s=22, facecolor="white", edgecolor=INK, linewidth=1.0, zorder=4, label="Ghia et al. (1982)")
            ax.set_xlabel("$y / L$" if row == 0 else "$x / L$")
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f"Re = {re}", loc="left")
            if row == 0 and col == len(values) - 1:
                ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(_describe(study), x=0.01, ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_training_curves(study: dict, path: Path) -> None:
    """Training loss and velocity error against the optimiser step, one line per run."""
    values = _reynolds(study)
    colours = reynolds_colours(values)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    seen = set()
    for run in study["summary"]["runs"]:
        h, re = study["histories"][run["run"]], run["reynolds"]
        label = f"Re = {re}" if re not in seen else None
        seen.add(re)
        axes[0].plot(h["step"], h["loss"], color=colours[re], linewidth=1.3, label=label)
        tracked = ~np.isnan(h["velocity_rel_l2"])
        axes[1].plot(h["step"][tracked], h["velocity_rel_l2"][tracked], color=colours[re], linewidth=1.3, marker="o", markersize=3.5, label=label)
    for ax, label, title in zip(
        axes,
        ("Training loss", "Relative $L^2$ error of the velocity"),
        ("Loss of the upstream example", "Error against the reference solution"),
    ):
        ax.set_yscale("log")
        ax.set_xlabel("Optimiser step")
        ax.set_ylabel(label)
        ax.set_title(title, loc="left")
    axes[0].legend(fontsize=9)
    fig.suptitle(f"One line per run. {_describe(study)}", x=0.01, ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_loss_terms(study: dict, path: Path) -> None:
    """Final value of every term of the loss, mean over seeds, by Reynolds number."""
    values = _reynolds(study)
    colours = reynolds_colours(values)
    terms = ["continuity", "momentum_x", "momentum_y", "u_no_slip", "v_no_slip", "u_lid", "v_lid"]
    names = ["continuity", "momentum $x$", "momentum $y$", "$u$, fixed walls", "$v$, fixed walls", "$u$, lid", "$v$, lid"]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    group_width = 0.8
    bar = group_width / len(values)
    for k, re in enumerate(values):
        runs = [r["run"] for r in study["summary"]["runs"] if r["reynolds"] == re]
        # mean over the last tenth of the history: a single step is noisy
        means = []
        for term in terms:
            tails = [study["histories"][run][term][-max(1, len(study["histories"][run][term]) // 10) :].mean() for run in runs]
            means.append(float(np.mean(tails)))
        x = np.arange(len(terms)) - group_width / 2 + (k + 0.5) * bar
        ax.bar(x, means, width=bar * 0.88, color=colours[re], label=f"Re = {re}")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(terms)), names)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Loss term, last tenth of training")
    ax.set_title("Terms of the training loss", loc="left")
    ax.legend(fontsize=9, ncols=len(values))
    fig.suptitle(f"Mean over seeds. {_describe(study)}", x=0.01, ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_reference_check(check: dict, path: Path) -> None:
    """Centreline profiles of the reference solver with the tabulated values, and its grid convergence."""
    values = sorted(int(r) for r in check["profiles"])
    colours = reynolds_colours(values)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    for re in values:
        p = check["profiles"][str(re)]
        axes[0].plot(p["s"], p["u_vertical"], color=colours[re], label=f"Re = {re}")
        axes[1].plot(p["s"], p["v_horizontal"], color=colours[re])
        if re in ref.GHIA_U:
            axes[0].scatter(ref.GHIA_Y, ref.GHIA_U[re], s=20, facecolor="white", edgecolor=INK, linewidth=0.9, zorder=4)
            axes[1].scatter(ref.GHIA_X, ref.GHIA_V[re], s=20, facecolor="white", edgecolor=INK, linewidth=0.9, zorder=4)
        c = check["grid_convergence"][str(re)]
        axes[2].plot(c["grids"], c["velocity_rel_l2_to_finest"], color=colours[re], marker="o")
        axes[2].plot(c["grids"], c["velocity_rel_l2_core_to_finest"], color=colours[re], marker="s", linestyle=":")
    axes[0].scatter([], [], s=20, facecolor="white", edgecolor=INK, linewidth=0.9, label="Ghia et al. (1982)")
    axes[0].legend(fontsize=8.5)
    axes[0].set_xlabel("$y / L$")
    axes[0].set_ylabel("$u / U$")
    axes[0].set_title("Vertical centreline", loc="left")
    axes[1].set_xlabel("$x / L$")
    axes[1].set_ylabel("$v / U$")
    axes[1].set_title("Horizontal centreline", loc="left")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    grids = check["grid_convergence"][str(values[0])]["grids"]
    axes[2].set_xticks(grids, [str(g) for g in grids])
    axes[2].minorticks_off()
    axes[2].set_xlabel("Nodes per side")
    axes[2].set_ylabel(f"Relative $L^2$ difference to the {check['finest_grid']}-node grid")
    axes[2].set_title("Grid convergence of the velocity", loc="left")
    axes[2].plot([], [], color=MUTED, marker="o", label="whole cavity")
    axes[2].plot([], [], color=MUTED, marker="s", linestyle=":", label="central 80% per direction")
    axes[2].legend(fontsize=8.5)
    fig.suptitle(
        f"Finite-difference reference solver, profiles on the {check['profile_grid']}-node grid", x=0.01, ha="left", fontsize=8.5, color=MUTED
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_all(study_dir: Path, out_dir: Path, prefix: str = "") -> list[Path]:
    study = load_study(study_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""
    figures = [
        ("error_vs_reynolds", plot_error_vs_reynolds),
        ("fields", plot_fields),
        ("centrelines", plot_centrelines),
        ("training_curves", plot_training_curves),
        ("loss_terms", plot_loss_terms),
    ]
    paths = []
    for name, function in figures:
        paths.append(out_dir / f"{stem}{name}.png")
        function(study, paths[-1])
    return paths
