"""Check the finite-difference reference solver before it is used to score the networks.

    python scripts/verify_reference.py

Three checks, written to ``results/reference_check.json`` and ``figures/reference_check.png``:

1. grid convergence of the velocity field on 129, 257 and 513 nodes per side;
2. extrema of the centreline profiles and of the stream function, Richardson-extrapolated
   from the two finest grids, against published high-accuracy values;
3. centreline profiles against the values tabulated by Ghia, Ghia and Shin (1982).
"""

import argparse
from pathlib import Path

import numpy as np
from _bootstrap import ROOT
from pinn_cavity import reference as ref
from pinn_cavity.plotting import plot_reference_check
from pinn_cavity.provenance import write_json
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

# Re = 1000: Botella & Peyret, Computers & Fluids 27 (1998), Chebyshev collocation.
# Re = 100: Marchi, Suero & Araki, J. Braz. Soc. Mech. Sci. Eng. 31 (2009), 1024 x 1024 grid
# with Richardson extrapolation.
PUBLISHED = {
    100: {"u_min": -0.2140417, "v_max": 0.1795728, "v_min": -0.2538030},
    1000: {"u_min": -0.3885698, "y_u_min": 0.1717, "v_max": 0.3769447, "x_v_max": 0.1578, "v_min": -0.5270771, "x_v_min": 0.9092, "psi_min": -0.1189366},
}


def extrema(sol: dict) -> dict:
    """Extrema of the centreline profiles from cubic splines through the nodal values."""
    lines = ref.centrelines(sol)
    u, v = CubicSpline(lines["s"], lines["u_vertical"]), CubicSpline(lines["s"], lines["v_horizontal"])
    a = minimize_scalar(u, bounds=(0.05, 0.7), method="bounded", options={"xatol": 1e-10})
    b = minimize_scalar(lambda x: -v(x), bounds=(0.05, 0.5), method="bounded", options={"xatol": 1e-10})
    c = minimize_scalar(v, bounds=(0.5, 0.98), method="bounded", options={"xatol": 1e-10})
    return {
        "u_min": float(a.fun), "y_u_min": float(a.x), "v_max": float(-b.fun), "x_v_max": float(b.x),
        "v_min": float(c.fun), "x_v_min": float(c.x), "psi_min": float(sol["psi"].min()),
    }  # fmt: skip


def velocity_difference(coarse: dict, fine: dict, core: bool = False) -> float:
    """Relative L2 difference of the velocity on the nodes of the coarse grid.

    ``core=True`` restricts it to nodes at least a tenth of the width away from every wall.
    """
    n = coarse["u"].shape[0]
    k = (fine["u"].shape[0] - 1) // (n - 1)
    s = np.linspace(0.0, 1.0, n)
    keep = (s >= 0.1) & (s <= 0.9) if core else np.ones(n, dtype=bool)
    pick = np.ix_(keep, keep)
    uf, vf = fine["u"][::k, ::k][pick], fine["v"][::k, ::k][pick]
    return float(np.sqrt(((coarse["u"][pick] - uf) ** 2 + (coarse["v"][pick] - vf) ** 2).sum() / (uf**2 + vf**2).sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reynolds", type=int, nargs="+", default=[10, 100, 400, 1000])
    parser.add_argument("--grids", type=int, nargs="+", default=[129, 257, 513], help="nodes per side, each 2 * previous - 1")
    parser.add_argument("--out-root", type=Path, default=ROOT)
    args = parser.parse_args()
    grids = sorted(args.grids)
    if any(b != 2 * a - 1 for a, b in zip(grids, grids[1:])):
        raise SystemExit("every grid must have 2 n - 1 nodes of the previous one")

    check = {"grids": grids, "finest_grid": grids[-1], "profile_grid": grids[-1], "grid_convergence": {}, "extrema": {}, "ghia": {}, "profiles": {}, "solver": {}}
    for re in args.reynolds:
        sols = {n: ref.solve_with_continuation(re, n) for n in grids}
        fine, mid = sols[grids[-1]], sols[grids[-2]]
        check["solver"][str(re)] = {
            str(n): {"seconds": s["solve_seconds"], "newton_iterations": [len(v) for v in s["newton_residuals"].values()],
                     "final_residual": list(s["newton_residuals"].values())[-1][-1]} for n, s in sols.items()
        }  # fmt: skip
        diffs = [velocity_difference(sols[n], fine) for n in grids[:-1]]
        core = [velocity_difference(sols[n], fine, core=True) for n in grids[:-1]]
        check["grid_convergence"][str(re)] = {
            "grids": grids[:-1],
            "velocity_rel_l2_to_finest": diffs,
            "velocity_rel_l2_core_to_finest": core,
        }
        e_fine, e_mid = extrema(fine), extrema(mid)
        # second-order scheme: f_h = f + C h^2, so f = f_h + (f_h - f_2h) / 3
        extrapolated = {k: e_fine[k] + (e_fine[k] - e_mid[k]) / 3 for k in e_fine}
        entry = {f"grid_{grids[-2]}": e_mid, f"grid_{grids[-1]}": e_fine, "richardson": extrapolated}
        if re in PUBLISHED:
            entry["published"] = PUBLISHED[re]
            entry["richardson_minus_published"] = {k: extrapolated[k] - v for k, v in PUBLISHED[re].items()}
            entry["finest_minus_published"] = {k: e_fine[k] - v for k, v in PUBLISHED[re].items()}
        check["extrema"][str(re)] = entry
        if re in ref.GHIA_U:
            check["ghia"][str(re)] = ref.compare_with_ghia(fine, re)
        lines = ref.centrelines(fine)
        check["profiles"][str(re)] = {k: v.tolist() for k, v in lines.items()}

        line = f"Re {re:>4}: difference to the {grids[-1]}-node grid " + ", ".join(f"{n}: {d:.2e} (core {c:.2e})" for n, d, c in zip(grids, diffs, core))
        if re in PUBLISHED:
            worst = max(abs(v) for v in entry["richardson_minus_published"].values())
            line += f" | extrapolated extrema within {worst:.1e} of published values"
        if re in ref.GHIA_U:
            g = check["ghia"][str(re)]
            line += f" | Ghia: u rms {g['u_rms_difference']:.4f}, v rms {g['v_rms_difference']:.4f}"
        print(line, flush=True)

    write_json(args.out_root / "results" / "reference_check.json", check)
    (args.out_root / "figures").mkdir(parents=True, exist_ok=True)
    plot_reference_check(check, args.out_root / "figures" / "reference_check.png")
    print(args.out_root / "results" / "reference_check.json")


if __name__ == "__main__":
    main()
