"""Finite-difference reference solution of the steady lid-driven cavity.

Independent of PhysicsNeMo and of the network: stream function and vorticity on a uniform
grid, second-order central differences, Thom's wall vorticity, Newton's method with a sparse
direct solve of the full Jacobian. The solver works on the unit square with unit lid speed,
where the only parameter is the Reynolds number; :func:`load_or_solve` maps the result to
the cavity of the NVIDIA example.

    laplace(psi) = -omega,      u = d(psi)/dy,   v = -d(psi)/dx
    u d(omega)/dx + v d(omega)/dy = (1/Re) laplace(omega)

The tabulated values of Ghia, Ghia and Shin (1982) are included to check the solver.
"""

import hashlib
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

# Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982), Tables I and II.
# u on the vertical centreline (x = 0.5) and v on the horizontal centreline (y = 0.5).
GHIA_Y = [1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0]
GHIA_X = [1.0, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0]
GHIA_U = {
    100: [1.0, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.0],
    400: [1.0, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135, -0.11477, -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.0],
    1000: [1.0, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702, -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.0],
}
GHIA_V = {
    100: [0.0, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0.0],
    400: [0.0, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, -0.38598, 0.05186, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 0.19713, 0.18360, 0.0],
    1000: [0.0, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012, 0.27485, 0.0],
}
# Stream function at the centre of the primary vortex (Ghia et al., Table III).
GHIA_PSI_MIN = {100: -0.103423, 400: -0.113909, 1000: -0.117929}
# Index of a tabulated v value that is inconsistent with its neighbours and with every grid of
# the reference solver (Re = 400, x = 0.9063). It is kept in the table and reported separately.
GHIA_V_SUSPECT = {400: 5}

SOLVER_VERSION = 1


def _operators(n: int):
    """Central-difference operators on all ``n * n`` nodes, index ``i * n + j`` for ``(x_i, y_j)``."""
    h = 1.0 / (n - 1)
    eye = sp.identity(n, format="csr")
    d1 = sp.diags([-1.0, 1.0], [-1, 1], shape=(n, n), format="csr") / (2 * h)
    d2 = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="csr") / h**2
    return sp.kron(d1, eye, format="csr"), sp.kron(eye, d1, format="csr"), (sp.kron(d2, eye) + sp.kron(eye, d2)).tocsr()


def _wall_vorticity(n: int):
    """Thom's formula ``omega_wall = -2 psi_adjacent / h^2 - 2 U_tangential / h`` as ``A psi + b``."""
    h = 1.0 / (n - 1)
    idx = np.arange(n * n).reshape(n, n)
    inner = np.arange(1, n - 1)
    rows, cols = [], []
    for wall, adjacent in (
        (idx[0, inner], idx[1, inner]),
        (idx[-1, inner], idx[-2, inner]),
        (idx[inner, 0], idx[inner, 1]),
        (idx[inner, -1], idx[inner, -2]),
    ):
        rows.append(wall)
        cols.append(adjacent)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    a = sp.csr_matrix((np.full(rows.size, -2.0 / h**2), (rows, cols)), shape=(n * n, n * n))
    b = np.zeros(n * n)
    b[idx[inner, -1]] = -2.0 / h  # moving lid at y = 1, unit speed
    return a, b


def solve_cavity(reynolds: float, n: int, tol: float = 1e-8, max_newton: int = 30, guess=None) -> dict:
    """Steady cavity flow on the unit square with ``n`` nodes per side.

    Returns nodal arrays of shape ``(n, n)`` indexed ``[i, j]`` for ``(x_i, y_j)``, and the
    Newton history. ``guess`` is a ``(psi, omega)`` pair used as the starting point.
    """
    dx, dy, lap = _operators(n)
    a_wall, b_wall = _wall_vorticity(n)
    size = n * n
    interior = np.zeros((n, n), dtype=bool)
    interior[1:-1, 1:-1] = True
    interior = interior.ravel()
    m_int = sp.diags(interior.astype(float), format="csr")
    m_bnd = sp.diags((~interior).astype(float), format="csr")

    psi = np.zeros(size) if guess is None else guess[0].ravel().copy()
    omega = np.zeros(size) if guess is None else guess[1].ravel().copy()

    def residual(psi, omega):
        f_psi = np.where(interior, lap @ psi + omega, psi)
        transport = (dy @ psi) * (dx @ omega) - (dx @ psi) * (dy @ omega) - (lap @ omega) / reynolds
        f_omega = np.where(interior, transport, omega - a_wall @ psi - b_wall)
        return np.concatenate([f_psi, f_omega])

    history = []
    for _ in range(max_newton):
        f = residual(psi, omega)
        norm = float(np.abs(f).max())
        history.append(norm)
        if norm < tol:
            break
        j_pp = m_bnd + m_int @ lap
        j_po = m_int
        j_op = -a_wall + m_int @ (sp.diags(dx @ omega) @ dy - sp.diags(dy @ omega) @ dx)
        j_oo = m_bnd + m_int @ (sp.diags(dy @ psi) @ dx - sp.diags(dx @ psi) @ dy - lap / reynolds)
        jac = sp.bmat([[j_pp, j_po], [j_op, j_oo]], format="csc")
        step = splu(jac).solve(-f)
        psi += step[:size]
        omega += step[size:]
    else:
        raise RuntimeError(f"Newton did not converge at Re={reynolds}, n={n}: residuals {history}")

    psi, omega = psi.reshape(n, n), omega.reshape(n, n)
    u = (dy @ psi.ravel()).reshape(n, n)
    v = -(dx @ psi.ravel()).reshape(n, n)
    # wall values are the boundary conditions, not differences of psi
    u[0, :] = u[-1, :] = u[:, 0] = 0.0
    u[:, -1] = 1.0
    v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 0.0
    return {"psi": psi, "omega": omega, "u": u, "v": v, "newton_residuals": history}


def continuation(reynolds: float) -> list[float]:
    """Reynolds numbers solved in sequence so that every Newton solve starts close to its solution."""
    return [r for r in (100.0, 400.0) if r < reynolds] + [float(reynolds)]


def solve_with_continuation(reynolds: float, n: int) -> dict:
    guess, history = None, {}
    start = time.perf_counter()
    for re in continuation(reynolds):
        sol = solve_cavity(re, n, guess=guess)
        guess = (sol["psi"], sol["omega"])
        history[re] = sol["newton_residuals"]
    sol["newton_residuals"] = history
    sol["solve_seconds"] = time.perf_counter() - start
    return sol


def centrelines(sol: dict) -> dict:
    """``u`` on the vertical and ``v`` on the horizontal centreline. Needs an odd number of nodes."""
    n = sol["u"].shape[0]
    if n % 2 == 0:
        raise ValueError("centrelines need an odd number of nodes per side")
    s = np.linspace(0.0, 1.0, n)
    return {"s": s, "u_vertical": sol["u"][n // 2, :], "v_horizontal": sol["v"][:, n // 2]}


def compare_with_ghia(sol: dict, reynolds: int) -> dict:
    """Differences between the centreline profiles of ``sol`` and the values tabulated by Ghia et al."""
    lines = centrelines(sol)
    u = np.interp(GHIA_Y, lines["s"], lines["u_vertical"])
    v = np.interp(GHIA_X, lines["s"], lines["v_horizontal"])
    du = u - np.array(GHIA_U[reynolds])
    dv = v - np.array(GHIA_V[reynolds])
    keep = np.ones(dv.size, dtype=bool)
    if reynolds in GHIA_V_SUSPECT:
        keep[GHIA_V_SUSPECT[reynolds]] = False
    return {
        "reynolds": reynolds,
        "u_max_abs_difference": float(np.abs(du).max()),
        "u_rms_difference": float(np.sqrt(np.mean(du**2))),
        "v_max_abs_difference": float(np.abs(dv).max()),
        "v_rms_difference": float(np.sqrt(np.mean(dv**2))),
        "v_max_abs_difference_without_suspect": float(np.abs(dv[keep]).max()),
        "v_rms_difference_without_suspect": float(np.sqrt(np.mean(dv[keep] ** 2))),
        "v_differences": dv.tolist(),
        "u_differences": du.tolist(),
        "psi_min": float(sol["psi"].min()),
        "psi_min_ghia": GHIA_PSI_MIN[reynolds],
    }


def load_or_solve(reynolds: float, n: int, width: float, height: float, lid_velocity: float, data_dir: Path) -> dict:
    """Reference velocity on the cavity of the example, cached under ``data_dir``.

    Coordinates are ``x = (x* - 1/2) width`` and velocities ``u = U u*``. The cavity must be
    square: the dimensionless problem is solved on the unit square.
    """
    if abs(width - height) > 1e-12 * width:
        raise ValueError("the reference solver handles a square cavity only")
    key = hashlib.sha256(f"{SOLVER_VERSION}|{reynolds!r}|{n}|{width!r}|{lid_velocity!r}".encode()).hexdigest()[:10]
    path = Path(data_dir) / f"reference_re{reynolds:g}_n{n}_{key}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files} | {"file": path.name}
    sol = solve_with_continuation(reynolds, n)
    s = np.linspace(0.0, 1.0, n)
    out = {
        "x": (s - 0.5) * width,
        "y": (s - 0.5) * height,
        "u": lid_velocity * sol["u"],
        "v": lid_velocity * sol["v"],
        "psi": lid_velocity * width * sol["psi"],
        "reynolds": np.float64(reynolds),
        "solve_seconds": np.float64(sol["solve_seconds"]),
        "newton_final_residual": np.float64(list(sol["newton_residuals"].values())[-1][-1]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out | {"file": path.name}
