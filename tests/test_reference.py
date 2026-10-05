"""The finite-difference reference solver, on grids small enough for a test."""

import numpy as np
import pytest
from pinn_cavity import reference as ref


@pytest.fixture(scope="module")
def re100():
    return ref.solve_with_continuation(100, 65)


def test_newton_converges_quadratically(re100):
    history = re100["newton_residuals"][100.0]
    assert history[-1] < 1e-8 and len(history) <= 8


def test_boundary_conditions(re100):
    u, v, psi = re100["u"], re100["v"], re100["psi"]
    assert np.all(u[:, -1] == 1.0) and np.all(u[:, 0] == 0.0) and np.all(u[0, :-1] == 0.0) and np.all(u[-1, :-1] == 0.0)  # the two top corners belong to the lid
    assert np.all(v[0, :] == 0.0) and np.all(v[-1, :] == 0.0) and np.all(v[:, 0] == 0.0) and np.all(v[:, -1] == 0.0)
    assert np.abs(psi[[0, -1], :]).max() < 1e-15 and np.abs(psi[:, [0, -1]]).max() < 1e-15


def test_discrete_continuity(re100):
    # u = d(psi)/dy and v = -d(psi)/dx with commuting central differences: the discrete divergence vanishes
    n = re100["u"].shape[0]
    h = 1.0 / (n - 1)
    psi = re100["psi"]
    u = (psi[:, 2:] - psi[:, :-2]) / (2 * h)
    v = -(psi[2:, :] - psi[:-2, :]) / (2 * h)
    div = (u[2:, :] - u[:-2, :]) / (2 * h) + (v[:, 2:] - v[:, :-2]) / (2 * h)
    assert np.abs(div).max() < 1e-9


def test_agrees_with_ghia_at_re100(re100):
    check = ref.compare_with_ghia(re100, 100)
    assert check["u_rms_difference"] < 0.01 and check["v_rms_difference"] < 0.01
    assert abs(check["psi_min"] - check["psi_min_ghia"]) < 2e-3


def test_ghia_tables_are_complete():
    for table, coordinate in ((ref.GHIA_U, ref.GHIA_Y), (ref.GHIA_V, ref.GHIA_X)):
        assert set(table) == {100, 400, 1000}
        assert all(len(values) == len(coordinate) == 17 for values in table.values())


def test_stokes_limit_is_symmetric():
    # at vanishing Reynolds number the flow is symmetric about the vertical centreline,
    # and the asymmetry grows with the Reynolds number
    def asymmetry(reynolds):
        sol = ref.solve_cavity(reynolds, 33)
        return max(np.abs(sol["u"] - sol["u"][::-1, :]).max(), np.abs(sol["v"] + sol["v"][::-1, :]).max())

    assert asymmetry(1e-3) < 1e-4 < 1e-2 < asymmetry(10.0)


def test_second_order_in_the_core():
    sols = {n: ref.solve_with_continuation(10, n) for n in (17, 33, 65)}
    centre = [sols[n]["u"][n // 2, n // 2] for n in (17, 33, 65)]
    order = np.log2(abs(centre[0] - centre[1]) / abs(centre[1] - centre[2]))
    assert 1.6 < order < 2.4


def test_cache_and_scaling(tmp_path):
    a = ref.load_or_solve(10.0, 17, 0.1, 0.1, 2.0, tmp_path)
    b = ref.load_or_solve(10.0, 17, 0.1, 0.1, 2.0, tmp_path)
    assert a["file"] == b["file"] and np.array_equal(a["u"], b["u"]) and len(list(tmp_path.iterdir())) == 1
    assert a["x"][0] == pytest.approx(-0.05) and a["x"][-1] == pytest.approx(0.05)
    assert a["u"][:, -1] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        ref.load_or_solve(10.0, 17, 0.1, 0.2, 1.0, tmp_path)
