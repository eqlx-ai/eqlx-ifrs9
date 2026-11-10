import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.stats import norm

from eqlx_ifrs9 import InputError, PDCurve, cycle, transitions

P3 = np.array([[0.90, 0.08, 0.02], [0.10, 0.75, 0.15], [0.0, 0.0, 1.0]])


def test_cumulative_pd_by_hand():
    # row A of P squared: default = 0.9*0.02 + 0.08*0.15 + 0.02 = 0.05
    c = transitions.cumulative_pd(P3, 3)
    assert c[0] == pytest.approx([0.02, 0.05, 0.08616])
    assert c[1] == pytest.approx([0.15, 0.2645, 0.353375])
    assert c[2] == pytest.approx([1.0, 1.0, 1.0])


def test_cumulative_pd_equals_matrix_powers():
    rng = np.random.default_rng(3)
    for _ in range(20):
        k = int(rng.integers(2, 9))
        m = rng.random((k, k))
        m[-1] = 0
        m[-1, -1] = 1
        m = m / m.sum(axis=1, keepdims=True)
        c = transitions.cumulative_pd(m, 7)
        for t in range(1, 8):
            assert c[:, t - 1] == pytest.approx(np.linalg.matrix_power(m, t)[:, -1], abs=1e-12)
        assert np.all(np.diff(c, axis=1) >= -1e-12)


def test_two_state_chain_is_a_flat_curve():
    c = transitions.curve([[0.97, 0.03], [0.0, 1.0]], 0, 5)
    assert c.cumulative == pytest.approx(1 - 0.97 ** np.arange(1, 6))
    assert c.cumulative == pytest.approx(PDCurve.flat(0.03, 5, 1).cumulative)
    with pytest.raises(InputError):
        transitions.curve(P3, 2, 5)


def test_path_of_changing_matrices_keeps_the_order():
    good, bad = transitions.shift(P3, 1.0, 0.1), transitions.shift(P3, -1.0, 0.1)
    path = transitions.cumulative_pd_path([good, bad])
    assert path[:, 1] == pytest.approx((good @ bad)[:, -1])
    other = transitions.cumulative_pd_path([bad, good])
    assert path[0, 0] < other[0, 0]
    assert transitions.cumulative_pd_path([P3, P3, P3]) == pytest.approx(transitions.cumulative_pd(P3, 3))
    with pytest.raises(InputError):
        transitions.cumulative_pd_path([])


@pytest.mark.parametrize(
    "bad",
    [
        [[0.9, 0.1], [0.1, 0.9]],  # default not absorbing
        [[0.9, 0.2], [0.0, 1.0]],  # row sum
        [[1.1, -0.1], [0.0, 1.0]],  # negative
        [[0.9, 0.1, 0.0], [0.0, 1.0, 0.0]],  # not square
        [[1.0]],
    ],
)
def test_bad_matrices_are_rejected(bad):
    with pytest.raises(InputError):
        transitions.check_matrix(bad)


def test_rounded_matrix_needs_explicit_normalisation():
    rounded = [[0.9, 0.0998], [0.0, 1.0]]
    with pytest.raises(InputError, match="normalize_rows"):
        transitions.check_matrix(rounded)
    assert transitions.normalize_rows(rounded).sum(axis=1) == pytest.approx([1.0, 1.0])
    transitions.check_matrix(rounded, tol=1e-3)


def test_generator_of_an_embeddable_matrix_is_exact():
    q = np.array([[-0.12, 0.10, 0.02], [0.08, -0.25, 0.17], [0.0, 0.0, 0.0]])
    p = expm(q)
    for method in ("log", "da", "wa"):
        g = transitions.generator(p, method)  # type: ignore[arg-type]
        assert g.q == pytest.approx(q, abs=1e-10)
        assert g.l1_error < 1e-10 and g.negative_mass == pytest.approx(0.0, abs=1e-12) and not g.warnings
    monthly = transitions.sub_period_matrix(p, 12)
    assert np.linalg.matrix_power(monthly, 12) == pytest.approx(p, abs=1e-10)
    assert monthly == pytest.approx(expm(q / 12), abs=1e-10)


def test_adjusted_generators_are_valid_generators():
    from tests.test_published import JLT, ROUNDED

    for method in ("da", "wa", "jlt"):
        g = transitions.generator(JLT, method, tol=ROUNDED)  # type: ignore[arg-type]
        off = g.q[~np.eye(8, dtype=bool)]
        assert np.all(off >= 0.0)
        assert np.all(np.diag(g.q) <= 0.0)
        assert g.q[-1] == pytest.approx(np.zeros(8))
        if method != "jlt":
            # rows of the published matrix are off by up to 2e-4, and so are the generator rows
            assert g.q.sum(axis=1) == pytest.approx(np.zeros(8), abs=3e-4)
        m = g.matrix(1 / 12)
        assert m.sum(axis=1) == pytest.approx(np.ones(8)) and np.all(m >= 0)
    with pytest.raises(InputError):
        transitions.generator(JLT, "nope", tol=ROUNDED)  # type: ignore[arg-type]
    with pytest.raises(InputError, match="negative probabilities"):
        transitions.generator(JLT, "log", tol=ROUNDED).matrix(1 / 12)  # not repaired silently
    with pytest.raises(InputError):
        transitions.generator(JLT, "wa", tol=ROUNDED).matrix(-1)


def test_shift_by_hand():
    # row [0.90, 0.08, 0.02], rho 0.10: cut-offs Phi^-1(0.10) and Phi^-1(0.02)
    for z, row in [
        (-1.0, [0.845552, 0.120936, 0.033512]),
        (0.0, [0.911632, 0.073168, 0.015200]),
        (1.0, [0.953929, 0.039829, 0.006242]),
    ]:
        assert transitions.shift(P3, z, 0.10)[0] == pytest.approx(row, abs=1e-6)
    # default column agrees with the one-factor formula applied to the PD
    assert transitions.shift(P3, -1.0, 0.10)[:2, -1] == pytest.approx(cycle.conditional_pd([0.02, 0.15], 0.10, -1.0))


def test_shift_properties():
    for z in (-2.5, -0.3, 0.0, 1.7):
        s = transitions.shift(P3, z, [0.1, 0.2, 0.1])
        assert s.sum(axis=1) == pytest.approx(np.ones(3))
        assert np.all(s >= 0) and s[-1] == pytest.approx([0, 0, 1])
    assert transitions.shift(P3, 1.0, 0.0) == pytest.approx(P3)  # no correlation, no shift
    # more defaults in a bad year, fewer in a good one
    assert transitions.shift(P3, -1, 0.1)[0, -1] > P3[0, -1] > transitions.shift(P3, 1, 0.1)[0, -1]
    with pytest.raises(InputError):
        transitions.shift(P3, 0.0, 1.0)
    with pytest.raises(InputError):
        transitions.shift(P3, float("inf"), 0.1)


def test_shift_averages_back_to_the_long_run_matrix():
    for i in range(2):
        for j in range(3):
            mean = quad(lambda z, i=i, j=j: transitions.shift(P3, z, 0.15)[i, j] * norm.pdf(z), -9, 9)[0]
            assert mean == pytest.approx(P3[i, j], abs=1e-8)
    # the matrix at z = 0 is not the long-run matrix
    assert transitions.shift(P3, 0.0, 0.15)[0, -1] < P3[0, -1]


def test_shift_handles_zero_cells():
    p = np.array([[0.95, 0.05, 0.0], [0.0, 0.9, 0.1], [0, 0, 1.0]])
    s = transitions.shift(p, -2.0, 0.2)
    assert s[0, 2] == 0.0 and s[1, 0] == 0.0
    assert s.sum(axis=1) == pytest.approx(np.ones(3))
