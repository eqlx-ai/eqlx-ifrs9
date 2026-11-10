import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from eqlx_ifrs9 import InputError, PDCurve, cycle


def test_conditional_pd_by_hand():
    got = cycle.conditional_pd(0.02, 0.12, [-2.0, -1.0, 0.0, 1.0])
    assert got == pytest.approx([0.073424, 0.034377, 0.014287, 0.005255], abs=1e-6)
    direct = norm.cdf((norm.ppf(0.02) - np.sqrt(0.12) * -1.0) / np.sqrt(0.88))
    assert got[1] == pytest.approx(direct)


def test_long_run_pd_is_the_average_over_the_cycle_not_the_value_at_zero():
    mean = quad(lambda z: cycle.conditional_pd(0.02, 0.12, z)[0] * norm.pdf(z), -9, 9)[0]
    assert mean == pytest.approx(0.02, abs=1e-9)
    assert cycle.conditional_pd(0.02, 0.12, 0.0)[0] < 0.02


def test_edge_cases():
    assert cycle.conditional_pd([0.0, 1.0], 0.2, -3.0) == pytest.approx([0.0, 1.0])
    assert cycle.conditional_pd(0.05, 0.0, 2.0) == pytest.approx([0.05])
    for bad_rho in (-0.1, 1.0, float("nan")):
        with pytest.raises(InputError):
            cycle.conditional_pd(0.05, bad_rho, 0.0)
    with pytest.raises(InputError):
        cycle.conditional_pd(1.5, 0.1, 0.0)


def test_implied_z_round_trip():
    for z in (-2.2, -0.4, 0.0, 1.3):
        dr = float(cycle.conditional_pd(0.03, 0.15, z)[0])
        assert cycle.implied_z(0.03, 0.15, dr) == pytest.approx(z, abs=1e-9)
    with pytest.raises(InputError):
        cycle.implied_z(0.03, 0.0, 0.05)


def test_reversion_weights():
    assert cycle.reversion_weights(6, 3) == pytest.approx([1, 1, 1, 0, 0, 0])
    assert cycle.reversion_weights(8, 3, 4) == pytest.approx([1, 1, 1, 0.8, 0.6, 0.4, 0.2, 0])
    assert cycle.reversion_weights(5, 2, 1) == pytest.approx([1, 1, 0.5, 0, 0])  # one transitional year
    assert cycle.reversion_weights(5, 2, method="exponential", decay=0.5) == pytest.approx([1, 1, 0.5, 0.25, 0.125])
    assert cycle.reversion_weights(2, 0, 2) == pytest.approx([2 / 3, 1 / 3])
    with pytest.raises(InputError):
        cycle.reversion_weights(5, -1)
    with pytest.raises(InputError):
        cycle.reversion_weights(5, 2, method="cubic")  # type: ignore[arg-type]


def test_pit_curve_on_a_flat_curve_gives_the_formula_year_by_year():
    base = PDCurve.flat(0.02, 36)
    z = [-1.5, -0.5, 0.5]
    shifted = cycle.pit_curve(base, 0.12, z)
    s = shifted.survival()
    yearly = [1 - s[11], 1 - s[23] / s[11], 1 - s[35] / s[23]]
    assert yearly == pytest.approx(cycle.conditional_pd(0.02, 0.12, z))
    # constant hazard inside each year
    p = shifted.conditional()
    assert np.allclose(p[:12], p[0]) and np.allclose(p[12:24], p[12]) and not np.isclose(p[0], p[12])


def test_pit_curve_keeps_the_shape_inside_a_year():
    base = PDCurve.from_conditional(np.linspace(0.001, 0.004, 24))
    shifted = cycle.pit_curve(base, 0.2, [-1.0])
    ratio = np.log1p(-shifted.conditional()) / np.log1p(-base.conditional())
    assert np.allclose(ratio[:12], ratio[0]) and np.allclose(ratio[12:], ratio[12])
    assert shifted.pd_over() > base.pd_over()


def test_pit_curve_holds_the_last_z_and_handles_a_part_year():
    base = PDCurve.flat(0.03, 30)
    a = cycle.pit_curve(base, 0.1, [-1.0])
    b = cycle.pit_curve(base, 0.1, [-1.0, -1.0, -1.0])
    assert a.cumulative == pytest.approx(b.cumulative)
    # the last block has six months: its annualised PD is shifted, then applied for six months
    s = a.survival()
    assert 1 - (s[29] / s[23]) ** 2 == pytest.approx(float(cycle.conditional_pd(0.03, 0.1, -1.0)[0]))


def test_reversion_goes_to_the_long_run_pd_not_to_z_zero():
    base = PDCurve.flat(0.02, 72)
    w = cycle.reversion_weights(6, 2, 2)
    shifted = cycle.pit_curve(base, 0.12, [-1.0, -1.0], pit_weight=w)
    s = np.concatenate(([1.0], shifted.survival()))
    yearly = [1 - s[12 * (y + 1)] / s[12 * y] for y in range(6)]
    stressed = float(cycle.conditional_pd(0.02, 0.12, -1.0)[0])
    assert yearly[:2] == pytest.approx([stressed, stressed])
    assert yearly[2] == pytest.approx(2 / 3 * stressed + 1 / 3 * 0.02)
    assert yearly[3] == pytest.approx(1 / 3 * stressed + 2 / 3 * 0.02)
    assert yearly[4:] == pytest.approx([0.02, 0.02])


def test_pit_curve_with_zero_weight_or_zero_correlation_is_the_input():
    base = PDCurve.weibull(8, 1.4, 40)
    assert cycle.pit_curve(base, 0.15, [-2.0], pit_weight=[0.0]).cumulative == pytest.approx(base.cumulative)
    assert cycle.pit_curve(base, 0.0, [-2.0]).cumulative == pytest.approx(base.cumulative)
    with pytest.raises(InputError):
        cycle.pit_curve(base, 0.15, [])
    with pytest.raises(InputError):
        cycle.pit_curve(base, 0.15, [0.0], pit_weight=[1.5])
