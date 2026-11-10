import numpy as np
import pytest

from eqlx_ifrs9 import InputError, PDCurve


def test_identities_on_a_hand_example():
    c = PDCurve.from_cumulative([0.02, 0.05, 0.09], periods_per_year=1)
    assert c.marginal() == pytest.approx([0.02, 0.03, 0.04])
    assert c.conditional() == pytest.approx([0.02, 0.03 / 0.98, 0.04 / 0.95])
    assert c.survival() == pytest.approx([0.98, 0.95, 0.91])
    assert c.survival_at_start() == pytest.approx([1.0, 0.98, 0.95])
    assert c.hazard() == pytest.approx([0.020203, 0.031091, 0.043017], abs=1e-6)
    assert c.pd_over(0) == 0.0 and c.pd_over() == 0.09 and c.pd_over(2) == 0.05


def test_three_constructors_agree():
    a = PDCurve.from_cumulative([0.02, 0.05, 0.09])
    assert PDCurve.from_marginal([0.02, 0.03, 0.04]).cumulative == pytest.approx(a.cumulative)
    assert PDCurve.from_conditional(a.conditional()).cumulative == pytest.approx(a.cumulative)


def test_flat_curve_is_constant_hazard_not_pd_divided_by_twelve():
    c = PDCurve.flat(0.02, 36)
    assert c.conditional() == pytest.approx(np.full(36, 0.001682142552739574))
    assert c.pd_over(12) == pytest.approx(0.02)
    assert c.pd_over(6) == pytest.approx(0.010050506339, abs=1e-12)
    assert c.pd_over(36) == pytest.approx(1 - 0.98**3)
    assert c.annualized() == pytest.approx(0.02) and c.annualized(7) == pytest.approx(0.02)
    assert c.twelve_month_pd() == pytest.approx(0.02)
    assert PDCurve.flat(0.02, 5).twelve_month_pd() == pytest.approx(1 - 0.98 ** (5 / 12))


def test_forward_curve_for_the_remaining_window():
    # Origination curve over five years; two years later the three remaining years, given survival.
    orig = PDCurve.from_cumulative([0.01, 0.025, 0.045, 0.07, 0.10], periods_per_year=1)
    rest = orig.forward(2)
    assert rest.n_periods == 3
    assert rest.pd_over() == pytest.approx((0.10 - 0.025) / (1 - 0.025))
    assert rest.pd_over() == pytest.approx(0.076923, abs=1e-6)
    assert rest.annualized() == pytest.approx(0.026328, abs=1e-6)
    # survival multiplies: S(0,5) = S(0,2) * S(2,5)
    assert (1 - orig.pd_over(2)) * (1 - rest.pd_over()) == pytest.approx(1 - orig.pd_over())
    assert orig.forward(0) is orig
    with pytest.raises(InputError):
        orig.forward(5)


def test_refining_the_frequency_keeps_the_original_points():
    annual = PDCurve.from_cumulative([0.02, 0.05, 0.09], periods_per_year=1)
    monthly = annual.to_frequency(12)
    assert monthly.n_periods == 36 and monthly.periods_per_year == 12
    assert monthly.cumulative[[11, 23, 35]] == pytest.approx([0.02, 0.05, 0.09])
    assert np.allclose(monthly.conditional()[:12], monthly.conditional()[0])
    assert annual.to_frequency(1) is annual
    with pytest.raises(InputError):
        monthly.to_frequency(1)


def test_extend_and_truncate():
    c = PDCurve.from_cumulative([0.02, 0.05])
    e = c.extend(4)
    assert e.cumulative[:2] == pytest.approx([0.02, 0.05])
    assert e.conditional()[2:] == pytest.approx([0.03 / 0.98] * 2)
    assert c.extend(1).n_periods == 1
    assert e.truncate(2).cumulative == pytest.approx(c.cumulative)
    with pytest.raises(InputError):
        c.truncate(3)


def test_weibull():
    c = PDCurve.weibull(scale_years=50, shape=1.2, n_periods=3, periods_per_year=1)
    assert c.cumulative == pytest.approx([0.009104, 0.020793, 0.033603], abs=1e-6)


def test_certain_default_does_not_divide_by_zero():
    c = PDCurve.from_cumulative([0.5, 1.0, 1.0])
    assert c.conditional() == pytest.approx([0.5, 1.0, 1.0])
    assert c.marginal() == pytest.approx([0.5, 0.5, 0.0])
    assert np.isinf(c.hazard()[1])


@pytest.mark.parametrize(
    "bad",
    [[0.05, 0.02], [0.02, 1.2], [-0.01, 0.02], [], [0.02, float("nan")]],
)
def test_bad_cumulative_curves_are_rejected(bad):
    with pytest.raises(InputError):
        PDCurve.from_cumulative(bad)


def test_conditional_pds_given_as_marginal_are_caught():
    with pytest.raises(InputError, match="conditional"):
        PDCurve.from_marginal([0.6, 0.6])
    with pytest.raises(InputError):
        PDCurve.from_marginal([0.1, -0.1])


def test_curve_is_read_only_and_serialisable():
    c = PDCurve.flat(0.02, 3)
    with pytest.raises(ValueError):
        c.cumulative[0] = 0.5
    assert c.to_dict()["n_periods"] == 3
    with pytest.raises(InputError):
        PDCurve.flat(0.02, 0)
    with pytest.raises(InputError):
        PDCurve.flat(1.5, 3)
