"""The ECL sum against independent truth: closed forms, direct enumeration and simulation."""

import numpy as np
import pytest

from eqlx_ifrs9 import InputError, PDCurve, discount_factors, expected_credit_loss, schedule

E0, LGD, PD, EIR = 1000.0, 0.45, 0.02, 0.06
H = 1 - (1 - PD) ** (1 / 12)
V = (1 + EIR) ** (-1 / 12)


def closed_form(n: int) -> float:
    """Bullet, constant hazard, flat LGD, default at period end: a geometric series."""
    q = (1 - H) * V
    return LGD * E0 * H * V * (1 - q**n) / (1 - q)


def flat_bullet(n: int, timing: str = "end"):
    return expected_credit_loss(PDCurve.flat(PD, n), np.full(n, E0), LGD, EIR, timing=timing)


@pytest.mark.parametrize(("n", "value"), [(6, 4.446772038869), (12, 8.722449843941), (36, 24.242138103115)])
def test_closed_form_bullet(n, value):
    assert closed_form(n) == pytest.approx(value, abs=1e-9)
    assert flat_bullet(n).lifetime == pytest.approx(value, abs=1e-9)


def test_twelve_month_is_the_truncated_lifetime_sum():
    res = flat_bullet(36)
    assert res.twelve_month == pytest.approx(closed_form(12), abs=1e-12)
    assert res.twelve_month == pytest.approx(float(res.contribution[:12].sum()))
    assert res.lifetime > res.twelve_month
    assert res.for_stage(1) == res.twelve_month and res.for_stage(2) == res.lifetime
    with pytest.raises(InputError):
        res.for_stage(3)


def test_life_shorter_than_twelve_months_uses_the_shorter_horizon():
    res = flat_bullet(6)
    assert res.twelve_month == res.lifetime
    # the PD over six months, not the 12-month PD
    assert res.twelve_month_pd == pytest.approx(1 - (1 - PD) ** 0.5)
    assert res.twelve_month_pd < PD


def test_mid_period_timing_is_end_timing_times_half_a_period_of_interest():
    assert flat_bullet(36, "mid").lifetime == pytest.approx(flat_bullet(36).lifetime * (1 + EIR) ** (1 / 24), rel=1e-13)


def test_continuous_time_limit():
    # hazard lam, force of interest delta: LGD * E * lam / (lam + delta) * (1 - exp(-(lam + delta) T))
    lam, delta, t = -np.log(1 - PD), np.log(1 + EIR), 3.0
    exact = LGD * E0 * lam / (lam + delta) * (1 - np.exp(-(lam + delta) * t))
    k = 365 * 4
    fine = expected_credit_loss(PDCurve.flat(PD, int(t * k), k), np.full(int(t * k), E0), LGD, EIR, timing="mid")
    assert fine.lifetime == pytest.approx(exact, rel=1e-7)


def test_undiscounted_bullet_is_lgd_times_exposure_times_cumulative_pd():
    res = expected_credit_loss(PDCurve.flat(PD, 36), np.full(36, E0), LGD, 0.0)
    assert res.lifetime == pytest.approx(LGD * E0 * (1 - (1 - PD) ** 3), rel=1e-13)
    assert res.lifetime_pd == pytest.approx(1 - (1 - PD) ** 3)


def test_annual_buckets_understate_a_monthly_loan():
    annual = expected_credit_loss(PDCurve.flat(PD, 3, 1), np.full(3, E0), LGD, EIR)
    assert annual.lifetime == pytest.approx(23.597667873479, abs=1e-9)
    assert annual.twelve_month == pytest.approx(PD * LGD * E0 / 1.06)
    assert annual.lifetime < flat_bullet(36).lifetime


def test_summing_conditional_pd_overstates():
    # The frequent error: conditional PDs summed without the survival weighting.
    wrong = sum(H * LGD * E0 * V**t for t in range(1, 37))
    assert wrong == pytest.approx(24.941179606227, abs=1e-9)
    assert flat_bullet(36).lifetime < wrong


def test_against_direct_enumeration_with_everything_varying():
    rng = np.random.default_rng(11)
    for _ in range(50):
        n = int(rng.integers(1, 61))
        p = rng.uniform(0, 0.08, n)
        ead = rng.uniform(0, 5000, n)
        lgd = rng.uniform(0, 1, n)
        eir = float(rng.uniform(-0.02, 0.4))
        k = int(rng.choice([1, 4, 12]))
        total, first_year, survival = 0.0, 0.0, 1.0
        for t in range(1, n + 1):
            term = survival * p[t - 1] * lgd[t - 1] * ead[t - 1] / (1 + eir) ** (t / k)
            total += term
            if t <= k:
                first_year += term
            survival *= 1 - p[t - 1]
        res = expected_credit_loss(PDCurve.from_conditional(p, k), ead, lgd, eir)
        assert res.lifetime == pytest.approx(total, rel=1e-12, abs=1e-12)
        assert res.twelve_month == pytest.approx(first_year, rel=1e-12, abs=1e-12)
        assert res.lifetime_pd == pytest.approx(1 - survival)


def test_against_simulated_default_times():
    """Draw the month of default for two million loans on an amortising schedule; average the loss."""
    n = 48
    sched = schedule.annuity(E0, 0.12, n)
    curve = PDCurve.weibull(scale_years=9.0, shape=1.3, n_periods=n)
    res = expected_credit_loss(curve, sched.ead(), LGD, EIR)
    rng = np.random.default_rng(2024)
    draws = rng.random(2_000_000)
    month = np.searchsorted(curve.cumulative, draws, side="right") + 1  # first month with c_t > u
    defaulted = month <= n
    idx = np.where(defaulted, month, 1) - 1
    loss = np.where(defaulted, LGD * sched.balance_start[idx] * V ** (idx + 1), 0.0)
    se = loss.std() / np.sqrt(loss.size)
    assert abs(loss.mean() - res.lifetime) < 4 * se
    assert se / res.lifetime < 0.005  # the simulation is precise enough to notice a 2% error


def test_discount_factors():
    assert discount_factors(0.06, 2, 1) == pytest.approx([1 / 1.06, 1 / 1.06**2])
    assert discount_factors(0.06, 1, 12, "mid") == pytest.approx([1.06 ** (-0.5 / 12)])
    assert discount_factors(0.0, 3) == pytest.approx([1, 1, 1])
    with pytest.raises(InputError):
        discount_factors(-1.0, 3)
    with pytest.raises(InputError):
        discount_factors(0.05, 3, timing="start")  # type: ignore[arg-type]


def test_inputs_are_not_repaired_silently():
    curve = PDCurve.flat(PD, 12)
    with pytest.raises(InputError, match="covers 12 periods"):
        expected_credit_loss(curve, np.full(13, E0), LGD, EIR)
    with pytest.raises(InputError):
        expected_credit_loss(curve, np.full(12, E0), 1.2, EIR)
    with pytest.raises(InputError):
        expected_credit_loss(curve, [-1.0], LGD, EIR)
    with pytest.raises(InputError):
        expected_credit_loss(curve, np.full(12, E0), np.full(11, LGD), EIR)
    with pytest.raises(InputError):
        expected_credit_loss(curve, [], LGD, EIR)


def test_result_dictionary_adds_up():
    d = flat_bullet(18).to_dict()
    assert d["n_periods"] == 18 and len(d["periods"]) == 18
    assert sum(p["ecl"] for p in d["periods"]) == pytest.approx(d["lifetime_ecl"])
    assert "periods" not in flat_bullet(18).to_dict(periods=False)
