import numpy as np
import pytest

from eqlx_ifrs9 import InputError, lgd, schedule, simplified
from eqlx_ifrs9.scenarios import check_weights, weighted_ecl, weighted_value


def test_annuity_by_formula_and_by_recursion():
    s = schedule.annuity(1000, 0.12, 12)
    assert s.payment == pytest.approx(np.full(12, 88.848789), abs=1e-6)
    assert s.balance_start[1:4] == pytest.approx([921.1512, 841.5139, 761.0803], abs=1e-4)
    assert s.balance_end[-1] == pytest.approx(0.0, abs=1e-9)
    pay = 1000 * 0.01 / (1 - 1.01**-12)
    b = 1000.0
    for t in range(12):
        assert s.balance_start[t] == pytest.approx(b, abs=1e-9)
        b = b * 1.01 - pay
    assert s.principal.sum() == pytest.approx(1000.0)


def test_zero_rate_annuity_and_effective_rate():
    s = schedule.annuity(1200, 0.0, 12)
    assert s.payment == pytest.approx(np.full(12, 100.0)) and s.interest == pytest.approx(np.zeros(12))
    assert schedule.period_rate(0.12, 12) == pytest.approx(0.01)
    assert schedule.period_rate(0.12, 12, "effective") == pytest.approx(1.12 ** (1 / 12) - 1)
    with pytest.raises(InputError):
        schedule.period_rate(0.12, 12, "simple")  # type: ignore[arg-type]


def test_annuity_with_a_tiny_rate_is_stable():
    flat = schedule.annuity(1200, 0.0, 12).balance_start
    for tiny in (1e-17, 1e-15, 1e-12, -1e-12):
        s = schedule.annuity(1200, tiny, 12)
        assert s.balance_start == pytest.approx(flat, rel=1e-9) and s.payment == pytest.approx(
            np.full(12, 100.0), rel=1e-9
        )
    negative = schedule.annuity(1000, -0.02, 24)
    assert negative.balance_end[-1] == pytest.approx(0.0, abs=1e-9) and negative.principal.sum() == pytest.approx(1000)


def test_equal_principal_by_hand():
    s = schedule.equal_principal(1000, 0.03, 4, periods_per_year=1)
    assert s.balance_start == pytest.approx([1000, 750, 500, 250])
    assert s.balance_end == pytest.approx([750, 500, 250, 0])
    assert s.payment == pytest.approx([280, 272.5, 265, 257.5])


def test_bullet_has_exposure_in_its_last_period():
    s = schedule.bullet(1000, 0.12, 6)
    assert s.ead() == pytest.approx(np.full(6, 1000.0))
    assert s.ead("start_plus_interest") == pytest.approx(np.full(6, 1010.0))
    assert s.ead("mid")[-1] == pytest.approx(500.0)
    assert s.principal == pytest.approx([0, 0, 0, 0, 0, 1000])
    one = schedule.bullet(1000, 0.12, 1)
    assert one.ead() == pytest.approx([1000.0])
    with pytest.raises(InputError):
        s.ead("end")  # type: ignore[arg-type]


def test_prepayment():
    assert schedule.smm_from_cpr(0.06) == pytest.approx(0.005143012832, abs=1e-12)
    assert 1 - (1 - schedule.smm_from_cpr(0.06)) ** 12 == pytest.approx(0.06)
    base = schedule.annuity(1000, 0.12, 12)
    pp = schedule.with_prepayment(base, 0.06)
    assert pp.balance_end[:3] == pytest.approx([916.4137, 832.8804, 749.3978], abs=1e-4)
    assert pp.balance_start[0] == 1000.0
    assert pp.balance_start[1:] == pytest.approx(pp.balance_end[:-1])
    assert np.all(pp.balance_start <= base.balance_start + 1e-12)
    assert schedule.with_prepayment(base, 0.0).balance_start == pytest.approx(base.balance_start)
    with pytest.raises(InputError):
        schedule.with_prepayment(base, [0.1, 0.2])


def test_commitment_exposure_is_capped_at_the_limit():
    assert schedule.commitment_ead(60, 40, 0.5) == pytest.approx([80.0])
    assert schedule.commitment_ead([10, 20], [5, 0], [1.0, 1.0]) == pytest.approx([15, 20])
    with pytest.raises(InputError):
        schedule.commitment_ead(60, 40, 1.2)


def test_collateral_lgd_by_hand():
    # EAD 100, collateral 80, haircut 20%, costs 5%, two years to recovery, 5%:
    # net 60.8, present value 60.8 / 1.05^2 = 55.147392, LGD 0.448526
    kw = dict(haircut=0.2, cost_rate=0.05, years_to_recovery=2, discount_rate=0.05)
    assert lgd.collateral_lgd(100, 80, **kw) == pytest.approx([0.448526], abs=1e-6)
    assert lgd.collateral_lgd(100, 80, cure_probability=0.3, **kw) == pytest.approx([0.313968], abs=1e-6)
    # 1.5 years already in default: half a year left
    assert lgd.collateral_lgd(100, 80, haircut=0.2, cost_rate=0.05, years_to_recovery=0.5, discount_rate=0.05) == (
        pytest.approx([0.406653], abs=1e-6)
    )


def test_collateral_lgd_limits():
    assert lgd.collateral_lgd(100, 0) == pytest.approx([1.0])
    assert lgd.collateral_lgd(100, 500) == pytest.approx([0.0])  # recovery capped at the claim
    assert lgd.collateral_lgd(100, 500, years_to_recovery=1, discount_rate=0.1) == pytest.approx([1 - 1 / 1.1])
    assert lgd.collateral_lgd(100, 40, unsecured_recovery_rate=0.25) == pytest.approx([0.45])
    assert lgd.collateral_lgd(0, 40) == pytest.approx([1.0])
    # a term structure: the loan amortises, the collateral does not
    path = lgd.collateral_lgd([100, 75, 50, 25], 50)
    assert path == pytest.approx([0.5, 1 / 3, 0.0, 0.0])
    with pytest.raises(InputError):
        lgd.collateral_lgd(100, 80, haircut=1.5)
    with pytest.raises(InputError):
        lgd.collateral_lgd([100, 90], [80, 70, 60])


def test_workout_lgd():
    # 40 after one year and 30 after two, costs 5 with the first, 10%: 35/1.1 + 30/1.21 = 56.6116
    assert lgd.workout_lgd(100, [40, 30], [1, 2], 0.10, costs=[5, 0]) == pytest.approx(1 - 56.611570 / 100, abs=1e-6)
    assert lgd.workout_lgd(100, [120], [0], 0.1) == pytest.approx(-0.2)
    with pytest.raises(InputError):
        lgd.workout_lgd(0, [1], [1], 0.1)
    with pytest.raises(InputError):
        lgd.workout_lgd(100, [1, 2], [1], 0.1)


def test_credit_impaired_allowance():
    # gross 1,000; 300 expected in one year and 400 in two, EIR 8%
    pv = 300 / 1.08 + 400 / 1.08**2
    assert lgd.credit_impaired_allowance(1000, [300, 400], [1, 2], 0.08) == pytest.approx(1000 - pv)
    # two recovery scenarios, 60% and 40%
    weighted = lgd.credit_impaired_allowance(1000, [[300, 400], [100, 100]], [1, 2], 0.08, probabilities=[0.6, 0.4])
    low = 100 / 1.08 + 100 / 1.08**2
    assert weighted == pytest.approx(1000 - (0.6 * pv + 0.4 * low))
    with pytest.raises(InputError):
        lgd.credit_impaired_allowance(1000, [[300, 400]], [1, 2], 0.08, probabilities=[0.6, 0.3])


def test_loss_rates_from_history():
    # Sales 10,000; still unpaid after 30 / 60 / 90 days: 8,000 / 4,500 / 1,500; 300 never paid.
    rates = simplified.loss_rates_from_history([10_000, 8_000, 4_500, 1_500], 300)
    assert rates == pytest.approx([0.03, 0.0375, 0.066667, 0.20], abs=1e-6)
    forward = simplified.loss_rates_from_history([10_000, 8_000, 4_500, 1_500], 400)
    assert forward == pytest.approx([0.04, 0.05, 0.088889, 0.266667], abs=1e-6)
    assert forward == pytest.approx(simplified.adjust_loss_rates(rates, 400 / 300))
    pm = simplified.provision_matrix([50, 40, 30, 20], forward)
    assert pm.total_allowance == pytest.approx(2 + 2 + 30 * 400 / 4500 + 20 * 400 / 1500)
    with pytest.raises(InputError):
        simplified.loss_rates_from_history([10_000, 11_000], 300)
    with pytest.raises(InputError):
        simplified.loss_rates_from_history([10_000, 200], 300)


def test_loss_rates_from_flow_rates():
    rates = simplified.loss_rates_from_flow_rates([0.2, 0.5, 0.6, 0.8])
    assert rates == pytest.approx([0.048, 0.24, 0.48, 0.8])
    assert simplified.loss_rates_from_flow_rates([0.2, 0.5], 0.5) == pytest.approx([0.05, 0.25])
    assert simplified.adjust_loss_rates([0.6, 0.9], 1.5) == pytest.approx([0.9, 1.0])
    with pytest.raises(InputError):
        simplified.loss_rates_from_flow_rates([0.2, 1.5])


def test_provision_matrix_checks_and_dictionary():
    pm = simplified.provision_matrix([100, 50], [0.01, 0.1])
    d = pm.to_dict()
    assert d["total_allowance"] == pytest.approx(6.0) and d["coverage"] == pytest.approx(0.04)
    assert [b["bucket"] for b in d["buckets"]] == ["bucket_1", "bucket_2"]
    with pytest.raises(InputError):
        simplified.provision_matrix([100, 50], [0.01])
    with pytest.raises(InputError):
        simplified.provision_matrix([100, 50], [0.01, 1.1])
    with pytest.raises(InputError):
        simplified.provision_matrix([100, 50], [0.01, 0.1], ["a"])


def test_scenario_weights():
    with pytest.raises(InputError, match="add up to 1"):
        check_weights([0.5, 0.4])
    with pytest.raises(InputError):
        check_weights([1.2, -0.2])
    with pytest.raises(InputError):
        weighted_ecl([1, 2], [0.5, 0.5], ["a", "a"])
    with pytest.raises(InputError):
        weighted_ecl([1, 2], [0.5, 0.5], base="x")
    res = weighted_ecl([10, 30], [0.25, 0.75])
    d = res.to_dict()
    assert d["weighted_ecl"] == 25.0 and d["weighted_to_base"] is None
    assert d["scenarios"][0]["difference_to_weighted"] == -15.0
    assert weighted_value([[0.01, 0.02], [0.03, 0.06]], [0.5, 0.5]) == pytest.approx([0.02, 0.04])


def test_weighted_ecl_is_not_the_ecl_of_the_average_scenario():
    from eqlx_ifrs9 import cycle

    z, w = np.array([1.0, 0.0, -1.5]), np.array([0.3, 0.5, 0.2])
    assert float(z @ w) == pytest.approx(0.0)
    pds = cycle.conditional_pd(0.02, 0.12, z)
    weighted = weighted_ecl(pds * 0.45 * 1000, w, ["up", "base", "down"], base="base")
    assert weighted.weighted == pytest.approx(8.5126, abs=1e-4)
    assert weighted.base_ecl == pytest.approx(6.4293, abs=1e-4)  # the ECL at the average z
    assert weighted.weighted_to_base == pytest.approx(1.324, abs=1e-3)
