"""Turkey profile. Article numbers refer to the BDDK provisioning regulation (Official Gazette 29750)."""

import datetime as dt

import numpy as np
import pytest

from eqlx_ifrs9 import InputError
from eqlx_ifrs9.profiles import tr_bddk as tr


def test_day_limits_of_article_4():
    days = [0, 30, 31, 90, 91, 180, 181, 365, 366, 2000]
    assert tr.group_from_days_past_due(days).tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert tr.stage_from_group([1, 2, 3, 4, 5]).tolist() == [1, 2, 3, 3, 3]
    assert tr.group_from_days_past_due([361], tr.day_limits(days_in_year=360)).tolist() == [5]
    with pytest.raises(InputError):
        tr.group_from_days_past_due([-1])
    with pytest.raises(InputError):
        tr.stage_from_group([0])
    with pytest.raises(InputError):
        tr.day_limits(days_in_year=364)


def test_pandemic_relief_period():
    # Board decisions 8948 / 8970 (from 17 March 2020) to 9795 (30 September 2021): 90 and 180 days
    relief = tr.day_limits(dt.date(2020, 12, 31))
    assert (relief.group2, relief.group3) == (90, 180)
    assert tr.group_from_days_past_due([45, 120, 181], relief).tolist() == [1, 2, 4]
    for day in (dt.date(2020, 3, 16), dt.date(2021, 10, 1), dt.date(2026, 11, 10)):
        assert (tr.day_limits(day).group2, tr.day_limits(day).group3) == (30, 90)
    assert tr.day_limits(dt.date(2020, 3, 17)).group2 == 90 and tr.day_limits(dt.date(2021, 9, 30)).group2 == 90


def test_flags_lift_the_group():
    c = tr.classify(
        [0, 0, 0, 10, 45, 10, 0],
        significant_increase=[0, 1, 0, 0, 0, 0, 0],
        restructured=[0, 0, 1, 0, 0, 0, 0],
        unlikely_to_pay=[0, 0, 0, 1, 0, 0, 0],
        in_monitoring_year=[0, 0, 0, 0, 1, 1, 1],
        restructured_again=[0, 0, 0, 0, 0, 0, 1],
    )
    assert c.group.tolist() == [1, 2, 2, 3, 3, 1, 3]
    assert c.stage.tolist() == [1, 2, 2, 3, 3, 1, 3]
    assert "4(1)(b)6" in c.reasons[1][0] and "4(1)(b)8" in c.reasons[2][0] and "5(1)" in c.reasons[3][0]
    assert any("4(1)(c)5" in r for r in c.reasons[4]) and any("restructured again" in r for r in c.reasons[6])
    assert c.to_dict()[0] == {"group": 1, "stage": 1, "reasons": []}


def test_judgement_group_can_only_worsen():
    c = tr.classify([0, 200, 0], judgement_group=[4, 1, 2])
    assert c.group.tolist() == [4, 4, 2]
    assert "bank's assessment" in c.reasons[0][0]
    with pytest.raises(InputError):
        tr.classify([0], judgement_group=[6])


def test_contagion_across_a_borrowers_loans():
    # Article 5(5): borrower A has a loan 200 days past due (group 4): its other commercial loans follow
    c = tr.classify([200, 0, 100, 0, 0], borrower=["A", "A", "A", "B", "B"])
    assert c.group.tolist() == [4, 4, 4, 1, 1]
    assert "Art. 5(5)" in c.reasons[1][-1] and "Art. 5(5)" in c.reasons[2][-1]


def test_consumer_loans_may_be_assessed_loan_by_loan():
    dpd = [100, 0, 45]
    kw = dict(borrower=["A", "A", "A"], consumer_loan=[1, 1, 1])
    assert tr.classify(dpd, **kw).group.tolist() == [3, 2, 2]  # other consumer loans: not group 1
    assert tr.classify(dpd, consumer_loans_at_loan_level=False, **kw).group.tolist() == [3, 3, 3]
    # a non-performing commercial loan pulls commercial loans to its group, consumer loans to group 2
    mixed = tr.classify([200, 0, 0], borrower=["A", "A", "A"], consumer_loan=[0, 0, 1])
    assert mixed.group.tolist() == [4, 4, 2]
    # a non-performing consumer loan moves the borrower's other loans to group 2, not into default
    other = tr.classify([200, 0, 0], borrower=["A", "A", "A"], consumer_loan=[1, 0, 1])
    assert other.group.tolist() == [4, 2, 2]


def test_classification_does_not_depend_on_row_order():
    rng = np.random.default_rng(8)
    n = 300
    dpd = rng.choice([0, 0, 0, 20, 45, 100, 200, 400], n).astype(float)
    borrower = [f"B{int(b)}" for b in rng.integers(0, 90, n)]
    consumer = rng.random(n) < 0.4
    base = tr.classify(dpd, borrower=borrower, consumer_loan=consumer)
    perm = rng.permutation(n)
    shuffled = tr.classify(dpd[perm], borrower=[borrower[i] for i in perm], consumer_loan=consumer[perm])
    assert shuffled.group.tolist() == base.group[perm].tolist()
    # brute force of the rule
    own = tr.group_from_days_past_due(dpd)
    for i in range(n):
        mates = [j for j in range(n) if borrower[j] == borrower[i]]
        pulling = [own[j] for j in mates if own[j] >= 3 and not consumer[j]]
        any_npl = any(own[j] >= 3 for j in mates)
        expect = own[i]
        if not consumer[i] and pulling:
            expect = max(expect, max(pulling))
        if expect < 2 and any_npl:
            expect = 2
        assert base.group[i] == expect


def test_return_to_performing_article_6():
    ok = dict(
        arrears_collected_without_collateral=True,
        no_amount_past_due=True,
        last_two_payments_on_time=True,
        meets_group_1_or_2=True,
    )
    assert tr.performing_again(**ok).eligible
    v = tr.performing_again(**{**ok, "last_two_payments_on_time": False})
    assert not v.eligible and "6(3)(b)" in v.unmet[0]
    barred = tr.performing_again(**ok, partially_written_off=True)
    assert not barred.eligible and "6(4)" in barred.unmet[0]
    assert barred.to_dict() == {"eligible": False, "unmet": list(barred.unmet)}
    with pytest.raises(InputError, match="true or false"):
        tr.performing_again(**{**ok, "no_amount_past_due": "false"})


def test_restructured_non_performing_loan_article_7_5():
    ok = dict(
        non_performing_conditions_gone=True,
        no_delay_and_no_doubt=True,
        arrears_and_written_off_principal_collected=True,
    )
    assert not tr.restructured_npl_to_group_2(months_as_non_performing_since_restructuring=11, **ok).eligible
    assert tr.restructured_npl_to_group_2(months_as_non_performing_since_restructuring=12, **ok).eligible


def test_leaving_restructured_status_article_7_6_and_7_8():
    kw = dict(share_of_remaining_debt_repaid=0.0, max_days_past_due_any_loan=0, financial_difficulty_ended=True)
    # banks applying TFRS 9: three months, without the one-year and 10% conditions
    assert tr.exit_restructured_status(months_monitored_in_group_2=3, **kw).eligible
    assert not tr.exit_restructured_status(months_monitored_in_group_2=2, **kw).eligible
    # other banks: one year and 10% repaid
    assert not tr.exit_restructured_status(months_monitored_in_group_2=12, applies_tfrs9=False, **kw).eligible
    full = dict(kw, share_of_remaining_debt_repaid=0.10)
    assert tr.exit_restructured_status(months_monitored_in_group_2=12, applies_tfrs9=False, **full).eligible
    # loans that came back from non-performing status keep the one-year rule (literal reading of 7(8))
    assert not tr.exit_restructured_status(
        months_monitored_in_group_2=6, came_from_non_performing=True, **full
    ).eligible
    late = dict(full, max_days_past_due_any_loan=31)
    v = tr.exit_restructured_status(months_monitored_in_group_2=12, **late)
    assert not v.eligible and "7(6)(c)" in v.unmet[0]
    with pytest.raises(InputError):
        tr.exit_restructured_status(months_monitored_in_group_2=12, **{**full, "financial_difficulty_ended": "no"})
    with pytest.raises(InputError):
        tr.exit_restructured_status(months_monitored_in_group_2="12", **full)


def test_fixed_rates_for_banks_not_applying_tfrs9():
    assert tr.fixed_rate_provision(1, 1_000_000)["provision"] == pytest.approx(15_000)  # Art. 10(1)(a)
    assert tr.fixed_rate_provision(2, 1_000_000)["provision"] == pytest.approx(30_000)  # Art. 10(1)(b)
    assert [tr.fixed_rate_provision(g, 100)["provision"] for g in (3, 4, 5)] == pytest.approx([20, 50, 100])
    with pytest.raises(InputError):
        tr.fixed_rate_provision(6, 100)
    with pytest.raises(InputError):
        tr.fixed_rate_provision(1, -100)


def test_collateral_consideration_of_article_15():
    # receivable 100, mortgage (group 3, 60%) worth 100: 60 exempt, 40 subject to provision
    base, steps = tr.amount_subject_to_specific_provision(100, [(3, 100)])
    assert base == pytest.approx(40) and steps[0]["exempt"] == pytest.approx(60)
    # collateral above the receivable is ignored
    assert tr.amount_subject_to_specific_provision(100, [(3, 500)])[0] == pytest.approx(40)
    # most liquid first: cash 30 (100%) then mortgage on the remaining 70 (60%) -> exempt 30 + 42
    base, steps = tr.amount_subject_to_specific_provision(100, [(3, 200), (1, 30)])
    assert base == pytest.approx(28)
    assert [s["collateral_group"] for s in steps] == [1, 3] and steps[1]["value_used"] == pytest.approx(70)
    assert tr.fixed_rate_provision(4, 100, [(3, 200), (1, 30)])["provision"] == pytest.approx(14)
    # fully cash-covered: nothing left
    assert tr.amount_subject_to_specific_provision(100, [(1, 100), (3, 50)])[0] == pytest.approx(0)
    with pytest.raises(InputError):
        tr.amount_subject_to_specific_provision(100, [(7, 10)])
