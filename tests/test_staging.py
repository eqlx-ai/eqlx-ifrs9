import numpy as np
import pytest

from eqlx_ifrs9 import InputError, PDCurve
from eqlx_ifrs9.staging import DEFAULT_PRECEDENCE, Reason, StagingPolicy, assess, names

NAN = float("nan")


def one(policy, **kw):
    r = assess(policy, **{k: [v] for k, v in kw.items()})
    return int(r.stage[0]), set(r.reason_names(0)), r.primary_name(0)


def test_relative_threshold_is_a_ratio():
    p = StagingPolicy(relative_threshold=2.0, backstop_ratio=None)
    assert one(p, pd_now=0.0399, pd_origination=0.02)[0] == 1
    assert one(p, pd_now=0.04, pd_origination=0.02) == (2, {"pd_relative"}, "pd_relative")
    # the same absolute change matters more for a safer borrower (IFRS 9 B5.5.9)
    assert one(p, pd_now=0.002, pd_origination=0.001)[0] == 2  # +0.1 point, doubled
    assert one(p, pd_now=0.011, pd_origination=0.010)[0] == 1  # +0.1 point, +10%


def test_remaining_window_matching_changes_the_answer():
    # Origination curve over five years, reporting date after two; current three-year PD is 16%.
    orig = PDCurve.from_cumulative([0.01, 0.025, 0.045, 0.07, 0.10], periods_per_year=1)
    p = StagingPolicy(relative_threshold=2.0, backstop_ratio=None)
    matched = orig.forward(2).pd_over()
    assert 0.16 / matched == pytest.approx(2.08)
    assert one(p, pd_now=0.16, pd_origination=matched)[0] == 2
    # comparing with the full five-year origination PD would miss it (ratio 1.6)
    assert one(p, pd_now=0.16, pd_origination=orig.pd_over())[0] == 1


def test_bands_by_origination_pd():
    p = StagingPolicy(relative_threshold=1.5, relative_bands=((0.005, 3.0), (0.02, 2.0)), backstop_ratio=None)
    assert one(p, pd_now=0.008, pd_origination=0.004)[0] == 1  # needs x3 below 0.5%
    assert one(p, pd_now=0.012, pd_origination=0.004)[0] == 2
    assert one(p, pd_now=0.025, pd_origination=0.010)[0] == 2  # x2.5 against x2
    assert one(p, pd_now=0.08, pd_origination=0.05)[0] == 2  # x1.6 against x1.5 above the bands
    assert not p.warnings() or all("riskier" not in w for w in p.warnings())
    with pytest.raises(InputError):
        StagingPolicy(relative_threshold=2.0, relative_bands=((0.02, 2.0), (0.005, 3.0)))
    with pytest.raises(InputError):
        StagingPolicy(relative_bands=((0.02, 2.0),))


def test_absolute_threshold_and_combination():
    either = StagingPolicy(relative_threshold=2.0, absolute_threshold=0.01, backstop_ratio=None)
    both = either.with_changes(combine="and")
    # doubled but only +0.2 points
    assert one(either, pd_now=0.004, pd_origination=0.002) == (2, {"pd_relative"}, "pd_relative")
    assert one(both, pd_now=0.004, pd_origination=0.002)[0] == 1
    # +1.5 points but only x1.3
    assert one(either, pd_now=0.065, pd_origination=0.05) == (2, {"pd_absolute"}, "pd_absolute")
    assert one(both, pd_now=0.065, pd_origination=0.05)[0] == 1
    assert one(both, pd_now=0.03, pd_origination=0.01)[1] == {"pd_relative", "pd_absolute"}


def test_threefold_backstop_uses_its_own_pd_pair():
    p = StagingPolicy(relative_threshold=5.0, backstop_ratio=3.0)
    assert one(p, pd_now=0.03, pd_origination=0.01) == (2, {"pd_backstop"}, "pd_backstop")
    assert one(p, pd_now=0.0299, pd_origination=0.01)[0] == 1
    r = assess(
        StagingPolicy(relative_threshold=2.0),
        pd_now=[0.05],
        pd_origination=[0.04],
        backstop_pd_now=[0.03],
        backstop_pd_origination=[0.01],
    )
    assert r.stage[0] == 2 and r.reason_names(0) == ["pd_backstop"]


def test_days_past_due_presumptions():
    p = StagingPolicy(relative_threshold=2.0)
    kw = dict(pd_now=0.02, pd_origination=0.02)
    assert one(p, days_past_due=30, **kw)[0] == 1
    assert one(p, days_past_due=31, **kw) == (2, {"days_past_due_backstop"}, "days_past_due_backstop")
    assert one(p, days_past_due=90, **kw)[0] == 2
    stage, why, primary = one(p, days_past_due=91, **kw)
    assert stage == 3 and primary == "default_days_past_due" and "days_past_due_backstop" in why
    rebutted = p.with_changes(days_past_due_backstop=60, default_days_past_due=None)
    assert one(rebutted, days_past_due=45, **kw)[0] == 1 and one(rebutted, days_past_due=400, **kw)[0] == 2
    assert len([w for w in rebutted.warnings() if "presumption" in w]) == 2


def test_backstop_only_share_is_visible_through_the_primary_reason():
    p = StagingPolicy(relative_threshold=2.0)
    r = assess(p, pd_now=[0.05, 0.02], pd_origination=[0.02, 0.02], days_past_due=[40, 40])
    assert [r.primary_name(i) for i in range(2)] == ["pd_relative", "days_past_due_backstop"]
    assert r.has(Reason.DAYS_PAST_DUE_BACKSTOP).tolist() == [True, True]


def test_qualitative_flags_and_credit_impaired():
    p = StagingPolicy(relative_threshold=2.0)
    kw = dict(pd_now=0.02, pd_origination=0.02)
    assert one(p, forborne=True, **kw) == (2, {"forborne"}, "forborne")
    assert one(p, watchlist=True, **kw) == (2, {"watchlist"}, "watchlist")
    assert one(p, collective=True, **kw) == (2, {"collective"}, "collective")
    assert one(p, credit_impaired=True, **kw) == (3, {"credit_impaired"}, "credit_impaired")
    off = p.with_changes(forborne_is_stage2=False, watchlist_is_stage2=False)
    assert one(off, forborne=True, watchlist=True, **kw)[0] == 1


def test_grade_rules():
    p = StagingPolicy(notch_threshold=2, backstop_ratio=None)
    assert one(p, pd_now=0.01, grade_now=5, grade_origination=4)[0] == 1
    assert one(p, pd_now=0.01, grade_now=6, grade_origination=4) == (2, {"rating_notches"}, "rating_notches")
    # Illustrative Example 6: originated at grades 3-4 only, anything worse than 5 has increased significantly
    ie6 = StagingPolicy(max_grade_without_increase=5, backstop_ratio=None)
    assert [one(ie6, pd_now=0.01, grade_now=g)[0] for g in (3, 4, 5, 6, 9)] == [1, 1, 1, 2, 2]
    assert any("Example 6" in w for w in ie6.warnings())


def test_low_credit_risk_exemption_covers_the_pd_tests_only():
    p = StagingPolicy(relative_threshold=2.0, low_credit_risk_pd=0.003)
    stage, why, primary = one(p, pd_now=0.002, pd_origination=0.0005)
    assert (stage, why, primary) == (1, {"low_credit_risk"}, "none")
    assert one(p, pd_now=0.004, pd_origination=0.0005)[0] == 2  # no longer low risk: normal test applies
    stage, why, _ = one(p, pd_now=0.002, pd_origination=0.0005, days_past_due=45)
    assert stage == 2 and why == {"low_credit_risk", "days_past_due_backstop"}
    g = StagingPolicy(relative_threshold=2.0, low_credit_risk_grade=3)
    assert one(g, pd_now=0.01, pd_origination=0.001, grade_now=3)[0] == 1
    assert one(g, pd_now=0.01, pd_origination=0.001, grade_now=4)[0] == 2
    assert any("0.3%" in w for w in p.with_changes(low_credit_risk_pd=0.01).warnings())


def test_missing_origination_pd():
    p = StagingPolicy(relative_threshold=2.0)
    assert one(p, pd_now=0.02, pd_origination=NAN) == (2, {"no_origination_pd"}, "no_origination_pd")
    assert one(p.with_changes(missing_origination_pd="skip"), pd_now=0.02, pd_origination=NAN)[0] == 1
    with pytest.raises(InputError, match="no origination PD"):
        assess(p.with_changes(missing_origination_pd="error"), pd_now=[0.02], pd_origination=[NAN])
    # low credit risk at the reporting date avoids lifetime ECL (IFRS 9 7.2.20)
    lcr = p.with_changes(low_credit_risk_pd=0.003)
    assert one(lcr, pd_now=0.001, pd_origination=NAN)[0] == 1
    # an origination PD of zero and a positive PD now is an increase
    assert one(p, pd_now=0.01, pd_origination=0.0)[0] == 2
    assert one(p, pd_now=0.0, pd_origination=0.0)[0] == 1


def test_probation_and_cure_periods():
    p = StagingPolicy(relative_threshold=2.0, probation_periods=3, cure_periods=12)
    kw = dict(pd_now=0.02, pd_origination=0.02)
    assert one(p, previous_stage=2, periods_without_trigger=2, **kw) == (2, {"probation"}, "probation")
    assert one(p, previous_stage=2, periods_without_trigger=3, **kw)[0] == 1
    assert one(p, previous_stage=1, periods_without_trigger=0, **kw)[0] == 1
    assert one(p, previous_stage=3, periods_without_trigger=11, **kw) == (3, {"cure_period"}, "cure_period")
    # leaving stage 3 leads through stage 2: three more clean periods are needed
    assert one(p, previous_stage=3, periods_without_trigger=12, **kw) == (2, {"probation"}, "probation")
    assert one(p, previous_stage=3, periods_without_trigger=15, **kw)[0] == 1
    assert one(p.with_changes(probation_periods=0), previous_stage=3, periods_without_trigger=12, **kw)[0] == 1
    # a live trigger wins over probation as the primary reason
    assert one(p, previous_stage=2, periods_without_trigger=0, pd_now=0.05, pd_origination=0.02)[2] == "pd_relative"


def test_missing_origination_pd_counts_for_the_backstop_and_notch_rules_too():
    # the default policy has only the threefold backstop: still no comparison without an origination PD
    assert one(StagingPolicy(), pd_now=0.30, pd_origination=NAN) == (2, {"no_origination_pd"}, "no_origination_pd")
    notch = StagingPolicy(notch_threshold=2, backstop_ratio=None)
    assert one(notch, pd_now=0.30, pd_origination=NAN, grade_now=5)[0] == 2
    assert one(notch, pd_now=0.30, pd_origination=NAN, grade_now=5, grade_origination=5)[0] == 1
    assert one(StagingPolicy(backstop_ratio=None), pd_now=0.30, pd_origination=NAN)[0] == 1  # no test at all


def test_tiny_pds_do_not_trip_the_thresholds():
    p = StagingPolicy(relative_threshold=2.0)
    assert one(p, pd_now=1e-13, pd_origination=1e-13)[0] == 1
    assert one(p, pd_now=2e-13, pd_origination=5e-13)[0] == 1
    assert one(p, pd_now=2e-13, pd_origination=1e-13)[0] == 2


def test_precedence_can_be_changed():
    dpd_first = (Reason.DAYS_PAST_DUE_BACKSTOP, *[r for r in DEFAULT_PRECEDENCE if r != Reason.DAYS_PAST_DUE_BACKSTOP])
    p = StagingPolicy(relative_threshold=2.0, precedence=dpd_first)
    assert one(p, pd_now=0.05, pd_origination=0.02, days_past_due=40)[2] == "days_past_due_backstop"
    with pytest.raises(InputError):
        StagingPolicy(precedence=DEFAULT_PRECEDENCE[:-1])


def test_policy_warnings_follow_public_supervisory_findings():
    assert any("absolute" in w for w in StagingPolicy(absolute_threshold=0.01).warnings())
    assert any(
        "both be met" in w
        for w in StagingPolicy(relative_threshold=2, absolute_threshold=0.01, combine="and").warnings()
    )
    assert any("no quantitative" in w for w in StagingPolicy().warnings())
    assert any("backstop" in w for w in StagingPolicy(relative_threshold=2, backstop_ratio=None).warnings())
    assert any("looser than a threefold" in w for w in StagingPolicy(relative_threshold=2, backstop_ratio=4).warnings())
    assert any("only the backstop" in w for w in StagingPolicy(relative_threshold=4).warnings())
    upside_down = StagingPolicy(relative_threshold=3.0, relative_bands=((0.01, 2.0),), backstop_ratio=3.0)
    assert any("riskier" in w for w in upside_down.warnings())
    assert StagingPolicy(relative_threshold=2.0).warnings() == ()
    assert StagingPolicy(relative_threshold=2.0).to_dict()["relative_threshold"] == 2.0


@pytest.mark.parametrize(
    "kw",
    [
        {"relative_threshold": 0.5},
        {"backstop_ratio": 0.9},
        {"absolute_threshold": 0.0},
        {"combine": "xor"},
        {"probation_periods": -1},
        {"low_credit_risk_pd": 1.5},
        {"missing_origination_pd": "guess"},
        {"days_past_due_backstop": -3},
        {"forborne_is_stage2": "False"},
        {"backstop_ratio": "3"},
        {"relative_threshold": True},
        {"probation_periods": 1.5},
        {"notch_threshold": 0},
    ],
)
def test_bad_policies_are_rejected(kw):
    with pytest.raises(InputError):
        StagingPolicy(**kw)


def test_bad_inputs_are_rejected():
    p = StagingPolicy(relative_threshold=2.0)
    with pytest.raises(InputError):
        assess(p, pd_now=[1.2], pd_origination=[0.1])
    with pytest.raises(InputError):
        assess(p, pd_now=[0.1, 0.2], pd_origination=[0.1])
    with pytest.raises(InputError):
        assess(p, pd_now=[0.1], pd_origination=[-0.1])
    with pytest.raises(InputError):
        assess(p, pd_now=[0.1], pd_origination=[0.1], forborne=[True, False])
    for bad in (dict(days_past_due=[NAN]), dict(days_past_due=[-1]), dict(previous_stage=[7])):
        with pytest.raises(InputError):
            assess(p, pd_now=[0.1], pd_origination=[0.1], **bad)


def test_vectorised_result_equals_one_by_one():
    rng = np.random.default_rng(5)
    n = 400
    p = StagingPolicy(
        relative_threshold=2.0, absolute_threshold=0.02, probation_periods=2, cure_periods=3, notch_threshold=3
    )
    data = dict(
        pd_now=rng.uniform(0, 0.3, n),
        pd_origination=np.where(rng.random(n) < 0.1, np.nan, rng.uniform(0, 0.2, n)),
        days_past_due=rng.integers(0, 120, n).astype(float),
        grade_now=rng.integers(1, 10, n).astype(float),
        grade_origination=rng.integers(1, 10, n).astype(float),
        credit_impaired=rng.random(n) < 0.03,
        forborne=rng.random(n) < 0.05,
        watchlist=rng.random(n) < 0.05,
        previous_stage=rng.integers(1, 4, n).astype(float),
        periods_without_trigger=rng.integers(0, 5, n).astype(float),
    )
    whole = assess(p, **data)
    for i in range(n):
        single = assess(p, **{k: v[i : i + 1] for k, v in data.items()})
        assert single.stage[0] == whole.stage[i] and single.reasons[0] == whole.reasons[i]
        assert single.primary[0] == whole.primary[i]
    assert set(np.unique(whole.stage)) == {1, 2, 3}
    assert names(int(Reason.FORBORNE | Reason.PROBATION)) == ["forborne", "probation"]
