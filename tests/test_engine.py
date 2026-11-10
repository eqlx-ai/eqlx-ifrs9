"""The portfolio run against the scalar building blocks, the planted cases and its own invariants."""

import json

import numpy as np
import pytest

from eqlx_ifrs9 import (
    InputError,
    PDCurve,
    config,
    cycle,
    data,
    engine,
    expected_credit_loss,
    report,
    schedule,
    synthetic,
)
from eqlx_ifrs9.staging import Reason

MATRIX = [
    [0.92, 0.06, 0.015, 0.005],
    [0.05, 0.85, 0.07, 0.03],
    [0.01, 0.09, 0.75, 0.15],
    [0.0, 0.0, 0.0, 1.0],
]
SETTINGS = {
    "pd_basis": "long_run",
    "staging": {"relative_threshold": 2.0},
    "scenarios": [
        {"name": "base", "weight": 0.6, "z": [0.0, 0.2]},
        {"name": "down", "weight": 0.25, "z": [-1.5, -1.0]},
        {"name": "up", "weight": 0.15, "z": [1.0, 0.5]},
    ],
    "base_scenario": "base",
    "forecast": {"reversion_years": 2},
    "segments": {"*": {"rho": 0.12, "cpr": 0.05}, "corp": {"rho": 0.2, "grades": ["A", "B", "C"], "matrix": MATRIX}},
}


def random_portfolio(n: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    head = (
        "loan_id,segment,product,amortization,balance,undrawn,ccf,ccf_12m,rate,eir,remaining_months,"
        "months_on_book,pd_12m,grade,grade_origination,pd_12m_origination,lgd,dpd"
    )
    rows = [head]
    for i in range(n):
        corp = rng.random() < 0.4
        product = "revolving" if rng.random() < 0.15 else "term"
        rows.append(
            ",".join(
                str(v)
                for v in [
                    f"L{i}",
                    "corp" if corp else "retail",
                    product,
                    ["bullet", "annuity", "equal_principal"][int(rng.integers(3))],
                    round(float(rng.uniform(100, 1e5)), 2),
                    round(float(rng.uniform(0, 2e4)), 2) if product == "revolving" else 0,
                    0.5,
                    0.2,
                    round(float(rng.uniform(0, 0.3)), 4),
                    round(float(rng.uniform(0.01, 0.35)), 4),
                    int(rng.integers(1, 90)),
                    int(rng.integers(0, 40)),
                    "" if corp else round(float(rng.uniform(0.001, 0.2)), 5),
                    "ABC"[int(rng.integers(3))] if corp else "",
                    "ABC"[int(rng.integers(3))] if corp else "",
                    "" if corp else round(float(rng.uniform(0.001, 0.1)), 5),
                    round(float(rng.uniform(0.1, 0.9)), 3),
                    int(rng.choice([0, 0, 0, 10, 40, 100])),
                ]
            )
        )
    return "\n".join(rows) + "\n"


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"timing": "mid", "ead_basis": "mid"},
        {"ead_basis": "start_plus_interest", "rate_convention": "effective"},
        {
            "pd_basis": "current",
            "current_z": -0.7,
            "forecast": {"reversion_years": 3, "method": "exponential", "decay": 0.6},
        },
    ],
)
def test_array_run_equals_the_scalar_building_blocks(extra):
    loans = data.from_text(random_portfolio(250, seed=3))
    res = engine.run(loans, config.from_dict({**SETTINGS, **extra}))
    for i in range(loans.n):
        d = engine.loan_detail(res, f"L{i}")
        assert d["weighted"]["twelve_month_ecl"] == pytest.approx(res.ecl_12m[i], rel=1e-10, abs=1e-9)
        assert d["weighted"]["lifetime_ecl"] == pytest.approx(res.ecl_lifetime[i], rel=1e-10, abs=1e-9)
        assert d["ecl"] == pytest.approx(res.ecl[i], rel=1e-10, abs=1e-9)
        for k, s in enumerate(d["scenarios"]):
            assert s["lifetime_pd"] == pytest.approx(res.scenario_pd_lifetime[k, i], abs=1e-12)
            assert s["twelve_month_pd"] == pytest.approx(res.scenario_pd_12m[k, i], abs=1e-12)


def test_chunking_does_not_change_anything(monkeypatch):
    loans = data.from_text(random_portfolio(120, seed=9))
    cfg = config.from_dict(SETTINGS)
    whole = engine.run(loans, cfg)
    monkeypatch.setattr(engine, "CHUNK_CELLS", 200)
    pieces = engine.run(loans, cfg)
    assert pieces.ecl == pytest.approx(whole.ecl, rel=1e-13)
    assert pieces.stage.tolist() == whole.stage.tolist()


def test_result_does_not_depend_on_row_order():
    text = random_portfolio(150, seed=4).strip().split("\n")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(text) - 1)
    shuffled = "\n".join([text[0]] + [text[1 + i] for i in perm])
    cfg = config.from_dict({**SETTINGS, "profile": "tr_bddk"})
    a, b = engine.run(data.from_text("\n".join(text)), cfg), engine.run(data.from_text(shuffled), cfg)
    assert b.ecl == pytest.approx(a.ecl[perm], rel=1e-12)
    assert b.stage.tolist() == a.stage[perm].tolist() and b.reasons.tolist() == a.reasons[perm].tolist()
    assert report.summary(a)["total"]["ecl"] == pytest.approx(report.summary(b)["total"]["ecl"], rel=1e-12)


def one_loan(**cols) -> data.Loans:
    base = dict(loan_id="A", balance=1000, eir=0.06, remaining_months=36, lgd=0.45, pd_12m=0.02)
    base.update(cols)
    return data.from_text(",".join(base) + "\n" + ",".join(str(v) for v in base.values()) + "\n")


def test_single_loan_reproduces_the_closed_form():
    policy = {"staging": {"relative_threshold": 2.0}}
    res = engine.run(one_loan(), config.from_dict(policy))
    assert res.ecl_lifetime[0] == pytest.approx(24.242138103115, abs=1e-9)
    assert res.ecl_12m[0] == pytest.approx(8.722449843941, abs=1e-9)
    assert res.stage[0] == 2 and res.ecl[0] == res.ecl_lifetime[0]  # no origination PD: lifetime ECL (IFRS 9 7.2.20)
    assert any("no origination PD" in w for w in res.warnings)
    known = engine.run(one_loan(pd_12m_origination=0.02), config.from_dict(policy))
    assert known.stage[0] == 1 and known.ecl[0] == pytest.approx(8.722449843941, abs=1e-9)
    assert any("single scenario" in w for w in known.warnings)


def test_default_contract_rate_is_consistent_with_the_discounting():
    # no contract rate: the instalments are priced at the effective interest rate, so their present
    # value at the discount factors of the run is the balance
    loans = one_loan(amortization="annuity", eir=0.48)
    sched = engine.loan_schedule(loans, config.from_dict({}), 0)
    months = np.arange(1, 37)
    assert float(np.sum(sched.payment * 1.48 ** (-months / 12))) == pytest.approx(1000.0)
    # a contract rate, when given, is nominal by default
    quoted = engine.loan_schedule(one_loan(amortization="annuity", eir=0.48, rate=0.48), config.from_dict({}), 0)
    assert quoted.interest[0] == pytest.approx(1000 * 0.04) and sched.interest[0] == pytest.approx(
        1000 * (1.48 ** (1 / 12) - 1)
    )


def test_tiny_contract_rate_does_not_break_the_run():
    for rate in (1e-15, 1e-17, 0.0):
        res = engine.run(one_loan(amortization="annuity", rate=rate, pd_12m_origination=0.02), config.from_dict({}))
        assert np.isfinite(res.ecl[0]) and res.ecl_lifetime[0] == pytest.approx(
            engine.loan_detail(res, "A")["weighted"]["lifetime_ecl"]
        )


def test_segments_must_be_configured_when_scenarios_shift_pds():
    scen = {"pd_basis": "long_run", "scenarios": [{"name": "s", "weight": 1.0, "z": [-2.5]}]}
    with pytest.raises(InputError, match="no entry under 'segments'"):
        engine.run(one_loan(segment="Corporate"), config.from_dict({**scen, "segments": {"corporate": {"rho": 0.2}}}))
    engine.run(one_loan(segment="Corporate"), config.from_dict({**scen, "segments": {"*": {"rho": 0.2}}}))
    engine.run(one_loan(segment="Corporate"), config.from_dict({}))  # nothing to shift: no correlation needed


def test_grades_without_a_grade_scale():
    loans = one_loan(grade="7", grade_origination="1", pd_12m_origination=0.02)
    with pytest.raises(InputError, match="defines no 'grades'"):
        engine.run(loans, config.from_dict({"staging": {"notch_threshold": 2}}))
    assert any("grade columns" in w for w in engine.run(loans, config.from_dict({})).warnings)


def test_probation_without_history_is_reported():
    cfg = config.from_dict({"staging": {"relative_threshold": 2.0, "probation_periods": 3}})
    res = engine.run(one_loan(pd_12m_origination=0.02, previous_stage=2), cfg)
    assert res.stage[0] == 1 and any("periods_without_trigger" in w for w in res.warnings)


def test_life_under_twelve_months():
    res = engine.run(one_loan(remaining_months=6, pd_12m_origination=0.02), config.from_dict({}))
    assert res.ecl_12m[0] == res.ecl_lifetime[0] == pytest.approx(4.446772038869, abs=1e-9)
    assert res.scenario_pd_12m[0, 0] == pytest.approx(1 - 0.98**0.5)


def test_stage_three_uses_the_loss_rate_in_default():
    res = engine.run(one_loan(dpd=120, lgd_default=0.8, undrawn=500, ccf=0.4), config.from_dict({}))
    assert res.stage[0] == 3 and res.ecl[0] == pytest.approx((1000 + 0.4 * 500) * 0.8)
    fallback = engine.run(one_loan(dpd=120), config.from_dict({}))
    assert fallback.ecl[0] == pytest.approx(1000 * 0.45)


def test_undrawn_limit_uses_different_drawdown_for_12_months_and_lifetime():
    kw = dict(product="revolving", remaining_months=30, undrawn=400, ccf=0.5, ccf_12m=0.1, pd_12m_origination=0.02)
    res = engine.run(one_loan(**kw), config.from_dict({}))
    curve = PDCurve.flat(0.02, 30)
    life = expected_credit_loss(curve, np.full(30, 1000 + 0.5 * 400), 0.45, 0.06)
    first = expected_credit_loss(curve, np.full(30, 1000 + 0.1 * 400), 0.45, 0.06)
    assert res.ecl_lifetime[0] == pytest.approx(life.lifetime) and res.ecl_12m[0] == pytest.approx(first.twelve_month)
    assert res.exposure[0] == pytest.approx(1200.0)


def test_scenarios_are_weighted_on_ecl_not_on_inputs():
    cfg = config.from_dict(
        {
            "pd_basis": "long_run",
            "scenarios": [
                {"name": "base", "weight": 0.5, "z": [0.0]},
                {"name": "down", "weight": 0.2, "z": [-1.5]},
                {"name": "up", "weight": 0.3, "z": [1.0]},
            ],
            "base_scenario": "base",
            "segments": {"*": {"rho": 0.12}},
        }
    )
    res = engine.run(one_loan(remaining_months=12, eir=0.0, pd_12m_origination=0.02), cfg)
    pds = cycle.conditional_pd(0.02, 0.12, [0.0, -1.5, 1.0])
    assert res.scenario_pd_12m[:, 0] == pytest.approx(pds)
    assert res.ecl_12m[0] == pytest.approx(float(np.array([0.5, 0.2, 0.3]) @ pds) * 0.45 * 1000)
    table = report.scenario_table(res)
    assert table["weighted_ecl"] == pytest.approx(8.5126, abs=1e-4)
    assert table["scenarios"][0]["ecl_same_stages"] == pytest.approx(6.4293, abs=1e-4)
    assert table["weighted_to_base"] == pytest.approx(1.324, abs=1e-3)


def test_current_pd_basis_reproduces_the_given_pd_and_reverts_to_the_long_run():
    cfg = config.from_dict(
        {
            "pd_basis": "current",
            "current_z": -1.0,
            "scenarios": [{"name": "stay", "weight": 1.0, "z": [-1.0]}],
            "segments": {"*": {"rho": 0.15}},
        }
    )
    res = engine.run(one_loan(pd_12m=0.05), cfg)
    years = engine.loan_detail(res, "A")["scenarios"][0]["by_year"]
    long_run = 0.028469522232711406  # Phi(Phi^-1(0.05) sqrt(0.85) - sqrt(0.15))
    assert float(cycle.conditional_pd(long_run, 0.15, -1.0)[0]) == pytest.approx(0.05)
    assert years[0]["pd"] == pytest.approx(0.05)
    assert years[1]["pd"] == pytest.approx(0.95 * long_run)


def test_scenario_without_z_uses_the_pd_as_given():
    cfg = config.from_dict({"scenarios": [{"name": "only", "weight": 1.0}], "segments": {"*": {"rho": 0.3}}})
    assert engine.run(one_loan(), cfg).ecl_lifetime[0] == pytest.approx(24.242138103115, abs=1e-9)


def test_graded_loan_follows_the_matrix():
    loans = data.from_text(
        "loan_id,segment,grade,grade_origination,balance,eir,remaining_months,months_on_book,lgd\nG,corp,B,A,1000,0.0,24,12,1.0\n"
    )
    cfg = config.from_dict(
        {"segments": {"corp": {"grades": ["A", "B", "C"], "matrix": MATRIX}}, "staging": {"pd_measure": "lifetime"}}
    )
    res = engine.run(loans, cfg)
    two_years = np.linalg.matrix_power(np.array(MATRIX), 2)[1, 3]
    assert res.scenario_pd_lifetime[0, 0] == pytest.approx(two_years, abs=2e-4)  # through the fitted generator
    assert res.ecl_lifetime[0] == pytest.approx(1000 * res.scenario_pd_lifetime[0, 0])
    # origination: grade A, one year on book, two years left -> forward PD of years 2-3 from the matrix
    c = [np.linalg.matrix_power(np.array(MATRIX), k)[0, 3] for k in (1, 3)]
    assert res.pd_origination[0] == pytest.approx((c[1] - c[0]) / (1 - c[0]), abs=2e-4)


def test_staging_basis_weighted_or_base():
    loan = one_loan(pd_12m=0.03, pd_12m_origination=0.0165)
    weighted = {**SETTINGS, "segments": {"*": {"rho": 0.3}}}
    res = engine.run(loan, config.from_dict(weighted))
    base_only = engine.run(
        loan, config.from_dict({**weighted, "staging": {"relative_threshold": 2.0, "scenario_basis": "base"}})
    )
    assert res.pd_now[0] > base_only.pd_now[0]
    assert res.scenario_stage[:, 0].tolist() == [1, 2, 1] or res.scenario_stage[1, 0] == 2


def test_twelve_month_measure_warns_on_bullet_loans():
    cfg = config.from_dict({"staging": {"relative_threshold": 2.0, "pd_measure": "twelve_month"}})
    res = engine.run(one_loan(pd_12m_origination=0.02), cfg)
    assert any("B5.5.14" in w for w in res.warnings)
    assert res.pd_now[0] == pytest.approx(0.02) and res.pd_origination[0] == pytest.approx(0.02)


def test_planted_cases_of_the_sample_portfolio():
    for profile in ("tr_bddk", None):
        s = synthetic.generate(300, seed=7, profile=profile)
        loans = data.from_text(s.csv)
        res = engine.run(loans, config.from_dict(s.settings))
        for p in s.planted:
            i = loans.index_of(p.loan_id)
            primary = Reason(int(res.primary[i])).name.lower() if res.primary[i] else "none"
            assert (int(res.stage[i]), primary) == (p.stage, p.primary_reason), p.loan_id
        i = loans.index_of("CASE-SHORT-LIFE")
        assert res.ecl_12m[i] == res.ecl_lifetime[i]
        i = loans.index_of("CASE-SCENARIO")
        assert res.scenario_stage[1, i] == 2 and res.stage[i] == 1
        i = loans.index_of("CASE-LOW-RISK")
        assert res.reasons[i] & int(Reason.LOW_CREDIT_RISK)


def test_sample_is_reproducible_and_its_matrix_is_embeddable():
    assert synthetic.generate(50, seed=3).csv == synthetic.generate(50, seed=3).csv
    assert synthetic.generate(50, seed=3).csv != synthetic.generate(50, seed=4).csv
    res = engine.run(data.from_text(synthetic.generate(200).csv), config.from_dict(synthetic.settings()))
    assert not any("approximation" in w for w in res.warnings)


def test_turkey_profile_in_the_run():
    s = synthetic.generate(0)
    loans = data.from_text(s.csv)
    res = engine.run(loans, config.from_dict(s.settings))
    i, j = loans.index_of("CASE-GROUP-X-CURRENT"), loans.index_of("CASE-GROUP-X-LATE")
    assert res.group is not None and res.group[i] == 4 and res.group[j] == 4
    d = engine.loan_detail(res, "CASE-GROUP-X-CURRENT")
    assert d["regulatory_group"] == 4 and "Art. 5(5)" in d["regulatory_reasons"][-1]
    assert d["measurement"] == "exposure x loss rate in default"
    relief = engine.run(loans, config.from_dict({**s.settings, "profile_options": {"as_of": "2020-12-31"}}))
    k = loans.index_of("CASE-DPD-ONLY")
    assert res.stage[k] == 2 and relief.stage[k] == 1  # 45 days: under the 2020-21 relief still group 1
    assert any("90 and 180" in w for w in relief.warnings)
    with pytest.raises(InputError):
        engine.run(loans, config.from_dict({**s.settings, "profile_options": {"as_of": "31.12.2020"}}))


def test_summary_tables_add_up():
    s = synthetic.generate(400)
    res = engine.run(data.from_text(s.csv), config.from_dict(s.settings))
    table = report.summary(res)
    json.dumps(table)
    total = table["total"]
    assert total["loans"] == 411 and total["ecl"] == pytest.approx(float(res.ecl.sum()))
    for key in ("loans", "exposure", "ecl"):
        assert sum(r[key] for r in table["by_stage"]) == pytest.approx(total[key])
        assert sum(r[key] for r in table["by_segment"]) == pytest.approx(total[key])
        assert sum(r[key] for r in table["regulatory_groups"]) == pytest.approx(total[key])
    stage2 = table["by_stage"][1]
    assert sum(r["loans"] for r in table["stage_2_by_primary_reason"]) == stage2["loans"]
    sc = table["scenarios"]
    assert sc["weighted_ecl"] == pytest.approx(total["ecl"])
    same = [r["ecl_same_stages"] for r in sc["scenarios"]]
    stage3 = table["by_stage"][2]["ecl"]
    # stages 1 and 2 are linear in the scenario weights once the stages are fixed
    assert float(res.config.weights @ np.array(same)) == pytest.approx(total["ecl"])
    assert same[1] > same[0] > same[2] > stage3
    assert sc["scenarios"][1]["ecl_restaged"] > same[1]
    mig = table["migration_from_previous_stage"]
    assert sum(map(sum, mig["loans"])) + mig["loans_without_previous_stage"] == total["loans"]
    assert table["checks"]["stage_1_with_threefold_pd_increase"]["loans"] == 0  # the backstop is on
    assert "CASE" not in report.text_summary(table) and "stage 2 by primary reason" in report.text_summary(table)


def test_threefold_check_shows_what_a_loose_policy_leaves_in_stage_one():
    s = synthetic.generate(400)
    loose = {**s.settings, "staging": {"relative_threshold": 6.0, "backstop_ratio": None}}
    res = engine.run(data.from_text(s.csv), config.from_dict(loose))
    c = report.checks(res)["stage_1_with_threefold_pd_increase"]
    assert c["loans"] > 0 and 0 < c["share_of_stage_1_exposure"] < 1
    assert any("backstop" in w for w in res.warnings)


def test_loan_rows_filters_and_sorting():
    s = synthetic.generate(300)
    res = engine.run(data.from_text(s.csv), config.from_dict(s.settings))
    top = report.loan_rows(res, stage=2, sort="ecl", limit=5)
    assert top["returned"] == 5 and all(r["stage"] == 2 for r in top["loans"])
    assert [r["ecl"] for r in top["loans"]] == sorted((r["ecl"] for r in top["loans"]), reverse=True)
    forb = report.loan_rows(res, reason="forborne", limit=500)
    assert forb["matching"] == forb["returned"] and all("forborne" in r["reasons"] for r in forb["loans"])
    group = report.loan_rows(res, borrower_id="C-GROUP-X")
    assert {r["loan_id"] for r in group["loans"]} == {"CASE-GROUP-X-LATE", "CASE-GROUP-X-CURRENT"}
    assert report.loan_rows(res, segment="cards", limit=1)["loans"][0]["segment"] == "cards"
    for bad in (dict(stage=4), dict(reason="mood"), dict(sort="name")):
        with pytest.raises(InputError):
            report.loan_rows(res, **bad)


def test_compare_two_runs():
    s = synthetic.generate(300)
    loans = data.from_text(s.csv)
    a = engine.run(loans, config.from_dict(s.settings))
    b = engine.run(loans, config.from_dict(config.with_overrides(s.settings, {"staging.relative_threshold": 3.0})))
    c = report.compare(a, b, limit=3)
    assert c["ecl_change"] == pytest.approx(float(b.ecl.sum() - a.ecl.sum())) and c["ecl_change"] < 0
    moves = np.array(c["stage_moves_rows_before_columns_after"])
    assert moves.sum() == loans.n and moves[1, 0] == c["loans_changing_stage"] > 0 and moves[0, 1] == 0
    assert len(c["largest_changes"]) == 3
    assert report.compare(a, a)["ecl_change"] == 0.0
    with pytest.raises(InputError):
        report.compare(a, engine.run(one_loan(), config.from_dict({})))


def test_loan_detail():
    s = synthetic.generate(0)
    res = engine.run(data.from_text(s.csv), config.from_dict(s.settings))
    d = engine.loan_detail(res, "CASE-SHORTER-LIFE", periods=True)
    json.dumps(d)
    assert d["stage"] == 2 and d["staging_test"]["ratio"] > 2.0
    assert d["staging_test"]["pd_at_origination_for_the_remaining_window"] == pytest.approx(0.026328, abs=1e-6)
    assert len(d["scenarios"]) == 3 and len(d["scenarios"][0]["periods"]) == 36
    assert sum(y["ecl"] for y in d["scenarios"][0]["by_year"]) == pytest.approx(d["scenarios"][0]["lifetime_ecl"])
    assert d["ecl"] == d["weighted"]["lifetime_ecl"] and d["measurement"] == "lifetime ECL"
    assert d["inputs"]["remaining_months"] == 36
    with pytest.raises(InputError):
        engine.loan_detail(res, "NOPE")


def test_scalar_schedule_matches_the_library_schedules():
    loans = data.from_text(random_portfolio(40, seed=6))
    cfg = config.from_dict(SETTINGS)
    for i in range(loans.n):
        sched = engine.loan_schedule(loans, cfg, i)
        assert sched.n_periods == int(loans.num["remaining_months"][i])
        assert sched.balance_start[0] == pytest.approx(loans.num["balance"][i])
    assert isinstance(engine.loan_schedule(loans, cfg, 0), schedule.Schedule)


def test_large_portfolio_runs_in_reasonable_time():
    import time

    s = synthetic.generate(30_000, seed=1)
    loans = data.from_text(s.csv)
    t0 = time.perf_counter()
    res = engine.run(loans, config.from_dict(s.settings))
    assert time.perf_counter() - t0 < 30
    assert np.all(np.isfinite(res.ecl)) and np.all(res.ecl >= 0) and np.all(res.ecl <= res.exposure * 1.0000001 + 1e-9)
    assert np.all(res.ecl_12m <= res.ecl_lifetime + 1e-9)
