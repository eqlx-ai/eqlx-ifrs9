"""The numbers printed in the README are real."""

import re
from pathlib import Path

import pytest

from eqlx_ifrs9 import PDCurve, config, cycle, data, engine, expected_credit_loss, report, schedule, synthetic
from eqlx_ifrs9.staging import StagingPolicy, assess

README = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")


def test_library_example():
    loan = schedule.annuity(100_000, 0.30, 36)
    curve = PDCurve.flat(0.03, 36)
    res = expected_credit_loss(curve, loan.ead(), lgd=0.60, effective_interest_rate=0.31)
    assert (round(res.twelve_month, 2), round(res.lifetime, 2)) == (1406.99, 2351.36)
    down = cycle.pit_curve(curve, rho=0.06, z_by_year=[-1.6, -1.0, -0.4])
    assert round(expected_credit_loss(down, loan.ead(), 0.60, 0.31).lifetime, 2) == 4226.37
    at_origination = PDCurve.from_cumulative([0.01, 0.025, 0.045, 0.07, 0.10], periods_per_year=1)
    expected_then = at_origination.forward(2).pd_over()
    r = assess(StagingPolicy(relative_threshold=2.0), pd_now=[0.16], pd_origination=[expected_then])
    assert round(expected_then, 4) == 0.0769
    assert (int(r.stage[0]), r.reason_names(0), round(float(r.pd_ratio[0]), 2)) == (2, ["pd_relative"], 2.08)
    for shown in ("1406.99, 2351.36", "4226.37", "0.0769", "2.08"):
        assert shown in README


def test_command_line_output_shown_in_the_readme():
    s = synthetic.generate()
    res = engine.run(data.from_text(s.csv), config.from_dict(s.settings))
    text = report.text_summary(report.summary(res))
    block = re.search(r"```text\n(loans 2,011.*?)```", README, re.S)
    assert block is not None
    assert block.group(1).strip() == text.strip()
    d = engine.loan_detail(res, "CASE-GROUP-X-CURRENT")
    assert d["ecl"] == pytest.approx(2_750_000.0) and d["regulatory_group"] == 4
    assert '"ecl": 2750000.0' in README


def test_individual_assessment_shown_in_the_readme():
    from eqlx_ifrs9 import individual

    r = individual.assess(synthetic.individual_cases()["case_gone_concern.json"])
    e = r["exposures"][0]
    shown = [r["total"]["ecl"], e["gross_carrying_amount"]]
    for sc in e["scenarios"]:
        shown += [sc["recoverable"], sc["shortfall"]]
    sale = r["scenarios"][0]["collateral_sold"][0]["by_exposure"]["PROJECT-LOAN"]["flows"]
    shown += [f["present_value"] for f in sale if f["item"] == "sale at market value"]
    for value in shown:
        assert f"{value:,.0f}" in README
    assert f"{r['total']['coverage']:.1%} of the loan" in README
    pv = individual.collateral_recovery(
        200, 0.05, years_to_sale=5, haircut=0.10, selling_cost_rate=0.10, annual_cost_rate=0.05
    )["present_value"]
    assert f"# {pv:.2f}" in README


def test_review_and_period_sections_shown_in_the_readme():
    from eqlx_ifrs9 import period, review

    s = synthetic.generate()
    opening = engine.run(data.from_text(s.previous_csv), config.from_dict(s.previous_settings))
    closing = engine.run(data.from_text(s.csv), config.from_dict(s.settings))
    gone = s.write_offs_csv.split()[1:]

    block = re.search(r"```text\n(allowance .*?)```", README, re.S)
    assert block is not None
    assert block.group(1).strip() == period.text_movement(period.movement(opening, closing, write_offs=gone)).strip()

    rec = review.reconcile(closing)
    al = rec["allowance"]
    for name, x in [("all loans", al["total"])] + [(x["segment"], x) for x in al["by_segment"]]:
        row = f"| {name} | {x['reported']:,.0f} | {x['recalculated']:,.0f} | {x['relative']:+.1%} | {x['verdict']} |"
        assert row in README
    gap = al["gap_of_loans_with_both_figures"]
    assert f"{rec['stages']['reported_in_a_lower_stage']['loans']} loans are reported in stage 1" in README
    for value in (gap["difference"], gap["from_staging"], gap["from_measurement"]):
        assert f"{value:,.0f}" in README

    resp = review.response(closing.loans, closing.config)
    for pt in resp["points"]:
        shift = "0" if pt["shift"] == 0 else f"{pt['shift']:+g}"
        row = f"| {shift} | {pt['ecl_stages_held']:,.0f} | {pt['ecl_restaged']:,.0f} | {pt['stage_2_share_of_exposure']:.1%} |"
        assert row in README
    assert f"costs {resp['asymmetry']['ratio']:.1f} times" in README

    q = period.staging_quality(opening, closing, write_offs=gone)
    inflow, dr = q["stage_3_inflow"], q["default_rate_by_opening_stage"]
    assert f"{inflow['loans']} loans went to stage 3, {inflow['directly_from_stage_1']} of them straight" in README
    assert (
        f"default rate {dr['stage_1']['rate']:.2%} in stage 1 against {dr['stage_2']['rate']:.1%} in stage 2" in README
    )
    assert f"{q['what_became_of_stage_2']['back_to_stage_1']:.1%} of stage 2 went back" in README
    bt = period.backtest(opening, closing, write_offs=gone, realised_z=-0.6)["total"]
    assert f"{bt['expected_weighted']:.1f} defaults expected, {bt['defaults']} happened" in README
    assert (
        f"give {bt['expected_realised_economy']:.1f}, so the forecast was off by {bt['forecast_error']:+.1f}" in README
    )
    assert f"the model by {bt['model_error']:+.1f}" in README
    ch = review.challenger(opening, closing, write_offs=gone)
    seg = ch["segments"][0]
    assert (
        f"corporate allowance {seg['with_observed_matrix']:,.0f} against {seg['reported']:,.0f} reported: "
        f"{seg['relative']:+.1%}, {seg['verdict']} ({ch['observed'][0]['observations']} loans observed)"
    ) in README


def test_every_mcp_tool_is_listed():
    pytest.importorskip("mcp")
    import asyncio

    from mcp import Client

    from eqlx_ifrs9.mcp_server import build_server

    async def go():
        async with Client(build_server(Path(__file__).parent)) as c:
            return await c.list_tools()

    for tool in asyncio.run(go()).tools:
        assert f"`{tool.name}`" in README
