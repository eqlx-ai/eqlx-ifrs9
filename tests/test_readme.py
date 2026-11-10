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
