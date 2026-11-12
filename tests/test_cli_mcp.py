import asyncio
import csv
import json
import os

import pytest

from eqlx_ifrs9 import calc, synthetic
from eqlx_ifrs9.cli import main, parse_overrides
from eqlx_ifrs9.errors import InputError


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "book"
    assert main(["sample", str(d), "--loans", "150", "--seed", "7"]) == 0
    return d


def run(capsys, *args):
    code = main([str(a) for a in args])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_sample_writes_three_files(folder):
    assert sorted(p.name for p in folder.iterdir()) == [
        "case_going_concern.json",
        "case_gone_concern.json",
        "loans.csv",
        "loans_previous.csv",
        "planted_cases.json",
        "settings.json",
        "settings_previous.json",
        "write_offs.csv",
    ]
    assert (folder / "loans.csv").read_text() == synthetic.generate(150, 7).csv
    assert len(json.loads((folder / "planted_cases.json").read_text())) == 11


def test_parse_overrides():
    assert parse_overrides(["staging.relative_threshold=2.5", "profile=null", "timing=mid"]) == {
        "staging.relative_threshold": 2.5,
        "profile": None,
        "timing": "mid",
    }
    with pytest.raises(InputError):
        parse_overrides(["nothing"])


def test_run_text_json_and_csv_output(folder, capsys, tmp_path):
    loans, settings = folder / "loans.csv", folder / "settings.json"
    code, out, _ = run(capsys, "run", loans, "--config", settings)
    assert code == 0 and "stage 2 by primary reason" in out and "weighted / base" in out
    out_csv = tmp_path / "result.csv"
    code, out, _ = run(capsys, "run", loans, "--config", settings, "--json", "--out", out_csv)
    table = json.loads(out)
    rows = list(csv.DictReader(out_csv.open()))
    assert len(rows) == table["total"]["loans"] == 161
    assert sum(float(r["ecl"]) for r in rows) == pytest.approx(table["total"]["ecl"])
    planted = {r["loan_id"]: r for r in rows if r["loan_id"].startswith("CASE")}
    assert planted["CASE-DPD-ONLY"]["primary_reason"] == "days_past_due_backstop"
    assert planted["CASE-GROUP-X-CURRENT"]["regulatory_group"] == "4"


def test_run_with_override(folder, capsys):
    loans, settings = folder / "loans.csv", folder / "settings.json"
    _, base, _ = run(capsys, "run", loans, "--config", settings, "--json")
    _, loose, _ = run(capsys, "run", loans, "--config", settings, "--json", "--set", "staging.relative_threshold=3")
    assert json.loads(loose)["by_stage"][1]["loans"] < json.loads(base)["by_stage"][1]["loans"]
    assert json.loads(loose)["settings"]["staging"]["relative_threshold"] == 3


def test_loan_list_compare_describe(folder, capsys):
    loans, settings = folder / "loans.csv", folder / "settings.json"
    code, out, _ = run(capsys, "loan", loans, "CASE-PD-TRIPLED", "--config", settings)
    d = json.loads(out)
    assert code == 0 and d["stage"] == 2 and d["primary_reason"] == "pd_relative" and "periods" not in d["scenarios"][0]
    _, out, _ = run(capsys, "loan", loans, "CASE-PD-TRIPLED", "--config", settings, "--periods")
    assert len(json.loads(out)["scenarios"][0]["periods"]) == 36
    _, out, _ = run(capsys, "list", loans, "--config", settings, "--stage", "3", "--limit", "3")
    assert all(r["stage"] == 3 for r in json.loads(out)["loans"])
    _, out, _ = run(capsys, "compare", loans, "--config", settings, "--set", "staging.relative_threshold=3")
    assert json.loads(out)["ecl_change"] < 0
    _, out, _ = run(capsys, "describe", loans)
    assert json.loads(out)["loans"] == 161


def test_calculator_command(capsys):
    code, out, _ = run(capsys, "ecl", "--balance", 1000000, "--pd", 0.005, "--lgd", 0.25, "--eir", 0, "--months", 12)
    assert code == 0 and json.loads(out)["twelve_month_ecl"] == pytest.approx(1250.0)


def test_errors_exit_with_2(folder, capsys):
    loans, settings = folder / "loans.csv", folder / "settings.json"
    assert run(capsys, "run", str(loans) + ".missing")[0] == 2
    code, _, err = run(capsys, "loan", loans, "NOPE", "--config", settings)
    assert code == 2 and "unknown loan_id" in err
    code, _, err = run(capsys, "run", loans, "--config", settings, "--set", "staging.mood=1")
    assert code == 2 and "unknown setting" in err
    assert run(capsys, "compare", loans, "--config", settings)[0] == 2
    bad = folder / "bad.json"
    bad.write_text("{oops")
    assert run(capsys, "run", loans, "--config", bad)[0] == 2
    assert run(capsys, "mcp", "--root", folder / "nowhere")[0] == 2


def test_calculator_functions():
    flat = calc.instrument_ecl(balance=1000, lgd=0.45, eir=0.06, remaining_months=36, pd_12m=0.02)
    assert flat["lifetime_ecl"] == pytest.approx(24.242138103115, abs=1e-9)
    assert flat["undiscounted_lifetime_ecl"] == pytest.approx(0.45 * 1000 * (1 - 0.98**3))
    assert sum(y["ecl"] for y in flat["by_year"]) == pytest.approx(flat["lifetime_ecl"])
    # an annual cumulative curve refined to months gives the same lifetime PD
    curve = calc.instrument_ecl(
        balance=1000,
        lgd=0.45,
        eir=0.06,
        remaining_months=36,
        cumulative_pd=[0.02, 0.05, 0.09],
        cumulative_pd_frequency=1,
    )
    assert curve["lifetime_pd"] == pytest.approx(0.09) and curve["twelve_month_pd"] == pytest.approx(0.02)
    stressed = calc.instrument_ecl(
        balance=1000, lgd=0.45, eir=0.06, remaining_months=36, pd_12m=0.02, z_by_year=[-1.0], rho=0.12
    )
    assert stressed["twelve_month_pd"] == pytest.approx(0.034377, abs=1e-6)
    amort = calc.instrument_ecl(
        balance=1000,
        lgd=0.45,
        eir=0.06,
        remaining_months=36,
        pd_12m=0.02,
        amortization="annuity",
        undrawn=200,
        ccf=0.5,
        cpr=0.1,
    )
    assert amort["lifetime_ecl"] > 0 and amort["by_year"][0]["average_ead"] > amort["by_year"][2]["average_ead"]
    for bad in (
        dict(pd_12m=0.02, cumulative_pd=[0.1]),
        dict(),
        dict(cumulative_pd=[0.02], cumulative_pd_frequency=1),
        dict(pd_12m=0.02, amortization="balloon"),
    ):
        with pytest.raises(InputError):
            calc.instrument_ecl(balance=1000, lgd=0.45, eir=0.06, remaining_months=36, **bad)
    with pytest.raises(InputError):
        calc.instrument_ecl(balance=1000, lgd=0.45, eir=0.06, remaining_months=0, pd_12m=0.02)


def test_pd_term_structure_calculator():
    m = [[0.90, 0.08, 0.02], [0.10, 0.75, 0.15], [0, 0, 1]]
    annual = calc.pd_term_structure(m, 3, grades=["A", "B"])
    assert annual["cumulative_pd"]["A"] == pytest.approx([0.02, 0.05, 0.08616])
    monthly = calc.pd_term_structure(m, 2, monthly=True)
    assert len(monthly["cumulative_pd"]["grade_1"]) == 24
    assert monthly["cumulative_pd"]["grade_1"][11] == pytest.approx(0.02, abs=1e-6)
    assert monthly["generator"]["l1_error"] < 1e-6
    bad_year = calc.pd_term_structure(m, 2, z_by_year=[-1.0], rho=0.1)
    assert bad_year["cumulative_pd"]["grade_1"][0] == pytest.approx(0.033512, abs=1e-6)
    assert calc.provision_matrix([100, 50], [0.01, 0.1])["total_allowance"] == pytest.approx(6.0)
    for bad in (dict(years=0), dict(years=2, grades=["A"]), dict(years=2, z_by_year=[])):
        with pytest.raises(InputError):
            calc.pd_term_structure(m, **bad)


# --- MCP ----------------------------------------------------------------------------------------

mcp = pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from eqlx_ifrs9.mcp_server import build_server  # noqa: E402


@pytest.fixture
def root(folder, tmp_path):
    secret = tmp_path / "outside.csv"
    secret.write_text((folder / "loans.csv").read_text())
    os.symlink(secret, folder / "link_out.csv")
    (folder / "notes.txt").write_text("not a loan file")
    return folder


def call(server, name, args):
    async def go():
        async with Client(server) as c:
            return await c.call_tool(name, args)

    return asyncio.run(go())


TOOLS = {
    "list_files",
    "describe_loans",
    "run_portfolio",
    "list_loans",
    "explain_loan",
    "compare_settings",
    "calculate_ecl",
    "pd_term_structure",
    "provision_matrix",
    "turkey_reclassification_check",
    "assess_individual",
    "collateral_recovery",
    "reconcile_reported",
    "scenario_response",
    "allowance_movement",
    "staging_quality",
    "backtest_defaults",
    "challenger_matrix",
}


def test_tools_are_read_only(root):
    async def go():
        async with Client(build_server(root)) as c:
            return await c.list_tools()

    tools = {t.name: t for t in asyncio.run(go()).tools}
    assert set(tools) == TOOLS
    assert all(t.annotations.read_only_hint and not t.annotations.destructive_hint for t in tools.values())
    assert all(t.description and len(t.description) > 40 for t in tools.values())


def test_mcp_flow(root):
    server = build_server(root)
    files = call(server, "list_files", {}).structured_content["result"]
    assert [f["path"] for f in files] == [  # no symlink, no .txt
        "case_going_concern.json",
        "case_gone_concern.json",
        "loans.csv",
        "loans_previous.csv",
        "planted_cases.json",
        "settings.json",
        "settings_previous.json",
        "write_offs.csv",
    ]
    assert call(server, "describe_loans", {"loans": "loans.csv"}).structured_content["loans"] == 161
    args = {"loans": "loans.csv", "settings": "settings.json"}
    table = call(server, "run_portfolio", args).structured_content
    assert table["total"]["loans"] == 161 and len(table["scenarios"]["scenarios"]) == 3
    rows = call(
        server, "list_loans", {**args, "stage": 2, "reason": "days_past_due_backstop", "limit": 500}
    ).structured_content
    assert "CASE-DPD-ONLY" in {r["loan_id"] for r in rows["loans"]}
    one = call(server, "explain_loan", {**args, "loan_id": "CASE-SHORTER-LIFE"}).structured_content
    assert one["stage"] == 2 and one["staging_test"]["ratio"] > 2
    cmp_ = call(server, "compare_settings", {**args, "changes": {"staging.relative_threshold": 3.0}}).structured_content
    assert cmp_["ecl_change"] < 0 and cmp_["loans_changing_stage"] > 0
    looser = call(
        server, "run_portfolio", {**args, "overrides": {"staging.relative_threshold": 3.0}}
    ).structured_content
    assert looser["total"]["ecl"] == pytest.approx(cmp_["ecl_after"])
    assert not (root / "result.csv").exists() and sorted(p.name for p in root.iterdir()) == [
        "case_going_concern.json",
        "case_gone_concern.json",
        "link_out.csv",
        "loans.csv",
        "loans_previous.csv",
        "notes.txt",
        "planted_cases.json",
        "settings.json",
        "settings_previous.json",
        "write_offs.csv",
    ]


def test_mcp_calculators(root):
    server = build_server(root)
    ie8 = call(
        server,
        "calculate_ecl",
        {"balance": 1_000_000, "pd_12m": 0.005, "lgd": 0.25, "eir": 0.0, "remaining_months": 12},
    )
    assert ie8.structured_content["twelve_month_ecl"] == pytest.approx(1250.0)
    ie12 = call(
        server,
        "provision_matrix",
        {"gross_carrying_amount": [15e6, 7.5e6, 4e6, 2.5e6, 1e6], "loss_rate": [0.003, 0.016, 0.036, 0.066, 0.106]},
    )
    assert ie12.structured_content["total_allowance"] == pytest.approx(580_000)
    ts = call(server, "pd_term_structure", {"matrix": [[0.9, 0.08, 0.02], [0.1, 0.75, 0.15], [0, 0, 1]], "years": 2})
    assert ts.structured_content["cumulative_pd"]["grade_1"] == pytest.approx([0.02, 0.05])
    tr = call(
        server,
        "turkey_reclassification_check",
        {
            "check": "exit_restructured_status",
            "facts": {
                "months_monitored_in_group_2": 2,
                "share_of_remaining_debt_repaid": 0.0,
                "max_days_past_due_any_loan": 0,
                "financial_difficulty_ended": True,
            },
        },
    ).structured_content
    assert tr["eligible"] is False and "7(8)" in tr["unmet"][0]


def test_mcp_review_and_two_dates(root):
    from eqlx_ifrs9 import config, data, engine, period, review

    server = build_server(root)
    closing = engine.run(data.read_csv(root / "loans.csv"), config.load(root / "settings.json"))
    opening = engine.run(data.read_csv(root / "loans_previous.csv"), config.load(root / "settings_previous.json"))
    gone = period.read_write_offs(root / "write_offs.csv")
    plain = lambda x: json.loads(json.dumps(x))  # noqa: E731
    one = {"loans": "loans.csv", "settings": "settings.json"}
    two = {
        "opening_loans": "loans_previous.csv",
        "closing_loans": "loans.csv",
        "closing_settings": "settings.json",
        "opening_settings": "settings_previous.json",
        "write_offs": "write_offs.csv",
    }
    assert call(server, "reconcile_reported", one).structured_content == plain(review.reconcile(closing))
    got = call(server, "scenario_response", {**one, "shifts": [-1.0, 1.0]}).structured_content
    assert got == plain(review.response(closing.loans, closing.config, shifts=[-1.0, 1.0]))
    assert call(server, "allowance_movement", two).structured_content == plain(
        period.movement(opening, closing, write_offs=gone)
    )
    reordered = call(server, "allowance_movement", {**two, "order": ["lgd", "pd", "exposure", "settings", "other"]})
    assert reordered.structured_content["order"][0] == "lgd"
    assert call(server, "staging_quality", two).structured_content == plain(
        period.staging_quality(opening, closing, write_offs=gone)
    )
    assert call(server, "backtest_defaults", {**two, "realised_z": -0.4}).structured_content == plain(
        period.backtest(opening, closing, write_offs=gone, realised_z=-0.4)
    )
    assert call(server, "challenger_matrix", two).structured_content == plain(
        review.challenger(opening, closing, write_offs=gone)
    )
    for bad in ("../outside.csv", "link_out.csv", "settings.json", "missing.csv"):
        assert call(server, "allowance_movement", {**two, "write_offs": bad}).is_error
        assert call(server, "staging_quality", {**two, "opening_loans": bad}).is_error
    assert call(server, "allowance_movement", {**two, "order": ["pd"]}).is_error
    assert call(server, "reconcile_reported", {"loans": "loans_previous.csv", "settings": "settings.json"}).is_error
    assert sorted(p.name for p in root.iterdir() if p.suffix == ".csv") == [
        "link_out.csv",
        "loans.csv",
        "loans_previous.csv",
        "write_offs.csv",
    ]


def test_mcp_says_why_input_was_refused(root):
    server = build_server(root)
    two = {"opening_loans": "loans_previous.csv", "closing_loans": "loans.csv", "closing_settings": "settings.json"}

    def reason(tool, args):
        res = call(server, tool, args)
        assert res.is_error
        return " ".join(c.text for c in res.content)

    assert "months must be a whole number between 1 and 12" in reason("backtest_defaults", {**two, "months": 13})
    assert "unknown key" in reason("assess_individual", {"case": "settings.json"})
    assert "neither reported_stage nor reported_ecl" in reason(
        "reconcile_reported", {"loans": "loans_previous.csv", "settings": "settings_previous.json"}
    )
    assert "outside the allowed folder" in reason("describe_loans", {"loans": "../outside.csv"})
    assert "file not found" in reason("run_portfolio", {"loans": "nothing.csv"})
    assert "unknown loan_id" in reason(
        "explain_loan", {"loans": "loans.csv", "settings": "settings.json", "loan_id": "X"}
    )
    assert "haircut must be at most 1" in reason(
        "collateral_recovery", {"value": 1, "eir": 0.1, "years_to_sale": 1, "haircut": 2}
    )


def test_mcp_individual_assessment(root):
    from eqlx_ifrs9 import individual

    server = build_server(root)
    for name, case in synthetic.individual_cases().items():
        got = call(server, "assess_individual", {"case": name}).structured_content
        assert got == json.loads(json.dumps(individual.assess(case)))
    npl = call(
        server,
        "collateral_recovery",
        {
            "value": 200,
            "eir": 0.05,
            "years_to_sale": 5,
            "haircut": 0.10,
            "selling_cost_rate": 0.10,
            "annual_cost_rate": 0.05,
        },
    ).structured_content
    assert round(npl["present_value"]) == 84  # ECB guidance on non-performing loans, 7.4.3
    aqr = call(
        server,
        "collateral_recovery",
        {
            "value": 240,
            "eir": 0.05,
            "years_to_sale": 2,
            "haircut": 0.0,
            "costs": [{"year": 1, "amount": 12}, {"year": 2, "amount": 24}],
            "share": 0.4,
        },
    ).structured_content
    assert round(aqr["recoverable"], 2) == 73.80  # ECB AQR manual, 4.6.3
    for bad in ("../outside.csv", "link_out.csv", "loans.csv", "settings.json", "missing.json", "/etc/passwd"):
        assert call(server, "assess_individual", {"case": bad}).is_error
    assert call(server, "collateral_recovery", {"value": 1, "eir": 0.1, "years_to_sale": 1, "haircut": 2}).is_error


def test_mcp_cache_follows_file_changes(root):
    server = build_server(root)
    args = {"loans": "loans.csv", "settings": "settings.json"}
    before = call(server, "run_portfolio", args).structured_content["total"]["loans"]
    text = (root / "loans.csv").read_text().rstrip("\n").split("\n")
    (root / "loans.csv").write_text("\n".join(text[:-1]) + "\n")
    assert call(server, "run_portfolio", args).structured_content["total"]["loans"] == before - 1


@pytest.mark.parametrize("path", ["..", "../outside.csv", "/etc/passwd", "link_out.csv"])
def test_mcp_refuses_paths_outside_root(root, path):
    server = build_server(root)
    for tool, args in (("describe_loans", {"loans": path}), ("run_portfolio", {"loans": path})):
        res = call(server, tool, args)
        assert res.is_error
    res = call(server, "run_portfolio", {"loans": "loans.csv", "settings": path})
    assert res.is_error


def test_mcp_reports_input_errors(root):
    server = build_server(root)
    assert call(server, "run_portfolio", {"loans": "notes.txt"}).is_error
    assert call(server, "run_portfolio", {"loans": "settings.json"}).is_error
    assert call(server, "explain_loan", {"loans": "loans.csv", "settings": "settings.json", "loan_id": "X"}).is_error
    assert call(server, "compare_settings", {"loans": "loans.csv", "settings": "settings.json", "changes": {}}).is_error
    assert call(server, "turkey_reclassification_check", {"check": "performing_again", "facts": {"x": 1}}).is_error
    with pytest.raises(NotADirectoryError):
        build_server(root / "loans.csv")


# --- regressions from review --------------------------------------------------------------------


def test_calculator_reverts_to_the_long_run_pd_like_the_portfolio_run():
    kw = dict(balance=1000, lgd=0.45, eir=0.06, remaining_months=36, pd_12m=0.02, z_by_year=[-1.0], rho=0.12)
    stressed = calc.instrument_ecl(**kw)
    assert stressed["lifetime_pd"] == pytest.approx(1 - (1 - 0.034377) * 0.98**2, abs=1e-6)
    assert calc.instrument_ecl(**kw, reversion_years=1)["lifetime_pd"] > stressed["lifetime_pd"]


def test_calculator_default_schedule_is_priced_at_the_effective_rate():
    res = calc.instrument_ecl(
        balance=1000, lgd=1.0, eir=0.48, remaining_months=36, pd_12m=0.02, amortization="annuity", periods=True
    )
    first_month_interest = 1000 * (1.48 ** (1 / 12) - 1)
    second = res["periods"][1]["ead"]
    quoted = calc.instrument_ecl(
        balance=1000,
        lgd=1.0,
        eir=0.48,
        rate=0.48,
        remaining_months=36,
        pd_12m=0.02,
        amortization="annuity",
        periods=True,
    )
    assert second < quoted["periods"][1]["ead"]  # less interest, so more principal is repaid in month 1
    assert first_month_interest == pytest.approx(33.2, abs=0.1)


@pytest.mark.parametrize(
    "bad",
    [dict(cpr=-0.5), dict(rho=5.0), dict(z_by_year=[]), dict(remaining_months=12.9), dict(remaining_months=0)],
)
def test_calculator_rejects_values_out_of_range(bad):
    kw = dict(balance=1000, lgd=0.45, eir=0.06, remaining_months=36, pd_12m=0.02)
    with pytest.raises(InputError):
        calc.instrument_ecl(**{**kw, **bad})


def test_sample_does_not_overwrite(folder, capsys):
    code, _, err = run(capsys, "sample", folder)
    assert code == 2 and "already exist" in err
    assert run(capsys, "sample", folder, "--loans", 150, "--force")[0] == 0


def test_misspelt_settings_are_refused_on_the_command_line(folder, capsys):
    base = ["run", folder / "loans.csv", "--config", folder / "settings.json"]
    code, _, err = run(capsys, *base, "--set", "staging.forborne_is_stage2=False")
    assert code == 2 and "true or false" in err
    code, _, err = run(capsys, *base, "--set", "staging=5")
    assert code == 2 and "must be an object" in err
    code, _, err = run(capsys, *base, "--set", "staging.backstop_ratio=3,5")
    assert code == 2 and "must be a number" in err
    listed = folder / "list.json"
    listed.write_text("[1, 2]")
    assert run(capsys, "run", folder / "loans.csv", "--config", listed)[0] == 2


def test_mcp_yes_no_facts_must_be_booleans(root):
    facts = {
        "months_monitored_in_group_2": 2,
        "share_of_remaining_debt_repaid": 0.0,
        "max_days_past_due_any_loan": 0,
        "financial_difficulty_ended": "false",
    }
    res = call(
        build_server(root), "turkey_reclassification_check", {"check": "exit_restructured_status", "facts": facts}
    )
    assert res.is_error
