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
    assert sorted(p.name for p in folder.iterdir()) == ["loans.csv", "planted_cases.json", "settings.json"]
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
    assert [f["path"] for f in files] == ["loans.csv", "planted_cases.json", "settings.json"]  # no symlink, no .txt
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
        "link_out.csv",
        "loans.csv",
        "notes.txt",
        "planted_cases.json",
        "settings.json",
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
