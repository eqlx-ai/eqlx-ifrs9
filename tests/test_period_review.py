"""Two reporting dates (movement, staging quality, backtest, observed migration) and the review of
reported figures (reconcile, scenario response, challenger).

Hand-made portfolios use loans whose allowance has a closed form: bullet, effective interest rate 0,
PD taken as given. Then 12-month ECL = balance x LGD x (1 - (1 - PD)^(min(months, 12) / 12)), lifetime
ECL = balance x LGD x (1 - (1 - PD)^(months / 12)) and stage 3 = balance x LGD in default.
"""

import copy
import dataclasses
import itertools
import json
import random

import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from eqlx_ifrs9 import config, data, engine, period, report, review, synthetic
from eqlx_ifrs9.cli import main
from eqlx_ifrs9.errors import InputError

COLS = (
    "loan_id",
    "segment",
    "balance",
    "eir",
    "remaining_months",
    "lgd",
    "lgd_default",
    "pd_12m",
    "pd_12m_origination",
    "grade",
    "dpd",
    "reported_stage",
    "reported_ecl",
    "individual_ecl",
)
PLAIN = {"staging": {"relative_threshold": 2.0, "pd_measure": "twelve_month"}}


def loan(loan_id, balance=1000.0, months=12, pd=0.02, lgd=0.45, dpd=0, **kw):
    row = dict.fromkeys(COLS, "")
    row.update(loan_id=loan_id, segment="all", balance=balance, eir=0.0, remaining_months=months, lgd=lgd)
    row.update(lgd_default=0.8, pd_12m=pd, pd_12m_origination=0.02, dpd=dpd)
    row.update(kw)
    return row


def run(rows, settings=PLAIN):
    text = ",".join(COLS) + "\n" + "".join(",".join(str(r[c]) for c in COLS) + "\n" for r in rows)
    return engine.run(data.from_text(text), config.from_dict(copy.deepcopy(settings)))


def table(m, key="loss_allowance"):
    return {r["row"]: [r["stage_1"], r["stage_2"], r["stage_3"], r["total"]] for r in m[key]}


@pytest.fixture(scope="module")
def sample():
    s = synthetic.generate(400, 7)
    opening = engine.run(data.from_text(s.previous_csv), config.from_dict(s.previous_settings))
    closing = engine.run(data.from_text(s.csv), config.from_dict(s.settings))
    return s, opening, closing, s.write_offs_csv.split()[1:]


# --- movement --------------------------------------------------------------------------------


def test_movement_by_hand():
    """Four loans, every figure from the closed forms above.

    A: stage 1 -> 2 (PD 2% -> 12%), balance 1000 -> 900, life 24 -> 12 months, LGD 45% -> 50%.
    B: new, stage 1. C: stage 3, written off. D: stage 1, repaid."""
    opening = run([loan("A", months=24), loan("C", balance=500.0, dpd=120), loan("D", balance=2000.0)])
    closing = run([loan("A", balance=900.0, pd=0.12, lgd=0.50), loan("B", balance=4000.0, pd=0.01)])
    m = period.movement(opening, closing, write_offs=["C"], order=["settings", "exposure", "pd", "lgd", "other"])
    t = table(m)
    a_open = 1000 * 0.45 * 0.02  # 9.00
    a_lifetime_open = 1000 * 0.45 * (1 - 0.98**2)  # 17.82: opening inputs, measured for the life
    assert t["opening_balance"] == pytest.approx([a_open + 2000 * 0.45 * 0.02, 0, 500 * 0.8, 427.0])
    assert t["transfer_to_stage_2"] == pytest.approx([-9.0, 9.0, 0, 0])
    assert t["transfer_to_stage_1"] == t["transfer_to_stage_3"] == [0, 0, 0, 0]
    assert t["remeasurement_on_stage_transfer"] == pytest.approx([0, a_lifetime_open - a_open, 0, 8.82])
    assert t["originated_or_purchased"] == pytest.approx([4000 * 0.45 * 0.01, 0, 0, 18.0])
    assert t["derecognised"] == pytest.approx([-18.0, 0, 0, -18.0])
    assert t["written_off"] == pytest.approx([0, 0, -400.0, -400.0])
    assert t["change_settings"] == [0, 0, 0, 0]
    assert t["change_exposure"][1] == pytest.approx(900 * 0.45 * 0.02 - a_lifetime_open)  # 8.10 - 17.82
    assert t["change_pd"][1] == pytest.approx(900 * 0.45 * 0.12 - 8.10)  # +40.50
    assert t["change_lgd"][1] == pytest.approx(900 * 0.50 * 0.12 - 48.60)  # +5.40
    assert t["change_other"] == [0, 0, 0, 0]
    assert t["closing_balance"] == pytest.approx([18.0, 54.0, 0, 72.0])
    g = table(m, "gross_carrying_amount")
    assert g["opening_balance"] == [3000.0, 0, 500.0, 3500.0]
    assert g["transfer_to_stage_2"] == [-1000.0, 1000.0, 0, 0]
    assert g["net_repayments_and_drawdowns"] == [0, -100.0, 0, -100.0]
    assert g["closing_balance"] == [4000.0, 900.0, 0, 4900.0]
    assert m["loans"] == {
        "opening": 3,
        "closing": 2,
        "continuing": 1,
        "changing_stage": 1,
        "new": 1,
        "derecognised": 1,
        "written_off": 1,
    }
    top = m["largest_changes"][0]
    assert (top["loan_id"], top["what"], top["change"]) == ("C", "written off", -400.0)
    a = next(c for c in m["largest_changes"] if c["loan_id"] == "A")
    assert a["drivers"] == pytest.approx(
        {"stage_transfer": 8.82, "settings": 0, "exposure": -9.72, "pd": 40.5, "lgd": 5.4, "other": 0}
    )
    assert sum(a["drivers"].values()) == pytest.approx(a["change"]) == pytest.approx(45.0)


def test_another_order_moves_the_split_not_the_total():
    opening = run([loan("A", months=24)])
    closing = run([loan("A", balance=900.0, pd=0.12, lgd=0.50)])
    first = period.movement(opening, closing, order=["settings", "exposure", "pd", "lgd", "other"])
    other = period.movement(opening, closing, order=["lgd", "pd", "exposure", "other", "settings"])
    t = table(other)
    assert t["change_lgd"][1] == pytest.approx(1000 * (1 - 0.98**2) * 0.05)  # LGD first, on opening inputs
    assert t["change_pd"][1] == pytest.approx(1000 * 0.50 * ((1 - 0.88**2) - (1 - 0.98**2)))
    assert t["change_exposure"][1] == pytest.approx(54.0 - 1000 * 0.50 * (1 - 0.88**2))
    assert t["change_lgd"][1] != pytest.approx(table(first)["change_lgd"][1])
    assert other["change"] == pytest.approx(first["change"]) == pytest.approx(45.0)


def test_movement_rows_add_up_and_transfers_cancel(sample):
    _, opening, closing, gone = sample
    for order in (
        period.DRIVERS,
        ("other", "lgd", "pd", "exposure", "settings"),
        ("pd", "settings", "other", "exposure", "lgd"),
    ):
        m = period.movement(opening, closing, write_offs=gone, order=order)
        for key in ("loss_allowance", "gross_carrying_amount"):
            rows = m[key]
            for col in ("stage_1", "stage_2", "stage_3", "total"):
                assert sum(r[col] for r in rows[:-1]) == pytest.approx(rows[-1][col], rel=1e-10, abs=1e-6)
            for r in rows:
                assert r["stage_1"] + r["stage_2"] + r["stage_3"] == pytest.approx(r["total"], abs=1e-6)
                if r["row"].startswith("transfer_to"):
                    assert r["total"] == pytest.approx(0.0, abs=1e-6)
        t = table(m)
        assert t["opening_balance"][3] == pytest.approx(float(opening.ecl.sum()))
        assert t["closing_balance"][:3] == pytest.approx(
            [float(closing.ecl[closing.stage == s].sum()) for s in (1, 2, 3)]
        )
        assert m["largest_difference_between_rows_and_closing_balance"] < 1e-6
    assert m["loans"]["continuing"] + m["loans"]["new"] == closing.n
    assert m["loans"]["continuing"] + m["loans"]["derecognised"] + m["loans"]["written_off"] == opening.n


def test_drivers_equal_runs_on_hand_built_files(sample):
    """The same steps through loan files written out as text, one per step, without period._substitute."""
    s, opening, closing, gone = sample
    m = period.movement(opening, closing, write_offs=gone, limit=500)

    def rows_of(text):
        head, *lines = text.strip().split("\n")
        names = head.split(",")
        return names, {ln.split(",")[0]: dict(zip(names, ln.split(","), strict=True)) for ln in lines}

    names_o, prev = rows_of(s.previous_csv)
    _, now = rows_of(s.csv)
    stage_now = dict(zip(closing.loans.text["loan_id"].tolist(), closing.stage.tolist(), strict=True))
    common = sorted(set(prev) & set(now))
    steps = [
        (s.previous_settings, ()),
        (s.settings, ()),
        (s.settings, period.GROUPS["exposure"]),
        (s.settings, period.GROUPS["exposure"] + period.GROUPS["pd"]),
        (s.settings, period.GROUPS["exposure"] + period.GROUPS["pd"] + period.GROUPS["lgd"]),
    ]
    measured = []
    for settings, replaced in steps:
        lines = [",".join(names_o)]
        for lid in common:
            row = {**prev[lid], **{c: now[lid][c] for c in replaced if c in names_o}}
            lines.append(",".join(row[c] for c in names_o))
        r = engine.run(data.from_text("\n".join(lines) + "\n"), config.from_dict(settings))
        ids = r.loans.text["loan_id"].tolist()
        stage = np.array([stage_now[i] for i in ids], dtype=np.int8)
        measured.append(dict(zip(ids, r.ecl_for_stages(stage).tolist(), strict=True)))
    checked = 0
    for c in m["largest_changes"]:
        if c["what"] != "continuing":
            continue
        lid, d = c["loan_id"], c["drivers"]
        assert d["settings"] == pytest.approx(measured[1][lid] - measured[0][lid], abs=1e-6)
        assert d["exposure"] == pytest.approx(measured[2][lid] - measured[1][lid], abs=1e-6)
        assert d["pd"] == pytest.approx(measured[3][lid] - measured[2][lid], abs=1e-6)
        assert d["lgd"] == pytest.approx(measured[4][lid] - measured[3][lid], abs=1e-6)
        assert sum(d.values()) == pytest.approx(c["change"], abs=1e-6)
        checked += 1
    assert checked > 200


def test_movement_does_not_depend_on_row_order(sample):
    s, opening, closing, gone = sample
    rng = random.Random(5)

    def shuffled(text, settings):
        head, *lines = text.strip().split("\n")
        rng.shuffle(lines)
        return engine.run(data.from_text("\n".join([head, *lines]) + "\n"), config.from_dict(settings))

    a = period.movement(opening, closing, write_offs=gone)
    b = period.movement(
        shuffled(s.previous_csv, s.previous_settings), shuffled(s.csv, s.settings), write_offs=gone[::-1]
    )
    for ra, rb in zip(a["loss_allowance"], b["loss_allowance"], strict=True):
        assert [ra[k] for k in ("stage_1", "stage_2", "stage_3")] == pytest.approx(
            [rb[k] for k in ("stage_1", "stage_2", "stage_3")], rel=1e-9, abs=1e-6
        )
    assert [c["loan_id"] for c in a["largest_changes"]] == [c["loan_id"] for c in b["largest_changes"]]


def test_movement_refuses_bad_input():
    opening, closing = run([loan("A"), loan("C")]), run([loan("A")])
    with pytest.raises(InputError, match="order"):
        period.movement(opening, closing, order=["pd", "lgd"])
    with pytest.raises(InputError, match="not a loan of the opening file"):
        period.movement(opening, closing, write_offs=["Z"])
    with pytest.raises(InputError, match="still in the closing file"):
        period.movement(opening, closing, write_offs=["A"])
    nothing_in_common = period.movement(run([loan("X")]), run([loan("Y")]))
    assert nothing_in_common["loans"]["continuing"] == 0
    assert table(nothing_in_common)["closing_balance"][3] == pytest.approx(9.0)


# --- staging quality and backtest ------------------------------------------------------------


def hand_pair():
    """Stage follows days past due: 0 -> stage 1, 45 -> stage 2, 120 -> stage 3."""
    before = [loan(f"S1-{i}") for i in range(10)] + [loan(f"S2-{i}", dpd=45) for i in range(5)]
    before += [loan("S3-0", dpd=120), loan("GONE"), loan("LOST", dpd=45)]
    after = [loan(f"S1-{i}") for i in range(7)] + [loan("S1-7", dpd=45), loan("S1-8", dpd=45), loan("S1-9", dpd=120)]
    after += [loan("S2-0"), loan("S2-1", dpd=45), loan("S2-2", dpd=45), loan("S2-3", dpd=120), loan("S2-4", dpd=120)]
    after += [loan("S3-0", dpd=120), loan("NEW")]
    return run(before), run(after)


def test_staging_quality_by_hand():
    opening, closing = hand_pair()
    q = period.staging_quality(opening, closing, write_offs=["LOST"])
    rows = {r["opening_stage"]: r for r in q["transitions"]}
    assert [rows[1][f"to_stage_{s}"]["loans"] for s in (1, 2, 3)] == [7, 2, 1]
    assert [rows[2][f"to_stage_{s}"]["loans"] for s in (1, 2, 3)] == [1, 2, 2]
    assert rows[1]["left"]["loans"] == 1 and rows[2]["written_off"]["loans"] == 1
    assert q["stage_3_inflow"]["loans"] == 4  # one from stage 1, two from stage 2, one written off from stage 2
    assert q["stage_3_inflow"]["share_directly_from_stage_1"] == 0.25
    assert q["default_rate_by_opening_stage"]["stage_1"] == {"loans": 10, "defaults": 1, "rate": 0.1}
    assert q["default_rate_by_opening_stage"]["stage_2"] == {"loans": 6, "defaults": 3, "rate": 0.5}
    assert q["stage_2_inflow"]["loans"] == 2
    assert q["stage_2_inflow"]["share_of_opening_stage_1_exposure"] == pytest.approx(2 / 11)
    assert q["stage_2_inflow"]["share_only_because_of_days_past_due"] == 1.0
    assert q["what_became_of_stage_2"] == {"loans": 5, "back_to_stage_1": 0.2, "still_stage_2": 0.4, "to_stage_3": 0.4}
    assert q["warnings"] == []


def test_staging_quality_warns_when_stage_2_is_not_riskier():
    before = [loan("a"), loan("b"), loan("c", dpd=45), loan("d", dpd=45)]
    after = [loan("a", dpd=120), loan("b"), loan("c"), loan("d", dpd=45)]
    q = period.staging_quality(run(before), run(after))
    assert q["stage_3_inflow"]["share_directly_from_stage_1"] == 1.0
    assert len(q["warnings"]) == 1 and "did not single out" in q["warnings"][0]


def test_backtest_by_hand():
    opening, closing = hand_pair()
    b = period.backtest(opening, closing, write_offs=["LOST"])
    # 17 performing loans with a 12-month PD of 2%; four defaulted (three stage 3, one written off)
    assert b["total"]["loans"] == 17 and b["total"]["defaults"] == 4
    assert b["total"]["expected_weighted"] == pytest.approx(17 * 0.02)
    assert b["total"]["difference"] == pytest.approx(17 * 0.02 - 4)
    assert b["total"]["standard_deviation_if_independent"] == pytest.approx((17 * 0.02 * 0.98) ** 0.5)
    assert [x["defaults"] for x in b["by_opening_stage"]] == [1, 3]
    half = period.backtest(opening, closing, write_offs=["LOST"], months=6)
    assert half["total"]["expected_weighted"] == pytest.approx(17 * (1 - 0.98**0.5))
    for bad in (0, 13, 1.5, True):
        with pytest.raises(InputError):
            period.backtest(opening, closing, months=bad)
    with pytest.raises(InputError, match="no scenario path"):
        period.backtest(opening, closing, realised_z=-1.0)


def test_backtest_splits_forecast_error_from_model_error():
    """One-factor formula written out here: PD(z) = Phi((Phi^-1(PD) - sqrt(rho) z) / sqrt(1 - rho))."""
    rho, pd = 0.12, 0.02
    settings = {
        **PLAIN,
        "pd_basis": "long_run",
        "scenarios": [
            {"name": "base", "weight": 0.7, "z": [0.0]},
            {"name": "down", "weight": 0.3, "z": [-1.5]},
        ],
        "segments": {"all": {"rho": rho}},
    }
    before = [loan(f"L{i}") for i in range(50)]
    after = [loan(f"L{i}", dpd=120 if i < 3 else 0) for i in range(50)]
    opening, closing = run(before, settings), run(after, settings)

    def shifted(z):
        return float(ndtr((ndtri(pd) - rho**0.5 * z) / (1 - rho) ** 0.5))

    b = period.backtest(opening, closing, realised_z=-1.0)["total"]
    assert b["expected_base_scenario"] == pytest.approx(50 * shifted(0.0))
    assert b["expected_weighted"] == pytest.approx(50 * (0.7 * shifted(0.0) + 0.3 * shifted(-1.5)))
    assert b["expected_realised_economy"] == pytest.approx(50 * shifted(-1.0))
    assert b["forecast_error"] == pytest.approx(b["expected_weighted"] - 50 * shifted(-1.0))
    assert b["model_error"] == pytest.approx(50 * shifted(-1.0) - 3)
    assert b["forecast_error"] + b["model_error"] == pytest.approx(b["expected_weighted"] - b["defaults"])


# --- observed migration and challenger -------------------------------------------------------

GRADED = {
    "staging": {"relative_threshold": 2.0},
    "segments": {
        "corp": {
            "grades": ["A", "B"],
            "matrix": [[0.90, 0.08, 0.02], [0.10, 0.80, 0.10], [0, 0, 1]],
        }
    },
}


def graded(loan_id, grade, **kw):
    return loan(loan_id, segment="corp", pd="", pd_12m_origination="", grade=grade, **kw)


def graded_pair():
    moves = {("A", "A"): 40, ("A", "B"): 6, ("A", "D"): 2, ("B", "A"): 5, ("B", "B"): 30, ("B", "D"): 5}
    before, after = [], []
    for (start, end), n in moves.items():
        for i in range(n):
            lid = f"{start}{end}{i:02d}"
            before.append(graded(lid, start))
            after.append(graded(lid, start if end == "D" else end, dpd=120 if end == "D" else 0))
    before += [graded("LEFT", "A"), graded("WO", "B"), graded("OLD-DEFAULT", "B", dpd=120)]
    return run(before, GRADED), run(after, GRADED)


def test_observed_migration_counts():
    opening, closing = graded_pair()
    o = period.observed_migration(opening, closing, "corp", write_offs=["WO"])
    assert o["counts"] == [[40, 6, 2], [5, 30, 6]]  # the written-off loan is a default of grade B
    assert o["left_out"] == {"left_the_portfolio": 1, "no_grade_at_closing": 0}
    assert o["matrix"][0] == pytest.approx([40 / 48, 6 / 48, 2 / 48])
    assert o["matrix"][1] == pytest.approx([5 / 41, 30 / 41, 6 / 41])
    assert o["matrix"][2] == [0, 0, 1]
    assert o["annual_matrix"] == o["matrix"]
    assert o["annual_pd_by_grade"] == pytest.approx({"A": 2 / 48, "B": 6 / 41})
    assert o["warnings"] == []
    half = period.observed_migration(opening, closing, "corp", write_offs=["WO"], months=6)
    squared = np.array(o["matrix"]) @ np.array(o["matrix"])
    assert np.array(half["annual_matrix"]) == pytest.approx(squared, abs=2e-3)
    with pytest.raises(InputError, match="no grades"):
        period.observed_migration(opening, closing, "all")


def test_observed_migration_small_and_empty_grades():
    before = [graded(f"A{i}", "A") for i in range(5)]
    after = [graded(f"A{i}", "A") for i in range(5)]
    o = period.observed_migration(run(before, GRADED), run(after, GRADED), "corp")
    assert o["matrix"][0] == [1.0, 0, 0] and o["matrix"][1] == [0.10, 0.80, 0.10]
    assert any("only 5 observations" in w for w in o["warnings"])
    assert any("grade B: no observation" in w for w in o["warnings"])


def test_challenger_equals_a_run_with_the_observed_matrix_in_the_settings():
    opening, closing = graded_pair()
    c = review.challenger(opening, closing, write_offs=["WO"])
    by_hand = copy.deepcopy(GRADED)
    by_hand["segments"]["corp"]["matrix"] = c["observed"][0]["annual_matrix"]
    by_hand["segments"]["corp"]["matrix_tolerance"] = 1e-8
    again = engine.run(closing.loans, config.from_dict(by_hand))
    seg = c["segments"][0]
    assert seg["with_observed_matrix"] == pytest.approx(float(again.ecl.sum()))
    assert seg["with_settings_matrix"] == pytest.approx(float(closing.ecl.sum()))
    assert seg["against"] == "run with the settings' matrix"
    assert seg["difference"] == pytest.approx(seg["with_observed_matrix"] - seg["with_settings_matrix"])
    assert seg["verdict"] == "investigate"  # observed PDs of 4.2% and 14.6% against 2% and 10%
    assert c["observed"][0]["annual_pd_by_grade_in_settings"] == {"A": 0.02, "B": 0.10}
    with pytest.raises(InputError, match="nothing to observe"):
        review.challenger(run([loan("a")]), run([loan("a")]))


# --- reconcile -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("recalculated", "reported", "expected"),
    [
        (100.0, 100.0, "reported is not lower"),
        (99.0, 100.0, "reported is not lower"),
        (104.9, 100.0, "immaterial"),
        (105.0, 100.0, "explain"),
        (110.0, 100.0, "explain"),
        (110.1, 100.0, "investigate"),
        (1.0, 0.0, "investigate"),
        (0.0, 0.0, "reported is not lower"),
    ],
)
def test_verdict_thresholds(recalculated, reported, expected):
    """ECB AQR manual 7.1: below 5% immaterial, 5% to 10% to be explained, above 10% investigated."""
    assert review.verdict(recalculated, reported) == expected


def test_reconcile_by_hand():
    rows = [
        loan("same", reported_stage=1, reported_ecl=9.0),  # 1000 x 45% x 2%
        loan("kept-in-1", months=24, pd=0.12, reported_stage=1, reported_ecl=50.0),
        loan("too-early", reported_stage=2, reported_ecl=9.0),
        loan("no-figures"),
        loan("late", dpd=120, reported_stage=3, reported_ecl=700.0),
    ]
    r = run(rows)
    out = review.reconcile(r)
    st = out["stages"]
    assert st["rows_reported_columns_recalculated"]["loans"] == [[1, 1, 0], [1, 0, 0], [0, 0, 1]]
    assert (st["loans_compared"], st["loans_without_reported_stage"], st["same_stage"]) == (4, 1, 2)
    assert st["reported_in_a_lower_stage"]["loans"] == 1
    assert (
        st["reported_in_a_lower_stage"]["by_primary_reason_of_the_recalculated_stage"][0]["primary_reason"]
        == "pd_relative"
    )
    assert st["reported_in_a_higher_stage"]["loans"] == 1
    assert [d["loan_id"] for d in st["loans_that_differ"]] == ["kept-in-1", "too-early"]
    lifetime = 1000 * 0.45 * (1 - 0.88**2)  # 101.52
    twelve = 1000 * 0.45 * 0.12  # 54.00
    al = out["allowance"]
    assert al["total"]["reported"] == 768.0
    assert al["total"]["recalculated"] == pytest.approx(9.0 + lifetime + 9.0 + 800.0)
    gap = al["gap_of_loans_with_both_figures"]
    assert gap["from_staging"] == pytest.approx(lifetime - twelve)  # only the loan kept in stage 1 is affected
    assert gap["from_measurement"] == pytest.approx((twelve - 50.0) + (800.0 - 700.0))
    assert gap["from_staging"] + gap["from_measurement"] == pytest.approx(gap["difference"])
    assert al["by_recalculated_stage"][2]["verdict"] == "investigate"  # 800 against 700
    assert al["largest_differences"][0]["loan_id"] == "late"
    with pytest.raises(InputError, match="neither reported_stage nor reported_ecl"):
        review.reconcile(run([loan("x")]))
    with pytest.raises(InputError, match="thresholds"):
        review.reconcile(r, immaterial=0.2, investigate=0.1)


def test_reconcile_finds_what_the_sample_lender_did(sample):
    closing = sample[2]
    out = review.reconcile(closing, limit=500)
    lower = out["stages"]["reported_in_a_lower_stage"]
    assert lower["loans"] > 0 and out["stages"]["reported_in_a_higher_stage"]["loans"] == 0
    assert [x["primary_reason"] for x in lower["by_primary_reason_of_the_recalculated_stage"]] == ["pd_relative"]
    assert all(d["reported_stage"] == 1 and d["recalculated_stage"] == 2 for d in out["stages"]["loans_that_differ"])
    verdicts = {x["segment"]: x["verdict"] for x in out["allowance"]["by_segment"]}
    assert verdicts["cards"] == "investigate" and verdicts["mortgage"] == "explain"
    for x in out["allowance"]["by_segment"]:
        booked = synthetic.REPORTED_FACTOR.get(x["segment"], 1.0)
        assert x["recalculated"] / x["reported"] >= 1 / booked - 1e-6
    gap = out["allowance"]["gap_of_loans_with_both_figures"]
    assert gap["from_staging"] > 0 and gap["from_measurement"] > 0
    assert gap["from_staging"] + gap["from_measurement"] == pytest.approx(gap["difference"])


# --- scenario response -----------------------------------------------------------------------


def test_response_curve(sample):
    s, _, closing, _ = sample
    cfg = config.from_dict(s.settings)
    out = review.response(closing.loans, cfg, shifts=[-2, -1, 1, 2])
    pts = {p["shift"]: p for p in out["points"]}
    assert sorted(pts) == [-2.0, -1.0, 0.0, 1.0, 2.0]  # no shift is always included
    base = next(x for x in report.scenario_table(closing)["scenarios"] if x["name"] == "base")
    assert pts[0.0]["ecl_stages_held"] == pytest.approx(base["ecl_same_stages"])
    assert pts[0.0]["ecl_restaged"] == pytest.approx(base["ecl_restaged"])
    assert pts[0.0]["loans_changing_stage"] == base["loans_changing_stage"]
    for key in ("ecl_stages_held", "ecl_restaged", "stage_2_share_of_exposure"):
        series = [pts[x][key] for x in sorted(pts)]
        assert series == sorted(series, reverse=True)
    assert out["weighted_ecl"] == pytest.approx(float(closing.ecl.sum()))
    assert out["asymmetry"]["ratio"] > 1.0  # a step down costs more than a step up saves
    # the same as writing the shifted path into the settings by hand
    moved = copy.deepcopy(s.settings)
    moved["scenarios"] = [{"name": "base", "weight": 1.0, "z": [z - 1.0 for z in s.settings["scenarios"][0]["z"]]}]
    by_hand = engine.run(closing.loans, config.from_dict(moved))
    assert pts[-1.0]["ecl_restaged"] == pytest.approx(float(by_hand.ecl.sum()))
    assert pts[-1.0]["z"] == pytest.approx(moved["scenarios"][0]["z"])
    down = review.response(closing.loans, cfg, shifts=[0], scenario="downside")
    assert down["points"][0]["z"] == [-1.6, -1.0, -0.4]
    with pytest.raises(InputError, match="no scenario named"):
        review.response(closing.loans, cfg, scenario="boom")
    with pytest.raises(InputError, match="nothing to shift"):
        review.response(run([loan("a")]).loans, config.from_dict(PLAIN))


# --- individually assessed allowance in the portfolio run ------------------------------------


def test_individual_ecl_replaces_the_default_rate_in_stage_3():
    r = run(
        [
            loan("impaired", dpd=120, individual_ecl=612.5),
            loan("impaired-rate", dpd=120),
            loan("fine", individual_ecl=5.0),
        ]
    )
    assert r.ecl.tolist() == pytest.approx([612.5, 800.0, 9.0])
    assert any("individual_ecl but are not in stage 3" in w for w in r.warnings)
    one = engine.loan_detail(r, "impaired")
    assert one["ecl"] == 612.5 and one["measurement"] == "individually assessed allowance"
    assert engine.loan_detail(r, "impaired-rate")["measurement"] == "exposure x loss rate in default"
    assert engine.loan_detail(r, "fine")["ecl"] == pytest.approx(9.0)


# --- sample, command line --------------------------------------------------------------------


def test_sample_period_files_are_consistent():
    s = synthetic.generate(300, 3)
    assert s == synthetic.generate(300, 3)
    before, after = data.from_text(s.previous_csv), data.from_text(s.csv)
    gone = s.write_offs_csv.split()[1:]
    assert gone and set(gone) <= set(before.text["loan_id"].tolist()) - set(after.text["loan_id"].tolist())
    assert after.has("reported_stage") and after.has("reported_ecl") and not before.has("reported_stage")
    common, io, ic = np.intersect1d(before.text["loan_id"], after.text["loan_id"], return_indices=True)
    assert common.size > 150
    assert np.all(after.num["months_on_book"][ic] - before.num["months_on_book"][io] == 12)
    young = ~np.isin(after.text["loan_id"], before.text["loan_id"])
    assert np.all(after.num["months_on_book"][young] < 12) and np.all(after.num["dpd"][young] <= 90)


def test_command_line_for_two_dates(tmp_path, capsys):
    assert main(["sample", str(tmp_path), "--loans", "120"]) == 0
    capsys.readouterr()
    f = {
        n: str(tmp_path / n)
        for n in ("loans.csv", "loans_previous.csv", "settings.json", "settings_previous.json", "write_offs.csv")
    }
    two = [f["loans_previous.csv"], f["loans.csv"], "--config", f["settings.json"]]
    full = [*two, "--opening-config", f["settings_previous.json"], "--write-offs", f["write_offs.csv"]]
    opening = engine.run(data.read_csv(f["loans_previous.csv"]), config.load(f["settings_previous.json"]))
    closing = engine.run(data.read_csv(f["loans.csv"]), config.load(f["settings.json"]))
    gone = period.read_write_offs(f["write_offs.csv"])

    def out(*args):
        assert main(list(args)) == 0
        return capsys.readouterr().out

    as_json = lambda x: json.loads(json.dumps(x))  # noqa: E731
    assert json.loads(out("movement", *full, "--json")) == as_json(period.movement(opening, closing, write_offs=gone))
    text = out("movement", *full, "--order", "pd,lgd,exposure,settings,other")
    assert "loss allowance" in text and "gross carrying amount" in text and "order of the drivers: pd, lgd" in text
    assert "-0.00" not in text
    assert json.loads(out("staging-quality", *full)) == as_json(
        period.staging_quality(opening, closing, write_offs=gone)
    )
    assert json.loads(out("backtest", *full, "--realised-z", "-0.5", "--months", "12")) == as_json(
        period.backtest(opening, closing, write_offs=gone, realised_z=-0.5)
    )
    assert json.loads(out("challenger", *full)) == as_json(review.challenger(opening, closing, write_offs=gone))
    assert json.loads(out("reconcile", f["loans.csv"], "--config", f["settings.json"])) == as_json(
        review.reconcile(closing)
    )
    got = json.loads(out("response", f["loans.csv"], "--config", f["settings.json"], "--shifts=-1,1"))
    assert [p["shift"] for p in got["points"]] == [-1.0, 0.0, 1.0]
    # without --opening-config both dates use the same settings: no effect of settings
    same = json.loads(out("movement", *two, "--json"))
    assert next(r for r in same["loss_allowance"] if r["row"] == "change_settings")["total"] == 0.0
    assert main(["movement", *full, "--order", "pd"]) == 2
    assert main(["response", f["loans.csv"], "--config", f["settings.json"], "--shifts", "a,b"]) == 2
    (tmp_path / "bad.csv").write_text("id\nx\n")
    assert main(["movement", *two, "--write-offs", str(tmp_path / "bad.csv")]) == 2
    assert "error:" in capsys.readouterr().err


def test_write_off_file(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text("amount,LOAN_ID\n10, a \n20,b\n\n")
    assert period.read_write_offs(p) == ["a", "b"]
    p.write_text("loan_id\na\na\n")
    with pytest.raises(InputError, match="more than once"):
        period.read_write_offs(p)
    p.write_text("")
    with pytest.raises(InputError, match="empty"):
        period.read_write_offs(p)


def test_everything_is_json_ready(sample):
    _, opening, closing, gone = sample
    for result in (
        period.movement(opening, closing, write_offs=gone),
        period.staging_quality(opening, closing, write_offs=gone),
        period.backtest(opening, closing, write_offs=gone, realised_z=0.3),
        review.reconcile(closing),
        review.challenger(opening, closing, write_offs=gone),
        review.response(closing.loans, closing.config, shifts=[-1, 1]),
    ):
        assert json.loads(json.dumps(result)) is not None
    assert dataclasses.is_dataclass(period.pair(opening, closing, gone))
    assert len(list(itertools.permutations(period.DRIVERS))) == 120
