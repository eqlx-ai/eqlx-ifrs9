"""A made-up loan portfolio with known answers, for demos and tests. No real data is involved.

:func:`generate` returns the loan file as CSV text, a settings object and a list of planted cases:
loans built so that a specific rule decides their stage. The same seed always gives the same file.

The loan file also carries what a lender would report itself (``reported_stage``, ``reported_ecl``),
with planted differences, and the sample comes with the same portfolio one year earlier, its
settings and a list of loans written off in between, for the period-to-period reports.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import expm

GRADES = ("A", "B", "C", "D", "E")
# Annual transition intensities between the five grades and default (last column). Invented numbers.
_INTENSITY = np.array(
    [
        [0.000, 0.060, 0.010, 0.002, 0.000, 0.0005],
        [0.040, 0.000, 0.080, 0.012, 0.003, 0.0020],
        [0.006, 0.070, 0.000, 0.100, 0.015, 0.0080],
        [0.001, 0.012, 0.090, 0.000, 0.130, 0.0300],
        [0.000, 0.003, 0.020, 0.110, 0.000, 0.1200],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.0000],
    ]
)
COLUMNS = (
    "loan_id",
    "borrower_id",
    "segment",
    "product",
    "amortization",
    "balance",
    "undrawn",
    "ccf",
    "ccf_12m",
    "rate",
    "eir",
    "remaining_months",
    "months_on_book",
    "pd_12m",
    "pd_12m_origination",
    "pd_origination_remaining",
    "grade",
    "grade_origination",
    "lgd",
    "lgd_default",
    "dpd",
    "forborne",
    "watchlist",
    "credit_impaired",
    "previous_stage",
    "periods_without_trigger",
    "consumer",
    "in_monitoring_year",
)


def corporate_matrix() -> np.ndarray[Any, np.dtype[np.float64]]:
    """One-year transition matrix of the made-up corporate rating scale (it has an exact generator)."""
    q = _INTENSITY.copy()
    q[np.diag_indices(6)] = -q.sum(axis=1)
    return np.asarray(expm(q), dtype=np.float64)


@dataclass(frozen=True)
class Planted:
    loan_id: str
    stage: int
    primary_reason: str
    what: str


@dataclass(frozen=True)
class Sample:
    csv: str
    settings: dict[str, Any]
    planted: tuple[Planted, ...]
    previous_csv: str = ""
    """The same portfolio one year earlier."""
    previous_settings: dict[str, Any] | None = None
    write_offs_csv: str = ""
    """Loans of the earlier file that were written off during the year (column ``loan_id``)."""


REPORTED = ("reported_stage", "reported_ecl")
# What the made-up lender does differently from the recalculation.
KEPT_IN_STAGE_1_EVERY = 7  # every 7th loan that the PD test moves to stage 2 stays in stage 1
REPORTED_FACTOR = {"cards": 0.86, "mortgage": 0.93}  # share of the recalculated allowance that is booked


def settings(profile: str | None = "tr_bddk") -> dict[str, Any]:
    out: dict[str, Any] = {
        "timing": "end",
        "ead_basis": "start",
        "pd_basis": "long_run",
        "staging": {
            "pd_measure": "annualized_lifetime",
            "scenario_basis": "weighted",
            "relative_threshold": 2.0,
            "relative_bands": [[0.005, 3.0]],
            "backstop_ratio": 3.0,
            "low_credit_risk_pd": 0.003,
            "probation_periods": 3,
        },
        "scenarios": [
            {"name": "base", "weight": 0.6, "z": [0.0, 0.2, 0.3]},
            {"name": "downside", "weight": 0.25, "z": [-1.6, -1.0, -0.4]},
            {"name": "upside", "weight": 0.15, "z": [1.0, 0.8, 0.5]},
        ],
        "base_scenario": "base",
        "forecast": {"reversion_years": 2, "method": "linear"},
        "segments": {
            "corporate": {
                "rho": 0.18,
                "grades": list(GRADES),
                "matrix": [[round(float(v), 10) for v in row] for row in corporate_matrix()],
                "matrix_tolerance": 1e-8,
            },
            "mortgage": {"rho": 0.15, "cpr": 0.04},
            "consumer": {"rho": 0.06},
            "cards": {"rho": 0.04},
        },
    }
    if profile:
        out["profile"] = profile
    return out


def _row(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {c: "" for c in COLUMNS}
    base.update(product="term", amortization="annuity", undrawn=0, ccf=0, dpd=0, months_on_book=12)
    base.update(forborne=0, watchlist=0, credit_impaired=0, consumer=0, in_monitoring_year=0)
    base.update(previous_stage=1, periods_without_trigger=12)
    base.update(kw)
    if base["borrower_id"] == "":
        base["borrower_id"] = "B-" + str(base["loan_id"])
    return base


def generate(n_loans: int = 2000, seed: int = 7, profile: str | None = "tr_bddk") -> Sample:
    """A portfolio of ``n_loans`` random loans plus eleven planted cases."""
    if n_loans < 0:
        raise ValueError("n_loans must not be negative")
    rng = np.random.default_rng(seed)

    def u(lo: float, hi: float) -> float:
        return float(lo + (hi - lo) * rng.random())

    def pick(items: tuple[Any, ...], weights: tuple[float, ...]) -> Any:
        x, acc = rng.random() * sum(weights), 0.0
        for item, w in zip(items, weights, strict=True):
            acc += w
            if x < acc:
                return item
        return items[-1]

    rows: list[dict[str, Any]] = []
    for i in range(n_loans):
        seg = pick(("corporate", "mortgage", "consumer", "cards"), (0.25, 0.25, 0.30, 0.20))
        lid = f"L{i + 1:06d}"
        dpd = pick((0, 0, 12, 25, 45, 75, 120, 250), (0.80, 0.08, 0.04, 0.03, 0.02, 0.012, 0.012, 0.006))
        common = dict(loan_id=lid, segment=seg, dpd=dpd, months_on_book=int(u(1, 60)))
        if dpd > 90:  # a loan this far past due is not a new loan
            common["months_on_book"] = max(common["months_on_book"], 14)
        common["forborne"] = int(rng.random() < 0.02)
        common["watchlist"] = int(rng.random() < 0.02)
        if seg == "corporate":
            g0 = int(pick((0, 1, 2, 3, 4), (0.15, 0.35, 0.30, 0.15, 0.05)))
            g1 = min(4, max(0, g0 + int(pick((-1, 0, 1, 2), (0.10, 0.65, 0.18, 0.07)))))
            rate = round(u(0.08, 0.20), 4)
            rows.append(
                _row(
                    **common,
                    borrower_id=f"C{int(u(1, max(2, n_loans // 12))):05d}",
                    amortization=pick(("bullet", "equal_principal", "annuity"), (0.4, 0.3, 0.3)),
                    balance=round(u(200_000, 8_000_000), 2),
                    rate=rate,
                    eir=round(rate + u(0.0, 0.01), 4),
                    remaining_months=int(u(3, 84)),
                    grade=GRADES[g1],
                    grade_origination=GRADES[g0],
                    lgd=round(u(0.25, 0.55), 3),
                    lgd_default=round(u(0.40, 0.70), 3),
                )
            )
            continue
        pd0 = {"mortgage": u(0.002, 0.02), "consumer": u(0.01, 0.08), "cards": u(0.015, 0.10)}[seg]
        pd1 = min(0.6, pd0 * float(pick((0.7, 1.0, 1.4, 2.4, 4.0), (0.15, 0.55, 0.18, 0.08, 0.04))))
        if seg == "mortgage":
            rate = round(u(0.09, 0.16), 4)
            rows.append(
                _row(
                    **common,
                    consumer=1,
                    balance=round(u(150_000, 3_000_000), 2),
                    rate=rate,
                    eir=round(rate + 0.003, 4),
                    remaining_months=int(u(24, 180)),
                    pd_12m=round(pd1, 6),
                    pd_12m_origination=round(pd0, 6),
                    lgd=round(u(0.10, 0.25), 3),
                    lgd_default=round(u(0.20, 0.35), 3),
                )
            )
        elif seg == "consumer":
            rate = round(u(0.20, 0.45), 4)
            rows.append(
                _row(
                    **common,
                    consumer=1,
                    balance=round(u(5_000, 250_000), 2),
                    rate=rate,
                    eir=round(rate + 0.01, 4),
                    remaining_months=int(u(2, 48)),
                    pd_12m=round(pd1, 6),
                    pd_12m_origination=round(pd0, 6),
                    lgd=round(u(0.45, 0.75), 3),
                    lgd_default=round(u(0.60, 0.85), 3),
                )
            )
        else:
            limit = round(u(5_000, 150_000), 2)
            used = round(limit * u(0.05, 0.9), 2)
            rate = round(u(0.30, 0.50), 4)
            rows.append(
                _row(
                    **common,
                    consumer=1,
                    product="revolving",
                    amortization="bullet",
                    balance=used,
                    undrawn=round(limit - used, 2),
                    ccf=0.45,
                    ccf_12m=0.30,
                    rate=rate,
                    eir=rate,
                    remaining_months=30,
                    pd_12m=round(pd1, 6),
                    pd_12m_origination=round(pd0, 6),
                    lgd=round(u(0.55, 0.80), 3),
                    lgd_default=round(u(0.70, 0.90), 3),
                )
            )

    retail = dict(segment="consumer", consumer=1, balance=100_000.0, rate=0.30, eir=0.31, lgd=0.6, lgd_default=0.75)
    planted_rows = [
        _row(loan_id="CASE-PD-TRIPLED", remaining_months=36, pd_12m=0.09, pd_12m_origination=0.03, **retail),
        _row(
            loan_id="CASE-SHORTER-LIFE",
            remaining_months=36,
            months_on_book=24,
            pd_12m=0.0565,
            pd_origination_remaining=0.076923,
            **retail,
        ),
        _row(loan_id="CASE-DPD-ONLY", remaining_months=24, pd_12m=0.03, pd_12m_origination=0.03, dpd=45, **retail),
        _row(loan_id="CASE-DEFAULT", remaining_months=24, pd_12m=0.20, pd_12m_origination=0.03, dpd=120, **retail),
        _row(loan_id="CASE-FORBORNE", remaining_months=24, pd_12m=0.03, pd_12m_origination=0.03, forborne=1, **retail),
        _row(
            loan_id="CASE-LOW-RISK",
            segment="mortgage",
            consumer=1,
            balance=900_000.0,
            rate=0.10,
            eir=0.103,
            remaining_months=120,
            pd_12m=0.0016,
            pd_12m_origination=0.0004,
            lgd=0.15,
            lgd_default=0.25,
        ),
        _row(
            loan_id="CASE-PROBATION",
            remaining_months=24,
            pd_12m=0.03,
            pd_12m_origination=0.03,
            previous_stage=2,
            periods_without_trigger=1,
            **retail,
        ),
        _row(loan_id="CASE-SHORT-LIFE", remaining_months=5, pd_12m=0.03, pd_12m_origination=0.03, **retail),
        _row(loan_id="CASE-SCENARIO", remaining_months=36, pd_12m=0.03, pd_12m_origination=0.0165, **retail),
    ]
    company = dict(
        segment="corporate",
        borrower_id="C-GROUP-X",
        amortization="bullet",
        rate=0.15,
        eir=0.155,
        grade="B",
        grade_origination="B",
        lgd=0.4,
        lgd_default=0.55,
    )
    planted_rows.append(_row(loan_id="CASE-GROUP-X-LATE", balance=2_000_000.0, remaining_months=18, dpd=200, **company))
    planted_rows.append(
        _row(loan_id="CASE-GROUP-X-CURRENT", balance=5_000_000.0, remaining_months=30, dpd=0, **company)
    )
    rows += planted_rows

    with_profile = profile == "tr_bddk"
    planted = (
        Planted("CASE-PD-TRIPLED", 2, "pd_relative", "12-month PD went from 3% to 9%"),
        Planted(
            "CASE-SHORTER-LIFE",
            2,
            "pd_relative",
            "remaining-life PD is about 16% against 7.7% expected at origination for the same three years; "
            "against the full five-year origination PD of 10% it would look like no increase",
        ),
        Planted("CASE-DPD-ONLY", 2, "days_past_due_backstop", "45 days past due and no change in PD"),
        Planted("CASE-DEFAULT", 3, "default_days_past_due", "120 days past due"),
        Planted("CASE-FORBORNE", 2, "forborne", "restructured because of financial difficulty"),
        Planted("CASE-LOW-RISK", 1, "none", "PD four times higher, but still below the low credit risk cut-off"),
        Planted("CASE-PROBATION", 2, "probation", "left stage 2 conditions one period ago; three are required"),
        Planted("CASE-SHORT-LIFE", 1, "none", "five months left: 12-month ECL equals lifetime ECL"),
        Planted(
            "CASE-SCENARIO",
            1,
            "none",
            "stage 1 on the weighted PD, stage 2 if the downside scenario alone had full weight",
        ),
        Planted("CASE-GROUP-X-LATE", 3, "default_days_past_due", "200 days past due"),
        Planted(
            "CASE-GROUP-X-CURRENT",
            3 if with_profile else 1,
            "profile_default" if with_profile else "none",
            "not past due itself; another loan of the same borrower is non-performing (BDDK Article 5(5))",
        ),
    )
    cfg = settings(profile)
    text = _csv(rows, COLUMNS)
    previous_rows, written_off = _previous(rows, seed)
    return Sample(
        _with_reported(text, cfg),
        cfg,
        planted,
        _csv(previous_rows, COLUMNS),
        previous_settings(profile),
        "loan_id\n" + "".join(f"{x}\n" for x in written_off),
    )


def _csv(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    buf = io.StringIO()
    buf.write(",".join(columns) + "\n")
    for r in rows:
        buf.write(",".join(str(r[c]) for c in columns) + "\n")
    return buf.getvalue()


def previous_settings(profile: str | None = "tr_bddk") -> dict[str, Any]:
    """Settings of the earlier date: a milder outlook and a little less weight on the downside."""
    out = settings(profile)
    out["scenarios"] = [
        {"name": "base", "weight": 0.65, "z": [0.2, 0.3, 0.3]},
        {"name": "downside", "weight": 0.20, "z": [-1.2, -0.8, -0.3]},
        {"name": "upside", "weight": 0.15, "z": [1.0, 0.8, 0.5]},
    ]
    return out


def _with_reported(text: str, cfg: dict[str, Any]) -> str:
    """Append the lender's own stage and allowance to every row of the loan file."""
    from . import config, data, engine
    from .staging import Reason

    r = engine.run(data.from_text(text), config.from_dict(cfg))
    ids = r.loans.text["loan_id"]
    stage = r.stage.copy()
    by_pd = [
        i
        for i in np.argsort(ids, kind="stable").tolist()
        if r.stage[i] == 2 and r.primary[i] == int(Reason.PD_RELATIVE) and not str(ids[i]).startswith("CASE")
    ]
    stage[by_pd[KEPT_IN_STAGE_1_EVERY - 1 :: KEPT_IN_STAGE_1_EVERY]] = 1
    factor = np.array([REPORTED_FACTOR.get(str(sg), 1.0) for sg in r.loans.text["segment"].tolist()])
    booked = r.ecl_for_stages(stage) * factor
    lines = text.rstrip("\n").split("\n")
    out = [lines[0] + "," + ",".join(REPORTED)]
    out += [f"{line},{int(stage[i])},{float(booked[i]):.2f}" for i, line in enumerate(lines[1:])]
    return "\n".join(out) + "\n"


def _previous(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[str]]:
    """The portfolio one year earlier, and the ids of loans written off since.

    Loans younger than a year did not exist yet. The others are a year younger, owe more, and were
    mostly in better shape. Some loans exist only in the earlier file: repaid, or written off.
    """
    rng = np.random.default_rng(seed + 1000)

    def u(lo: float, hi: float) -> float:
        return float(lo + (hi - lo) * rng.random())

    def pick(items: tuple[Any, ...], weights: tuple[float, ...]) -> Any:
        x, acc = rng.random() * sum(weights), 0.0
        for item, w in zip(items, weights, strict=True):
            acc += w
            if x < acc:
                return item
        return items[-1]

    out: list[dict[str, Any]] = []
    for row in rows:
        age = int(row["months_on_book"])
        if age < 12:
            continue
        prev = dict(row)
        prev["months_on_book"] = age - 12
        prev["previous_stage"], prev["periods_without_trigger"] = 1, 12
        rem = int(row["remaining_months"])
        balance = float(row["balance"])
        if row["product"] == "revolving":
            limit = balance + float(row["undrawn"])
            used = round(min(limit, balance * u(0.7, 1.2)), 2)
            prev["balance"], prev["undrawn"] = used, round(limit - used, 2)
        else:
            prev["remaining_months"] = rem + 12
            if row["amortization"] != "bullet":
                prev["balance"] = round(balance * min(2.0, (rem + 12) / rem), 2)
        dpd = int(row["dpd"])
        if row["grade"] != "":
            if dpd > 90:  # loans that have defaulted since were mostly weak already
                prev["grade"] = pick(("C", "D", "E"), (0.15, 0.35, 0.50))
            else:
                g = GRADES.index(row["grade"]) + int(pick((-1, 0, 1), (0.25, 0.65, 0.10)))
                prev["grade"] = GRADES[min(4, max(0, g))]
        if row["pd_12m"] != "":
            earlier = float(row["pd_12m"]) / float(pick((0.8, 1.0, 1.3, 2.0), (0.10, 0.60, 0.20, 0.10)))
            prev["pd_12m"] = round(min(0.6, max(0.0005, earlier)), 6)
        if dpd > 90:
            prev["dpd"] = pick((dpd, 0, 15, 45, 60), (0.55, 0.05, 0.03, 0.20, 0.17))
        elif dpd > 0:
            prev["dpd"] = pick((0, dpd), (0.8, 0.2))
        for flag in ("forborne", "watchlist"):
            if int(row[flag]) and rng.random() < 0.7:
                prev[flag] = 0
        if rng.random() < 0.10:
            prev["lgd"] = round(max(0.01, float(row["lgd"]) - 0.03), 3)
        out.append(prev)

    written_off: list[str] = []
    for i in range(max(3, len(rows) // 12)):
        lid = f"P{i + 1:06d}"
        lost = i % 4 == 0
        rate = round(u(0.20, 0.45), 4)
        pd0 = u(0.01, 0.08)
        out.append(
            _row(
                loan_id=lid,
                segment="consumer",
                consumer=1,
                balance=round(u(5_000, 250_000), 2),
                rate=rate,
                eir=round(rate + 0.01, 4),
                remaining_months=int(u(2, 12)),
                months_on_book=int(u(12, 48)),
                pd_12m=round(pd0 * (4.0 if lost else 1.0), 6),
                pd_12m_origination=round(pd0, 6),
                lgd=round(u(0.45, 0.75), 3),
                lgd_default=round(u(0.60, 0.85), 3),
                dpd=pick((150, 45), (0.6, 0.4)) if lost else 0,
            )
        )
        if lost:
            written_off.append(lid)
    return out, written_off


def individual_cases() -> dict[str, dict[str, Any]]:
    """Two made-up debtors for the individual assessment: one that may survive, one that will not.

    Keyed by file name. Every figure is invented; the amounts are in one currency.
    """
    going = {
        "debtor": {
            "id": "DEMO-MACHINERY",
            "sector": "other",
            "months_past_due": 7,
            "ebitda_history": [18_200_000, 21_500_000],
        },
        "exposures": [
            {"id": "TERM-LOAN", "gross_carrying_amount": 42_000_000, "eir": 0.38, "rank": 2},
            {
                "id": "REVOLVING",
                "gross_carrying_amount": 8_000_000,
                "eir": 0.42,
                "rank": 2,
                "expected_draws": [{"year": 1, "amount": 2_000_000}],
            },
        ],
        "other_claims": [
            {"name": "taxes and employees", "rank": 1, "amount": 6_500_000},
            {"name": "other banks", "rank": 2, "amount": 55_000_000},
            {"name": "trade creditors", "rank": 3, "amount": 31_000_000},
        ],
        "collateral": [
            {
                "id": "FACTORY",
                "type": "industrial property",
                "value": 60_000_000,
                "valuation_age_months": 14,
                "secures": "TERM-LOAN",
                "central_to_cash_flows": True,
                "sale": {
                    "years": 2.5,
                    "haircut": 0.20,
                    "selling_cost_rate": 0.04,
                    "annual_cost_rate": 0.01,
                    "auction": True,
                },
            },
            {
                "id": "PLEDGED-DEPOSIT",
                "type": "cash",
                "value": 3_000_000,
                "secures": "REVOLVING",
                "central_to_cash_flows": False,
                "sale": {"years": 0, "haircut": 0.0},
            },
        ],
        "scenarios": [
            {
                "name": "restructuring holds",
                "probability": 0.55,
                "approach": "steady_state",
                "cash_flow": {
                    "revenue": 310_000_000,
                    "cost_of_sales": 262_000_000,
                    "distribution_costs": 14_000_000,
                    "administrative_expenses": 11_000_000,
                    "other_gains_losses": -1_500_000,
                    "tax_rate": 0.25,
                    "net_interest_expense": 21_000_000,
                    "depreciation": 6_000_000,
                    "owner_payout": 1_200_000,
                    "capex": 5_000_000,
                    "sustainability_deductions": 800_000,
                },
                "multiple": 6,
                "non_pledged_cash": 2_000_000,
            },
            {
                "name": "liquidation",
                "probability": 0.45,
                "approach": "gone_concern",
                "collateral": {"FACTORY": {"years": 3, "haircut": 0.25}},
                "unsecured_recovery": {"amount": 9_000_000, "year": 3},
            },
        ],
    }
    gone = {
        "debtor": {"id": "DEMO-DEVELOPER", "sector": "other", "months_past_due": 20},
        "exposures": [{"id": "PROJECT-LOAN", "gross_carrying_amount": 95_000_000, "eir": 0.35}],
        "collateral": [
            {
                "id": "LAND",
                "type": "land with unfinished building",
                "value": 120_000_000,
                "valuation_age_months": 9,
                "secures": "PROJECT-LOAN",
                "central_to_cash_flows": True,
                "prior_claims": 10_000_000,
                "sale": {
                    "years": 3,
                    "haircut": 0.15,
                    "selling_cost_rate": 0.05,
                    "annual_cost_rate": 0.01,
                    "costs": [{"year": 1, "amount": 1_500_000}],
                    "auction": True,
                },
            }
        ],
        "scenarios": [
            {"name": "sale as planned", "probability": 0.6, "approach": "gone_concern"},
            {
                "name": "slow market",
                "probability": 0.4,
                "approach": "gone_concern",
                "collateral": {"LAND": {"years": 4, "haircut": 0.30, "value_change": -0.10}},
            },
        ],
    }
    return {"case_going_concern.json": going, "case_gone_concern.json": gone}
