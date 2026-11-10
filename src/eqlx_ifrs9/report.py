"""Tables from a portfolio run. Plain dictionaries, ready for JSON, a terminal or an AI agent."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .engine import PortfolioResult
from .errors import InputError
from .staging import Reason, names

STAGES = (1, 2, 3)
LIST_LIMIT = 500


def _row(mask: NDArray[np.bool_], r: PortfolioResult) -> dict[str, float]:
    gross = float(r.loans.num["balance"][mask].sum())
    exposure = float(r.exposure[mask].sum())
    ecl = float(r.ecl[mask].sum())
    return {
        "loans": int(mask.sum()),
        "gross_carrying_amount": gross,
        "exposure": exposure,
        "ecl": ecl,
        "coverage": ecl / exposure if exposure > 0 else 0.0,
    }


def _reason_label(code: int) -> str:
    return Reason(code).name.lower() if code else "none"  # type: ignore[union-attr]


def summary(r: PortfolioResult) -> dict[str, Any]:
    """Totals, the split by stage and segment, why loans are in stage 2, scenarios and checks."""
    n = r.n
    everything = np.ones(n, dtype=bool)
    total = _row(everything, r)
    by_stage = []
    for s in STAGES:
        row = {"stage": s, **_row(r.stage == s, r)}
        row["share_of_exposure"] = row["exposure"] / total["exposure"] if total["exposure"] > 0 else 0.0
        by_stage.append(row)
    seg = r.loans.text["segment"]
    by_segment = [
        {"segment": str(name), "stage": s, **_row((seg == name) & (r.stage == s), r)}
        for name in sorted(set(seg.tolist()))
        for s in STAGES
        if np.any((seg == name) & (r.stage == s))
    ]
    out: dict[str, Any] = {
        "total": total,
        "by_stage": by_stage,
        "by_segment": by_segment,
        "stage_2_by_primary_reason": _by_reason(r, 2),
        "stage_3_by_primary_reason": _by_reason(r, 3),
        "scenarios": scenario_table(r),
        "checks": checks(r),
        "warnings": list(r.warnings),
        "settings": r.config.to_dict(),
    }
    if r.group is not None:
        out["regulatory_groups"] = [{"group": g, **_row(r.group == g, r)} for g in range(1, 6)]
    migration = _migration(r)
    if migration is not None:
        out["migration_from_previous_stage"] = migration
    return out


def _by_reason(r: PortfolioResult, stage: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    in_stage = r.stage == stage
    for code in sorted(set(r.primary[in_stage].tolist())):
        rows.append({"reason": _reason_label(int(code)), **_row(in_stage & (r.primary == code), r)})
    rows.sort(key=lambda x: (-x["exposure"], x["reason"]))
    return rows


def scenario_table(r: PortfolioResult) -> dict[str, Any]:
    """Weighted ECL and, per scenario, the ECL if that scenario alone had full weight.

    ``ecl_same_stages`` keeps the reported stages and changes only the measurement.
    ``ecl_restaged`` also lets each loan move to the stage it would have under that scenario alone,
    which is the sensitivity usually disclosed.
    """
    cfg = r.config
    weighted = float(r.ecl.sum())
    rows: list[dict[str, Any]] = []
    for k, s in enumerate(cfg.scenarios):
        same = float(r.ecl_for_stages(r.stage, k).sum())
        st = r.scenario_stage[k]
        restaged = r.ecl_for_stages(st, k)
        rows.append(
            {
                "name": s.name,
                "weight": s.weight,
                "ecl_same_stages": same,
                "ecl_restaged": float(restaged.sum()),
                "exposure_by_stage_restaged": {str(x): float(r.exposure[st == x].sum()) for x in STAGES},
                "loans_changing_stage": int(np.sum(st != r.stage)),
            }
        )
    base = rows[cfg.base_index]["ecl_restaged"]
    return {
        "weighted_ecl": weighted,
        "base_scenario": cfg.base_scenario,
        "weighted_to_base": weighted / base if base > 0 else None,
        "scenarios": rows,
    }


def checks(r: PortfolioResult) -> dict[str, Any]:
    """Diagnostics supervisors and auditors look at (EBA IFRS 9 monitoring reports 2021 and 2023)."""
    s1, s2 = r.stage == 1, r.stage == 2
    exp = r.exposure
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(r.annual_pd_origination > 0, r.annual_pd_now / r.annual_pd_origination, np.nan)
    exempt = (r.reasons & int(Reason.LOW_CREDIT_RISK)) != 0
    # EBA's filter: leave out low-credit-risk exemptions and annualised lifetime PDs below 0.3%
    tripled = s1 & (ratio >= 3.0) & ~exempt & (r.annual_pd_now >= 0.003)
    dpd_only = s2 & (r.primary == int(Reason.DAYS_PAST_DUE_BACKSTOP))
    e1, e2 = float(exp[s1].sum()), float(exp[s2].sum())
    out: dict[str, Any] = {
        "stage_1_with_threefold_pd_increase": {
            "loans": int(tripled.sum()),
            "exposure": float(exp[tripled].sum()),
            "share_of_stage_1_exposure": float(exp[tripled].sum()) / e1 if e1 > 0 else 0.0,
            "all_including_exempt_and_low_pd": int((s1 & (ratio >= 3.0)).sum()),
        },
        "stage_2_only_because_of_days_past_due": {
            "loans": int(dpd_only.sum()),
            "exposure": float(exp[dpd_only].sum()),
            "share_of_stage_2_exposure": float(exp[dpd_only].sum()) / e2 if e2 > 0 else 0.0,
        },
        "stage_1_kept_by_low_credit_risk_exemption": {
            "loans": int((s1 & exempt).sum()),
            "exposure": float(exp[s1 & exempt].sum()),
        },
        "loans_without_origination_pd": int(np.isnan(r.pd_origination).sum()),
        "life_under_12_months": int((r.loans.num["remaining_months"] < 12).sum()),
    }
    prev = r.loans.num["previous_stage"]
    known = ~np.isnan(prev)
    if np.any(known):
        was1 = known & (prev == 1)
        direct = was1 & (r.stage == 3)
        out["stage_1_directly_to_stage_3"] = {
            "loans": int(direct.sum()),
            "share_of_previous_stage_1_loans": float(direct.sum()) / int(was1.sum()) if was1.any() else 0.0,
        }
    return out


def _migration(r: PortfolioResult) -> dict[str, Any] | None:
    prev = r.loans.num["previous_stage"]
    known = ~np.isnan(prev)
    if not np.any(known):
        return None
    loans = [[int(np.sum(known & (prev == a) & (r.stage == b))) for b in STAGES] for a in STAGES]
    exposure = [[float(r.exposure[known & (prev == a) & (r.stage == b)].sum()) for b in STAGES] for a in STAGES]
    return {
        "rows_previous_stage_columns_current_stage": True,
        "loans": loans,
        "exposure": exposure,
        "loans_without_previous_stage": int(np.sum(~known)),
    }


def loan_rows(
    r: PortfolioResult,
    *,
    stage: int | None = None,
    reason: str | None = None,
    segment: str | None = None,
    borrower_id: str | None = None,
    sort: str = "ecl",
    limit: int = 50,
) -> dict[str, Any]:
    """Loans matching a filter, largest first by ``sort`` (ecl, exposure, coverage, pd_ratio, dpd)."""
    mask = np.ones(r.n, dtype=bool)
    if stage is not None:
        if stage not in STAGES:
            raise InputError("stage must be 1, 2 or 3")
        mask &= r.stage == stage
    if reason is not None:
        try:
            bit = int(Reason[reason.upper()])
        except KeyError:
            valid = ", ".join(x.name.lower() for x in Reason if x.value and x.name)
            raise InputError(f"unknown reason {reason!r}; one of: {valid}") from None
        mask &= (r.reasons & bit) != 0
    if segment is not None:
        mask &= r.loans.text["segment"] == segment
    if borrower_id is not None:
        mask &= r.loans.text["borrower_id"] == borrower_id
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage = np.where(r.exposure > 0, r.ecl / r.exposure, 0.0)
        ratio = np.where(r.pd_origination > 0, r.pd_now / r.pd_origination, np.nan)
    keys = {
        "ecl": r.ecl,
        "exposure": r.exposure,
        "coverage": coverage,
        "pd_ratio": np.nan_to_num(ratio, nan=-1.0),
        "dpd": r.loans.num["dpd"],
    }
    if sort not in keys:
        raise InputError(f"sort must be one of {', '.join(keys)}")
    idx = np.nonzero(mask)[0]
    ids = r.loans.text["loan_id"]
    order = sorted(idx.tolist(), key=lambda i: (-float(keys[sort][i]), str(ids[i])))
    lim = max(1, min(int(limit), LIST_LIMIT))
    rows = []
    for i in order[:lim]:
        row: dict[str, Any] = {
            "loan_id": str(ids[i]),
            "borrower_id": str(r.loans.text["borrower_id"][i]),
            "segment": str(r.loans.text["segment"][i]),
            "stage": int(r.stage[i]),
            "primary_reason": _reason_label(int(r.primary[i])),
            "reasons": names(int(r.reasons[i])),
            "exposure": float(r.exposure[i]),
            "ecl": float(r.ecl[i]),
            "coverage": float(coverage[i]),
            "pd_now": float(r.pd_now[i]),
            "pd_origination": None if np.isnan(r.pd_origination[i]) else float(r.pd_origination[i]),
            "pd_ratio": None if np.isnan(ratio[i]) else float(ratio[i]),
            "dpd": float(r.loans.num["dpd"][i]),
        }
        if r.group is not None:
            row["regulatory_group"] = int(r.group[i])
        rows.append(row)
    return {"matching": int(idx.size), "returned": len(rows), "sorted_by": sort, "loans": rows}


def compare(a: PortfolioResult, b: PortfolioResult, limit: int = 20) -> dict[str, Any]:
    """Difference between two runs on the same loans (b minus a): totals, stage moves, largest changes."""
    if a.n != b.n or not np.array_equal(a.loans.text["loan_id"], b.loans.text["loan_id"]):
        raise InputError("the two runs must be on the same loan file")
    delta = b.ecl - a.ecl
    moves = [[int(np.sum((a.stage == x) & (b.stage == y))) for y in STAGES] for x in STAGES]
    by_stage = []
    for s in STAGES:
        ra, rb = _row(a.stage == s, a), _row(b.stage == s, b)
        by_stage.append(
            {
                "stage": s,
                "loans_before": ra["loans"],
                "loans_after": rb["loans"],
                "exposure_before": ra["exposure"],
                "exposure_after": rb["exposure"],
                "ecl_before": ra["ecl"],
                "ecl_after": rb["ecl"],
            }
        )
    ids = a.loans.text["loan_id"]
    order = sorted(np.nonzero(delta != 0)[0].tolist(), key=lambda i: (-abs(float(delta[i])), str(ids[i])))
    lim = max(1, min(int(limit), LIST_LIMIT))
    return {
        "ecl_before": float(a.ecl.sum()),
        "ecl_after": float(b.ecl.sum()),
        "ecl_change": float(delta.sum()),
        "ecl_change_relative": float(delta.sum() / a.ecl.sum()) if a.ecl.sum() > 0 else None,
        "by_stage": by_stage,
        "stage_moves_rows_before_columns_after": moves,
        "loans_changing_stage": int(np.sum(a.stage != b.stage)),
        "largest_changes": [
            {
                "loan_id": str(ids[i]),
                "stage_before": int(a.stage[i]),
                "stage_after": int(b.stage[i]),
                "ecl_before": float(a.ecl[i]),
                "ecl_after": float(b.ecl[i]),
                "change": float(delta[i]),
            }
            for i in order[:lim]
        ],
        "warnings_after": list(b.warnings),
    }


def describe(loans: Any) -> dict[str, Any]:
    """What a loan file contains, before any calculation: size, totals and gaps in the data."""
    num, text, flag = loans.num, loans.text, loans.flag
    n = loans.n
    dpd = num["dpd"]
    has_orig = (
        ~np.isnan(num["pd_origination_remaining"])
        | ~np.isnan(num["pd_12m_origination"])
        | (text["grade_origination"] != "")
    )

    def counts(col: str) -> dict[str, int]:
        vals, c = np.unique(text[col], return_counts=True)
        return {str(v): int(k) for v, k in zip(vals.tolist(), c.tolist(), strict=True)}

    return {
        "loans": n,
        "borrowers": int(np.unique(text["borrower_id"]).size),
        "columns": list(loans.columns),
        "gross_carrying_amount": float(num["balance"].sum()),
        "undrawn": float(num["undrawn"].sum()),
        "segments": counts("segment"),
        "products": counts("product"),
        "amortization": counts("amortization"),
        "remaining_months": {
            "min": int(num["remaining_months"].min()),
            "median": float(np.median(num["remaining_months"])),
            "max": int(num["remaining_months"].max()),
        },
        "days_past_due": {
            "0": int(np.sum(dpd == 0)),
            "1-30": int(np.sum((dpd > 0) & (dpd <= 30))),
            "31-90": int(np.sum((dpd > 30) & (dpd <= 90))),
            "over 90": int(np.sum(dpd > 90)),
        },
        "pd_source": {
            "pd_12m": int(np.sum(~np.isnan(num["pd_12m"]))),
            "grade_only": int(np.sum(np.isnan(num["pd_12m"]) & (text["grade"] != ""))),
        },
        "origination_risk_known": int(has_orig.sum()),
        "origination_risk_missing": int(n - has_orig.sum()),
        "flags": {k: int(v.sum()) for k, v in flag.items() if k in loans.columns},
        "previous_stage_given": int(np.sum(~np.isnan(num["previous_stage"]))),
    }


# ---- text for the terminal ---------------------------------------------------------------------


def _money(x: float) -> str:
    return f"{x:,.2f}"


def text_summary(s: dict[str, Any]) -> str:
    t = s["total"]
    lines = [
        f"loans {t['loans']:,}   exposure {_money(t['exposure'])}   ECL {_money(t['ecl'])}   "
        f"coverage {t['coverage']:.2%}",
        "",
        f"{'stage':<7}{'loans':>9}{'exposure':>20}{'share':>9}{'ECL':>18}{'coverage':>10}",
    ]
    for row in s["by_stage"]:
        lines.append(
            f"{row['stage']:<7}{row['loans']:>9,}{_money(row['exposure']):>20}{row['share_of_exposure']:>9.1%}"
            f"{_money(row['ecl']):>18}{row['coverage']:>10.2%}"
        )
    if s["stage_2_by_primary_reason"]:
        lines += ["", "stage 2 by primary reason"]
        for row in s["stage_2_by_primary_reason"]:
            lines.append(
                f"  {row['reason']:<26}{row['loans']:>8,}{_money(row['exposure']):>20}{_money(row['ecl']):>18}"
            )
    sc = s["scenarios"]
    lines += ["", "scenarios (ECL if one scenario alone had full weight)"]
    for row in sc["scenarios"]:
        lines.append(
            f"  {row['name']:<14}weight {row['weight']:>5.0%}  same stages {_money(row['ecl_same_stages']):>16}"
            f"  restaged {_money(row['ecl_restaged']):>16}  ({row['loans_changing_stage']} loans change stage)"
        )
    if sc["weighted_to_base"] is not None:
        lines.append(f"  weighted / base: {sc['weighted_to_base']:.3f}")
    c = s["checks"]
    lines += [
        "",
        "checks",
        f"  stage 1 loans whose PD has tripled since origination: {c['stage_1_with_threefold_pd_increase']['loans']}",
        f"  stage 2 only because of days past due: {c['stage_2_only_because_of_days_past_due']['loans']} "
        f"({c['stage_2_only_because_of_days_past_due']['share_of_stage_2_exposure']:.1%} of stage 2 exposure)",
        f"  stage 1 kept by the low credit risk exemption: {c['stage_1_kept_by_low_credit_risk_exemption']['loans']}",
        f"  loans without origination PD: {c['loans_without_origination_pd']}",
    ]
    if s["warnings"]:
        lines += ["", "warnings"] + [f"  - {w}" for w in s["warnings"]]
    return "\n".join(lines)
