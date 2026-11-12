"""From one reporting date to the next: why the allowance changed, and how the staging behaved.

Two portfolio runs are compared, the opening one (previous loan file and settings) and the closing
one. Loans are matched by ``loan_id``.

- :func:`movement` reconciles the opening allowance to the closing allowance by stage, in the layout
  of the IFRS 7 35H table, and splits the change of continuing loans into drivers by replacing one
  group of inputs at a time. The rows add up to the closing balance exactly.
- :func:`staging_quality` gives the stage transition table and the indicators supervisors look at:
  where stage 3 came from, whether stage 2 was riskier than stage 1, how much left stage 2 again.
- :func:`backtest` compares the defaults expected at the opening date with those that happened, and
  separates the error of the economic forecast from the error of the model.
- :func:`observed_migration` counts the moves between rating grades for a challenger matrix.

The attribution depends on the order in which the drivers are replaced; IFRS 7 prescribes none. The
order is a parameter and is printed with the result.
"""

from __future__ import annotations

import csv
import dataclasses
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import engine, transitions
from ._checks import Array
from .config import Scenario
from .data import Loans
from .engine import PortfolioResult
from .errors import InputError
from .staging import Reason

STAGES = (1, 2, 3)
MONTHS = 12
DRIVERS = ("settings", "exposure", "pd", "lgd", "other")
# Loan-file columns replaced by each driver. "settings" replaces the settings file; "other" takes
# every column not named here (days past due, flags, origination risk), which can only move the stage.
GROUPS: dict[str, tuple[str, ...]] = {
    "exposure": (
        "balance",
        "undrawn",
        "ccf",
        "ccf_12m",
        "remaining_months",
        "months_on_book",
        "rate",
        "eir",
        "amortization",
        "product",
    ),
    "pd": ("pd_12m", "grade", "segment"),
    "lgd": ("lgd", "lgd_default", "individual_ecl"),
}
LABELS = {
    "opening_balance": "opening balance",
    "transfer_to_stage_1": "transfers to stage 1",
    "transfer_to_stage_2": "transfers to stage 2",
    "transfer_to_stage_3": "transfers to stage 3",
    "remeasurement_on_stage_transfer": "remeasurement on transfer between stages",
    "originated_or_purchased": "new loans originated or purchased",
    "derecognised": "loans derecognised (repaid or sold)",
    "written_off": "write-offs",
    "change_settings": "changes in models, scenarios and settings",
    "change_exposure": "repayments, drawdowns and passage of time",
    "change_pd": "changes in probability of default",
    "change_lgd": "changes in loss given default",
    "change_other": "other changes",
    "net_repayments_and_drawdowns": "repayments and drawdowns of continuing loans",
    "closing_balance": "closing balance",
}
MIN_OBSERVATIONS = 30


@dataclass(frozen=True, eq=False)
class Pair:
    """Two runs with their loans matched by ``loan_id``. All index arrays are sorted by ``loan_id``."""

    opening: PortfolioResult
    closing: PortfolioResult
    io: NDArray[np.intp]
    """Continuing loans: positions in the opening run."""
    ic: NDArray[np.intp]
    """Continuing loans: positions in the closing run (same order as ``io``)."""
    new: NDArray[np.intp]
    """Positions in the closing run of loans that were not in the opening run."""
    gone: NDArray[np.intp]
    """Positions in the opening run of loans that left without being written off."""
    written_off: NDArray[np.intp]


def pair(opening: PortfolioResult, closing: PortfolioResult, write_offs: Iterable[str] = ()) -> Pair:
    ids_o, ids_c = opening.loans.text["loan_id"], closing.loans.text["loan_id"]
    _, io, ic = np.intersect1d(ids_o, ids_c, assume_unique=True, return_indices=True)
    in_closing = np.isin(ids_o, ids_c, assume_unique=True)
    listed = sorted({str(x) for x in write_offs})
    known = set(ids_o.tolist())
    strangers = [x for x in listed if x not in known]
    if strangers:
        raise InputError(f"write-offs: {strangers[0]!r} is not a loan of the opening file")
    wo = np.isin(ids_o, np.array(listed, dtype=np.str_)) if listed else np.zeros(ids_o.size, dtype=bool)
    still = np.nonzero(wo & in_closing)[0]
    if still.size:
        raise InputError(f"write-offs: {ids_o[still[0]]!r} is still in the closing file")

    def by_id(ids: NDArray[np.str_], mask: NDArray[np.bool_]) -> NDArray[np.intp]:
        idx = np.nonzero(mask)[0]
        return np.asarray(idx[np.argsort(ids[idx], kind="stable")], dtype=np.intp)

    return Pair(
        opening=opening,
        closing=closing,
        io=np.asarray(io, dtype=np.intp),
        ic=np.asarray(ic, dtype=np.intp),
        new=by_id(ids_c, ~np.isin(ids_c, ids_o, assume_unique=True)),
        gone=by_id(ids_o, ~in_closing & ~wo),
        written_off=by_id(ids_o, wo),
    )


def _by_stage(values: Array, stage: NDArray[np.int8]) -> Array:
    return np.array([float(values[stage == s].sum()) for s in STAGES])


def _line(key: str, v: Array) -> dict[str, Any]:
    return {
        "row": key,
        "label": LABELS[key],
        "stage_1": float(v[0]) + 0.0,  # + 0.0 turns a negative zero into zero
        "stage_2": float(v[1]) + 0.0,
        "stage_3": float(v[2]) + 0.0,
        "total": float(v.sum()) + 0.0,
    }


def _substitute(base: Loans, source: Loans, columns: Iterable[str]) -> Loans:
    text, num, flag = dict(base.text), dict(base.num), dict(base.flag)
    for c in columns:
        if c in source.text:
            text[c] = source.text[c]
        elif c in source.num:
            num[c] = source.num[c]
        elif c in source.flag:
            flag[c] = source.flag[c]
    cols = tuple(dict.fromkeys((*base.columns, *source.columns)))
    return Loans(base.n, text, num, flag, cols)


def _transfers(values: Array, before: NDArray[np.int8], after: NDArray[np.int8]) -> list[Array]:
    """One vector per destination stage: the amount arrives there and leaves where it came from."""
    out = []
    for dest in STAGES:
        v = np.zeros(3)
        moved = (before != dest) & (after == dest)
        v[dest - 1] = float(values[moved].sum())
        for s in STAGES:
            v[s - 1] -= float(values[moved & (before == s)].sum())
        out.append(v)
    return out


def movement(
    opening: PortfolioResult,
    closing: PortfolioResult,
    *,
    write_offs: Iterable[str] = (),
    order: Sequence[str] = DRIVERS,
    limit: int = 20,
) -> dict[str, Any]:
    """Opening to closing allowance by stage, with the change of continuing loans split into drivers.

    For a loan in both files: its opening allowance moves to the column of its closing stage
    (transfer, at the opening amount); it is then measured on the basis of the closing stage with the
    opening inputs (remeasurement on transfer); then the inputs are replaced group by group in
    ``order`` and the allowance is recomputed after each step. The last step lands on the closing
    allowance, so nothing is left unexplained. ``write_offs`` are loan ids of the opening file that
    were written off; other loans missing from the closing file count as derecognised.
    """
    if sorted(order) != sorted(DRIVERS):
        raise InputError(f"order must contain each of {', '.join(DRIVERS)} once")
    order = tuple(order)
    p = pair(opening, closing, write_offs)
    so, sc = opening.stage[p.io], closing.stage[p.ic]
    open_ecl = opening.ecl[p.io]

    basis = opening.stage.copy()
    basis[p.io] = sc
    previous = opening.ecl_for_stages(basis)[p.io]
    remeasured = previous - open_ecl
    delta: dict[str, Array] = {}
    if p.io.size:
        hybrid = opening.loans.take(p.io)
        target = closing.loans.take(p.ic)
        named = {c for cols in GROUPS.values() for c in cols} | {"loan_id"}
        everything = (*target.text, *target.num, *target.flag)
        config = opening.config
        for k, driver in enumerate(order):
            if driver == "settings":
                config = closing.config
            elif driver == "other":
                hybrid = _substitute(hybrid, target, [c for c in everything if c not in named])
            else:
                hybrid = _substitute(hybrid, target, GROUPS[driver])
            last = k == len(order) - 1
            current = closing.ecl[p.ic] if last else engine.run(hybrid, config).ecl_for_stages(sc)
            delta[driver] = current - previous
            previous = current
    else:
        delta = {d: np.zeros(0) for d in order}

    rows = [_line("opening_balance", _by_stage(opening.ecl, opening.stage))]
    for dest, v in zip(STAGES, _transfers(open_ecl, so, sc), strict=True):
        rows.append(_line(f"transfer_to_stage_{dest}", v))
    rows.append(_line("remeasurement_on_stage_transfer", _by_stage(remeasured, sc)))
    rows.append(_line("originated_or_purchased", _by_stage(closing.ecl[p.new], closing.stage[p.new])))
    rows.append(_line("derecognised", -_by_stage(opening.ecl[p.gone], opening.stage[p.gone])))
    rows.append(_line("written_off", -_by_stage(opening.ecl[p.written_off], opening.stage[p.written_off])))
    for driver in order:
        rows.append(_line(f"change_{driver}", _by_stage(delta[driver], sc)))
    closing_row = _line("closing_balance", _by_stage(closing.ecl, closing.stage))
    keys = ("stage_1", "stage_2", "stage_3", "total")
    gap = max(abs(math.fsum(r[k] for r in rows) - closing_row[k]) for k in keys)
    rows.append(closing_row)

    bal_o, bal_c = opening.loans.num["balance"], closing.loans.num["balance"]
    gross = [_line("opening_balance", _by_stage(bal_o, opening.stage))]
    for dest, v in zip(STAGES, _transfers(bal_o[p.io], so, sc), strict=True):
        gross.append(_line(f"transfer_to_stage_{dest}", v))
    gross.append(_line("originated_or_purchased", _by_stage(bal_c[p.new], closing.stage[p.new])))
    gross.append(_line("derecognised", -_by_stage(bal_o[p.gone], opening.stage[p.gone])))
    gross.append(_line("written_off", -_by_stage(bal_o[p.written_off], opening.stage[p.written_off])))
    gross.append(_line("net_repayments_and_drawdowns", _by_stage(bal_c[p.ic] - bal_o[p.io], sc)))
    gross.append(_line("closing_balance", _by_stage(bal_c, closing.stage)))

    ids_o, ids_c = opening.loans.text["loan_id"], closing.loans.text["loan_id"]
    changes: list[dict[str, Any]] = []
    for j in range(p.io.size):
        item: dict[str, Any] = {
            "loan_id": str(ids_o[p.io[j]]),
            "what": "continuing",
            "stage_before": int(so[j]),
            "stage_after": int(sc[j]),
            "ecl_before": float(open_ecl[j]),
            "ecl_after": float(closing.ecl[p.ic[j]]),
            "drivers": {"stage_transfer": float(remeasured[j]), **{d: float(delta[d][j]) for d in order}},
        }
        changes.append(item)
    for what, idx, run, sign in (
        ("new", p.new, closing, 1.0),
        ("derecognised", p.gone, opening, -1.0),
        ("written off", p.written_off, opening, -1.0),
    ):
        ids = ids_c if run is closing else ids_o
        for i in idx.tolist():
            before = 0.0 if sign > 0 else float(run.ecl[i])
            after = float(run.ecl[i]) if sign > 0 else 0.0
            changes.append(
                {
                    "loan_id": str(ids[i]),
                    "what": what,
                    "stage_before": None if sign > 0 else int(run.stage[i]),
                    "stage_after": int(run.stage[i]) if sign > 0 else None,
                    "ecl_before": before,
                    "ecl_after": after,
                }
            )
    for c in changes:
        c["change"] = c["ecl_after"] - c["ecl_before"]
    changes.sort(key=lambda c: (-abs(c["change"]), c["loan_id"]))
    total_o, total_c = float(opening.ecl.sum()), float(closing.ecl.sum())
    return {
        "opening_ecl": total_o,
        "closing_ecl": total_c,
        "change": total_c - total_o,
        "change_relative": (total_c - total_o) / total_o if total_o > 0 else None,
        "loans": {
            "opening": opening.n,
            "closing": closing.n,
            "continuing": int(p.io.size),
            "changing_stage": int(np.sum(so != sc)),
            "new": int(p.new.size),
            "derecognised": int(p.gone.size),
            "written_off": int(p.written_off.size),
        },
        "order": list(order),
        "loss_allowance": rows,
        "gross_carrying_amount": gross,
        "largest_changes": changes[: max(1, min(int(limit), 500))],
        "largest_difference_between_rows_and_closing_balance": gap,
        "notes": [
            "transfers are shown at the opening amount and add up to zero across the stages",
            "the split into drivers depends on the order shown; the total does not",
            "write-offs and derecognitions release the opening allowance of the loan; a loss beyond that "
            "allowance is not in this table",
            "the unwinding of discount is part of 'repayments, drawdowns and passage of time'",
        ],
    }


def text_movement(m: dict[str, Any]) -> str:
    """The movement tables as plain text."""
    rel = "" if m["change_relative"] is None else f"  ({m['change_relative']:+.1%})"
    n = m["loans"]
    lines = [
        f"allowance {m['opening_ecl']:,.2f} -> {m['closing_ecl']:,.2f}{rel}",
        f"loans {n['opening']:,} -> {n['closing']:,}: {n['continuing']:,} continuing ({n['changing_stage']:,} changed "
        f"stage), {n['new']:,} new, {n['derecognised']:,} derecognised, {n['written_off']:,} written off",
    ]
    for title, key in (("loss allowance", "loss_allowance"), ("gross carrying amount", "gross_carrying_amount")):
        lines += ["", f"{title:<46}{'stage 1':>18}{'stage 2':>18}{'stage 3':>18}{'total':>18}"]
        for r in m[key]:
            cells = "".join(f"{r[k]:>18,.2f}" for k in ("stage_1", "stage_2", "stage_3", "total"))
            lines.append(f"{r['label']:<46}{cells}")
    lines += ["", "order of the drivers: " + ", ".join(m["order"])]
    return "\n".join(lines)


def read_write_offs(source: str | Path) -> list[str]:
    """Loan ids from a CSV file with a ``loan_id`` column (other columns are ignored)."""
    try:
        with open(source, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
    except UnicodeDecodeError as exc:
        raise InputError(f"{source}: the write-off file is not UTF-8 text") from exc
    if not rows:
        raise InputError(f"{source}: the write-off file is empty")
    header = [h.strip().lower() for h in rows[0]]
    if "loan_id" not in header:
        raise InputError(f"{source}: the write-off file needs a loan_id column")
    k = header.index("loan_id")
    ids = [r[k].strip() for r in rows[1:] if len(r) > k and r[k].strip()]
    if len(set(ids)) != len(ids):
        raise InputError(f"{source}: a loan_id appears more than once")
    return ids


# ---- staging quality ---------------------------------------------------------------------------


def _share(part: float, whole: float) -> float | None:
    return part / whole if whole > 0 else None


def staging_quality(
    opening: PortfolioResult, closing: PortfolioResult, *, write_offs: Iterable[str] = ()
) -> dict[str, Any]:
    """How the stages of the opening date turned out (EBA IFRS 9 monitoring report 2023; ECB 2024).

    A written-off loan counts as a default. A loan that left for another reason has an unknown
    outcome: it is shown in the table and left out of the rates.
    """
    p = pair(opening, closing, write_offs)
    so, sc = opening.stage[p.io], closing.stage[p.ic]
    exp_o = opening.exposure
    ce = exp_o[p.io]
    table = []
    for a in STAGES:
        was = so == a
        row: dict[str, Any] = {"opening_stage": a}
        for b in STAGES:
            row[f"to_stage_{b}"] = {"loans": int(np.sum(was & (sc == b))), "exposure": float(ce[was & (sc == b)].sum())}
        for name, idx in (("written_off", p.written_off), ("left", p.gone)):
            sel = idx[opening.stage[idx] == a]
            row[name] = {"loans": int(sel.size), "exposure": float(exp_o[sel].sum())}
        table.append(row)

    def defaults(a: int) -> tuple[int, int]:
        wo = int(np.sum(opening.stage[p.written_off] == a))
        return int(np.sum((so == a) & (sc == 3))) + wo, int(np.sum(so == a)) + wo

    d1, n1 = defaults(1)
    d2, n2 = defaults(2)
    inflow = d1 + d2

    def wo_exp(a: int) -> float:
        return float(exp_o[p.written_off[opening.stage[p.written_off] == a]].sum())

    inflow_exp_1 = float(ce[(so == 1) & (sc == 3)].sum()) + wo_exp(1)
    inflow_exp_2 = float(ce[(so == 2) & (sc == 3)].sum()) + wo_exp(2)
    was2 = so == 2
    entered = (so == 1) & (sc == 2)
    primary = closing.primary[p.ic]
    reasons: list[dict[str, Any]] = []
    for code in sorted({int(x) for x in primary[entered].tolist()}):
        sel = entered & (primary == code)
        reasons.append(
            {
                "primary_reason": Reason(code).name.lower() if code else "none",  # type: ignore[union-attr]
                "loans": int(sel.sum()),
                "exposure": float(ce[sel].sum()),
            }
        )
    reasons.sort(key=lambda r: (-r["exposure"], r["primary_reason"]))
    backstop = entered & (primary == int(Reason.DAYS_PAST_DUE_BACKSTOP))
    stage1_exposure = float(exp_o[opening.stage == 1].sum())
    out: dict[str, Any] = {
        "loans": {"continuing": int(p.io.size), "written_off": int(p.written_off.size), "left": int(p.gone.size)},
        "transitions": table,
        "stage_3_inflow": {
            "loans": inflow,
            "directly_from_stage_1": d1,
            "share_directly_from_stage_1": _share(d1, inflow),
            "exposure": inflow_exp_1 + inflow_exp_2,
            "exposure_share_directly_from_stage_1": _share(inflow_exp_1, inflow_exp_1 + inflow_exp_2),
        },
        "default_rate_by_opening_stage": {
            "stage_1": {"loans": n1, "defaults": d1, "rate": _share(d1, n1)},
            "stage_2": {"loans": n2, "defaults": d2, "rate": _share(d2, n2)},
        },
        "stage_2_inflow": {
            "loans": int(entered.sum()),
            "exposure": float(ce[entered].sum()),
            "share_of_opening_stage_1_exposure": _share(float(ce[entered].sum()), stage1_exposure),
            "by_primary_reason": reasons,
            "share_only_because_of_days_past_due": _share(float(ce[backstop].sum()), float(ce[entered].sum())),
        },
        "what_became_of_stage_2": {
            "loans": int(was2.sum()),
            "back_to_stage_1": _share(float(np.sum(was2 & (sc == 1))), float(was2.sum())),
            "still_stage_2": _share(float(np.sum(was2 & (sc == 2))), float(was2.sum())),
            "to_stage_3": _share(float(np.sum(was2 & (sc == 3))), float(was2.sum())),
        },
        "warnings": [],
    }
    r1, r2 = _share(d1, n1), _share(d2, n2)
    if r1 is not None and r2 is not None and r2 <= r1:
        out["warnings"].append(
            f"stage 2 defaulted at {r2:.2%}, stage 1 at {r1:.2%}: stage 2 did not single out the riskier loans"
        )
    if n2 == 0 and inflow > 0:
        out["warnings"].append("there was no stage 2 at the opening date; every new default came from stage 1")
    return out


# ---- backtest ----------------------------------------------------------------------------------


def _period_pd(pd_12: Array, remaining: Array, months: int) -> Array:
    """PD over ``months`` from the PD over the first 12 months (or the remaining life if shorter)."""
    window = np.minimum(remaining, MONTHS)
    horizon = np.minimum(remaining, months)
    return np.asarray(1.0 - (1.0 - pd_12) ** (horizon / window), dtype=np.float64)


def backtest(
    opening: PortfolioResult,
    closing: PortfolioResult,
    *,
    write_offs: Iterable[str] = (),
    months: int = 12,
    realised_z: float | None = None,
) -> dict[str, Any]:
    """Defaults expected at the opening date against defaults that happened by the closing date.

    Expected: the sum of the PDs over the period of the loans that were performing at the opening
    date, under the probability-weighted scenarios and under the base scenario. With ``realised_z``
    (the state of the economy that actually occurred, z > 0 good) the PDs are computed once more for
    that economy, and the difference splits in two: ``forecast_error`` is what the scenarios got
    wrong about the economy, ``model_error`` is what remains when the economy is known.

    Happened: stage 3 at the closing date, or written off. A loan that left for another reason counts
    as not defaulted.
    """
    if isinstance(months, bool) or not isinstance(months, int) or not 1 <= months <= MONTHS:
        raise InputError("months must be a whole number between 1 and 12")
    p = pair(opening, closing, write_offs)
    performing = opening.stage < 3
    remaining = opening.loans.num["remaining_months"]
    config = opening.config
    pds = {
        "weighted": _period_pd(config.weights @ opening.scenario_pd_12m, remaining, months),
        "base_scenario": _period_pd(opening.scenario_pd_12m[config.base_index], remaining, months),
    }
    if realised_z is not None:
        z = float(realised_z)
        if not math.isfinite(z):
            raise InputError("realised_z must be finite")
        if all(s.z is None for s in config.scenarios):
            raise InputError("the opening settings have no scenario path, so a realised economy cannot be applied")
        known = dataclasses.replace(
            config, scenarios=(Scenario("realised", 1.0, (z,)),), base_scenario="realised", scenario_basis="weighted"
        )
        pds["realised_economy"] = _period_pd(engine.run(opening.loans, known).scenario_pd_12m[0], remaining, months)
    defaulted = np.zeros(opening.n, dtype=bool)
    defaulted[p.io] = closing.stage[p.ic] == 3
    defaulted[p.written_off] = True

    def block(mask: NDArray[np.bool_]) -> dict[str, Any]:
        n = int(mask.sum())
        actual = int(defaulted[mask].sum())
        out: dict[str, Any] = {"loans": n, "defaults": actual, "default_rate": _share(actual, n)}
        for name, pd in pds.items():
            out[f"expected_{name}"] = float(pd[mask].sum())
        reference = pds.get("realised_economy", pds["weighted"])[mask]
        out["standard_deviation_if_independent"] = float(np.sqrt(np.sum(reference * (1.0 - reference))))
        if "realised_economy" in pds:
            out["forecast_error"] = out["expected_weighted"] - out["expected_realised_economy"]
            out["model_error"] = out["expected_realised_economy"] - actual
        else:
            out["difference"] = out["expected_weighted"] - actual
        return out

    seg = opening.loans.text["segment"]
    return {
        "months": months,
        "realised_z": realised_z,
        "total": block(performing),
        "by_opening_stage": [{"stage": s, **block(opening.stage == s)} for s in (1, 2)],
        "by_segment": [
            {"segment": str(name), **block(performing & (seg == name))} for name in sorted(set(seg.tolist()))
        ],
        "notes": [
            "expected minus actual: a positive figure means more defaults were expected than happened",
            "defaults of one period move together with the economy, so the spread around the expected number is "
            "wider than the independent standard deviation shown",
            "loans that left without a write-off count as not defaulted",
        ],
    }


# ---- observed migration ------------------------------------------------------------------------


def observed_migration(
    opening: PortfolioResult,
    closing: PortfolioResult,
    segment: str,
    *,
    write_offs: Iterable[str] = (),
    months: int = 12,
) -> dict[str, Any]:
    """Moves between the rating grades of one segment, counted from the two loan files.

    Rows are the grade at the opening date of loans that were not in stage 3; columns are the grade
    at the closing date, with default (stage 3 or written off) last. Loans that left, or that have no
    grade of this segment at the closing date, are left out and counted. A grade with no
    observation keeps the row of the settings' matrix. With ``months`` other than 12 the matrix is
    brought to one year through a fitted generator.
    """
    if isinstance(months, bool) or not isinstance(months, int) or not 1 <= months <= 120:
        raise InputError("months must be a whole number between 1 and 120")
    seg = closing.config.segment(segment)
    if not seg.grades:
        raise InputError(f"segment {segment!r} defines no grades")
    p = pair(opening, closing, write_offs)
    k = len(seg.grades)
    lookup = {g: i for i, g in enumerate(seg.grades)}
    counts = np.zeros((k, k + 1))
    left = unclassified = 0
    lo, lc = opening.loans, closing.loans
    position = dict(zip(p.io.tolist(), p.ic.tolist(), strict=True))
    written = set(p.written_off.tolist())
    for i in np.nonzero((lo.text["segment"] == segment) & (opening.stage < 3))[0].tolist():
        start = lookup.get(str(lo.text["grade"][i]))
        if start is None:
            continue
        if i in written:
            counts[start, k] += 1
        elif i not in position:
            left += 1
        else:
            j = position[i]
            end = lookup.get(str(lc.text["grade"][j]))
            if closing.stage[j] == 3:
                counts[start, k] += 1
            elif end is None or str(lc.text["segment"][j]) != segment:
                unclassified += 1
            else:
                counts[start, end] += 1
    n = counts.sum(axis=1)
    warnings: list[str] = []
    matrix = np.zeros((k + 1, k + 1))
    matrix[k, k] = 1.0
    for g in range(k):
        if n[g] > 0:
            matrix[g] = counts[g] / n[g]
            if n[g] < MIN_OBSERVATIONS:
                warnings.append(f"grade {seg.grades[g]}: only {int(n[g])} observations")
        elif seg.matrix is not None and months == MONTHS:
            matrix[g] = seg.matrix[g]
            warnings.append(f"grade {seg.grades[g]}: no observation, the row of the settings' matrix is kept")
        else:
            matrix[g, g] = 1.0
            warnings.append(f"grade {seg.grades[g]}: no observation, the grade is assumed not to move")
    annual = matrix
    if months != MONTHS:
        fitted = transitions.generator(matrix, seg.generator, tol=1e-9)
        annual = fitted.matrix(MONTHS / months)
        warnings.extend(f"annualising: {w}" for w in fitted.warnings)
    return {
        "segment": segment,
        "grades": list(seg.grades),
        "months": months,
        "observations": int(n.sum()),
        "left_out": {"left_the_portfolio": left, "no_grade_at_closing": unclassified},
        "counts": [[int(x) for x in row] for row in counts],
        "observations_by_grade": [int(x) for x in n],
        "matrix": [[float(x) for x in row] for row in matrix],
        "annual_matrix": [[float(x) for x in row] for row in annual],
        "annual_pd_by_grade": {g: float(annual[i, k]) for i, g in enumerate(seg.grades)},
        "warnings": warnings,
    }
