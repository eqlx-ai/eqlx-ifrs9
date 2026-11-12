"""Independent recalculation: what a reviewer does with a lender's loan file and its reported figures.

- :func:`reconcile` puts the lender's own stage and allowance (columns ``reported_stage`` and
  ``reported_ecl``) next to the recalculated ones: every loan staged again, the loans whose stage
  differs with the rule that fired, and the gap in the allowance split into the part that comes
  from staging and the part that comes from measurement.
- :func:`challenger` replaces the rating transition matrix of the settings with the one observed
  between two loan files and computes the allowance again.
- :func:`response` shows how the allowance reacts to the economy: one run per shift of the base
  scenario.

The 5% and 10% thresholds are those of the ECB asset quality review (Phase 2 Manual, 2018, section
7.1): a challenger above the lender's allowance by less than 5% is immaterial, between 5% and 10% it
needs an explanation, above 10% it is investigated. They are parameters.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import engine, period
from ._checks import Array
from .config import Config, Scenario
from .data import Loans
from .engine import PortfolioResult
from .errors import InputError
from .staging import Reason, names

STAGES = (1, 2, 3)
IMMATERIAL = 0.05
INVESTIGATE = 0.10
SHIFTS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
SOURCE = "ECB Asset Quality Review Phase 2 Manual (2018), section 7.1"


def verdict(
    recalculated: float, reported: float, immaterial: float = IMMATERIAL, investigate: float = INVESTIGATE
) -> str:
    """Where a recalculated allowance stands against the reported one."""
    if recalculated <= reported:
        return "reported is not lower"
    if reported <= 0.0:
        return "investigate"
    excess = recalculated / reported - 1.0
    if excess < immaterial - 1e-12:  # exactly 5% is no longer immaterial, exactly 10% is not yet investigated
        return "immaterial"
    return "explain" if excess <= investigate + 1e-12 else "investigate"


def _thresholds(immaterial: float, investigate: float) -> None:
    if not 0.0 <= immaterial <= investigate:
        raise InputError("thresholds: 0 <= immaterial <= investigate is required")


def _compare(recalculated: float, reported: float, immaterial: float, investigate: float) -> dict[str, Any]:
    return {
        "reported": reported,
        "recalculated": recalculated,
        "difference": recalculated - reported,
        "relative": (recalculated - reported) / reported if reported > 0 else None,
        "verdict": verdict(recalculated, reported, immaterial, investigate),
    }


def reconcile(
    r: PortfolioResult, *, immaterial: float = IMMATERIAL, investigate: float = INVESTIGATE, limit: int = 20
) -> dict[str, Any]:
    """The lender's reported stage and allowance against the recalculation, loan by loan.

    ``difference`` is always recalculated minus reported. The allowance gap of the loans that carry
    both reported figures is split exactly in two: ``from_staging`` is what the recalculation would
    change by moving loans to the recalculated stage (measured with the recalculated parameters),
    ``from_measurement`` is what is left at the lender's own stages.
    """
    _thresholds(immaterial, investigate)
    loans = r.loans
    rep_stage, rep_ecl = loans.num["reported_stage"], loans.num["reported_ecl"]
    has_stage, has_ecl = ~np.isnan(rep_stage), ~np.isnan(rep_ecl)
    if not has_stage.any() and not has_ecl.any():
        raise InputError("the loan file has neither reported_stage nor reported_ecl to compare with")
    out: dict[str, Any] = {"thresholds": {"immaterial": immaterial, "investigate": investigate, "source": SOURCE}}
    ids = loans.text["loan_id"]
    lim = max(1, min(int(limit), 500))

    if has_stage.any():
        rs = np.where(has_stage, rep_stage, 0.0).astype(np.int8)
        lower = has_stage & (rs < r.stage)
        higher = has_stage & (rs > r.stage)
        by_reason: list[dict[str, Any]] = []
        for code in sorted({int(x) for x in r.primary[lower].tolist()}):
            sel = lower & (r.primary == code)
            by_reason.append(
                {
                    "primary_reason": Reason(code).name.lower() if code else "none",  # type: ignore[union-attr]
                    "loans": int(sel.sum()),
                    "exposure": float(r.exposure[sel].sum()),
                }
            )
        by_reason.sort(key=lambda x: (-x["exposure"], x["primary_reason"]))
        differ = np.nonzero(lower | higher)[0].tolist()
        differ.sort(key=lambda i: (-float(r.exposure[i]), str(ids[i])))
        out["stages"] = {
            "loans_compared": int(has_stage.sum()),
            "loans_without_reported_stage": int((~has_stage).sum()),
            "same_stage": int((has_stage & (rs == r.stage)).sum()),
            "rows_reported_columns_recalculated": {
                "loans": [[int(np.sum((rs == a) & (r.stage == b))) for b in STAGES] for a in STAGES],
                "exposure": [[float(r.exposure[(rs == a) & (r.stage == b)].sum()) for b in STAGES] for a in STAGES],
            },
            "reported_in_a_lower_stage": {
                "loans": int(lower.sum()),
                "exposure": float(r.exposure[lower].sum()),
                "by_primary_reason_of_the_recalculated_stage": by_reason,
            },
            "reported_in_a_higher_stage": {"loans": int(higher.sum()), "exposure": float(r.exposure[higher].sum())},
            "loans_that_differ": [
                {
                    "loan_id": str(ids[i]),
                    "reported_stage": int(rs[i]),
                    "recalculated_stage": int(r.stage[i]),
                    "reasons": names(int(r.reasons[i])),
                    "exposure": float(r.exposure[i]),
                }
                for i in differ[:lim]
            ],
        }

    if has_ecl.any():
        seg = loans.text["segment"]

        def gap(mask: NDArray[np.bool_]) -> dict[str, Any]:
            return _compare(float(r.ecl[mask].sum()), float(rep_ecl[mask].sum()), immaterial, investigate)

        allowance: dict[str, Any] = {
            "loans_compared": int(has_ecl.sum()),
            "loans_without_reported_ecl": int((~has_ecl).sum()),
            "total": gap(has_ecl),
            "by_segment": [
                {"segment": str(nm), **gap(has_ecl & (seg == nm))}
                for nm in sorted(set(seg.tolist()))
                if np.any(has_ecl & (seg == nm))
            ],
            "by_recalculated_stage": [{"stage": s, **gap(has_ecl & (r.stage == s))} for s in STAGES],
        }
        both = has_ecl & has_stage
        if both.any():
            at_reported = r.ecl_for_stages(np.where(has_stage, rep_stage, r.stage).astype(np.int8))
            allowance["gap_of_loans_with_both_figures"] = {
                "loans": int(both.sum()),
                "difference": float((r.ecl - rep_ecl)[both].sum()),
                "from_staging": float((r.ecl - at_reported)[both].sum()),
                "from_measurement": float((at_reported - rep_ecl)[both].sum()),
            }
        diff = np.where(has_ecl, r.ecl - rep_ecl, 0.0)
        order = np.nonzero(diff != 0.0)[0].tolist()
        order.sort(key=lambda i: (-abs(float(diff[i])), str(ids[i])))
        allowance["largest_differences"] = [
            {
                "loan_id": str(ids[i]),
                "reported_ecl": float(rep_ecl[i]),
                "recalculated_ecl": float(r.ecl[i]),
                "difference": float(diff[i]),
                "reported_stage": int(rep_stage[i]) if has_stage[i] else None,
                "recalculated_stage": int(r.stage[i]),
            }
            for i in order[:lim]
        ]
        out["allowance"] = allowance
    out["warnings"] = list(r.warnings)
    return out


def response(
    loans: Loans, config: Config, *, shifts: Sequence[float] = SHIFTS, scenario: str | None = None
) -> dict[str, Any]:
    """Allowance as a function of the economy: the base scenario moved up and down by ``shifts``.

    Each shift is added to every forecast year of the scenario path (z, in standard deviations of
    the economy; negative is worse) and the portfolio is computed with that single scenario. The
    result is given with the stages of the probability-weighted run held fixed and with the loans
    staged again, so the two channels of a downturn can be told apart. An allowance that hardly moves
    is not a result of this tool; it is a property of the PDs, correlations and stages it was given.
    """
    name = config.base_scenario if scenario is None else scenario
    found = [s for s in config.scenarios if s.name == name]
    if not found:
        raise InputError(f"no scenario named {name!r}")
    base = found[0]
    if base.z is None:
        raise InputError(f"scenario {name!r} has no z path: there is nothing to shift")
    values = sorted({float(x) for x in shifts} | {0.0})
    if not all(np.isfinite(values)) or len(values) > 41:
        raise InputError("shifts must be finite numbers, at most 40 of them")
    main = engine.run(loans, config)
    points: list[dict[str, Any]] = []
    for shift in values:
        path = tuple(float(z) + shift for z in base.z)
        one = dataclasses.replace(config, scenarios=(Scenario("shifted", 1.0, path),), base_scenario="shifted")
        run = engine.run(loans, one)
        exposure = float(run.exposure.sum())
        points.append(
            {
                "shift": shift,
                "z": list(path),
                "ecl_stages_held": float(run.ecl_for_stages(main.stage).sum()),
                "ecl_restaged": float(run.ecl.sum()),
                "stage_2_share_of_exposure": float(run.exposure[run.stage == 2].sum()) / exposure if exposure else 0.0,
                "loans_changing_stage": int(np.sum(run.stage != main.stage)),
            }
        )
    zero = next(pt for pt in points if pt["shift"] == 0.0)
    for pt in points:
        for key in ("ecl_stages_held", "ecl_restaged"):
            pt[f"{key}_relative_to_no_shift"] = pt[key] / zero[key] - 1.0 if zero[key] > 0 else None
    out: dict[str, Any] = {
        "scenario": name,
        "weighted_ecl": float(main.ecl.sum()),
        "points": points,
    }
    by_shift = {pt["shift"]: pt for pt in points}
    if -1.0 in by_shift and 1.0 in by_shift and zero["ecl_restaged"] > 0:
        down = by_shift[-1.0]["ecl_restaged"] - zero["ecl_restaged"]
        up = zero["ecl_restaged"] - by_shift[1.0]["ecl_restaged"]
        out["asymmetry"] = {
            "increase_for_one_step_down": down,
            "decrease_for_one_step_up": up,
            "ratio": down / up if up > 0 else None,
        }
    return out


def challenger(
    opening: PortfolioResult,
    closing: PortfolioResult,
    *,
    write_offs: Iterable[str] = (),
    months: int = 12,
    immaterial: float = IMMATERIAL,
    investigate: float = INVESTIGATE,
) -> dict[str, Any]:
    """Allowance with rating transition matrices observed in the lender's own two loan files.

    For every segment of the closing settings that has grades and a matrix, the moves between grades
    from the opening to the closing file are counted (:func:`eqlx_ifrs9.period.observed_migration`)
    and the closing portfolio is computed again with the observed matrices. The result is compared
    with ``reported_ecl`` where the closing file has it, otherwise with the run on the settings'
    matrices. One period of observation is a small sample: read the observation counts first.
    """
    _thresholds(immaterial, investigate)
    write_offs = list(write_offs)
    config = closing.config
    segments = dict(config.segments)
    present = set(closing.loans.text["segment"].tolist())
    observed: list[dict[str, Any]] = []
    for nm, seg in config.segments.items():
        if nm == "*" or not seg.grades or seg.matrix is None or nm not in present:
            continue
        obs = period.observed_migration(opening, closing, nm, write_offs=write_offs, months=months)
        if obs["observations"] == 0:
            continue
        k = len(seg.grades)
        obs["annual_pd_by_grade_in_settings"] = {g: float(seg.matrix[i, k]) for i, g in enumerate(seg.grades)}
        segments[nm] = dataclasses.replace(seg, matrix=np.array(obs["annual_matrix"]), matrix_tolerance=1e-8)
        observed.append(obs)
    if not observed:
        raise InputError(
            "no segment with grades and a transition matrix has loans in both files: there is nothing to observe"
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alt = engine.run(closing.loans, dataclasses.replace(config, segments=segments))
    numerical = sorted({str(w.message) for w in caught})
    rep = closing.loans.num["reported_ecl"]
    has = ~np.isnan(rep)
    seg_of = closing.loans.text["segment"]
    used = [o["segment"] for o in observed]

    def row(mask: NDArray[np.bool_]) -> dict[str, Any]:
        bank: Array = np.where(has, rep, closing.ecl)
        out: dict[str, Any] = {
            "with_settings_matrix": float(closing.ecl[mask].sum()),
            "with_observed_matrix": float(alt.ecl[mask].sum()),
            "loans_changing_stage": int(np.sum((alt.stage != closing.stage) & mask)),
        }
        compared = _compare(float(alt.ecl[mask].sum()), float(bank[mask].sum()), immaterial, investigate)
        out["against"] = "reported_ecl" if has[mask].all() and mask.any() else "run with the settings' matrix"
        if has[mask].any() and not has[mask].all():
            out["against"] = "reported_ecl where given, otherwise the run with the settings' matrix"
        return {**out, **compared}

    return {
        "months": months,
        "thresholds": {"immaterial": immaterial, "investigate": investigate, "source": SOURCE},
        "segments": [{"segment": nm, **row(seg_of == nm)} for nm in used],
        "total_of_these_segments": row(np.isin(seg_of, np.array(used, dtype=np.str_))),
        "observed": observed,
        "warnings": [f"{o['segment']}: {w}" for o in observed for w in o["warnings"]]
        + [f"turning an observed matrix into monthly steps: {w} Too few observations?" for w in numerical]
        + list(alt.warnings),
    }
