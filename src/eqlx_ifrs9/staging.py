"""Stage allocation: which instruments carry 12-month ECL, which lifetime ECL, and why.

The standard never says "stage"; the three states are (1) 12-month ECL, (2) lifetime ECL because
credit risk has increased significantly since initial recognition, (3) lifetime ECL on a
credit-impaired asset. It also gives no number for "significant". Every threshold here is therefore
an input, set in a :class:`StagingPolicy`, and the result records every rule that fired for every
instrument.

What is compared (IFRS 9 5.5.9, B5.5.9-B5.5.11): the risk of default at the reporting date with the
risk of default expected at initial recognition **for the same remaining window**. Collateral and
loss amounts play no part in the test. The PD pair passed in must already be on the same measure
(for example annualised lifetime PD over the remaining life); see ``PDCurve.forward``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import IntFlag
from itertools import pairwise
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._checks import Array
from .errors import InputError


class Reason(IntFlag):
    """Why an instrument is in its stage. Several can apply at once."""

    NONE = 0
    CREDIT_IMPAIRED = 1
    DEFAULT_DAYS_PAST_DUE = 2
    CURE_PERIOD = 4
    PD_RELATIVE = 8
    PD_ABSOLUTE = 16
    PD_BACKSTOP = 32
    RATING_NOTCHES = 64
    GRADE_THRESHOLD = 128
    FORBORNE = 256
    WATCHLIST = 512
    COLLECTIVE = 1024
    NO_ORIGINATION_PD = 2048
    DAYS_PAST_DUE_BACKSTOP = 4096
    PROBATION = 8192
    PROFILE = 16384
    PROFILE_DEFAULT = 65536
    LOW_CREDIT_RISK = 32768  # informational: the exemption kept the instrument in stage 1


# Order in which the single "primary" reason is picked. Days past due comes after everything except
# probation, so that "primary reason = days past due backstop" means no earlier signal existed.
DEFAULT_PRECEDENCE: tuple[Reason, ...] = (
    Reason.CREDIT_IMPAIRED,
    Reason.DEFAULT_DAYS_PAST_DUE,
    Reason.PROFILE_DEFAULT,
    Reason.CURE_PERIOD,
    Reason.PD_RELATIVE,
    Reason.PD_ABSOLUTE,
    Reason.PD_BACKSTOP,
    Reason.RATING_NOTCHES,
    Reason.GRADE_THRESHOLD,
    Reason.FORBORNE,
    Reason.WATCHLIST,
    Reason.COLLECTIVE,
    Reason.PROFILE,
    Reason.NO_ORIGINATION_PD,
    Reason.DAYS_PAST_DUE_BACKSTOP,
    Reason.PROBATION,
)

_STAGE3 = Reason.CREDIT_IMPAIRED | Reason.DEFAULT_DAYS_PAST_DUE | Reason.CURE_PERIOD | Reason.PROFILE_DEFAULT


@dataclass(frozen=True)
class StagingPolicy:
    """Thresholds for stage allocation. ``None`` switches a rule off.

    Quantitative test on the PD pair ``(pd_now, pd_origination)``:

    * ``relative_threshold``: significant if ``pd_now >= relative_threshold * pd_origination``.
      It is a **ratio**: 2.0 means the PD has doubled (an increase of 100%), 3.0 that it has tripled.
    * ``relative_bands``: ``((upper origination PD, ratio), ...)`` in ascending order, to use a
      different ratio by origination PD; ``relative_threshold`` applies above the last band.
    * ``absolute_threshold``: an increase ``pd_now - pd_origination`` of at least this much.
    * ``combine``: ``"or"`` (either test) or ``"and"`` (both tests must be met).
    * ``backstop_ratio``: regardless of the above, significant once the PD has reached this multiple
      of the origination PD. Supervisors in the EU expect three (ECB letter of 4 December 2020; EBA
      IFRS 9 monitoring report 2023), applied to annualised lifetime PD.

    Other rules:

    * ``notch_threshold``: downgrade of at least this many grades since origination.
    * ``max_grade_without_increase``: any grade worse than this is stage 2 (valid only when all
      instruments had similar risk at origination; IFRS 9 Illustrative Example 6).
    * ``low_credit_risk_pd`` / ``low_credit_risk_grade``: the low credit risk exemption of 5.5.10.
      It switches off the PD and grade tests for instruments at or below the cut-off; days past due
      and the forborne, watch-list and collective flags still apply.
    * ``days_past_due_backstop``: more than this many days past due is stage 2 (rebuttable
      presumption of 30 days, 5.5.11).
    * ``default_days_past_due``: more than this many days past due is stage 3 (rebuttable
      presumption of 90 days, B5.5.37).
    * ``probation_periods``: after the last trigger has gone, an instrument stays in stage 2 until it
      has shown this many periods without any trigger.
    * ``cure_periods``: the same for leaving stage 3.
    * ``missing_origination_pd``: what to do without an origination PD. ``"stage2"`` follows 7.2.20
      (lifetime ECL unless low credit risk), ``"skip"`` leaves the PD tests out, ``"error"`` refuses.
    """

    relative_threshold: float | None = None
    relative_bands: tuple[tuple[float, float], ...] = ()
    absolute_threshold: float | None = None
    combine: Literal["or", "and"] = "or"
    backstop_ratio: float | None = 3.0
    notch_threshold: int | None = None
    max_grade_without_increase: int | None = None
    low_credit_risk_pd: float | None = None
    low_credit_risk_grade: int | None = None
    days_past_due_backstop: int | None = 30
    default_days_past_due: int | None = 90
    forborne_is_stage2: bool = True
    watchlist_is_stage2: bool = True
    probation_periods: int = 0
    cure_periods: int = 0
    missing_origination_pd: Literal["stage2", "skip", "error"] = "stage2"
    precedence: tuple[Reason, ...] = field(default=DEFAULT_PRECEDENCE)

    def __post_init__(self) -> None:
        for name in ("relative_threshold", "backstop_ratio"):
            v = getattr(self, name)
            if v is not None and not float(v) >= 1.0:
                raise InputError(f"{name} is a ratio of PDs and must be at least 1, got {v!r}")
        if self.absolute_threshold is not None and not self.absolute_threshold > 0.0:
            raise InputError("absolute_threshold must be positive")
        if self.combine not in ("or", "and"):
            raise InputError("combine must be 'or' or 'and'")
        if self.missing_origination_pd not in ("stage2", "skip", "error"):
            raise InputError("missing_origination_pd must be 'stage2', 'skip' or 'error'")
        last = 0.0
        for upper, ratio in self.relative_bands:
            if not (upper > last and ratio >= 1.0):
                raise InputError("relative_bands must be ascending in origination PD with ratios of at least 1")
            last = upper
        if self.relative_bands and self.relative_threshold is None:
            raise InputError("relative_bands needs relative_threshold for origination PDs above the last band")
        for name in ("probation_periods", "cure_periods"):
            if int(getattr(self, name)) < 0:
                raise InputError(f"{name} must not be negative")
        for name in ("days_past_due_backstop", "default_days_past_due", "notch_threshold"):
            v = getattr(self, name)
            if v is not None and int(v) < 0:
                raise InputError(f"{name} must not be negative")
        if self.low_credit_risk_pd is not None and not 0.0 < self.low_credit_risk_pd < 1.0:
            raise InputError("low_credit_risk_pd must lie in (0, 1)")
        if set(self.precedence) != set(DEFAULT_PRECEDENCE) or len(self.precedence) != len(DEFAULT_PRECEDENCE):
            raise InputError("precedence must list every reason exactly once")

    def with_changes(self, **changes: Any) -> StagingPolicy:
        return replace(self, **changes)

    def warnings(self) -> tuple[str, ...]:
        """Choices that supervisors and auditors have publicly criticised. Not errors."""
        out: list[str] = []
        has_rel = self.relative_threshold is not None
        has_abs = self.absolute_threshold is not None
        if has_abs and not has_rel:
            out.append(
                "only an absolute PD threshold is set: the standard asks for a change relative to the "
                "risk at initial recognition (IFRS 9 B5.5.9)"
            )
        if has_abs and has_rel and self.combine == "and":
            out.append(
                "relative AND absolute thresholds must both be met: this keeps more exposures in "
                "stage 1 than either test alone and has been criticised by the EBA (2021, 2023 reports)"
            )
        if not has_rel and not has_abs and self.notch_threshold is None and self.max_grade_without_increase is None:
            out.append("no quantitative test of the change in default risk is set")
        if self.backstop_ratio is None:
            out.append("no PD backstop is set (EU supervisors expect a threefold increase at the latest)")
        elif self.backstop_ratio > 3.0:
            out.append("the PD backstop is looser than a threefold increase")
        if (
            self.relative_threshold is not None
            and self.backstop_ratio is not None
            and self.relative_threshold > self.backstop_ratio
        ):
            out.append("the relative threshold is looser than the backstop, so only the backstop will bind")
        ratios = [r for _, r in self.relative_bands]
        if self.relative_threshold is not None:
            ratios.append(self.relative_threshold)
        if any(b > a for a, b in pairwise(ratios)):
            out.append("a higher relative threshold applies to riskier origination PDs than to safer ones")
        if self.days_past_due_backstop is None or self.days_past_due_backstop > 30:
            out.append("the 30 days past due presumption is rebutted or switched off: this needs evidence (5.5.11)")
        if self.default_days_past_due is None or self.default_days_past_due > 90:
            out.append("the 90 days past due presumption is rebutted or switched off: this needs evidence (B5.5.37)")
        if self.low_credit_risk_pd is not None and self.low_credit_risk_pd > 0.003:
            out.append(
                "the low credit risk cut-off is above a PD of 0.3%: check that it still corresponds to "
                "an investment-grade level of risk (B5.5.23)"
            )
        if self.max_grade_without_increase is not None:
            out.append(
                "a single grade threshold is only valid if all instruments had similar credit risk at "
                "initial recognition (Illustrative Example 6)"
            )
        if self.notch_threshold is not None and not has_rel:
            out.append("a notch rule alone is not linear in PD: one notch means different things across the scale")
        return tuple(out)

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_threshold": self.relative_threshold,
            "relative_bands": [list(b) for b in self.relative_bands],
            "absolute_threshold": self.absolute_threshold,
            "combine": self.combine,
            "backstop_ratio": self.backstop_ratio,
            "notch_threshold": self.notch_threshold,
            "max_grade_without_increase": self.max_grade_without_increase,
            "low_credit_risk_pd": self.low_credit_risk_pd,
            "low_credit_risk_grade": self.low_credit_risk_grade,
            "days_past_due_backstop": self.days_past_due_backstop,
            "default_days_past_due": self.default_days_past_due,
            "forborne_is_stage2": self.forborne_is_stage2,
            "watchlist_is_stage2": self.watchlist_is_stage2,
            "probation_periods": self.probation_periods,
            "cure_periods": self.cure_periods,
            "missing_origination_pd": self.missing_origination_pd,
        }


@dataclass(frozen=True, eq=False)
class StagingResult:
    stage: NDArray[np.int8]
    reasons: NDArray[np.int64]
    primary: NDArray[np.int64]
    pd_ratio: Array
    """pd_now / pd_origination; NaN where there is no origination PD or it is zero."""
    pd_change: Array

    def __len__(self) -> int:
        return int(self.stage.size)

    def reason_names(self, i: int) -> list[str]:
        return names(int(self.reasons[i]))

    def primary_name(self, i: int) -> str:
        p = int(self.primary[i])
        return Reason(p).name.lower() if p else "none"  # type: ignore[union-attr]

    def has(self, reason: Reason) -> NDArray[np.bool_]:
        return np.asarray((self.reasons & int(reason)) != 0)


def names(bits: int) -> list[str]:
    return [r.name.lower() for r in sorted(Reason, key=lambda r: r.value) if r.value and bits & r.value and r.name]


def stage3_bits() -> int:
    return int(_STAGE3)


def _f(name: str, values: ArrayLike | None, n: int, default: float) -> Array:
    if values is None:
        return np.full(n, default)
    a = np.asarray(values, dtype=np.float64)
    if a.ndim == 0:
        return np.full(n, float(a))
    if a.shape != (n,):
        raise InputError(f"{name} has shape {a.shape}, expected ({n},)")
    return a


def _b(name: str, values: ArrayLike | None, n: int) -> NDArray[np.bool_]:
    if values is None:
        return np.zeros(n, dtype=bool)
    a = np.asarray(values)
    if a.ndim == 0:
        return np.full(n, bool(a))
    if a.shape != (n,):
        raise InputError(f"{name} has shape {a.shape}, expected ({n},)")
    return a.astype(bool)


def assess(
    policy: StagingPolicy,
    *,
    pd_now: ArrayLike,
    pd_origination: ArrayLike | None = None,
    backstop_pd_now: ArrayLike | None = None,
    backstop_pd_origination: ArrayLike | None = None,
    low_credit_risk_pd_now: ArrayLike | None = None,
    grade_now: ArrayLike | None = None,
    grade_origination: ArrayLike | None = None,
    days_past_due: ArrayLike | None = None,
    credit_impaired: ArrayLike | None = None,
    forborne: ArrayLike | None = None,
    watchlist: ArrayLike | None = None,
    collective: ArrayLike | None = None,
    previous_stage: ArrayLike | None = None,
    periods_without_trigger: ArrayLike | None = None,
) -> StagingResult:
    """Allocate stages for many instruments at once. All arrays have one entry per instrument.

    * ``pd_now``, ``pd_origination``: the PD pair on the policy's measure; NaN in ``pd_origination``
      means unknown. Grades are integers, larger = worse.
    * ``backstop_pd_*``: the pair for the backstop ratio if it differs from the main pair (the
      backstop is defined on annualised lifetime PD); defaults to the main pair.
    * ``low_credit_risk_pd_now``: the PD compared with ``low_credit_risk_pd``; defaults to ``pd_now``.
    * ``collective``: instruments moved to lifetime ECL by a collective assessment (B5.5.1-B5.5.6).
    * ``previous_stage`` and ``periods_without_trigger`` feed the probation and cure periods:
      the number of consecutive past periods in which no trigger of the instrument's previous stage
      applied.
    """
    now = np.asarray(pd_now, dtype=np.float64)
    if now.ndim == 0:
        now = now.reshape(1)
    if now.ndim != 1:
        raise InputError("pd_now must be one-dimensional")
    n = int(now.size)
    if np.any(~np.isfinite(now)) or np.any(now < 0.0) or np.any(now > 1.0):
        raise InputError("pd_now must lie in [0, 1]")
    orig = _f("pd_origination", pd_origination, n, np.nan)
    known = np.isfinite(orig)
    if np.any(orig[known] < 0.0) or np.any(orig[known] > 1.0):
        raise InputError("pd_origination must lie in [0, 1] or be NaN for unknown")
    bs_now = now if backstop_pd_now is None else _f("backstop_pd_now", backstop_pd_now, n, np.nan)
    bs_orig = orig if backstop_pd_origination is None else _f("backstop_pd_origination", backstop_pd_origination, n, 0)
    lcr_pd = now if low_credit_risk_pd_now is None else _f("low_credit_risk_pd_now", low_credit_risk_pd_now, n, 0)
    dpd = _f("days_past_due", days_past_due, n, 0.0)
    g_now = _f("grade_now", grade_now, n, np.nan)
    g_orig = _f("grade_origination", grade_origination, n, np.nan)
    prev = _f("previous_stage", previous_stage, n, 1.0)
    clean = _f("periods_without_trigger", periods_without_trigger, n, np.inf)

    reasons = np.zeros(n, dtype=np.int64)

    def flag(mask: NDArray[np.bool_], reason: Reason) -> None:
        reasons[mask] |= int(reason)

    # ---- quantitative tests on the PD pair
    usable = known & (orig > 0.0)
    ratio = np.full(n, np.nan)
    np.divide(now, orig, out=ratio, where=usable)
    change = np.where(known, now - orig, np.nan)
    eps = 1e-12

    rel = np.zeros(n, dtype=bool)
    if policy.relative_threshold is not None:
        thr = np.full(n, float(policy.relative_threshold))
        for upper, r in reversed(policy.relative_bands):
            thr[orig <= upper] = r
        rel = usable & (now >= thr * orig - eps)
        rel |= known & (orig == 0.0) & (now > 0.0)
    absolute = np.zeros(n, dtype=bool)
    if policy.absolute_threshold is not None:
        absolute = known & (now - orig >= policy.absolute_threshold - eps)
    if policy.relative_threshold is not None and policy.absolute_threshold is not None and policy.combine == "and":
        both = rel & absolute
        rel, absolute = both, both
    backstop = np.zeros(n, dtype=bool)
    if policy.backstop_ratio is not None:
        ok = np.isfinite(bs_orig) & (bs_orig > 0.0) & np.isfinite(bs_now)
        backstop = ok & (bs_now >= policy.backstop_ratio * bs_orig - eps)

    notch = np.zeros(n, dtype=bool)
    if policy.notch_threshold is not None:
        notch = np.isfinite(g_now) & np.isfinite(g_orig) & (g_now - g_orig >= policy.notch_threshold)
    grade_rule = np.zeros(n, dtype=bool)
    if policy.max_grade_without_increase is not None:
        grade_rule = np.isfinite(g_now) & (g_now > policy.max_grade_without_increase)

    missing = np.zeros(n, dtype=bool)
    needs_pd = policy.relative_threshold is not None or policy.absolute_threshold is not None
    if needs_pd and np.any(~known):
        if policy.missing_origination_pd == "error":
            raise InputError(f"{int(np.sum(~known))} instruments have no origination PD")
        if policy.missing_origination_pd == "stage2":
            missing = ~known

    # ---- low credit risk exemption: switches the PD and grade tests off, nothing else
    exempt = np.zeros(n, dtype=bool)
    if policy.low_credit_risk_pd is not None:
        exempt |= lcr_pd <= policy.low_credit_risk_pd
    if policy.low_credit_risk_grade is not None:
        exempt |= np.isfinite(g_now) & (g_now <= policy.low_credit_risk_grade)
    quantitative = rel | absolute | backstop | notch | grade_rule | missing
    flag(exempt & quantitative, Reason.LOW_CREDIT_RISK)
    flag(rel & ~exempt, Reason.PD_RELATIVE)
    flag(absolute & ~exempt, Reason.PD_ABSOLUTE)
    flag(backstop & ~exempt, Reason.PD_BACKSTOP)
    flag(notch & ~exempt, Reason.RATING_NOTCHES)
    flag(grade_rule & ~exempt, Reason.GRADE_THRESHOLD)
    flag(missing & ~exempt, Reason.NO_ORIGINATION_PD)

    # ---- qualitative flags and days past due
    if policy.forborne_is_stage2:
        flag(_b("forborne", forborne, n), Reason.FORBORNE)
    if policy.watchlist_is_stage2:
        flag(_b("watchlist", watchlist, n), Reason.WATCHLIST)
    flag(_b("collective", collective, n), Reason.COLLECTIVE)
    if policy.days_past_due_backstop is not None:
        flag(dpd > policy.days_past_due_backstop, Reason.DAYS_PAST_DUE_BACKSTOP)

    # ---- stage 3
    flag(_b("credit_impaired", credit_impaired, n), Reason.CREDIT_IMPAIRED)
    if policy.default_days_past_due is not None:
        flag(dpd > policy.default_days_past_due, Reason.DEFAULT_DAYS_PAST_DUE)

    # ---- probation and cure: stay where you were until enough clean periods have passed
    stage3_now = (reasons & int(_STAGE3)) != 0
    stage2_now = (reasons & ~int(_STAGE3) & ~int(Reason.LOW_CREDIT_RISK)) != 0
    if policy.cure_periods > 0:
        flag((prev == 3) & ~stage3_now & (clean < policy.cure_periods), Reason.CURE_PERIOD)
    if policy.probation_periods > 0:
        flag((prev == 2) & ~stage3_now & ~stage2_now & (clean < policy.probation_periods), Reason.PROBATION)

    return finish(reasons, ratio, change, policy.precedence)


def finish(
    reasons: NDArray[np.int64], ratio: Array, change: Array, precedence: Sequence[Reason] = DEFAULT_PRECEDENCE
) -> StagingResult:
    """Turn reason bits into stages and primary reasons."""
    stage3 = (reasons & int(_STAGE3)) != 0
    stage2 = (reasons & ~int(_STAGE3) & ~int(Reason.LOW_CREDIT_RISK)) != 0
    stage = np.where(stage3, 3, np.where(stage2, 2, 1)).astype(np.int8)
    primary = np.zeros(reasons.size, dtype=np.int64)
    for r in reversed(tuple(precedence)):
        primary[(reasons & int(r)) != 0] = int(r)
    primary[stage == 1] = 0
    return StagingResult(stage=stage, reasons=reasons, primary=primary, pd_ratio=ratio, pd_change=change)
