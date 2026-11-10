"""Turkey: loan classification under the BDDK provisioning regulation.

Source: "Kredilerin Sınıflandırılması ve Bunlar İçin Ayrılacak Karşılıklara İlişkin Usul ve Esaslar
Hakkında Yönetmelik" (Official Gazette 22 June 2016, no. 29750; in force since 1 January 2018; text
as amended up to 24 May 2022, no. 31845). Article numbers below refer to it.

The regulation sorts loans into five groups. Groups 1 and 2 are performing, groups 3 to 5
non-performing (Article 5). For banks applying TFRS 9 the groups line up with the stages: group 1
carries 12-month ECL, group 2 lifetime ECL for a significant increase in credit risk, groups 3-5
lifetime ECL for default (Articles 4 and 9).

What this module encodes is the mechanical part: day counts, restructuring, contagion across a
borrower's loans, the conditions for moving back up, and the fixed rates for banks that do not apply
TFRS 9. Most criteria of Article 4 are judgements about the borrower ("creditworthiness has
weakened"); those enter as flags the bank supplies. A classification from day counts alone is a
floor, not the regulation's answer.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..errors import InputError

GROUP_NAMES = {
    1: "standard (standart nitelikli)",
    2: "under close monitoring (yakın izlemedeki)",
    3: "limited recovery (tahsil imkânı sınırlı)",
    4: "doubtful (tahsili şüpheli)",
    5: "loss (zarar niteliğindeki)",
}

# Article 10(1) and 11(1): minimum rates for banks that do NOT apply TFRS 9.
GENERAL_PROVISION_RATE = {1: 0.015, 2: 0.03}
SPECIFIC_PROVISION_RATE = {3: 0.20, 4: 0.50, 5: 1.00}
# Article 15(1)(a): share of the net realisable value of collateral that is taken into account, by
# collateral group, for banks that do NOT apply TFRS 9.
COLLATERAL_CONSIDERATION_RATE = {1: 1.00, 2: 0.80, 3: 0.60, 4: 0.40, 5: 0.20}

# Board decisions 8948 and 8970 (March 2020), extended by 9312 and 9624, ended by 9795: between these
# dates the 30-day and 90-day limits were applied as 90 and 180 days.
_RELIEF_START = dt.date(2020, 3, 17)
_RELIEF_END = dt.date(2021, 9, 30)


@dataclass(frozen=True)
class DayLimits:
    """Days past due above which a loan belongs to group 2, 3, 4 and 5."""

    group2: int = 30
    group3: int = 90
    group4: int = 180
    group5: int = 365


def day_limits(as_of: dt.date | None = None, days_in_year: int = 365) -> DayLimits:
    """The day limits of Article 4 on a given date.

    The regulation says "one year" for the last limit without a day count; ``days_in_year`` makes
    the choice explicit. Between 17 March 2020 and 30 September 2021 the first two limits were 90
    and 180 days by Board decision. Loans already past due on 1 October 2021 kept the relief; that
    grandfathering is not modelled here.
    """
    if days_in_year not in (360, 365, 366):
        raise InputError("days_in_year must be 360, 365 or 366")
    if as_of is not None and _RELIEF_START <= as_of <= _RELIEF_END:
        return DayLimits(90, 180, 180, days_in_year)
    return DayLimits(30, 90, 180, days_in_year)


def group_from_days_past_due(days_past_due: ArrayLike, limits: DayLimits | None = None) -> NDArray[np.int8]:
    """Group implied by the delay alone (Article 4(1)(a)3, (b)5, (c)3, (ç)3, (d)3)."""
    lim = limits or DayLimits()
    d = np.asarray(days_past_due, dtype=np.float64)
    if np.any(~np.isfinite(d)) or np.any(d < 0):
        raise InputError("days past due must be non-negative numbers")
    g = np.ones(d.shape, dtype=np.int8)
    g[d > lim.group2] = 2
    g[d > lim.group3] = 3
    g[d > lim.group4] = 4
    g[d > lim.group5] = 5
    return g


def stage_from_group(group: ArrayLike) -> NDArray[np.int8]:
    """Group 1 -> stage 1, group 2 -> stage 2, groups 3 to 5 -> stage 3."""
    g = np.asarray(group, dtype=np.int64)
    if np.any(g < 1) or np.any(g > 5):
        raise InputError("groups run from 1 to 5")
    return np.minimum(g, 3).astype(np.int8)


@dataclass(frozen=True, eq=False)
class Classification:
    group: NDArray[np.int8]
    stage: NDArray[np.int8]
    reasons: tuple[tuple[str, ...], ...]

    def to_dict(self) -> list[dict[str, object]]:
        return [
            {"group": int(g), "stage": int(s), "reasons": list(r)}
            for g, s, r in zip(self.group, self.stage, self.reasons, strict=True)
        ]


def _flags(name: str, values: ArrayLike | None, n: int) -> NDArray[np.bool_]:
    if values is None:
        return np.zeros(n, dtype=bool)
    a = np.asarray(values).astype(bool)
    if a.shape != (n,):
        raise InputError(f"{name} needs one entry per loan")
    return a


def classify(
    days_past_due: ArrayLike,
    *,
    borrower: Sequence[object] | None = None,
    consumer_loan: ArrayLike | None = None,
    significant_increase: ArrayLike | None = None,
    unlikely_to_pay: ArrayLike | None = None,
    restructured: ArrayLike | None = None,
    in_monitoring_year: ArrayLike | None = None,
    restructured_again: ArrayLike | None = None,
    judgement_group: ArrayLike | None = None,
    limits: DayLimits | None = None,
    consumer_loans_at_loan_level: bool = True,
) -> Classification:
    """Group and stage of each loan, with the rule behind it.

    Inputs are arrays with one entry per loan.

    * ``significant_increase``: TFRS 9 says credit risk has increased significantly -> at least
      group 2 (Article 4(1)(b)6).
    * ``unlikely_to_pay``: the bank concludes the borrower will not pay in full without recourse to
      collateral, or is in default under the IRB communiqué -> at least group 3 (Article 5(1)).
    * ``restructured``: a concession was granted because of financial difficulty while the loan was
      performing -> at least group 2 (Article 4(1)(b)8). A change of terms for a group 1 borrower
      who is *not* in difficulty is not a restructuring (Article 7(9)) and should not be flagged.
    * ``in_monitoring_year`` with more than 30 days past due, or ``restructured_again``: a
      restructured performing loan that slips within its one-year monitoring period -> group 3
      (Article 4(1)(c)5).
    * ``judgement_group``: the group the bank assigns on the qualitative criteria of Article 4, if
      worse than what the rules above give.
    * ``borrower``: with it, Article 5(5) is applied. If one loan of a borrower is non-performing,
      all loans of that borrower go to the same (worst) non-performing group. Consumer loans may be
      assessed loan by loan: with ``consumer_loans_at_loan_level`` they are only moved to at least
      group 2, and a non-performing consumer loan does not pull the borrower's other loans into
      default. The regulation spells this out only for the borrower's *other consumer loans*; for a
      borrower with both kinds of loan this is a reading of the text, so check your own policy.
    """
    lim = limits or DayLimits()
    d = np.asarray(days_past_due, dtype=np.float64)
    if d.ndim != 1:
        raise InputError("days_past_due must be one-dimensional")
    n = int(d.size)
    group = group_from_days_past_due(d, lim).astype(np.int64)
    why: list[list[str]] = [[] for _ in range(n)]

    def note(mask: NDArray[np.bool_], text: str) -> None:
        for i in np.nonzero(mask)[0]:
            why[int(i)].append(text)

    def lift(mask: NDArray[np.bool_], to: int, text: str) -> None:
        note(mask, text)
        group[mask & (group < to)] = to

    for g, lo, art in ((5, lim.group5, "4(1)(d)3"), (4, lim.group4, "4(1)(ç)3"), (3, lim.group3, "4(1)(c)3")):
        note(group == g, f"more than {lo} days past due (Art. {art})")
    note(group == 2, f"more than {lim.group2} days past due (Art. 4(1)(b)5)")

    sicr = _flags("significant_increase", significant_increase, n)
    utp = _flags("unlikely_to_pay", unlikely_to_pay, n)
    forb = _flags("restructured", restructured, n)
    monitoring = _flags("in_monitoring_year", in_monitoring_year, n)
    again = _flags("restructured_again", restructured_again, n)

    lift(sicr, 2, "significant increase in credit risk under TFRS 9 (Art. 4(1)(b)6)")
    lift(forb, 2, "restructured while performing (Art. 4(1)(b)8)")
    lift(utp, 3, "unlikely to pay / default (Art. 5(1))")
    lift(
        monitoring & (d > lim.group2),
        3,
        f"restructured loan more than {lim.group2} days past due within its monitoring year (Art. 4(1)(c)5)",
    )
    lift(monitoring & again, 3, "restructured again within its monitoring year (Art. 4(1)(c)5)")
    if judgement_group is not None:
        j = np.asarray(judgement_group, dtype=np.int64)
        if j.shape != (n,) or np.any(j < 1) or np.any(j > 5):
            raise InputError("judgement_group needs one group from 1 to 5 per loan")
        for g in (2, 3, 4, 5):
            lift((j == g) & (group < g), g, f"bank's assessment: group {g} (Art. 4)")

    if borrower is not None:
        if len(borrower) != n:
            raise InputError("borrower needs one entry per loan")
        consumer = _flags("consumer_loan", consumer_loan, n)
        loan_level = consumer & consumer_loans_at_loan_level
        by_borrower: dict[object, list[int]] = {}
        for i, b in enumerate(borrower):
            by_borrower.setdefault(b, []).append(i)
        own = group.copy()
        for idx in by_borrower.values():
            if len(idx) < 2:
                continue
            ix = np.asarray(idx)
            pulling = ix[(own[ix] >= 3) & ~loan_level[ix]]
            worst = int(own[pulling].max()) if pulling.size else 0
            any_npl = bool(np.any(own[ix] >= 3))
            for i in (int(v) for v in ix):
                if worst and not loan_level[i] and group[i] < worst:
                    group[i] = worst
                    why[int(i)].append(
                        f"another loan of the borrower is non-performing: same group {worst} (Art. 5(5))"
                    )
                elif loan_level[i] and group[i] < 2 and (worst or (any_npl and own[i] < 3)):
                    group[i] = 2
                    why[int(i)].append(
                        "consumer loan of a borrower with a non-performing loan: not group 1 (Art. 5(5))"
                    )

    g8 = group.astype(np.int8)
    return Classification(g8, stage_from_group(g8), tuple(tuple(w) for w in why))


# ---- moving back up ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    unmet: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"eligible": self.eligible, "unmet": list(self.unmet)}


def _verdict(conditions: Sequence[tuple[bool, str]]) -> Eligibility:
    unmet = tuple(text for ok, text in conditions if not ok)
    return Eligibility(not unmet, unmet)


def performing_again(
    *,
    arrears_collected_without_collateral: bool,
    no_amount_past_due: bool,
    last_two_payments_on_time: bool,
    meets_group_1_or_2: bool,
    partially_written_off: bool = False,
    collateral_taken_over: bool = False,
    repaid_in_kind: bool = False,
) -> Eligibility:
    """May a non-performing loan be reclassified as performing? (Article 6(3) and 6(4))"""
    return _verdict(
        [
            (not partially_written_off, "partially written off: cannot be reclassified as performing (Art. 6(4))"),
            (not collateral_taken_over, "collateral was acquired against the debt (Art. 6(4))"),
            (not repaid_in_kind, "repaid in kind (Art. 6(4))"),
            (
                arrears_collected_without_collateral,
                "the overdue payments behind the classification were not all collected without recourse "
                "to collateral (Art. 6(3)(a))",
            ),
            (no_amount_past_due, "an amount is past due at the reclassification date (Art. 6(3)(b))"),
            (last_two_payments_on_time, "the last two payments were not made on time and in full (Art. 6(3)(b))"),
            (meets_group_1_or_2, "the conditions of group 1 or 2 are not met (Art. 6(3)(c))"),
        ]
    )


def restructured_npl_to_group_2(
    *,
    months_as_non_performing_since_restructuring: float,
    non_performing_conditions_gone: bool,
    no_delay_and_no_doubt: bool,
    arrears_and_written_off_principal_collected: bool,
) -> Eligibility:
    """May a restructured non-performing loan move to group 2 as a restructured loan? (Article 7(5))"""
    return _verdict(
        [
            (
                non_performing_conditions_gone,
                "the conditions for non-performing classification still exist (Art. 7(5)(a))",
            ),
            (
                months_as_non_performing_since_restructuring >= 12,
                "not yet monitored as non-performing for one year after the restructuring (Art. 7(5)(b))",
            ),
            (no_delay_and_no_doubt, "a payment is delayed or future payments are in doubt (Art. 7(5)(c))"),
            (
                arrears_and_written_off_principal_collected,
                "overdue payments or written-off principal have not been collected (Art. 7(5)(ç))",
            ),
        ]
    )


def exit_restructured_status(
    *,
    months_monitored_in_group_2: float,
    share_of_remaining_debt_repaid: float,
    max_days_past_due_any_loan: float,
    financial_difficulty_ended: bool,
    applies_tfrs9: bool = True,
    came_from_non_performing: bool = False,
) -> Eligibility:
    """May a restructured loan in group 2 leave the restructured status? (Article 7(6) and 7(8))

    The general rule is one year of monitoring with at least 10% of the remaining principal and
    interest repaid. Article 7(8) lets banks applying TFRS 9 move loans "that are performing and
    classified as restructured under group 2" to group 1 after three months, without the one-year
    and 10% conditions. Read literally that covers loans restructured while performing, not loans
    that came back from non-performing status under Article 7(5); this function follows that
    reading (``came_from_non_performing=True`` keeps the one-year rule).
    """
    short = applies_tfrs9 and not came_from_non_performing
    conditions = [
        (
            max_days_past_due_any_loan <= 30,
            "a loan of the borrower was more than 30 days past due during monitoring (Art. 7(6)(c))",
        ),
        (financial_difficulty_ended, "the financial difficulty behind the restructuring has not ended (Art. 7(6)(ç))"),
    ]
    if short:
        conditions.append((months_monitored_in_group_2 >= 3, "not yet monitored for three months (Art. 7(8))"))
    else:
        conditions.append((months_monitored_in_group_2 >= 12, "not yet monitored for one year (Art. 7(6)(a))"))
        conditions.append(
            (
                share_of_remaining_debt_repaid >= 0.10,
                "less than 10% of the remaining principal and interest repaid in the monitoring year (Art. 7(6)(b))",
            )
        )
    return _verdict(conditions)


# ---- fixed rates for banks that do not apply TFRS 9 --------------------------------------------


def amount_subject_to_specific_provision(
    non_performing_amount: float, collateral: Sequence[tuple[int, float]]
) -> tuple[float, list[dict[str, float]]]:
    """Article 15: the non-performing amount left after collateral, per borrower.

    ``collateral`` is a list of ``(collateral group 1..5, net realisable value)``. Collateral is used
    in order of group, most liquid first. Of each item only the part up to the balance still
    uncovered counts; that part times the group's consideration rate is exempt from specific
    provision, and the balance still uncovered falls by the part used. Value above the receivable is
    ignored.
    """
    amount = float(non_performing_amount)
    if not amount >= 0.0:
        raise InputError("the non-performing amount must not be negative")
    remaining = amount
    exempt = 0.0
    steps: list[dict[str, float]] = []
    for grp, value in sorted(collateral, key=lambda c: c[0]):
        if grp not in COLLATERAL_CONSIDERATION_RATE or not float(value) >= 0.0:
            raise InputError("collateral needs a group from 1 to 5 and a non-negative value")
        used = min(float(value), remaining)
        rate = COLLATERAL_CONSIDERATION_RATE[grp]
        exempt += used * rate
        remaining -= used
        steps.append({"collateral_group": float(grp), "value_used": used, "rate": rate, "exempt": used * rate})
    return amount - exempt, steps


def fixed_rate_provision(group: int, amount: float, collateral: Sequence[tuple[int, float]] = ()) -> dict[str, object]:
    """Minimum provision of one borrower's exposure for a bank that does not apply TFRS 9.

    Groups 1 and 2: general provision of 1.5% and 3% of cash loans (Article 10). Groups 3 to 5:
    specific provision of at least 20% and 50%, and exactly 100%, of the amount left after
    collateral (Articles 11 and 15). The exclusions of Article 10(4) and the Board's power to ask
    for more are not modelled.
    """
    if group in GENERAL_PROVISION_RATE:
        rate = GENERAL_PROVISION_RATE[group]
        return {
            "type": "general",
            "group": group,
            "rate": rate,
            "base": float(amount),
            "provision": float(amount) * rate,
        }
    if group in SPECIFIC_PROVISION_RATE:
        base, steps = amount_subject_to_specific_provision(amount, collateral)
        rate = SPECIFIC_PROVISION_RATE[group]
        return {
            "type": "specific",
            "group": group,
            "rate": rate,
            "base": base,
            "provision": base * rate,
            "collateral": steps,
        }
    raise InputError("groups run from 1 to 5")
