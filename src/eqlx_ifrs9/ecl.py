"""Expected credit loss from a PD curve, exposure, LGD and the effective interest rate.

``ECL = sum over periods of marginal PD x LGD x EAD x discount factor``

* ``marginal PD`` is unconditional: the probability, seen from the reporting date, that default
  happens in that period.
* Losses are discounted to the **reporting date** at the effective interest rate (IFRS 9 B5.5.44).
* 12-month ECL is the same sum over the periods of the next 12 months only, or over the remaining
  life when that is shorter (B5.5.43). The LGD inside it is still the full loss given default.

The standard does not prescribe the period length or when inside a period default is assumed to
happen. Both are explicit here: the period length comes from the PD curve, ``timing`` is a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._checks import Array, as_1d, broadcast, in_unit_interval, non_negative, rate
from .curves import PDCurve
from .errors import InputError

Timing = Literal["end", "mid"]


@dataclass(frozen=True, eq=False)
class ECLResult:
    twelve_month: float
    lifetime: float
    twelve_month_pd: float
    lifetime_pd: float
    periods_per_year: int
    timing: str
    effective_interest_rate: float
    marginal_pd: Array
    lgd: Array
    ead: Array
    discount_factor: Array
    contribution: Array

    @property
    def n_periods(self) -> int:
        return int(self.contribution.size)

    def for_stage(self, stage: int) -> float:
        """12-month ECL for stage 1, lifetime ECL for stage 2."""
        if stage == 1:
            return self.twelve_month
        if stage == 2:
            return self.lifetime
        raise InputError("for_stage covers stages 1 and 2; credit-impaired assets are measured separately")

    def to_dict(self, periods: bool = True) -> dict[str, object]:
        out: dict[str, object] = {
            "twelve_month_ecl": self.twelve_month,
            "lifetime_ecl": self.lifetime,
            "twelve_month_pd": self.twelve_month_pd,
            "lifetime_pd": self.lifetime_pd,
            "n_periods": self.n_periods,
            "periods_per_year": self.periods_per_year,
            "timing": self.timing,
            "effective_interest_rate": self.effective_interest_rate,
        }
        if periods:
            out["periods"] = [
                {
                    "period": t + 1,
                    "marginal_pd": float(self.marginal_pd[t]),
                    "lgd": float(self.lgd[t]),
                    "ead": float(self.ead[t]),
                    "discount_factor": float(self.discount_factor[t]),
                    "ecl": float(self.contribution[t]),
                }
                for t in range(self.n_periods)
            ]
        return out


def discount_factors(
    effective_interest_rate: float, n_periods: int, periods_per_year: int = 12, timing: Timing = "end"
) -> Array:
    """``(1 + rate) ** -(t / periods_per_year)`` for t = 1..n, or t - 0.5 with ``timing="mid"``."""
    r = rate("effective_interest_rate", effective_interest_rate)
    if timing not in ("end", "mid"):
        raise InputError(f"timing must be 'end' or 'mid', got {timing!r}")
    t = np.arange(1, int(n_periods) + 1, dtype=np.float64) - (0.5 if timing == "mid" else 0.0)
    return np.asarray((1.0 + r) ** (-t / periods_per_year), dtype=np.float64)


def expected_credit_loss(
    curve: PDCurve,
    ead: ArrayLike,
    lgd: ArrayLike,
    effective_interest_rate: float,
    *,
    timing: Timing = "end",
) -> ECLResult:
    """12-month and lifetime ECL of one instrument.

    ``ead`` has one entry per remaining period (see :mod:`eqlx_ifrs9.schedule`); its length is the
    remaining life. ``lgd`` is one number or one per period. The PD curve must cover the remaining
    life and use the same period length; it is not extended silently.
    """
    e = non_negative("ead", as_1d("ead", ead))
    n = int(e.size)
    if n == 0:
        raise InputError("ead needs at least one period")
    if curve.n_periods < n:
        raise InputError(
            f"the PD curve covers {curve.n_periods} periods but the exposure runs for {n}; "
            "extend the curve explicitly (PDCurve.extend) if that is intended"
        )
    loss = in_unit_interval("lgd", broadcast("lgd", lgd, n))
    m = curve.marginal()[:n]
    k = curve.periods_per_year
    df = discount_factors(effective_interest_rate, n, k, timing)
    contribution = m * loss * e * df
    first_year = min(n, k)
    return ECLResult(
        twelve_month=float(np.sum(contribution[:first_year])),
        lifetime=float(np.sum(contribution)),
        twelve_month_pd=curve.pd_over(first_year),
        lifetime_pd=curve.pd_over(n),
        periods_per_year=k,
        timing=timing,
        effective_interest_rate=float(effective_interest_rate),
        marginal_pd=m,
        lgd=loss,
        ead=e,
        discount_factor=df,
        contribution=contribution,
    )


def loss_rate_ecl(exposure: ArrayLike, loss_rate: ArrayLike) -> Array:
    """ECL as exposure times an expected loss rate (the loss-rate approach of IFRS 9 B5.5.12)."""
    e = non_negative("exposure", as_1d("exposure", exposure))
    r = in_unit_interval("loss_rate", broadcast("loss_rate", loss_rate, e.size))
    return np.asarray(e * r, dtype=np.float64)
