"""Repayment schedules and exposure at default.

A :class:`Schedule` lists, for each remaining period, the balance outstanding at the start of the
period (before that period's payment), the interest and principal due at its end, and the balance
after the payment. Period 1 starts at the reporting date.

Exposure at default for a default in period t is taken from the balance at the *start* of the period
by default. Using the balance after the payment would give a bullet loan no exposure in its last
period.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._checks import Array, as_1d, frozen, non_negative, positive_int, probability
from .errors import InputError

EadBasis = Literal["start", "start_plus_interest", "mid"]
RateConvention = Literal["nominal", "effective"]


def period_rate(annual_rate: float, periods_per_year: int, convention: RateConvention = "nominal") -> float:
    """Interest rate per period. ``nominal``: annual / periods; ``effective``: (1 + annual)^(1/periods) - 1."""
    r = float(annual_rate)
    k = positive_int("periods_per_year", periods_per_year)
    if not np.isfinite(r) or r <= -1.0:
        raise InputError("the interest rate must be greater than -100%")
    if convention == "nominal":
        return r / k
    if convention == "effective":
        return float((1.0 + r) ** (1.0 / k) - 1.0)
    raise InputError(f"unknown rate convention {convention!r}")


@dataclass(frozen=True, eq=False)
class Schedule:
    balance_start: Array
    interest: Array
    principal: Array
    periods_per_year: int = 12

    def __post_init__(self) -> None:
        b = non_negative("balance", as_1d("balance", self.balance_start))
        i = as_1d("interest", self.interest)
        p = as_1d("principal", self.principal)
        if not (b.size == i.size == p.size) or b.size == 0:
            raise InputError("balance, interest and principal need the same, non-zero number of periods")
        object.__setattr__(self, "balance_start", frozen(b))
        object.__setattr__(self, "interest", frozen(i))
        object.__setattr__(self, "principal", frozen(p))
        object.__setattr__(self, "periods_per_year", positive_int("periods_per_year", self.periods_per_year))

    @property
    def n_periods(self) -> int:
        return int(self.balance_start.size)

    @property
    def balance_end(self) -> Array:
        return np.asarray(np.maximum(self.balance_start - self.principal, 0.0), dtype=np.float64)

    @property
    def payment(self) -> Array:
        return np.asarray(self.interest + self.principal, dtype=np.float64)

    def ead(self, basis: EadBasis = "start") -> Array:
        """Exposure at default per period.

        * ``start``: balance before the period's payment (default).
        * ``start_plus_interest``: that balance plus the interest accrued over the period.
        * ``mid``: average of the balance before and after the payment.
        """
        if basis == "start":
            return np.array(self.balance_start)
        if basis == "start_plus_interest":
            return np.asarray(self.balance_start + self.interest, dtype=np.float64)
        if basis == "mid":
            return np.asarray((self.balance_start + self.balance_end) / 2.0, dtype=np.float64)
        raise InputError(f"unknown EAD basis {basis!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "periods_per_year": self.periods_per_year,
            "balance_start": [float(x) for x in self.balance_start],
            "interest": [float(x) for x in self.interest],
            "principal": [float(x) for x in self.principal],
        }


def _setup(
    balance: float, annual_rate: float, n: int, periods_per_year: int, convention: RateConvention
) -> tuple[float, float, int, int]:
    b = float(balance)
    if not np.isfinite(b) or b < 0.0:
        raise InputError("the balance must not be negative")
    k = positive_int("periods_per_year", periods_per_year)
    return b, period_rate(annual_rate, k, convention), positive_int("n_periods", n), k


def bullet(
    balance: float,
    annual_rate: float,
    n_periods: int,
    periods_per_year: int = 12,
    rate_convention: RateConvention = "nominal",
) -> Schedule:
    """Interest every period, the whole principal at the end."""
    b, i, n, k = _setup(balance, annual_rate, n_periods, periods_per_year, rate_convention)
    principal = np.zeros(n)
    principal[-1] = b
    return Schedule(np.full(n, b), np.full(n, b * i), principal, k)


def annuity(
    balance: float,
    annual_rate: float,
    n_periods: int,
    periods_per_year: int = 12,
    rate_convention: RateConvention = "nominal",
) -> Schedule:
    """Equal instalments: ``payment = balance * i / (1 - (1 + i) ** -n)``."""
    b, i, n, k = _setup(balance, annual_rate, n_periods, periods_per_year, rate_convention)
    t = np.arange(n, dtype=np.float64)
    if i == 0.0:
        start = b * (1.0 - t / n)
        pay = b / n
    else:
        # (g - (1+i)^t) / (g - 1) with g = (1+i)^n, written so that tiny rates do not cancel
        log_growth = np.log1p(i)
        start = b * (1.0 - np.expm1(t * log_growth) / np.expm1(n * log_growth))
        pay = b * i / -np.expm1(-n * log_growth)
    interest = start * i
    return Schedule(start, interest, pay - interest, k)


def equal_principal(
    balance: float,
    annual_rate: float,
    n_periods: int,
    periods_per_year: int = 12,
    rate_convention: RateConvention = "nominal",
) -> Schedule:
    """The same principal amount every period; interest on the falling balance."""
    b, i, n, k = _setup(balance, annual_rate, n_periods, periods_per_year, rate_convention)
    start = b * (1.0 - np.arange(n, dtype=np.float64) / n)
    return Schedule(start, start * i, np.full(n, b / n), k)


def smm_from_cpr(cpr: float) -> float:
    """Monthly prepayment rate from an annual one: ``1 - (1 - CPR) ** (1/12)``."""
    return float(1.0 - (1.0 - probability("cpr", cpr)) ** (1.0 / 12.0))


def with_prepayment(schedule: Schedule, cpr: ArrayLike) -> Schedule:
    """Scale a schedule by the share of the loan that has not been prepaid.

    ``cpr`` is the annual prepayment rate, one value or one per period. The balance at the start of
    period t is the scheduled balance times the product of ``(1 - rate)`` over the earlier periods,
    with the per-period rate ``1 - (1 - CPR) ** (1 / periods_per_year)``. Prepayment is an exit
    without loss: keep the PD curve credit-only and do not also shorten it for prepayment.
    """
    n = schedule.n_periods
    c = as_1d("cpr", cpr)
    c = np.full(n, c[0]) if c.size == 1 else c
    if c.size != n or np.any(c < 0.0) or np.any(c > 1.0):
        raise InputError("cpr must lie in [0, 1], one value or one per period")
    per = 1.0 - (1.0 - c) ** (1.0 / schedule.periods_per_year)
    remaining_start = np.concatenate(([1.0], np.cumprod(1.0 - per)[:-1]))
    start = schedule.balance_start * remaining_start
    interest = schedule.interest * remaining_start
    end = schedule.balance_end * remaining_start * (1.0 - per)
    return Schedule(start, interest, start - end, schedule.periods_per_year)


def commitment_ead(drawn: ArrayLike, undrawn: ArrayLike, drawdown_share: ArrayLike) -> Array:
    """Exposure of a facility: drawn balance plus the expected share of the undrawn limit.

    IFRS 9 B5.5.31 asks for the drawdown expected within 12 months for 12-month ECL and over the
    expected life for lifetime ECL, so the share may differ between the two measurements. Future
    drawdowns are capped at the contractual limit: the share must lie in [0, 1].
    """
    d = non_negative("drawn", as_1d("drawn", drawn))
    u = non_negative("undrawn", as_1d("undrawn", undrawn))
    s = as_1d("drawdown_share", drawdown_share)
    if np.any(s < 0.0) or np.any(s > 1.0):
        raise InputError("drawdown_share must lie in [0, 1]")
    return np.asarray(d + s * u, dtype=np.float64)
