"""Probability-of-default term structures.

A :class:`PDCurve` holds the cumulative probability of default at the end of each period, counted
from the reporting date. Everything else is derived from it, and the names are kept apart on purpose
because texts use "marginal PD" for two different things:

* ``marginal``     m_t = c_t - c_(t-1)        unconditional: default in period t, seen from today
* ``conditional``  p_t = m_t / S_(t-1)        default in period t given survival to its start
* ``survival``     S_t = 1 - c_t

Expected credit loss sums the *unconditional* marginal PD. Summing the conditional PD instead leaves
out the survival weighting and overstates the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._checks import Array, as_1d, frozen, in_unit_interval, positive_int, probability
from .errors import InputError

_TOL = 1e-12


@dataclass(frozen=True, eq=False)
class PDCurve:
    """Cumulative PD at the end of periods 1..n. ``periods_per_year`` is 12 for monthly, 1 for annual."""

    cumulative: Array
    periods_per_year: int = 12

    def __post_init__(self) -> None:
        c = in_unit_interval("cumulative PD", as_1d("cumulative PD", self.cumulative))
        if c.size == 0:
            raise InputError("a PD curve needs at least one period")
        steps = np.diff(np.concatenate(([0.0], c)))
        if np.any(steps < -_TOL):
            raise InputError("cumulative PD must not decrease from one period to the next")
        object.__setattr__(self, "cumulative", frozen(np.maximum.accumulate(c)))
        object.__setattr__(self, "periods_per_year", positive_int("periods_per_year", self.periods_per_year))

    # ---- constructors -------------------------------------------------------------------------

    @classmethod
    def from_cumulative(cls, cumulative: ArrayLike, periods_per_year: int = 12) -> PDCurve:
        return cls(as_1d("cumulative PD", cumulative), periods_per_year)

    @classmethod
    def from_marginal(cls, marginal: ArrayLike, periods_per_year: int = 12) -> PDCurve:
        """From unconditional per-period PDs (they add up to the cumulative PD)."""
        m = as_1d("marginal PD", marginal)
        if np.any(m < 0.0):
            raise InputError("marginal PD must not be negative")
        c = np.cumsum(m)
        if c[-1] > 1.0 + 1e-9:
            raise InputError("marginal PDs add up to more than 1; are these conditional PDs?")
        return cls(np.minimum(c, 1.0), periods_per_year)

    @classmethod
    def from_conditional(cls, conditional: ArrayLike, periods_per_year: int = 12) -> PDCurve:
        """From per-period PDs that are conditional on survival to the start of each period."""
        p = in_unit_interval("conditional PD", as_1d("conditional PD", conditional))
        return cls(1.0 - np.cumprod(1.0 - p), periods_per_year)

    @classmethod
    def flat(cls, annual_pd: float, n_periods: int, periods_per_year: int = 12) -> PDCurve:
        """Constant hazard: the PD over any 12 months of survival is ``annual_pd``.

        The per-period conditional PD is ``1 - (1 - annual_pd) ** (1 / periods_per_year)``, not
        ``annual_pd / periods_per_year``.
        """
        pd = probability("annual_pd", annual_pd)
        n = positive_int("n_periods", n_periods)
        k = positive_int("periods_per_year", periods_per_year)
        t = np.arange(1, n + 1, dtype=np.float64) / k
        return cls(1.0 - (1.0 - pd) ** t, k)

    @classmethod
    def weibull(cls, scale_years: float, shape: float, n_periods: int, periods_per_year: int = 12) -> PDCurve:
        """Survival ``exp(-(t / scale_years) ** shape)`` with t in years."""
        if not (scale_years > 0.0 and shape > 0.0):
            raise InputError("Weibull scale and shape must be positive")
        n = positive_int("n_periods", n_periods)
        k = positive_int("periods_per_year", periods_per_year)
        t = np.arange(1, n + 1, dtype=np.float64) / k
        return cls(1.0 - np.exp(-((t / scale_years) ** shape)), k)

    # ---- views --------------------------------------------------------------------------------

    @property
    def n_periods(self) -> int:
        return int(self.cumulative.size)

    def survival(self) -> Array:
        """S_1..S_n, survival to the end of each period."""
        return 1.0 - self.cumulative

    def survival_at_start(self) -> Array:
        """S_0..S_(n-1), survival to the start of each period."""
        return np.concatenate(([1.0], 1.0 - self.cumulative[:-1]))

    def marginal(self) -> Array:
        """Unconditional PD of each period: c_t - c_(t-1). This is what ECL sums."""
        return np.diff(np.concatenate(([0.0], self.cumulative)))

    def conditional(self) -> Array:
        """PD of each period given survival to its start. 1.0 once default is already certain."""
        s0 = self.survival_at_start()
        m = self.marginal()
        out = np.ones_like(m)
        np.divide(m, s0, out=out, where=s0 > 0.0)
        return np.minimum(out, 1.0)

    def hazard(self) -> Array:
        """Annualised piecewise-constant hazard rate of each period (infinite where p = 1)."""
        p = self.conditional()
        with np.errstate(divide="ignore"):
            return -np.log1p(-p) * self.periods_per_year

    def pd_over(self, n_periods: int | None = None) -> float:
        """Cumulative PD over the first ``n_periods`` (the whole curve when omitted)."""
        n = self.n_periods if n_periods is None else int(n_periods)
        if n == 0:
            return 0.0
        if n < 0 or n > self.n_periods:
            raise InputError(f"the curve covers {self.n_periods} periods, {n} requested")
        return float(self.cumulative[n - 1])

    def twelve_month_pd(self) -> float:
        """PD over the next 12 months, or over the whole curve when it is shorter than that."""
        return self.pd_over(min(self.n_periods, self.periods_per_year))

    def annualized(self, n_periods: int | None = None) -> float:
        """The constant annual PD that gives the same cumulative PD over the window."""
        n = self.n_periods if n_periods is None else int(n_periods)
        c = self.pd_over(n)
        if n == 0:
            return 0.0
        return float(1.0 - (1.0 - c) ** (self.periods_per_year / n))

    # ---- transformations ----------------------------------------------------------------------

    def truncate(self, n_periods: int) -> PDCurve:
        n = positive_int("n_periods", n_periods)
        if n > self.n_periods:
            raise InputError(f"the curve covers {self.n_periods} periods, {n} requested")
        return PDCurve(self.cumulative[:n], self.periods_per_year)

    def forward(self, start: int) -> PDCurve:
        """The curve for periods after ``start``, conditional on survival to ``start``.

        ``(c_(start+k) - c_start) / (1 - c_start)``. Used to compare today's remaining-life PD with
        the PD that was expected at origination for the same remaining window.
        """
        s = int(start)
        if s == 0:
            return self
        if s < 0 or s >= self.n_periods:
            raise InputError(f"start must lie in 0..{self.n_periods - 1}, got {start!r}")
        c0 = float(self.cumulative[s - 1])
        if c0 >= 1.0:
            raise InputError("survival to the start period has probability zero")
        return PDCurve((self.cumulative[s:] - c0) / (1.0 - c0), self.periods_per_year)

    def extend(self, n_periods: int) -> PDCurve:
        """Lengthen the curve to ``n_periods`` by repeating the last period's conditional PD."""
        n = positive_int("n_periods", n_periods)
        if n <= self.n_periods:
            return self.truncate(n)
        p = self.conditional()
        tail = np.full(n - self.n_periods, p[-1])
        return PDCurve.from_conditional(np.concatenate((p, tail)), self.periods_per_year)

    def to_frequency(self, periods_per_year: int) -> PDCurve:
        """Split each period into equal sub-periods with a constant hazard inside the period.

        Only refinement is supported (annual to monthly, quarterly to monthly); the cumulative PD at
        the original period ends is unchanged.
        """
        k_new = positive_int("periods_per_year", periods_per_year)
        if k_new == self.periods_per_year:
            return self
        if k_new % self.periods_per_year != 0:
            raise InputError("the new frequency must be a multiple of the current one")
        f = k_new // self.periods_per_year
        p_sub = 1.0 - (1.0 - self.conditional()) ** (1.0 / f)
        return PDCurve.from_conditional(np.repeat(p_sub, f), k_new)

    def to_dict(self) -> dict[str, object]:
        return {
            "periods_per_year": self.periods_per_year,
            "n_periods": self.n_periods,
            "cumulative": [float(x) for x in self.cumulative],
        }
