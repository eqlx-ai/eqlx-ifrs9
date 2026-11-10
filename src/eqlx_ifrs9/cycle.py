"""Point-in-time adjustment of PDs with the one-factor (Vasicek) model.

``PD(z) = Phi((Phi^-1(PD) - sqrt(rho) z) / sqrt(1 - rho))`` where z is the standardised state of the
economy. **z > 0 is a good year, z < 0 a bad one.**

The long-run PD is the *average* of PD(z) over the cycle, not PD(0): at z = 0 the conditional PD is
lower than the long-run PD. Reversion to the long-run level is therefore done on the PD itself
(``pit_weight``), not by letting z go to zero.

This module does not forecast z. The scenario paths are inputs.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr, ndtri

from ._checks import Array, as_1d, in_unit_interval, positive_int, probability
from .curves import PDCurve
from .errors import InputError


def _rho(rho: float) -> float:
    r = float(rho)
    if not np.isfinite(r) or r < 0.0 or r >= 1.0:
        raise InputError(f"rho must lie in [0, 1), got {rho!r}")
    return r


def conditional_pd(pd: ArrayLike, rho: float, z: ArrayLike) -> Array:
    """PD conditional on the systematic factor z (z > 0 good year). 0 and 1 stay 0 and 1."""
    p = in_unit_interval("pd", np.asarray(as_1d("pd", pd)))
    r = _rho(rho)
    zz = as_1d("z", z)
    with np.errstate(divide="ignore"):
        x = ndtri(p)
    out = ndtr((x - np.sqrt(r) * zz) / np.sqrt(1.0 - r))
    return np.asarray(out, dtype=np.float64)


def implied_z(pd: float, rho: float, default_rate: float) -> float:
    """The z that turns the long-run ``pd`` into the observed ``default_rate``."""
    p = probability("pd", pd, open_interval=True)
    d = probability("default_rate", default_rate, open_interval=True)
    r = _rho(rho)
    if r == 0.0:
        raise InputError("rho must be positive to back out z")
    return float((ndtri(p) - np.sqrt(1.0 - r) * ndtri(d)) / np.sqrt(r))


def reversion_weights(
    n_years: int,
    forecast_years: int,
    reversion_years: int = 0,
    method: Literal["linear", "exponential"] = "linear",
    decay: float = 0.5,
) -> Array:
    """Weight of the point-in-time PD in each year: 1 inside the forecast horizon, then falling to 0.

    * ``linear``: year ``forecast_years + k`` gets ``max(0, 1 - k / reversion_years)``;
      ``reversion_years=0`` switches to the long-run PD immediately after the horizon.
    * ``exponential``: year ``forecast_years + k`` gets ``decay ** k``.
    """
    n = positive_int("n_years", n_years)
    h = int(forecast_years)
    r = int(reversion_years)
    if h < 0 or r < 0:
        raise InputError("forecast_years and reversion_years must not be negative")
    k = np.arange(1, n + 1, dtype=np.float64) - h
    if method == "linear":
        w = np.where(k <= 0, 1.0, np.maximum(0.0, 1.0 - k / r) if r > 0 else 0.0)
    elif method == "exponential":
        d = probability("decay", decay)
        w = np.where(k <= 0, 1.0, d ** np.maximum(k, 0.0))
    else:
        raise InputError(f"unknown reversion method {method!r}")
    return np.asarray(w, dtype=np.float64)


def pit_curve(curve: PDCurve, rho: float, z_by_year: ArrayLike, pit_weight: ArrayLike | None = None) -> PDCurve:
    """Shift a long-run PD curve with one z per year.

    For each 12-month block the annualised conditional PD ``A`` of the block is replaced by
    ``w * PD(z) + (1 - w) * A`` with ``PD(z)`` from :func:`conditional_pd`, and the hazard of every
    period inside the block is scaled by the same factor, so the shape within the year is kept.
    A flat 2% curve with z for year 1 therefore has a first-year PD of exactly ``conditional_pd(0.02,
    rho, z)``.

    ``z_by_year`` shorter than the curve: the last value is held. ``pit_weight`` (default all 1) is
    the weight of the shifted PD per year; see :func:`reversion_weights`.
    """
    r = _rho(rho)
    k = curve.periods_per_year
    n = curve.n_periods
    n_years = -(-n // k)
    z = as_1d("z_by_year", z_by_year)
    if z.size == 0:
        raise InputError("z_by_year is empty")
    z = np.concatenate((z, np.full(max(0, n_years - z.size), z[-1])))[:n_years]
    if pit_weight is None:
        w = np.ones(n_years)
    else:
        w = in_unit_interval("pit_weight", as_1d("pit_weight", pit_weight))
        w = np.concatenate((w, np.full(max(0, n_years - w.size), w[-1])))[:n_years]
    p = curve.conditional()
    out = np.empty_like(p)
    for y in range(n_years):
        block = slice(y * k, min((y + 1) * k, n))
        pb = p[block]
        surv = float(np.prod(1.0 - pb))
        annual = 1.0 - surv ** (k / pb.size)
        if annual <= 0.0 or annual >= 1.0:
            out[block] = pb
            continue
        target = w[y] * float(conditional_pd(annual, r, z[y])[0]) + (1.0 - w[y]) * annual
        if target >= 1.0:
            out[block] = 1.0
            continue
        factor = np.log1p(-target) / np.log1p(-annual)
        out[block] = 1.0 - (1.0 - pb) ** factor
    return PDCurve.from_conditional(out, k)
