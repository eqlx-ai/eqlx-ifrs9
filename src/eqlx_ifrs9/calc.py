"""Single calculations with plain numbers in and a dictionary out.

These are the functions behind the calculator commands of the CLI and the MCP server: one instrument,
one transition matrix, one provision matrix. No files are involved.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from . import cycle, schedule, simplified, transitions
from .curves import PDCurve
from .ecl import ECLResult, expected_credit_loss
from .errors import InputError

MONTHS = 12


def instrument_ecl(
    *,
    balance: float,
    lgd: float,
    eir: float,
    remaining_months: int,
    pd_12m: float | None = None,
    cumulative_pd: Sequence[float] | None = None,
    cumulative_pd_frequency: int = 12,
    amortization: Literal["bullet", "annuity", "equal_principal"] = "bullet",
    rate: float | None = None,
    undrawn: float = 0.0,
    ccf: float = 0.0,
    cpr: float = 0.0,
    timing: Literal["end", "mid"] = "end",
    ead_basis: Literal["start", "start_plus_interest", "mid"] = "start",
    z_by_year: Sequence[float] | None = None,
    rho: float = 0.0,
    reversion_years: int = 0,
    periods: bool = False,
) -> dict[str, Any]:
    """12-month and lifetime ECL of one instrument, on monthly periods.

    The PD comes either from ``pd_12m`` (constant hazard) or from ``cumulative_pd``, a list of
    cumulative PDs at the end of each period (``cumulative_pd_frequency`` periods a year; annual or
    quarterly points are refined to months with a constant hazard inside each period). With
    ``z_by_year`` the curve is read as long-run and shifted year by year (z > 0 good year); after
    the last year given the PD reverts to the long-run PD over ``reversion_years`` transitional
    years, as in the portfolio run. ``rate`` is the contract rate (nominal, divided by 12); without
    it the schedule uses ``eir`` as an effective rate.
    """
    if isinstance(remaining_months, bool) or int(remaining_months) != remaining_months or remaining_months < 1:
        raise InputError("remaining_months must be a whole number of at least 1")
    n = int(remaining_months)
    if not 0.0 <= cpr <= 1.0:
        raise InputError("cpr must lie in [0, 1]")
    if not 0.0 <= rho < 1.0:
        raise InputError("rho must lie in [0, 1)")
    if (pd_12m is None) == (cumulative_pd is None):
        raise InputError("give either pd_12m or cumulative_pd")
    if pd_12m is not None:
        curve = PDCurve.flat(pd_12m, n, MONTHS)
        source = "pd_12m as a constant hazard"
    else:
        raw = PDCurve.from_cumulative(list(cumulative_pd or []), cumulative_pd_frequency).to_frequency(MONTHS)
        if raw.n_periods < n:
            raise InputError(f"cumulative_pd covers {raw.n_periods} months, the instrument runs for {n}")
        curve = raw.truncate(n)
        source = "cumulative PD curve"
    if z_by_year is not None:
        z = list(z_by_year)
        if not z:
            raise InputError("z_by_year is empty")
        weights = cycle.reversion_weights(-(-n // MONTHS), len(z), reversion_years)
        curve = cycle.pit_curve(curve, rho, z, weights)
        source += ", shifted with the one-factor model"
    contract = eir if rate is None else rate
    convention: Literal["nominal", "effective"] = "effective" if rate is None else "nominal"
    builder = {"bullet": schedule.bullet, "annuity": schedule.annuity, "equal_principal": schedule.equal_principal}
    if amortization not in builder:
        raise InputError("amortization must be bullet, annuity or equal_principal")
    sched = builder[amortization](balance, contract, n, MONTHS, convention)
    if cpr > 0.0:
        sched = schedule.with_prepayment(sched, cpr)
    ead = sched.ead(ead_basis) + schedule.commitment_ead(0.0, undrawn, ccf)[0]
    res = expected_credit_loss(curve, ead, lgd, eir, timing=timing)
    out = res.to_dict(periods=periods)
    out["pd_source"] = source
    out["annualized_lifetime_pd"] = curve.annualized()
    out["by_year"] = by_year(res)
    out["undiscounted_lifetime_ecl"] = float(np.sum(res.marginal_pd * res.lgd * res.ead))
    return out


def by_year(res: ECLResult) -> list[dict[str, float]]:
    k = res.periods_per_year
    out = []
    for y in range(-(-res.n_periods // k)):
        sl = slice(y * k, (y + 1) * k)
        out.append(
            {
                "year": y + 1,
                "pd": float(res.marginal_pd[sl].sum()),
                "average_ead": float(res.ead[sl].mean()),
                "ecl": float(res.contribution[sl].sum()),
            }
        )
    return out


def pd_term_structure(
    matrix: Sequence[Sequence[float]],
    years: int,
    *,
    grades: Sequence[str] | None = None,
    monthly: bool = False,
    generator: transitions.GeneratorMethod = "wa",
    z_by_year: Sequence[float] | None = None,
    rho: float = 0.0,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Cumulative PD by grade from a one-year transition matrix (last state default).

    Annual points come from matrix powers. With ``monthly`` a generator is fitted and monthly points
    are returned; the diagnostics say how well it reproduces the matrix. ``z_by_year`` shifts the
    matrix year by year before it is applied.
    """
    p = transitions.check_matrix(matrix, tol=tolerance)
    k = p.shape[0]
    y = int(years)
    if not 1 <= y <= 100:
        raise InputError("years must lie between 1 and 100")
    names = list(grades) if grades is not None else [f"grade_{i + 1}" for i in range(k - 1)]
    if len(names) != k - 1:
        raise InputError(f"{k - 1} grade names are needed (default is the last state and is not named)")
    if z_by_year is None:
        annual = [p] * y
    else:
        z = list(z_by_year)
        if not z:
            raise InputError("z_by_year is empty")
        z = (z + [z[-1]] * y)[:y]
        annual = [transitions.shift(p, zy, rho, tol=tolerance) for zy in z]
    out: dict[str, Any] = {"diagnostics": transitions.embedding_diagnostics(p, tol=tolerance)}
    if monthly:
        gens = [transitions.generator(a, generator, tol=max(tolerance, 1e-9)) for a in annual]
        g0 = transitions.generator(p, generator, tol=max(tolerance, 1e-9))
        path = [g.matrix(1.0 / MONTHS) for g in gens for _ in range(MONTHS)]
        cum = transitions.cumulative_pd_path(path, tol=1e-9)
        out["periods_per_year"] = MONTHS
        out["generator"] = {"method": g0.method, "l1_error": g0.l1_error, "warnings": list(g0.warnings)}
    else:
        cum = transitions.cumulative_pd_path(annual, tol=tolerance)
        out["periods_per_year"] = 1
    out["cumulative_pd"] = {nm: [float(v) for v in cum[i]] for i, nm in enumerate(names)}
    return out


def provision_matrix(
    gross_carrying_amount: Sequence[float], loss_rate: Sequence[float], buckets: Sequence[str] | None = None
) -> dict[str, Any]:
    return simplified.provision_matrix(list(gross_carrying_amount), list(loss_rate), buckets).to_dict()
