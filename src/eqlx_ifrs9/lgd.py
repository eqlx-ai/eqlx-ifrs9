"""Loss given default from collateral and recoveries, and the allowance for credit-impaired assets.

LGD here is measured **at the default date**: recoveries are discounted back to the moment of
default. The ECL sum then discounts from the default date to the reporting date. Discounting
recoveries to the reporting date inside the LGD as well would count the time value twice.

IFRS 9 wants a best estimate. Regulatory downturn LGD, margins of conservatism and floors do not
belong in these inputs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._checks import Array, as_1d, in_unit_interval, non_negative, rate
from .errors import InputError


def collateral_lgd(
    ead: ArrayLike,
    collateral_value: ArrayLike,
    *,
    haircut: float = 0.0,
    cost_rate: float = 0.0,
    years_to_recovery: float = 0.0,
    discount_rate: float = 0.0,
    unsecured_recovery_rate: float = 0.0,
    cure_probability: float = 0.0,
    lgd_if_cured: float = 0.0,
) -> Array:
    """LGD of a secured exposure, per period if ``ead`` and ``collateral_value`` are arrays.

    ``net proceeds = collateral * (1 - haircut) * (1 - cost_rate)``, received ``years_to_recovery``
    after default and discounted at ``discount_rate`` (the effective interest rate under IFRS 9).
    The recovery from collateral cannot exceed the claim at the default date (the cap is applied
    before discounting, so interest accruing on the claim during the workout is not recovered and a
    fully covered loan still loses the time value). On the part not covered by collateral
    ``unsecured_recovery_rate`` is recovered at the same time. With probability ``cure_probability``
    the loan cures and loses ``lgd_if_cured`` instead. Costs of obtaining and selling the collateral
    are deducted whether or not foreclosure is likely (IFRS 9 B5.5.55).
    """
    e = non_negative("ead", as_1d("ead", ead))
    c = non_negative("collateral_value", as_1d("collateral_value", collateral_value))
    try:
        e, c = np.broadcast_arrays(e, c)
    except ValueError as exc:
        raise InputError("ead and collateral_value need the same number of periods") from exc
    for name, v in (
        ("haircut", haircut),
        ("cost_rate", cost_rate),
        ("unsecured_recovery_rate", unsecured_recovery_rate),
        ("cure_probability", cure_probability),
        ("lgd_if_cured", lgd_if_cured),
    ):
        if not 0.0 <= float(v) <= 1.0:
            raise InputError(f"{name} must lie in [0, 1]")
    if years_to_recovery < 0.0:
        raise InputError("years_to_recovery must not be negative")
    df = (1.0 + rate("discount_rate", discount_rate)) ** (-float(years_to_recovery))
    net = c * (1.0 - haircut) * (1.0 - cost_rate)
    secured = np.minimum(net, e)
    recovered = (secured + unsecured_recovery_rate * (e - secured)) * df
    liq = np.ones_like(e, dtype=np.float64)
    np.divide(e - recovered, e, out=liq, where=e > 0.0)
    liq = np.clip(liq, 0.0, 1.0)
    return np.asarray(cure_probability * lgd_if_cured + (1.0 - cure_probability) * liq, dtype=np.float64)


def workout_lgd(
    ead: float,
    recoveries: ArrayLike,
    years_after_default: ArrayLike,
    discount_rate: float,
    costs: ArrayLike | None = None,
) -> float:
    """Realised or expected LGD of one default from its recovery cash flows.

    ``1 - (PV of recoveries - PV of costs) / ead``, present values taken at the default date. Not
    clipped: a workout can recover more than the exposure, or cost more than it brings in.
    """
    e = float(ead)
    if not e > 0.0:
        raise InputError("ead must be positive")
    rec = as_1d("recoveries", recoveries)
    t = non_negative("years_after_default", as_1d("years_after_default", years_after_default))
    if rec.size != t.size:
        raise InputError("recoveries and years_after_default need the same length")
    cost = np.zeros_like(rec) if costs is None else as_1d("costs", costs)
    if cost.size != rec.size:
        raise InputError("costs and recoveries need the same length")
    df = (1.0 + rate("discount_rate", discount_rate)) ** (-t)
    return float(1.0 - np.sum((rec - cost) * df) / e)


def credit_impaired_allowance(
    gross_carrying_amount: float,
    expected_cash_flows: ArrayLike,
    years_from_reporting_date: ArrayLike,
    effective_interest_rate: float,
    probabilities: ArrayLike | None = None,
) -> float:
    """Allowance of a credit-impaired asset: gross carrying amount minus the present value of the
    cash flows still expected, discounted at the original effective interest rate (IFRS 9 B5.5.33).

    With ``probabilities`` each row of ``expected_cash_flows`` is one recovery scenario (rows by
    scenario, columns by date) and the present values are probability-weighted; the weights must
    add up to 1.
    """
    g = float(gross_carrying_amount)
    if not g >= 0.0:
        raise InputError("gross_carrying_amount must not be negative")
    t = non_negative("years_from_reporting_date", as_1d("years_from_reporting_date", years_from_reporting_date))
    df = (1.0 + rate("effective_interest_rate", effective_interest_rate)) ** (-t)
    cf = np.asarray(expected_cash_flows, dtype=np.float64)
    if probabilities is None:
        flows = as_1d("expected_cash_flows", cf)
        if flows.size != t.size:
            raise InputError("cash flows and dates need the same length")
        return float(g - np.sum(flows * df))
    w = in_unit_interval("probabilities", as_1d("probabilities", probabilities))
    if cf.ndim != 2 or cf.shape != (w.size, t.size):
        raise InputError("with probabilities, cash flows must have one row per scenario and one column per date")
    if abs(float(w.sum()) - 1.0) > 1e-9:
        raise InputError("probabilities must add up to 1")
    return float(g - np.sum(w * (cf @ df)))
