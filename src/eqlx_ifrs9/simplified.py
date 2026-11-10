"""Simplified approach for trade receivables, contract assets and lease receivables.

Under IFRS 9 5.5.15 these are always measured at lifetime ECL (mandatory for trade receivables
without a significant financing component), and B5.5.35 allows a provision matrix: one loss rate per
days-past-due bucket.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._checks import Array, as_1d, in_unit_interval, non_negative
from .errors import InputError


@dataclass(frozen=True, eq=False)
class ProvisionMatrix:
    buckets: tuple[str, ...]
    gross_carrying_amount: Array
    loss_rate: Array
    allowance: Array

    @property
    def total_gross_carrying_amount(self) -> float:
        return float(self.gross_carrying_amount.sum())

    @property
    def total_allowance(self) -> float:
        return float(self.allowance.sum())

    def to_dict(self) -> dict[str, object]:
        return {
            "buckets": [
                {
                    "bucket": b,
                    "gross_carrying_amount": float(g),
                    "loss_rate": float(r),
                    "allowance": float(a),
                }
                for b, g, r, a in zip(
                    self.buckets, self.gross_carrying_amount, self.loss_rate, self.allowance, strict=True
                )
            ],
            "total_gross_carrying_amount": self.total_gross_carrying_amount,
            "total_allowance": self.total_allowance,
            "coverage": self.total_allowance / self.total_gross_carrying_amount
            if self.total_gross_carrying_amount > 0
            else 0.0,
        }


def provision_matrix(
    gross_carrying_amount: ArrayLike, loss_rate: ArrayLike, buckets: Sequence[str] | None = None
) -> ProvisionMatrix:
    """Allowance per bucket: gross carrying amount times lifetime expected loss rate."""
    g = non_negative("gross_carrying_amount", as_1d("gross_carrying_amount", gross_carrying_amount))
    r = in_unit_interval("loss_rate", as_1d("loss_rate", loss_rate))
    if g.size != r.size:
        raise InputError("amounts and loss rates need the same number of buckets")
    names = tuple(buckets) if buckets is not None else tuple(f"bucket_{i + 1}" for i in range(g.size))
    if len(names) != g.size:
        raise InputError("bucket names and amounts need the same length")
    return ProvisionMatrix(names, g, r, g * r)


def loss_rates_from_history(amount_reaching_bucket: ArrayLike, final_loss: float) -> Array:
    """Historical loss rate per bucket: final loss divided by the amount that reached the bucket.

    ``amount_reaching_bucket[0]`` is the total billed (every receivable starts current), the next
    entries the amounts still unpaid when each later bucket began. Every loss passed through every
    bucket, so the same final loss is the numerator throughout.
    """
    a = as_1d("amount_reaching_bucket", amount_reaching_bucket)
    loss = float(final_loss)
    if np.any(a <= 0.0) or loss < 0.0:
        raise InputError("amounts must be positive and the loss not negative")
    if np.any(np.diff(a) > 0.0):
        raise InputError("the amount reaching a later bucket cannot exceed the amount reaching an earlier one")
    if loss > a[-1]:
        raise InputError("the final loss exceeds the amount that reached the last bucket")
    return np.asarray(loss / a, dtype=np.float64)


def loss_rates_from_flow_rates(flow_rate: ArrayLike, loss_given_write_off: float = 1.0) -> Array:
    """Lifetime loss rate per bucket from bucket-to-bucket flow rates (net flow method).

    ``flow_rate[b]`` is the share of bucket b that moves on to the next bucket; the last entry is
    the share of the last bucket that is written off. The loss rate of bucket b is the product of
    the flow rates from b onwards times ``loss_given_write_off`` (1 minus recoveries after
    write-off).
    """
    f = in_unit_interval("flow_rate", as_1d("flow_rate", flow_rate))
    if not 0.0 <= float(loss_given_write_off) <= 1.0:
        raise InputError("loss_given_write_off must lie in [0, 1]")
    return np.asarray(np.cumprod(f[::-1])[::-1] * float(loss_given_write_off), dtype=np.float64)


def adjust_loss_rates(loss_rate: ArrayLike, factor: float) -> Array:
    """Scale historical loss rates for current conditions and forecasts; the result is capped at 1."""
    r = in_unit_interval("loss_rate", as_1d("loss_rate", loss_rate))
    if not float(factor) >= 0.0:
        raise InputError("the adjustment factor must not be negative")
    return np.asarray(np.minimum(r * float(factor), 1.0), dtype=np.float64)
