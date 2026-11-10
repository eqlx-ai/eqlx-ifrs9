"""Probability weighting across economic scenarios.

IFRS 9 5.5.17(a) asks for an unbiased, probability-weighted amount. The weighted ECL is the weighted
average of the ECL computed *under each scenario*. It is not the ECL of the average scenario: ECL is
not linear in the economy, so the two differ, usually with the weighted figure higher.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._checks import Array, as_1d
from .errors import InputError

WEIGHT_TOL = 1e-9


def check_weights(weights: ArrayLike) -> Array:
    w = as_1d("weights", weights)
    if w.size == 0:
        raise InputError("at least one scenario is needed")
    if np.any(w < 0.0):
        raise InputError("scenario weights must not be negative")
    if abs(float(w.sum()) - 1.0) > WEIGHT_TOL:
        raise InputError(f"scenario weights must add up to 1, they add up to {float(w.sum()):.10g}")
    return w


@dataclass(frozen=True, eq=False)
class WeightedECL:
    names: tuple[str, ...]
    weights: Array
    ecl: Array
    weighted: float
    base: str | None

    @property
    def base_ecl(self) -> float | None:
        return float(self.ecl[self.names.index(self.base)]) if self.base is not None else None

    @property
    def weighted_to_base(self) -> float | None:
        """Weighted ECL divided by the ECL of the base scenario: the size of the non-linearity effect."""
        b = self.base_ecl
        return None if b is None or b == 0.0 else self.weighted / b

    def to_dict(self) -> dict[str, object]:
        return {
            "weighted_ecl": self.weighted,
            "base_scenario": self.base,
            "weighted_to_base": self.weighted_to_base,
            "scenarios": [
                {
                    "name": n,
                    "weight": float(w),
                    "ecl_at_100_percent": float(e),
                    "difference_to_weighted": float(e - self.weighted),
                }
                for n, w, e in zip(self.names, self.weights, self.ecl, strict=True)
            ],
        }


def weighted_ecl(
    ecl_by_scenario: ArrayLike,
    weights: ArrayLike,
    names: Sequence[str] | None = None,
    base: str | None = None,
) -> WeightedECL:
    """Probability-weighted ECL from one ECL figure per scenario.

    ``ecl_at_100_percent`` in the result is the sensitivity usually disclosed: the ECL if that
    scenario alone were given full weight.
    """
    w = check_weights(weights)
    e = as_1d("ecl_by_scenario", ecl_by_scenario)
    if e.size != w.size:
        raise InputError("one ECL figure per scenario is needed")
    labels = tuple(names) if names is not None else tuple(f"scenario_{i + 1}" for i in range(w.size))
    if len(labels) != w.size or len(set(labels)) != len(labels):
        raise InputError("scenario names must be unique and match the number of scenarios")
    if base is not None and base not in labels:
        raise InputError(f"unknown base scenario {base!r}")
    return WeightedECL(labels, w, e, float(np.sum(w * e)), base)


def weighted_value(values_by_scenario: ArrayLike, weights: ArrayLike) -> Array:
    """Weighted average along the first axis (scenarios), for PDs or any other per-scenario array."""
    w = check_weights(weights)
    v = np.asarray(values_by_scenario, dtype=np.float64)
    if v.shape[0] != w.size:
        raise InputError("the first axis must have one entry per scenario")
    return np.asarray(np.tensordot(w, v, axes=1), dtype=np.float64)
