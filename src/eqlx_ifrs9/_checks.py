"""Input validation shared by the calculation modules. Nothing is clipped or repaired silently."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import InputError

Array = NDArray[np.float64]


def as_1d(name: str, values: ArrayLike) -> Array:
    try:
        a = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be numeric") from exc
    if a.ndim == 0:
        a = a.reshape(1)
    if a.ndim != 1:
        raise InputError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(a)):
        raise InputError(f"{name} contains NaN or infinite values")
    return a


def broadcast(name: str, values: ArrayLike, n: int) -> Array:
    """A scalar is repeated ``n`` times; an array must have exactly ``n`` entries."""
    a = as_1d(name, values)
    if a.size == 1 and n != 1:
        return np.full(n, float(a[0]))
    if a.size != n:
        raise InputError(f"{name} has {a.size} entries, expected {n}")
    return a


def in_unit_interval(name: str, a: Array) -> Array:
    if np.any(a < 0.0) or np.any(a > 1.0):
        raise InputError(f"{name} must lie in [0, 1]")
    return a


def non_negative(name: str, a: Array) -> Array:
    if np.any(a < 0.0):
        raise InputError(f"{name} must not be negative")
    return a


def probability(name: str, value: float, *, open_interval: bool = False) -> float:
    v = float(value)
    if not np.isfinite(v) or v < 0.0 or v > 1.0 or (open_interval and v in (0.0, 1.0)):
        rng = "(0, 1)" if open_interval else "[0, 1]"
        raise InputError(f"{name} must lie in {rng}, got {value!r}")
    return v


def positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise InputError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def rate(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= -1.0:
        raise InputError(f"{name} must be greater than -100%, got {value!r}")
    return v


def frozen(a: Array) -> Array:
    out = np.array(a, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out
