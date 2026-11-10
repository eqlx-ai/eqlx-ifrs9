"""Rating transition matrices: multi-period default probabilities, generators, cycle shifts.

Conventions used throughout this module:

* ``p[i, j]`` is the probability of moving from state i to state j in one period.
* States run from best to worst and the **last state is default**, which is absorbing.
* Rows must add up to 1. Published matrices are rounded, so a small tolerance is accepted; nothing
  is renormalised unless :func:`normalize_rows` is called.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm, logm
from scipy.special import ndtr, ndtri

from ._checks import Array, as_1d, positive_int
from .curves import PDCurve
from .errors import InputError

GeneratorMethod = Literal["wa", "da", "jlt", "log"]


def check_matrix(p: ArrayLike, *, tol: float = 1e-6) -> Array:
    """Validate a transition matrix and return it as a float array (not modified)."""
    try:
        a = np.asarray(p, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputError("a transition matrix must be numeric") from exc
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 2:
        raise InputError("a transition matrix must be square with at least two states")
    if not np.all(np.isfinite(a)):
        raise InputError("the transition matrix contains NaN or infinite values")
    if np.any(a < 0.0) or np.any(a > 1.0 + tol):
        raise InputError("transition probabilities must lie in [0, 1]")
    off = np.abs(a.sum(axis=1) - 1.0)
    if np.any(off > tol):
        i = int(np.argmax(off))
        raise InputError(
            f"row {i} adds up to {a[i].sum():.6f}, not 1 (tolerance {tol:g}); "
            "use normalize_rows() if the matrix is only rounded"
        )
    last = a[-1]
    if abs(last[-1] - 1.0) > tol or np.any(np.abs(last[:-1]) > tol):
        raise InputError("the last state must be default and absorbing (last row 0, ..., 0, 1)")
    return a


def normalize_rows(p: ArrayLike) -> Array:
    """Divide every row by its sum. For matrices whose rows are off only because of rounding."""
    a = np.asarray(p, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise InputError("a transition matrix must be square")
    s = a.sum(axis=1, keepdims=True)
    if np.any(s <= 0.0):
        raise InputError("a row adds up to zero")
    return np.asarray(a / s, dtype=np.float64)


def cumulative_pd(p: ArrayLike, n_periods: int, *, tol: float = 1e-6) -> Array:
    """Cumulative PD by starting state and horizon: ``out[i, t-1] = (p ** t)[i, default]``."""
    a = check_matrix(p, tol=tol)
    n = positive_int("n_periods", n_periods)
    out = np.empty((a.shape[0], n))
    col = np.zeros(a.shape[0])
    col[-1] = 1.0
    for t in range(n):
        col = a @ col
        out[:, t] = col
    return np.clip(out, 0.0, 1.0)


def cumulative_pd_path(matrices: Sequence[ArrayLike], *, tol: float = 1e-6) -> Array:
    """Cumulative PD when the matrix changes from period to period.

    ``out[i, t-1] = (P_1 @ P_2 @ ... @ P_t)[i, default]``; the order matters.
    """
    if len(matrices) == 0:
        raise InputError("at least one matrix is needed")
    mats = [check_matrix(m, tol=tol) for m in matrices]
    k = mats[0].shape[0]
    if any(m.shape[0] != k for m in mats):
        raise InputError("all matrices must have the same number of states")
    out = np.empty((k, len(mats)))
    prod = np.eye(k)
    for t, m in enumerate(mats):
        prod = prod @ m
        out[:, t] = prod[:, -1]
    return np.clip(np.maximum.accumulate(out, axis=1), 0.0, 1.0)


def curve(p: ArrayLike, state: int, n_periods: int, periods_per_year: int = 1, *, tol: float = 1e-6) -> PDCurve:
    """PD curve of one starting state from repeated application of a one-period matrix."""
    c = cumulative_pd(p, n_periods, tol=tol)
    if not 0 <= int(state) < c.shape[0] - 1:
        raise InputError(f"state must be a non-default state 0..{c.shape[0] - 2}")
    return PDCurve(np.maximum.accumulate(c[int(state)]), periods_per_year)


# ---- continuous time ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Generator:
    """A generator matrix Q (rows add up to 0, off-diagonals >= 0) with ``exp(Q)`` close to ``p``."""

    q: Array
    method: str
    l1_error: float
    """Sum of absolute differences between the input matrix and exp(Q), over all cells."""
    negative_mass: float
    """Sum of the negative off-diagonal entries of the principal logarithm, before adjustment."""
    warnings: tuple[str, ...]

    def matrix(self, years: float) -> Array:
        """Transition matrix over ``years`` (for example 1/12 for one month)."""
        if not years >= 0.0:
            raise InputError("years must not be negative")
        m = np.asarray(expm(self.q * float(years)), dtype=np.float64)
        if m.min() < -1e-9:
            raise InputError(
                f"the '{self.method}' generator gives negative probabilities; use an adjusted generator ('wa' or 'da')"
            )
        m = np.clip(m, 0.0, 1.0)  # rounding noise only
        return np.asarray(m / m.sum(axis=1, keepdims=True), dtype=np.float64)


def _principal_log(a: Array) -> Array:
    raw = logm(a)
    raw_arr = np.asarray(raw)
    if np.iscomplexobj(raw_arr):
        if np.max(np.abs(raw_arr.imag)) > 1e-8:
            raise InputError("the matrix has no real logarithm (negative or complex eigenvalues)")
        raw_arr = raw_arr.real
    return np.asarray(raw_arr, dtype=np.float64)


def embedding_diagnostics(p: ArrayLike, *, tol: float = 1e-6) -> dict[str, object]:
    """Checks of whether a matrix can come from a continuous-time Markov chain at all.

    Follows Israel, Rosenthal and Wei (2001): no generator exists when the determinant is not
    positive, when it exceeds the product of the diagonal, or when some state can be reached from
    another over several periods although the one-period probability is zero.
    """
    a = check_matrix(p, tol=tol)
    k = a.shape[0]
    det = float(np.linalg.det(a))
    reach = (a > 0.0) | np.eye(k, dtype=bool)
    for m in range(k):
        reach = reach | (reach[:, [m]] & reach[[m], :])
    blocked = [(int(i), int(j)) for i, j in zip(*np.nonzero(reach & (a <= 0.0)), strict=True)]
    diag_ok = bool(np.all(np.diag(a) > 0.5))
    return {
        "determinant": det,
        "determinant_positive": det > 0.0,
        "determinant_within_diagonal_product": det <= float(np.prod(np.diag(a))) + 1e-15,
        "all_diagonals_above_half": diag_ok,
        "reachable_but_zero": blocked,
        "may_have_generator": det > 0.0 and det <= float(np.prod(np.diag(a))) + 1e-15 and not blocked,
    }


def generator(p: ArrayLike, method: GeneratorMethod = "wa", *, tol: float = 1e-6) -> Generator:
    """Estimate a generator from a one-period transition matrix.

    * ``"log"`` principal matrix logarithm, unadjusted; off-diagonals may be negative.
    * ``"da"``  diagonal adjustment: negative off-diagonals set to 0 and added to the diagonal.
    * ``"wa"``  weighted adjustment: the removed negative mass is spread over the other entries of
      the row in proportion to their size. Usually the closest of the simple fixes.
    * ``"jlt"`` Jarrow-Lando-Turnbull approximation ``q_ii = ln p_ii``,
      ``q_ij = p_ij ln(p_ii) / (p_ii - 1)``; needs no logarithm but is the least accurate.

    Formulas for da, wa and jlt are those of Israel, Rosenthal and Wei (2001), equations 2, 2' and 3.
    """
    a = check_matrix(p, tol=tol)
    k = a.shape[0]
    notes: list[str] = []
    if method == "jlt":
        d = np.diag(a).copy()
        if np.any(d[:-1] <= 0.0):
            raise InputError("the JLT approximation needs positive diagonal entries")
        q = np.zeros_like(a)
        for i in range(k - 1):
            factor = np.log(d[i]) / (d[i] - 1.0) if d[i] < 1.0 else 1.0
            q[i] = a[i] * factor
            q[i, i] = np.log(d[i])
        neg = 0.0
    else:
        raw = _principal_log(a)
        raw[-1] = 0.0
        offdiag = ~np.eye(k, dtype=bool)
        neg = float(-np.sum(np.minimum(raw, 0.0)[offdiag]))
        if method == "log":
            q = raw
            if neg > 0.0:
                notes.append("principal logarithm has negative off-diagonal entries; not a valid generator")
        elif method == "da":
            q = np.where(offdiag, np.maximum(raw, 0.0), raw)
            q[np.diag_indices(k)] = np.diag(raw) + np.sum(np.where(offdiag, np.minimum(raw, 0.0), 0.0), axis=1)
        elif method == "wa":
            q = raw.copy()
            for i in range(k - 1):
                row = raw[i]
                mask = offdiag[i]
                g = abs(row[i]) + float(np.sum(np.maximum(row[mask], 0.0)))
                b = float(np.sum(np.maximum(-row[mask], 0.0)))
                if b == 0.0 or g <= 0.0:
                    continue
                new = row - b * np.abs(row) / g
                new[mask & (row < 0.0)] = 0.0
                q[i] = new
        else:
            raise InputError(f"unknown generator method {method!r}")
    diag = embedding_diagnostics(a, tol=tol)
    if diag["reachable_but_zero"]:
        notes.append("some states are reachable over several periods but have a one-period probability of zero")
    if not diag["determinant_positive"]:
        notes.append("determinant is not positive; no exact generator exists")
    err = float(np.sum(np.abs(a - expm(q))))
    return Generator(q=q, method=method, l1_error=err, negative_mass=neg, warnings=tuple(notes))


def sub_period_matrix(
    p: ArrayLike, periods_per_year: int, method: GeneratorMethod = "wa", *, tol: float = 1e-6
) -> Array:
    """A matrix for a shorter period (annual to monthly: ``periods_per_year=12``) via a generator."""
    k = positive_int("periods_per_year", periods_per_year)
    return generator(p, method, tol=tol).matrix(1.0 / k)


# ---- credit-cycle shift ------------------------------------------------------------------------


def shift(p: ArrayLike, z: float, rho: ArrayLike, *, tol: float = 1e-6) -> Array:
    """Condition a through-the-cycle matrix on the state of the economy (one-factor model).

    Each row is read as cut-offs of a standard normal variable: ``x_j`` is the point below which the
    obligor ends in state j or worse. Conditional on the systematic factor ``z`` the cut-offs become
    ``(x_j - sqrt(rho) z) / sqrt(1 - rho)``. **z > 0 is a good year** (fewer downgrades and
    defaults), z < 0 a bad one. Averaged over z ~ N(0, 1) the result is the input matrix again;
    note that ``shift(p, 0, rho)`` is *not* the input matrix.

    ``rho`` is the asset correlation, one number or one per row. This is the construction of Belkin,
    Suchower and Forest (1998) with the convention above.
    """
    a = check_matrix(p, tol=tol)
    k = a.shape[0]
    r = as_1d("rho", rho)
    r = np.full(k, r[0]) if r.size == 1 else r
    if r.size != k or np.any(r < 0.0) or np.any(r >= 1.0):
        raise InputError("rho must lie in [0, 1), one value or one per state")
    zz = float(z)
    if not np.isfinite(zz):
        raise InputError("z must be finite")
    tails = np.clip(np.cumsum(a[:, ::-1], axis=1)[:, ::-1], 0.0, 1.0)
    tails[:, 0] = 1.0
    with np.errstate(divide="ignore"):
        x = ndtri(tails)
    cond = ndtr((x - np.sqrt(r)[:, None] * zz) / np.sqrt(1.0 - r)[:, None])
    cond[:, 0] = 1.0
    out = cond - np.concatenate((cond[:, 1:], np.zeros((k, 1))), axis=1)
    out[-1] = 0.0
    out[-1, -1] = 1.0
    return np.asarray(np.clip(out, 0.0, 1.0), dtype=np.float64)
