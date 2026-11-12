"""Reading a loan file.

One row per instrument, comma-separated, with a header. Numbers use a dot as decimal separator and
rates are fractions (0.05 for 5%). Nothing is guessed: an unknown column, an empty required cell or
a value out of range stops the run with the row number.

Required columns
    ``loan_id``, ``balance`` (gross carrying amount), ``eir`` (effective interest rate, annual),
    ``remaining_months``, ``lgd`` and one of ``pd_12m`` (current 12-month PD) or ``grade``.

Optional columns
    ``borrower_id``, ``segment``, ``product`` (term | revolving), ``amortization``
    (bullet | annuity | equal_principal), ``rate`` (contract rate of the repayment schedule, read under
    the ``rate_convention`` setting; without it the schedule uses ``eir`` as an effective rate, so
    that it is consistent with the discounting), ``undrawn``,
    ``ccf`` and ``ccf_12m`` (share of the undrawn limit expected to be drawn over the remaining life
    / within 12 months), ``months_on_book``, ``lgd_default`` (loss rate once credit-impaired),
    origination risk as one of ``pd_origination_remaining`` (lifetime PD expected at origination for
    today's remaining window), ``pd_12m_origination`` or ``grade_origination``, and the status
    columns ``dpd``, ``forborne``, ``watchlist``, ``credit_impaired``, ``collective``,
    ``previous_stage``, ``periods_without_trigger``. With the Turkey profile also ``consumer``,
    ``unlikely_to_pay``, ``in_monitoring_year``, ``restructured_again``, ``judgement_group``.

Figures that are compared or taken over, never recomputed
    ``reported_stage`` and ``reported_ecl`` (the lender's own stage and allowance, for
    :func:`eqlx_ifrs9.review.reconcile`) and ``individual_ecl`` (the allowance of an individually
    assessed credit-impaired loan, e.g. from :mod:`eqlx_ifrs9.individual`; it replaces exposure times
    ``lgd_default`` when the loan is in stage 3).
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._checks import Array
from .errors import InputError

REQUIRED = ("loan_id", "balance", "eir", "remaining_months", "lgd")
NUMERIC: dict[str, tuple[float, float, float]] = {
    # name: (default, min, max); NaN default means "not given"
    "balance": (np.nan, 0.0, 1e18),
    "eir": (np.nan, -0.99, 100.0),
    "remaining_months": (np.nan, 1.0, 1200.0),
    "lgd": (np.nan, 0.0, 1.0),
    "pd_12m": (np.nan, 0.0, 1.0),
    "rate": (np.nan, -0.99, 100.0),
    "undrawn": (0.0, 0.0, 1e18),
    "ccf": (0.0, 0.0, 1.0),
    "ccf_12m": (np.nan, 0.0, 1.0),
    "months_on_book": (0.0, 0.0, 1200.0),
    "lgd_default": (np.nan, 0.0, 1.0),
    "pd_origination_remaining": (np.nan, 0.0, 1.0),
    "pd_12m_origination": (np.nan, 0.0, 1.0),
    "dpd": (0.0, 0.0, 100000.0),
    "previous_stage": (np.nan, 1.0, 3.0),
    "periods_without_trigger": (np.nan, 0.0, 100000.0),
    "judgement_group": (np.nan, 1.0, 5.0),
    "reported_stage": (np.nan, 1.0, 3.0),
    "reported_ecl": (np.nan, 0.0, 1e18),
    "individual_ecl": (np.nan, 0.0, 1e18),
}
WHOLE = {
    "remaining_months",
    "months_on_book",
    "previous_stage",
    "judgement_group",
    "dpd",
    "periods_without_trigger",
    "reported_stage",
}
_NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", re.ASCII)
FLAGS = (
    "forborne",
    "watchlist",
    "credit_impaired",
    "collective",
    "consumer",
    "unlikely_to_pay",
    "in_monitoring_year",
    "restructured_again",
)
TEXT = {"loan_id": "", "borrower_id": "", "segment": "all", "product": "term", "amortization": "bullet"}
TEXT_ALSO = ("grade", "grade_origination")
PRODUCTS = ("term", "revolving")
AMORTIZATIONS = ("bullet", "annuity", "equal_principal")
ALL_COLUMNS = (*TEXT, *TEXT_ALSO, *NUMERIC, *FLAGS)
_TRUE, _FALSE = {"1", "true", "yes", "y", "t"}, {"0", "false", "no", "n", "f", ""}


@dataclass(frozen=True, eq=False)
class Loans:
    n: int
    text: dict[str, NDArray[np.str_]]
    num: dict[str, Array]
    flag: dict[str, NDArray[np.bool_]]
    columns: tuple[str, ...]
    """Columns present in the file, in file order."""

    def has(self, column: str) -> bool:
        return column in self.columns

    def index_of(self, loan_id: str) -> int:
        hits = np.nonzero(self.text["loan_id"] == loan_id)[0]
        if hits.size == 0:
            raise InputError(f"unknown loan_id {loan_id!r}")
        return int(hits[0])

    def row(self, i: int) -> dict[str, object]:
        out: dict[str, object] = {}
        for c in self.columns:
            if c in self.text:
                out[c] = str(self.text[c][i])
            elif c in self.num:
                v = float(self.num[c][i])
                out[c] = None if np.isnan(v) else (int(v) if c in WHOLE else v)
            else:
                out[c] = bool(self.flag[c][i])
        return out

    def take(self, idx: NDArray[np.intp]) -> Loans:
        return Loans(
            int(idx.size),
            {k: v[idx] for k, v in self.text.items()},
            {k: v[idx] for k, v in self.num.items()},
            {k: v[idx] for k, v in self.flag.items()},
            self.columns,
        )


def read_csv(source: str | Path | io.TextIOBase, max_rows: int = 5_000_000) -> Loans:
    """Read a loan file from a path or an open text stream."""
    if isinstance(source, (str, Path)):
        try:
            with open(source, newline="", encoding="utf-8-sig") as fh:
                return _read(fh, max_rows)
        except UnicodeDecodeError as exc:
            raise InputError(f"{source}: the loan file is not UTF-8 text ({exc.reason})") from exc
    return _read(source, max_rows)


def from_text(text: str, max_rows: int = 5_000_000) -> Loans:
    return _read(io.StringIO(text), max_rows)


def _read(fh: io.TextIOBase, max_rows: int) -> Loans:
    reader = csv.reader(fh)
    try:
        header = [h.strip().lstrip("\ufeff").lower() for h in next(reader)]
    except StopIteration:
        raise InputError("the loan file is empty") from None
    if len(set(header)) != len(header):
        raise InputError("the loan file has duplicate column names")
    unknown = [h for h in header if h not in ALL_COLUMNS]
    if unknown:
        raise InputError(f"unknown column(s): {', '.join(unknown)}; allowed: {', '.join(ALL_COLUMNS)}")
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        raise InputError(f"missing required column(s): {', '.join(missing)}")
    if "pd_12m" not in header and "grade" not in header:
        raise InputError("the loan file needs a pd_12m or a grade column")

    cols: dict[str, list[str]] = {h: [] for h in header}
    width = len(header)
    n = 0
    lines: list[int] = []
    for rec in reader:
        if not rec or all(not c.strip() for c in rec):
            continue
        line = reader.line_num
        if len(rec) != width:
            raise InputError(f"row {line}: {len(rec)} cells, the header has {width}")
        n += 1
        lines.append(line)
        if n > max_rows:
            raise InputError(f"the loan file has more than {max_rows} rows")
        for h, cell in zip(header, rec, strict=True):
            cols[h].append(cell.strip())
    if n == 0:
        raise InputError("the loan file has no rows")

    def fail(column: str, i: int, why: str) -> InputError:
        return InputError(f"row {lines[i]}, column {column}: {why}")

    text: dict[str, NDArray[np.str_]] = {}
    for c, default in TEXT.items():
        vals = [v if v else default for v in cols[c]] if c in cols else [default] * n
        text[c] = np.array(vals, dtype=np.str_)
    for c in TEXT_ALSO:
        text[c] = np.array(cols.get(c, [""] * n), dtype=np.str_)
    ids = text["loan_id"]
    for i in np.nonzero(ids == "")[0][:1]:
        raise fail("loan_id", int(i), "empty")
    uniq, counts = np.unique(ids, return_counts=True)
    if uniq.size != n:
        raise InputError(f"loan_id {uniq[counts > 1][0]!r} appears more than once")
    if "borrower_id" not in cols:
        text["borrower_id"] = ids.copy()
    else:
        text["borrower_id"] = np.where(text["borrower_id"] == "", ids, text["borrower_id"])
    text["product"] = np.char.lower(text["product"])
    text["amortization"] = np.char.lower(text["amortization"])
    for c, allowed in (("product", PRODUCTS), ("amortization", AMORTIZATIONS)):
        bad = np.nonzero(~np.isin(text[c], allowed))[0]
        if bad.size:
            raise fail(c, int(bad[0]), f"{text[c][bad[0]]!r} is not one of {', '.join(allowed)}")

    num: dict[str, Array] = {}
    for c, (missing_value, lo, hi) in NUMERIC.items():
        if c not in cols:
            num[c] = np.full(n, missing_value)
            continue
        arr = np.empty(n)
        for i, cell in enumerate(cols[c]):
            if cell == "":
                arr[i] = missing_value
                continue
            if not _NUMBER.fullmatch(cell):
                raise fail(c, i, f"{cell!r} is not a number")
            arr[i] = float(cell)
            if not np.isfinite(arr[i]):
                raise fail(c, i, f"{cell!r} is too large")
        given = ~np.isnan(arr)
        bad = np.nonzero(given & ~((arr >= lo) & (arr <= hi)))[0]
        if bad.size:
            raise fail(c, int(bad[0]), f"{cols[c][bad[0]]} is outside [{lo:g}, {hi:g}]")
        if c in WHOLE:
            frac = np.nonzero(given & (arr != np.floor(arr)))[0]
            if frac.size:
                raise fail(c, int(frac[0]), "must be a whole number")
        num[c] = arr
    for c in REQUIRED[1:]:
        empty = np.nonzero(np.isnan(num[c]))[0]
        if empty.size:
            raise fail(c, int(empty[0]), "empty")
    need_pd = np.nonzero(np.isnan(num["pd_12m"]) & (text["grade"] == ""))[0]
    if need_pd.size:
        raise InputError(f"row {lines[int(need_pd[0])]}: neither pd_12m nor grade is given")

    flag: dict[str, NDArray[np.bool_]] = {}
    for c in FLAGS:
        if c not in cols:
            flag[c] = np.zeros(n, dtype=bool)
            continue
        vals_b = np.zeros(n, dtype=bool)
        for i, cell in enumerate(cols[c]):
            low = cell.lower()
            if low in _TRUE:
                vals_b[i] = True
            elif low not in _FALSE:
                raise fail(c, i, f"{cell!r} is not a yes/no value (use 1 or 0)")
        flag[c] = vals_b
    return Loans(n, text, num, flag, tuple(header))
