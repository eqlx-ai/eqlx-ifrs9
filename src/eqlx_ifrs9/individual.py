"""Individually assessed credit-impaired exposures: what can still be collected from one debtor.

For a credit-impaired asset the allowance is the gross carrying amount minus the present value of
the cash flows still expected, discounted at the original effective interest rate (IFRS 9 B5.5.33),
as a probability-weighted amount over a range of outcomes (5.5.17). The methods here are the ones
the ECB has published for reviewing such files (Asset Quality Review Phase 2 Manual, 2018, section
4.6; Guidance to banks on non-performing loans, 2017, sections 6.2 and 7.4):

- **gone concern**: operating cash flows stop and the collateral is sold;
- **going concern, steady state**: one sustainable year of cash flow times a multiple;
- **going concern, two step**: a year-by-year projection plus a terminal value;
- **market price**: an observable price as a practical expedient.

Cash is shared among all creditors by effective seniority. Everything is arithmetic on the inputs:
whether a business plan is credible, what a property is worth and how the claims rank in law are
judgements that enter as numbers. Where the input departs from a published supervisory expectation
the result carries a warning; nothing is corrected silently.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import InputError

AQR = "ECB Asset Quality Review Phase 2 Manual (2018)"
NPL = "ECB Guidance to banks on non-performing loans (2017)"

# Benchmark multiples of sustainable cash flow (AQR manual 4.6.6.1.3). Debt capacity, not equity value.
BENCHMARK_MULTIPLES = {"infrastructure": 12.0, "utilities": 10.0, "other": 6.0}
MIN_HAIRCUT = 0.10  # auction, long-held foreclosed asset, vendor finance (AQR 4.6.3; NPL guidance 7.4.3)
GONE_CONCERN_ARREARS_MONTHS = 18  # rebuttable presumption (NPL guidance 6.2.4)
MAX_VALUATION_AGE_MONTHS = 12
MIN_PROJECTION_YEARS, MAX_PROJECTION_YEARS = 3, 10  # AQR 4.6.7
MARKET_PRICE_TOLERANCE = 0.10  # AQR 4.6.4.2
APPROACHES = ("gone_concern", "steady_state", "two_step", "market_price")
_EPS = 1e-12


def _num(where: str, value: Any, *, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{where} must be a number, got {value!r}")
    x = float(value)
    if not math.isfinite(x):
        raise InputError(f"{where} must be finite")
    if low is not None and x < low:
        raise InputError(f"{where} must be at least {low:g}, got {x:g}")
    if high is not None and x > high:
        raise InputError(f"{where} must be at most {high:g}, got {x:g}")
    return x


def _rate(where: str, value: Any) -> float:
    x = _num(where, value)
    if x <= -1.0:
        raise InputError(f"{where} must be greater than -100%")
    return x


def _obj(where: str, value: Any, allowed: Sequence[str], required: Sequence[str] = ()) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InputError(f"{where} must be an object")
    unknown = sorted(str(k) for k in value if k not in allowed)
    if unknown:
        raise InputError(f"{where}: unknown key(s) {', '.join(unknown)}")
    missing = [k for k in required if k not in value]
    if missing:
        raise InputError(f"{where}: missing {', '.join(missing)}")
    return dict(value)


def _list(where: str, value: Any) -> list[Any]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise InputError(f"{where} must be a list")
    return list(value)


def _text(where: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{where} must be a non-empty text")
    return value


def _flag(where: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise InputError(f"{where} must be true or false")
    return value


def _rank(where: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InputError(f"{where} must be a whole number of at least 1 (1 is paid first)")
    return value


def _flows(where: str, value: Any) -> list[tuple[float, float]]:
    """``[{"year": 1, "amount": 12.0}, ...]`` or ``[(1, 12.0), ...]`` -> ``[(1.0, 12.0), ...]``."""
    out = []
    for i, item in enumerate(_list(where, value)):
        at = f"{where}[{i}]"
        if isinstance(item, Mapping):
            o = _obj(at, item, ("year", "amount"), ("year", "amount"))
            pair: tuple[Any, Any] = (o["year"], o["amount"])
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            pair = (item[0], item[1])
        else:
            raise InputError(f"{at} must be an object with year and amount")
        out.append((_num(f"{at}.year", pair[0], low=0.0), _num(f"{at}.amount", pair[1], low=0.0)))
    return out


def collateral_recovery(
    value: float,
    eir: float,
    *,
    years_to_sale: float,
    haircut: float,
    value_change: float = 0.0,
    selling_cost_rate: float = 0.0,
    annual_cost_rate: float = 0.0,
    costs: Sequence[Any] = (),
    income: Sequence[Any] = (),
    prior_claims: float = 0.0,
    share: float = 1.0,
) -> dict[str, Any]:
    """Present value of selling one collateral item, and the part of it that belongs to the lender.

    The item is sold ``years_to_sale`` years after the reporting date at ``value * (1 + value_change)``.
    ``haircut`` is the market price discount on that price; ``selling_cost_rate`` is taken on the
    price after the discount; ``annual_cost_rate`` is a holding cost on ``value``, paid at the end of
    each year until the sale (the last part-year pro rata, at the sale). ``costs`` and ``income`` are
    cash amounts at given years. All flows are discounted at ``eir`` to the reporting date.

    ``recoverable = share * max(present value - prior_claims, 0)``: claims that rank before the
    lender are served first, the rest is shared with claims of equal rank.
    """
    v = _num("value", value, low=0.0)
    r = _rate("eir", eir)
    t_sale = _num("years_to_sale", years_to_sale, low=0.0)
    h = _num("haircut", haircut, low=0.0, high=1.0)
    change = _num("value_change", value_change, low=-1.0)
    sell = _num("selling_cost_rate", selling_cost_rate, low=0.0, high=1.0)
    hold = _num("annual_cost_rate", annual_cost_rate, low=0.0)
    prior = _num("prior_claims", prior_claims, low=0.0)
    part = _num("share", share, low=0.0, high=1.0)
    price = v * (1.0 + change)
    rows: list[tuple[float, str, float]] = [(t_sale, "sale at market value", price)]
    if h:
        rows.append((t_sale, "market price discount", -price * h))
    if sell:
        rows.append((t_sale, "selling costs", -price * (1.0 - h) * sell))
    if hold:
        whole = math.floor(t_sale + _EPS)
        rows.extend((float(k), "holding costs", -v * hold) for k in range(1, whole + 1))
        if t_sale - whole > _EPS:
            rows.append((t_sale, "holding costs", -v * hold * (t_sale - whole)))
    for name, sign, items in (("costs", -1.0, costs), ("income", 1.0, income)):
        for t, amount in _flows(name, items):
            if t > t_sale + _EPS:
                raise InputError(f"{name}: year {t:g} is after the sale at year {t_sale:g}")
            rows.append((t, name, sign * amount))
    rows.sort(key=lambda row: row[0])
    flows = [
        {"year": t, "item": item, "amount": amount, "present_value": amount * (1.0 + r) ** (-t)}
        for t, item, amount in rows
    ]
    pv = math.fsum(f["present_value"] for f in flows)
    return {
        "present_value": pv,
        "prior_claims": prior,
        "share": part,
        "recoverable": part * max(pv - prior, 0.0),
        "flows": flows,
    }


def sustainable_cash_flow(
    *,
    ebitda: float | None = None,
    revenue: float | None = None,
    cost_of_sales: float = 0.0,
    distribution_costs: float = 0.0,
    administrative_expenses: float = 0.0,
    payroll_taxes: float = 0.0,
    other_gains_losses: float = 0.0,
    income_tax: float | None = None,
    tax_rate: float | None = None,
    net_interest_expense: float = 0.0,
    depreciation: float = 0.0,
    owner_payout: float = 0.0,
    capex: float = 0.0,
    sustainability_deductions: float = 0.0,
) -> dict[str, float]:
    """One sustainable year of the debtor's operating cash flow (AQR manual 4.6.6.1).

    ``EBITDA - income tax - owner's remuneration or essential dividends - required capex -
    sustainability deductions``. Give ``ebitda`` directly, or ``revenue`` with the cost lines
    (positive numbers, deducted; ``other_gains_losses`` is signed). Give ``income_tax`` directly, or
    ``tax_rate``: the tax is then the rate on ``EBITDA - net interest expense - depreciation`` when
    that is positive. Sustainability deductions neutralise items that flatter the year, such as
    reversals of provisions.
    """
    if (ebitda is None) == (revenue is None):
        raise InputError("give either ebitda or revenue with the cost lines")
    lines = {
        "cost_of_sales": cost_of_sales,
        "distribution_costs": distribution_costs,
        "administrative_expenses": administrative_expenses,
        "payroll_taxes": payroll_taxes,
    }
    if ebitda is not None:
        if any(lines.values()) or other_gains_losses:
            raise InputError("cost lines belong with revenue, not with a given ebitda")
        e = _num("ebitda", ebitda)
    else:
        costs = math.fsum(_num(k, x, low=0.0) for k, x in lines.items())
        e = _num("revenue", revenue) - costs + _num("other_gains_losses", other_gains_losses)
    if (income_tax is None) == (tax_rate is None):
        raise InputError("give either income_tax or tax_rate")
    pbt = e - _num("net_interest_expense", net_interest_expense) - _num("depreciation", depreciation, low=0.0)
    if income_tax is not None:
        tax = _num("income_tax", income_tax, low=0.0)
    else:
        tax = max(pbt, 0.0) * _num("tax_rate", tax_rate, low=0.0, high=1.0)
    payout = _num("owner_payout", owner_payout, low=0.0)
    cap = _num("capex", capex, low=0.0)
    sustain = _num("sustainability_deductions", sustainability_deductions, low=0.0)
    out = {} if income_tax is not None else {"profit_before_tax": pbt}
    return {
        "ebitda": e,
        **out,
        "income_tax": tax,
        "owner_payout": payout,
        "capex": cap,
        "cash_flow_adjustment": -(tax + payout + cap),
        "sustainability_adjustment": -sustain,
        "sustainable_cash_flow": e - tax - payout - cap - sustain,
    }


def allocate(amount: float, exposure: float, preferred_claims: float = 0.0, pari_passu_claims: float = 0.0) -> float:
    """Part of ``amount`` that reaches one exposure (AQR manual 4.6.6.2).

    ``min(max(amount - preferred claims, 0) / (exposure + pari passu claims), 1) * exposure``
    """
    a = _num("amount", amount, low=0.0)
    e = _num("exposure", exposure, low=0.0)
    pref = _num("preferred_claims", preferred_claims, low=0.0)
    pari = _num("pari_passu_claims", pari_passu_claims, low=0.0)
    if e == 0.0:
        return 0.0
    return min(max(a - pref, 0.0) / (e + pari), 1.0) * e


def waterfall(amount: float, claims: Sequence[tuple[int, float]]) -> list[float]:
    """Share ``amount`` among ``(rank, claim)`` pairs: rank 1 is paid first, equal ranks pro rata."""
    left = _num("amount", amount, low=0.0)
    ranks = [_rank(f"claims[{i}] rank", r) for i, (r, _) in enumerate(claims)]
    owed = [_num(f"claims[{i}] amount", c, low=0.0) for i, (_, c) in enumerate(claims)]
    paid = [0.0] * len(owed)
    for rank in sorted(set(ranks)):
        idx = [i for i, r in enumerate(ranks) if r == rank]
        total = math.fsum(owed[i] for i in idx)
        if total <= 0.0:
            continue
        ratio = min(left / total, 1.0)
        for i in idx:
            paid[i] = owed[i] * ratio
        left = max(left - total * ratio, 0.0)
    return paid


_SALE_KEYS = (
    "years",
    "haircut",
    "value_change",
    "selling_cost_rate",
    "annual_cost_rate",
    "costs",
    "income",
    "auction",
    "vendor_finance",
    "foreclosed_months_ago",
)
_CASH_FLOW_KEYS = (
    "ebitda",
    "revenue",
    "cost_of_sales",
    "distribution_costs",
    "administrative_expenses",
    "payroll_taxes",
    "other_gains_losses",
    "income_tax",
    "tax_rate",
    "net_interest_expense",
    "depreciation",
    "owner_payout",
    "capex",
    "sustainability_deductions",
)
_GOING = ("steady_state", "two_step")


def _warning(code: str, where: str, message: str, source: str) -> dict[str, str]:
    return {"code": code, "where": where, "message": message, "source": source}


def _one_year(where: str, sc: Mapping[str, Any]) -> tuple[float, dict[str, float] | None]:
    """Sustainable cash flow of a scenario or terminal value: a number, or built from the accounts."""
    if ("sustainable_cash_flow" in sc) == ("cash_flow" in sc):
        raise InputError(f"{where}: give either sustainable_cash_flow or cash_flow")
    if "sustainable_cash_flow" in sc:
        return _num(f"{where}.sustainable_cash_flow", sc["sustainable_cash_flow"]), None
    args = _obj(f"{where}.cash_flow", sc["cash_flow"], _CASH_FLOW_KEYS)
    try:
        built = sustainable_cash_flow(**args)
    except InputError as exc:
        raise InputError(f"{where}.cash_flow: {exc}") from exc
    return built["sustainable_cash_flow"], built


def _check_sale(where: str, sale: Mapping[str, Any]) -> None:
    """Sale assumptions are checked where they are given, also when no scenario ends up using them."""
    try:
        collateral_recovery(
            0.0,
            0.0,
            years_to_sale=sale["years"],
            haircut=sale["haircut"],
            value_change=sale.get("value_change", 0.0),
            selling_cost_rate=sale.get("selling_cost_rate", 0.0),
            annual_cost_rate=sale.get("annual_cost_rate", 0.0),
            costs=sale.get("costs", []),
            income=sale.get("income", []),
        )
    except InputError as exc:
        raise InputError(f"{where}: {exc}") from exc
    for key in ("auction", "vendor_finance"):
        _flag(f"{where}.{key}", sale.get(key, False))
    if sale.get("foreclosed_months_ago") is not None:
        _num(f"{where}.foreclosed_months_ago", sale["foreclosed_months_ago"], low=0.0)


def assess(case: Mapping[str, Any]) -> dict[str, Any]:
    """Allowance of every exposure to one debtor, from a case file.

    The case has a ``debtor``, the lender's ``exposures``, the ``other_claims`` on the same debtor,
    the ``collateral`` and one or more ``scenarios`` with probabilities that add up to 1. Each
    scenario names its approach: ``gone_concern``, ``steady_state``, ``two_step`` or
    ``market_price``. The result gives, per exposure and scenario, the cash flow value, the
    collateral value, the recoverable amount (never more than the exposure) and the shortfall; the
    allowance is the probability-weighted shortfall. See the README for a complete example.
    """
    top = _obj(
        "case",
        case,
        ("debtor", "exposures", "other_claims", "collateral", "scenarios"),
        ("debtor", "exposures", "scenarios"),
    )
    debtor = _obj("debtor", top["debtor"], ("id", "sector", "months_past_due", "ebitda_history"), ("id",))
    _text("debtor.id", debtor["id"])
    sector = debtor.get("sector")
    if sector is not None and sector not in BENCHMARK_MULTIPLES:
        raise InputError(f"debtor.sector must be one of {', '.join(BENCHMARK_MULTIPLES)}")
    benchmark = BENCHMARK_MULTIPLES[sector or "other"]
    warnings: list[dict[str, str]] = []

    # --- exposures -------------------------------------------------------------------------
    exposures: list[dict[str, Any]] = []
    for i, raw in enumerate(_list("exposures", top["exposures"])):
        at = f"exposures[{i}]"
        e = _obj(
            at,
            raw,
            ("id", "gross_carrying_amount", "eir", "rank", "expected_draws", "market_price"),
            ("id", "gross_carrying_amount", "eir"),
        )
        gross = _num(f"{at}.gross_carrying_amount", e["gross_carrying_amount"], low=0.0)
        eir = _rate(f"{at}.eir", e["eir"])
        draws = _flows(f"{at}.expected_draws", e.get("expected_draws", []))
        undrawn_pv = math.fsum(a * (1.0 + eir) ** (-t) for t, a in draws)
        exposures.append(
            {
                "id": _text(f"{at}.id", e["id"]),
                "gross": gross,
                "eir": eir,
                "rank": None if "rank" not in e else _rank(f"{at}.rank", e["rank"]),
                "draws": draws,
                "undrawn_pv": undrawn_pv,
                "total": gross + undrawn_pv,
                "price": None if "market_price" not in e else _num(f"{at}.market_price", e["market_price"], low=0.0),
            }
        )
    if not exposures:
        raise InputError("exposures is empty")
    ids = [e["id"] for e in exposures]
    if len(set(ids)) != len(ids):
        raise InputError("exposure ids must be unique")
    by_id = {e["id"]: e for e in exposures}

    others: list[tuple[str, int, float]] = []
    for i, raw in enumerate(_list("other_claims", top.get("other_claims", []))):
        at = f"other_claims[{i}]"
        o = _obj(at, raw, ("name", "rank", "amount"), ("name", "rank", "amount"))
        others.append(
            (
                _text(f"{at}.name", o["name"]),
                _rank(f"{at}.rank", o["rank"]),
                _num(f"{at}.amount", o["amount"], low=0.0),
            )
        )

    # --- collateral ------------------------------------------------------------------------
    collateral: list[dict[str, Any]] = []
    for i, raw in enumerate(_list("collateral", top.get("collateral", []))):
        at = f"collateral[{i}]"
        c = _obj(
            at,
            raw,
            (
                "id",
                "type",
                "value",
                "valuation_age_months",
                "secures",
                "central_to_cash_flows",
                "sale",
                "share",
                "prior_claims",
                "pari_passu_claims",
            ),
            ("id", "value", "secures", "central_to_cash_flows", "sale"),
        )
        cid = _text(f"{at}.id", c["id"])
        secures = c["secures"]
        if isinstance(secures, str):
            split = {secures: 1.0}
        else:
            split = {
                str(k): _num(f"{at}.secures[{k}]", x, low=0.0, high=1.0)
                for k, x in _obj(f"{at}.secures", secures, ids).items()
            }
            if not split or abs(math.fsum(split.values()) - 1.0) > 1e-9:
                raise InputError(f"{at}.secures: the shares must add up to 1")
        for k in split:
            if k not in by_id:
                raise InputError(f"{at}.secures: no exposure with id {k!r}")
        if "share" in c and ("prior_claims" in c or "pari_passu_claims" in c):
            raise InputError(f"{at}: give either share or prior_claims / pari_passu_claims")
        prior = _num(f"{at}.prior_claims", c.get("prior_claims", 0.0), low=0.0)
        if "share" in c:
            share = _num(f"{at}.share", c["share"], low=0.0, high=1.0)
        else:
            pari = _num(f"{at}.pari_passu_claims", c.get("pari_passu_claims", 0.0), low=0.0)
            own = math.fsum(by_id[k]["gross"] for k in split)
            share = own / (own + pari) if own + pari > 0.0 else 0.0
        age = c.get("valuation_age_months")
        if age is not None and _num(f"{at}.valuation_age_months", age, low=0.0) > MAX_VALUATION_AGE_MONTHS:
            warnings.append(
                _warning(
                    "stale_valuation",
                    cid,
                    f"the valuation is {age:g} months old; older than {MAX_VALUATION_AGE_MONTHS} months it should "
                    "be renewed or indexed to the date of sale",
                    f"{AQR}, section 5",
                )
            )
        base = _obj(f"{at}.sale", c["sale"], _SALE_KEYS, ("years", "haircut"))
        _check_sale(f"{at}.sale", base)
        collateral.append(
            {
                "id": cid,
                "value": _num(f"{at}.value", c["value"], low=0.0),
                "split": split,
                "central": _flag(f"{at}.central_to_cash_flows", c["central_to_cash_flows"]),
                "sale": base,
                "share": share,
                "prior": prior,
            }
        )
    cids = [c["id"] for c in collateral]
    if len(set(cids)) != len(cids):
        raise InputError("collateral ids must be unique")

    def sell(c: dict[str, Any], sc_name: str, override: Mapping[str, Any]) -> dict[str, Any]:
        """Collateral item ``c`` under one scenario: per secured exposure, discounted at its own EIR."""
        sale = {**c["sale"], **override}
        haircut = _num(f"{c['id']}.sale.haircut", sale["haircut"], low=0.0, high=1.0)
        change = _num(f"{c['id']}.sale.value_change", sale.get("value_change", 0.0), low=-1.0)
        where = f"{c['id']} / {sc_name}"
        forced = [
            label
            for key, label in (("auction", "sold by auction"), ("vendor_finance", "sold with vendor finance"))
            if _flag(f"{c['id']}.sale.{key}", sale.get(key, False))
        ]
        held = sale.get("foreclosed_months_ago")
        if held is not None and _num(f"{c['id']}.sale.foreclosed_months_ago", held, low=0.0) >= 24:
            forced.append("foreclosed two years ago and not sold")
        if forced and haircut < MIN_HAIRCUT - _EPS:
            warnings.append(
                _warning(
                    "low_haircut",
                    where,
                    f"market price discount of {haircut:.0%} on collateral {' and '.join(forced)}; at least "
                    f"{MIN_HAIRCUT:.0%} is expected unless the discount is already inside the costs given",
                    f"{AQR}, 4.6.3; {NPL}, 7.4.3",
                )
            )
        elif haircut == 0.0:
            warnings.append(
                _warning(
                    "no_haircut",
                    where,
                    "no market price discount: this needs evidence that the collateral is liquid and not "
                    "distressed, ideally a binding offer",
                    f"{AQR}, 4.6.3, footnote 37",
                )
            )
        if change > 0.0:
            warnings.append(
                _warning(
                    "value_increase_assumed",
                    where,
                    f"the value is assumed to rise by {change:.1%} before the sale; expected falls are to be "
                    "included, rises are not",
                    f"{AQR}, 4.6.3; {NPL}, 7.4.3",
                )
            )
        parts = {}
        for eid, fraction in c["split"].items():
            parts[eid] = collateral_recovery(
                c["value"] * fraction,
                by_id[eid]["eir"],
                years_to_sale=sale["years"],
                haircut=haircut,
                value_change=change,
                selling_cost_rate=sale.get("selling_cost_rate", 0.0),
                annual_cost_rate=sale.get("annual_cost_rate", 0.0),
                costs=[(t, a * fraction) for t, a in _flows(f"{c['id']}.sale.costs", sale.get("costs", []))],
                income=[(t, a * fraction) for t, a in _flows(f"{c['id']}.sale.income", sale.get("income", []))],
                prior_claims=c["prior"] * fraction,
                share=c["share"],
            )
        return {
            "id": c["id"],
            "years_to_sale": float(sale["years"]),
            "haircut": haircut,
            "share": c["share"],
            "by_exposure": parts,
        }

    def ranked(where: str) -> list[tuple[int, float]]:
        """All claims as (rank, amount): the lender's exposures first, then the other creditors."""
        missing = [e["id"] for e in exposures if e["rank"] is None]
        if missing:
            raise InputError(f"{where}: sharing cash among creditors needs a rank for {', '.join(missing)}")
        return [(e["rank"], e["total"]) for e in exposures] + [(r, a) for _, r, a in others]

    def check_multiple(m: float, where: str) -> None:
        if m > benchmark + _EPS:
            warnings.append(
                _warning(
                    "multiple_above_benchmark",
                    where,
                    f"multiple {m:g} is above the benchmark of {benchmark:g} for "
                    f"{sector or 'other'} exposures; a higher debt-capacity multiple needs an explanation",
                    f"{AQR}, 4.6.6.1.3",
                )
            )

    # --- scenarios -------------------------------------------------------------------------
    scenarios = _list("scenarios", top["scenarios"])
    if not scenarios:
        raise InputError("scenarios is empty")
    results: list[dict[str, Any]] = []
    names: list[str] = []
    for i, raw in enumerate(scenarios):
        at = f"scenarios[{i}]"
        if not isinstance(raw, Mapping) or raw.get("approach") not in APPROACHES:
            raise InputError(f"{at}.approach must be one of {', '.join(APPROACHES)}")
        approach = raw["approach"]
        common = ("name", "probability", "approach", "collateral")
        extra = {
            "gone_concern": ("unsecured_recovery",),
            "steady_state": (
                "sustainable_cash_flow",
                "cash_flow",
                "multiple",
                "non_pledged_cash",
                "non_pledged_financial_assets",
            ),
            "two_step": ("cash_flows", "terminal"),
            "market_price": ("price",),
        }[approach]
        sc = _obj(at, raw, (*common, *extra), ("name", "probability"))
        name = _text(f"{at}.name", sc["name"])
        names.append(name)
        prob = _num(f"{at}.probability", sc["probability"], low=0.0, high=1.0)
        overrides = _obj(f"{at}.collateral", sc.get("collateral", {}), cids)
        for cid, o in overrides.items():
            merged = {**collateral[cids.index(cid)]["sale"], **_obj(f"{at}.collateral.{cid}", o, _SALE_KEYS)}
            _check_sale(f"{at}.collateral.{cid}", merged)
        cash = dict.fromkeys(ids, 0.0)
        detail: dict[str, Any] = {}

        if approach in _GOING:
            arrears = debtor.get("months_past_due")
            if arrears is not None and _num("debtor.months_past_due", arrears, low=0.0) > GONE_CONCERN_ARREARS_MONTHS:
                warnings.append(
                    _warning(
                        "gone_concern_presumed",
                        name,
                        f"{arrears:g} months past due: beyond {GONE_CONCERN_ARREARS_MONTHS} months the allowance "
                        "is presumed to be estimated on a gone-concern basis unless that is rebutted",
                        f"{NPL}, 6.2.4",
                    )
                )
            history = [
                _num("debtor.ebitda_history", x) for x in _list("ebitda_history", debtor.get("ebitda_history", []))
            ]
            if len(history) >= 2 and history[-1] < 0.0 and history[-2] < 0.0:
                warnings.append(
                    _warning(
                        "negative_ebitda",
                        name,
                        "EBITDA was negative in the two previous years, which points to cash flows too uncertain "
                        "for a going-concern estimate",
                        f"{NPL}, 6.2.4",
                    )
                )

        if approach == "gone_concern":
            if "unsecured_recovery" in sc:
                u = _obj(f"{at}.unsecured_recovery", sc["unsecured_recovery"], ("amount", "year"), ("amount", "year"))
                amount = _num(f"{at}.unsecured_recovery.amount", u["amount"], low=0.0)
                year = _num(f"{at}.unsecured_recovery.year", u["year"], low=0.0)
                paid = waterfall(amount, ranked(at))
                for e, p in zip(exposures, paid, strict=False):
                    cash[e["id"]] = p * (1.0 + e["eir"]) ** (-year)
                detail["unsecured_recovery"] = {"amount": amount, "year": year}
        elif approach == "steady_state":
            cfs, built = _one_year(at, sc)
            if "multiple" not in sc:
                raise InputError(f"{at}: missing multiple")
            multiple = _num(f"{at}.multiple", sc["multiple"], low=0.0)
            check_multiple(multiple, name)
            free_cash = _num(f"{at}.non_pledged_cash", sc.get("non_pledged_cash", 0.0), low=0.0)
            free_assets = _num(
                f"{at}.non_pledged_financial_assets", sc.get("non_pledged_financial_assets", 0.0), low=0.0
            )
            pv_operating = max(cfs, 0.0) * multiple
            total = pv_operating + free_cash + free_assets
            paid = waterfall(total, ranked(at))
            for e, p in zip(exposures, paid, strict=False):
                cash[e["id"]] = p
            detail = {
                "sustainable_cash_flow": cfs,
                "multiple": multiple,
                "present_value_of_operating_cash_flows": pv_operating,
                "amount_for_allocation": total,
            }
            if built is not None:
                detail["cash_flow"] = built
        elif approach == "two_step":
            if "cash_flows" not in sc:
                raise InputError(f"{at}: missing cash_flows")
            flows = [
                _num(f"{at}.cash_flows[{k}]", x) for k, x in enumerate(_list(f"{at}.cash_flows", sc["cash_flows"]))
            ]
            if not flows:
                raise InputError(f"{at}.cash_flows is empty")
            if not MIN_PROJECTION_YEARS <= len(flows) <= MAX_PROJECTION_YEARS:
                warnings.append(
                    _warning(
                        "projection_length",
                        name,
                        f"the projection covers {len(flows)} years; it should cover at least "
                        f"{MIN_PROJECTION_YEARS} and not more than {MAX_PROJECTION_YEARS}",
                        f"{AQR}, 4.6.7",
                    )
                )
            terminal = 0.0
            if "terminal" in sc:
                t = _obj(
                    f"{at}.terminal",
                    sc["terminal"],
                    ("liquidation_value", "sustainable_cash_flow", "cash_flow", "multiple"),
                )
                if "liquidation_value" in t:
                    if len(t) != 1:
                        raise InputError(f"{at}.terminal: liquidation_value stands alone")
                    terminal = _num(f"{at}.terminal.liquidation_value", t["liquidation_value"], low=0.0)
                    detail["terminal"] = {"liquidation_value": terminal}
                else:
                    cfs, built = _one_year(f"{at}.terminal", t)
                    if "multiple" not in t:
                        raise InputError(f"{at}.terminal: missing multiple")
                    multiple = _num(f"{at}.terminal.multiple", t["multiple"], low=0.0)
                    check_multiple(multiple, f"{name} / terminal value")
                    terminal = max(cfs, 0.0) * multiple
                    end: dict[str, Any] = {"sustainable_cash_flow": cfs, "multiple": multiple, "value": terminal}
                    if built is not None:
                        end["cash_flow"] = built
                    detail["terminal"] = end
            claims = ranked(at)
            balance = [e["gross"] for e in exposures] + [a for _, _, a in others]
            rank_of = [r for r, _ in claims]
            years = []
            for yr, flow in enumerate(flows, start=1):
                for j, e in enumerate(exposures):
                    balance[j] += math.fsum(a for t, a in e["draws"] if yr - 1 < t <= yr or (yr == 1 and t == 0.0))
                available = max(flow, 0.0) + (terminal if yr == len(flows) else 0.0)
                paid = waterfall(available, list(zip(rank_of, balance, strict=True)))
                balance = [max(b - p, 0.0) for b, p in zip(balance, paid, strict=True)]
                row: dict[str, Any] = {"year": yr, "cash_flow": flow, "available": available, "to_lender": {}}
                for j, e in enumerate(exposures):
                    cash[e["id"]] += paid[j] * (1.0 + e["eir"]) ** (-yr)
                    row["to_lender"][e["id"]] = paid[j]
                row["to_other_creditors"] = math.fsum(paid[len(exposures) :])
                years.append(row)
            late = [e["id"] for e in exposures if any(t > len(flows) for t, _ in e["draws"])]
            if late:
                raise InputError(f"{at}: expected_draws of {', '.join(late)} fall after the last projected year")
            detail["years"] = years
        else:
            if "price" not in sc:
                raise InputError(f"{at}: missing price")
            price = _num(f"{at}.price", sc["price"], low=0.0)
            for e in exposures:
                cash[e["id"]] = price * e["gross"]
            detail = {"price": price}

        sold = []
        coll = dict.fromkeys(ids, 0.0)
        if approach != "market_price":
            for c in collateral:
                if approach in _GOING and c["central"]:
                    continue
                item = sell(c, name, overrides.get(c["id"], {}))
                for eid, part in item["by_exposure"].items():
                    coll[eid] += part["recoverable"]
                sold.append(item)
        per_exposure = {}
        for e in exposures:
            gross_recoverable = cash[e["id"]] + coll[e["id"]]
            recoverable = min(gross_recoverable, e["total"])
            per_exposure[e["id"]] = {
                "cash_flow_value": cash[e["id"]],
                "collateral_value": coll[e["id"]],
                "recoverable": recoverable,
                "capped_at_exposure": gross_recoverable > e["total"],
                "shortfall": e["total"] - recoverable,
            }
            if approach in _GOING and e["price"] is not None:
                at_price = e["price"] * e["gross"]
                if recoverable > at_price * (1.0 + MARKET_PRICE_TOLERANCE) + _EPS:
                    warnings.append(
                        _warning(
                            "above_market_price",
                            f"{e['id']} / {name}",
                            f"the going-concern recoverable amount {recoverable:,.2f} is more than "
                            f"{MARKET_PRICE_TOLERANCE:.0%} above the observable market price {at_price:,.2f}",
                            f"{AQR}, 4.6.4.2",
                        )
                    )
        results.append(
            {
                "name": name,
                "probability": prob,
                "approach": approach,
                "detail": detail,
                "collateral_sold": sold,
                "exposures": per_exposure,
            }
        )

    if len(set(names)) != len(names):
        raise InputError("scenario names must be unique")
    weight = math.fsum(r["probability"] for r in results)
    if abs(weight - 1.0) > 1e-9:
        raise InputError(f"scenario probabilities add up to {weight:g}, not 1")
    if len(results) == 1:
        warnings.append(
            _warning(
                "single_scenario",
                names[0],
                "one scenario only: expected credit losses are a probability-weighted amount over a range of "
                "outcomes, including the one that is less likely",
                "IFRS 9 5.5.17(a), 5.5.18",
            )
        )

    out_exposures = []
    for e in exposures:
        ecl = math.fsum(r["probability"] * r["exposures"][e["id"]]["shortfall"] for r in results)
        on_loan = ecl * e["gross"] / e["total"] if e["total"] > 0.0 else 0.0
        out_exposures.append(
            {
                "id": e["id"],
                "gross_carrying_amount": e["gross"],
                "present_value_of_expected_draws": e["undrawn_pv"],
                "exposure": e["total"],
                "ecl": ecl,
                "ecl_on_loan": on_loan,
                "ecl_on_commitment": ecl - on_loan,
                "coverage": ecl / e["total"] if e["total"] > 0.0 else 0.0,
                "scenarios": [
                    {"name": r["name"], "probability": r["probability"], "approach": r["approach"]}
                    | r["exposures"][e["id"]]
                    for r in results
                ],
            }
        )
    exposure = math.fsum(x["exposure"] for x in out_exposures)
    ecl_total = math.fsum(x["ecl"] for x in out_exposures)
    unique: list[dict[str, str]] = []
    for w in warnings:
        if not any(u["code"] == w["code"] and u["where"] == w["where"] for u in unique):
            unique.append(w)
    return {
        "debtor": debtor["id"],
        "total": {"exposure": exposure, "ecl": ecl_total, "coverage": ecl_total / exposure if exposure > 0.0 else 0.0},
        "exposures": out_exposures,
        "scenarios": [
            {
                "name": r["name"],
                "probability": r["probability"],
                "approach": r["approach"],
                "detail": r["detail"],
                "collateral_sold": r["collateral_sold"],
            }
            for r in results
        ],
        "warnings": unique,
    }
