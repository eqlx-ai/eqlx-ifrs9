"""Portfolio calculation: stage and expected credit loss for every loan in a file.

The steps, in order:

1. A PD curve per loan and scenario, monthly, over the remaining life. From ``pd_12m`` as a constant
   hazard, or from the loan's grade and the segment's transition matrix. Scenario paths shift it
   (see :mod:`eqlx_ifrs9.cycle` and :func:`eqlx_ifrs9.transitions.shift`).
2. Exposure per month from the repayment type, plus the expected drawdown of undrawn limits.
3. 12-month and lifetime ECL per scenario; probability-weighted across scenarios.
4. Stage from the staging policy, comparing today's PD with the PD expected at origination for the
   same remaining window. By default the scenario-weighted PD is used, so that staging and
   measurement see the same economy.
5. Reported ECL: 12-month ECL in stage 1, lifetime ECL in stage 2, exposure times the loss rate in
   default in stage 3.

Arithmetic is done on arrays, in chunks. :func:`loan_detail` recomputes one loan with the scalar
building blocks of the library; the test-suite checks that both routes agree.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr, ndtri

from . import cycle, schedule, transitions
from ._checks import Array
from .config import Config, Scenario, Segment
from .curves import PDCurve
from .data import Loans
from .ecl import expected_credit_loss
from .errors import InputError
from .profiles import tr_bddk
from .staging import Reason, StagingPolicy, StagingResult, assess, finish

CHUNK_CELLS = 2_000_000
MONTHS = 12


@dataclass(frozen=True, eq=False)
class PortfolioResult:
    loans: Loans
    config: Config
    policy: StagingPolicy
    """The staging policy actually applied (a profile may override day limits)."""
    stage: NDArray[np.int8]
    reasons: NDArray[np.int64]
    primary: NDArray[np.int64]
    pd_now: Array
    pd_origination: Array
    annual_pd_now: Array
    annual_pd_origination: Array
    exposure: Array
    """Gross carrying amount plus the expected drawdown of the undrawn limit over the remaining life."""
    ecl: Array
    ecl_12m: Array
    ecl_lifetime: Array
    ecl_default: Array
    scenario_ecl_12m: Array
    scenario_ecl_lifetime: Array
    scenario_pd_lifetime: Array
    scenario_pd_12m: Array
    scenario_stage: NDArray[np.int8]
    """Stage each loan would have if one scenario alone had full weight (scenarios x loans)."""
    group: NDArray[np.int8] | None
    profile_reasons: tuple[tuple[str, ...], ...] | None
    warnings: tuple[str, ...]

    @property
    def n(self) -> int:
        return self.loans.n

    def ecl_for_stages(self, stage: NDArray[np.int8], scenario: int | None = None) -> Array:
        """ECL per loan under a given stage vector, weighted or for one scenario."""
        if scenario is None:
            e12, life = self.ecl_12m, self.ecl_lifetime
        else:
            e12, life = self.scenario_ecl_12m[scenario], self.scenario_ecl_lifetime[scenario]
        return np.asarray(np.where(stage == 1, e12, np.where(stage == 2, life, self.ecl_default)), dtype=np.float64)


# ---- scenario paths ----------------------------------------------------------------------------


def scenario_years(config: Config, scenario: Scenario, n_years: int) -> tuple[Array, Array]:
    """z and the weight of the shifted PD for each of ``n_years`` years of a scenario."""
    if scenario.z is None:
        return np.zeros(n_years), np.zeros(n_years)
    z = np.asarray(scenario.z, dtype=np.float64)
    horizon = int(z.size)
    z = np.concatenate((z, np.full(max(0, n_years - horizon), z[-1])))[:n_years]
    w = cycle.reversion_weights(
        n_years, horizon, config.reversion_years, config.reversion_method, config.reversion_decay
    )
    return z, w


def _monthly_matrices(seg: Segment, z: Array, w: Array) -> list[Array]:
    """One monthly transition matrix per year of the scenario path."""
    assert seg.matrix is not None
    out: list[Array] = []
    cache: dict[tuple[float, float], Array] = {}
    for zy, wy in zip(z, w, strict=True):
        key = (float(zy), float(wy)) if wy > 0.0 else (0.0, 0.0)
        if key not in cache:
            annual = seg.matrix
            if wy > 0.0:
                annual = (
                    wy * transitions.shift(seg.matrix, float(zy), seg.rho, tol=seg.matrix_tolerance)
                    + (1.0 - wy) * seg.matrix
                )
            cache[key] = transitions.generator(annual, seg.generator, tol=max(seg.matrix_tolerance, 1e-9)).matrix(
                1.0 / MONTHS
            )
        out.append(cache[key])
    return out


def _matrix_table(monthly: list[Array], n_months: int) -> Array:
    """Cumulative PD by starting grade and month: (states, n_months)."""
    k = monthly[0].shape[0]
    out = np.empty((k, n_months))
    prod = np.eye(k)
    for t in range(n_months):
        prod = prod @ monthly[min(t // MONTHS, len(monthly) - 1)]
        out[:, t] = prod[:, -1]
    return np.clip(np.maximum.accumulate(out, axis=1), 0.0, 1.0)


# ---- loan attributes ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Prepared:
    segments: list[Segment]
    seg_index: NDArray[np.intp]
    rho: Array
    cpr: Array
    use_matrix: NDArray[np.bool_]
    grade: NDArray[np.int64]
    """0-based grade index in the segment's matrix; -1 if the loan has no usable grade."""
    grade_origination: NDArray[np.int64]
    rem: NDArray[np.int64]
    mob: NDArray[np.int64]
    rate_month: Array
    pd_long_run: Array
    """pd_12m read as a long-run PD (NaN where the loan uses a matrix)."""
    ccf_12m: Array
    lgd_default: Array
    warnings: list[str]


def _prepare(loans: Loans, config: Config) -> _Prepared:
    names = sorted(set(loans.text["segment"].tolist()))
    segments = [config.segment(nm) for nm in names]
    pos = {nm: i for i, nm in enumerate(names)}
    seg_index = np.array([pos[s] for s in loans.text["segment"].tolist()], dtype=np.intp)
    rho = np.array([s.rho for s in segments])[seg_index]
    cpr = np.array([s.cpr for s in segments])[seg_index]
    n = loans.n
    grade = np.full(n, -1, dtype=np.int64)
    grade_orig = np.full(n, -1, dtype=np.int64)
    warnings: list[str] = []
    for si, seg in enumerate(segments):
        rows = np.nonzero(seg_index == si)[0]
        if not seg.grades:
            continue
        lookup = {g: i for i, g in enumerate(seg.grades)}
        for col, target in (("grade", grade), ("grade_origination", grade_orig)):
            labels = loans.text[col][rows]
            for r, lab in zip(rows.tolist(), labels.tolist(), strict=True):
                if lab == "":
                    continue
                if lab not in lookup:
                    raise InputError(
                        f"loan {loans.text['loan_id'][r]}: {col} {lab!r} is not a grade of segment {seg.name!r}"
                    )
                target[r] = lookup[lab]
    has_pd = ~np.isnan(loans.num["pd_12m"])
    has_matrix = np.array([s.matrix is not None for s in segments])[seg_index]
    use_matrix = ~has_pd & has_matrix & (grade >= 0)
    stuck = np.nonzero(~has_pd & ~use_matrix)[0]
    if stuck.size:
        i = int(stuck[0])
        raise InputError(
            f"loan {loans.text['loan_id'][i]}: no pd_12m, and segment {loans.text['segment'][i]!r} has no "
            "transition matrix for its grade"
        )
    if np.any(has_pd & has_matrix & (grade >= 0)):
        warnings.append("some loans have both pd_12m and a grade with a matrix: pd_12m was used for the PD curve")
    eir = loans.num["eir"]
    rate = np.where(np.isnan(loans.num["rate"]), eir, loans.num["rate"])
    rate_month = rate / MONTHS if config.rate_convention == "nominal" else (1.0 + rate) ** (1.0 / MONTHS) - 1.0
    pd_long_run = loans.num["pd_12m"].copy()
    if config.pd_basis == "current":
        # PD(z0) = Phi((x - sqrt(rho) z0) / sqrt(1 - rho))  =>  long-run PD = Phi(x)
        ok = ~np.isnan(pd_long_run)
        with np.errstate(divide="ignore"):
            x = ndtri(pd_long_run[ok]) * np.sqrt(1.0 - rho[ok]) + np.sqrt(rho[ok]) * config.current_z
        pd_long_run[ok] = ndtr(x)
    ccf = loans.num["ccf"]
    ccf_12m = np.where(np.isnan(loans.num["ccf_12m"]), ccf, loans.num["ccf_12m"])
    lgd_default = np.where(np.isnan(loans.num["lgd_default"]), loans.num["lgd"], loans.num["lgd_default"])
    return _Prepared(
        segments=segments,
        seg_index=seg_index,
        rho=rho,
        cpr=cpr,
        use_matrix=use_matrix,
        grade=grade,
        grade_origination=grade_orig,
        rem=loans.num["remaining_months"].astype(np.int64),
        mob=loans.num["months_on_book"].astype(np.int64),
        rate_month=rate_month,
        pd_long_run=pd_long_run,
        ccf_12m=ccf_12m,
        lgd_default=lgd_default,
        warnings=warnings,
    )


def _ead_matrix(loans: Loans, prep: _Prepared, config: Config, rows: NDArray[np.intp], t_max: int) -> Array:
    """Drawn exposure per month (rows x t_max), zero after the remaining life."""
    bal = loans.num["balance"][rows][:, None]
    n = prep.rem[rows][:, None].astype(np.float64)
    i = prep.rate_month[rows][:, None]
    t = np.arange(t_max + 1, dtype=np.float64)[None, :]
    amort = loans.text["amortization"][rows]
    revolving = loans.text["product"][rows] == "revolving"
    start = np.broadcast_to(bal, (rows.size, t_max + 1)).copy()
    lin = (amort == "equal_principal") & ~revolving
    ann = (amort == "annuity") & ~revolving
    if np.any(lin):
        start[lin] = (bal * (1.0 - t / n))[lin]
    if np.any(ann):
        zero = ann & (i[:, 0] == 0.0)
        pos = ann & ~zero
        if np.any(zero):
            start[zero] = (bal * (1.0 - t / n))[zero]
        if np.any(pos):
            g = (1.0 + i[pos]) ** n[pos]
            start[pos] = bal[pos] * (g - (1.0 + i[pos]) ** np.minimum(t, n[pos])) / (g - 1.0)
    start = np.maximum(start, 0.0)
    # column t is the balance at the start of month t+1; the balance at the end of the last month is 0
    beyond = t >= n
    start[np.broadcast_to(beyond, start.shape)] = 0.0
    cpr = prep.cpr[rows]
    if np.any(cpr > 0.0):
        keep = (1.0 - cpr) ** (1.0 / MONTHS)
        factor = np.where(revolving, 1.0, keep)[:, None] ** t
        start = start * factor
    opening, closing = start[:, :-1], start[:, 1:].copy()
    # a revolving balance is not repaid at the end of its behavioural life
    closing[revolving] = opening[revolving]
    if config.ead_basis == "start":
        return opening
    if config.ead_basis == "start_plus_interest":
        return np.asarray(opening * (1.0 + i), dtype=np.float64)
    return np.asarray((opening + closing) / 2.0, dtype=np.float64)


def _flat_cum(pd: Array, rho: Array, z: Array, w: Array, t_max: int, given: Array | None = None) -> Array:
    """Cumulative PD (loans x t_max) of constant-hazard loans under a scenario path.

    ``pd`` is the long-run PD. ``given`` (the PD as it stands in the file) is used instead, unshifted,
    when the scenario has no z path.
    """
    n_years = -(-t_max // MONTHS)
    if given is not None:
        pd = given
    annual = np.broadcast_to(pd[:, None], (pd.size, n_years)).copy()
    active = np.nonzero(w[:n_years] > 0.0)[0]
    if active.size and given is None:
        with np.errstate(divide="ignore"):
            x = ndtri(pd)[:, None]
        shifted = ndtr((x - np.sqrt(rho)[:, None] * z[None, active]) / np.sqrt(1.0 - rho)[:, None])
        annual[:, active] = w[active][None, :] * shifted + (1.0 - w[active])[None, :] * pd[:, None]
    monthly_survival = (1.0 - annual) ** (1.0 / MONTHS)
    year_of_month = np.arange(t_max) // MONTHS
    return np.asarray(1.0 - np.cumprod(monthly_survival[:, year_of_month], axis=1), dtype=np.float64)


# ---- the run -----------------------------------------------------------------------------------


def run(loans: Loans, config: Config) -> PortfolioResult:
    """Stage and ECL for every loan."""
    prep = _prepare(loans, config)
    n, n_scen = loans.n, len(config.scenarios)
    warnings = list(prep.warnings)
    rem = prep.rem
    t_all = int(rem.max())
    n_years_all = -(-t_all // MONTHS)

    paths = [scenario_years(config, s, n_years_all) for s in config.scenarios]
    tables: dict[tuple[int, int], Array] = {}
    for si, seg in enumerate(prep.segments):
        rows = np.nonzero((prep.seg_index == si) & prep.use_matrix)[0]
        if rows.size == 0 or seg.matrix is None:
            continue
        g = transitions.generator(seg.matrix, seg.generator, tol=max(seg.matrix_tolerance, 1e-9))
        for note in g.warnings:
            warnings.append(f"segment {seg.name!r}: {note}; the '{seg.generator}' generator is an approximation")
        for k, (z, w) in enumerate(paths):
            tables[(si, k)] = _matrix_table(_monthly_matrices(seg, z, w), t_all)

    ecl12 = np.zeros((n_scen, n))
    life = np.zeros((n_scen, n))
    pd_life = np.zeros((n_scen, n))
    pd_12 = np.zeros((n_scen, n))
    lgd = loans.num["lgd"]
    eir = loans.num["eir"]
    undrawn = loans.num["undrawn"]
    ccf = loans.num["ccf"]
    offset = 0.5 if config.timing == "mid" else 0.0

    order = np.argsort(rem, kind="stable")
    start = 0
    while start < n:
        t_max = int(rem[order[start]])
        stop = start + 1
        while stop < n and int(rem[order[stop]]) * (stop - start + 1) <= CHUNK_CELLS:
            stop += 1
        rows = order[start:stop]
        t_max = int(rem[rows].max())
        start = stop
        r = rem[rows]
        alive = np.arange(t_max)[None, :] < r[:, None]
        ead = _ead_matrix(loans, prep, config, rows, t_max)
        months = np.arange(1, t_max + 1, dtype=np.float64)[None, :] - offset
        kernel = lgd[rows][:, None] * (1.0 + eir[rows])[:, None] ** (-months / MONTHS)
        first_year = np.arange(t_max)[None, :] < MONTHS
        last = (r - 1)[:, None]
        last12 = (np.minimum(r, MONTHS) - 1)[:, None]
        flat = ~prep.use_matrix[rows]
        for k, (z, w) in enumerate(paths):
            cum = np.empty((rows.size, t_max))
            if np.any(flat):
                fr = rows[flat]
                as_given = loans.num["pd_12m"][fr] if config.scenarios[k].z is None else None
                cum[flat] = _flat_cum(prep.pd_long_run[fr], prep.rho[fr], z, w, t_max, as_given)
            for si in np.unique(prep.seg_index[rows[~flat]]).tolist():
                sel = ~flat & (prep.seg_index[rows] == si)
                cum[sel] = tables[(si, k)][prep.grade[rows[sel]], :t_max]
            marginal = np.diff(cum, axis=1, prepend=0.0) * alive
            weight = marginal * kernel
            drawn = weight * ead
            u = undrawn[rows]
            life[k, rows] = drawn.sum(axis=1) + ccf[rows] * u * weight.sum(axis=1)
            ecl12[k, rows] = (drawn * first_year).sum(axis=1) + prep.ccf_12m[rows] * u * (weight * first_year).sum(
                axis=1
            )
            pd_life[k, rows] = np.take_along_axis(cum, last, axis=1)[:, 0]
            pd_12[k, rows] = np.take_along_axis(cum, last12, axis=1)[:, 0]

    weights = config.weights
    ecl12_w = weights @ ecl12
    life_w = weights @ life
    exposure = loans.num["balance"] + ccf * undrawn
    ecl_default = exposure * prep.lgd_default

    orig_life, orig_12 = _origination_pds(loans, prep, config)
    policy, limits = _effective_policy(config, warnings)

    def stage_with(pd_l: Array, pd_y: Array) -> tuple[StagingResult, Array, Array, Array, Array]:
        years = rem / MONTHS
        annual_now = 1.0 - (1.0 - pd_l) ** (1.0 / years)
        annual_orig = 1.0 - (1.0 - orig_life) ** (1.0 / years)
        if config.pd_measure == "annualized_lifetime":
            now, orig = annual_now, annual_orig
        elif config.pd_measure == "lifetime":
            now, orig = pd_l, orig_life
        else:
            now, orig = pd_y, orig_12
        res = assess(
            policy,
            pd_now=now,
            pd_origination=orig,
            backstop_pd_now=annual_now,
            backstop_pd_origination=annual_orig,
            low_credit_risk_pd_now=annual_now,
            grade_now=np.where(prep.grade >= 0, prep.grade + 1.0, np.nan),
            grade_origination=np.where(prep.grade_origination >= 0, prep.grade_origination + 1.0, np.nan),
            days_past_due=loans.num["dpd"],
            credit_impaired=loans.flag["credit_impaired"],
            forborne=loans.flag["forborne"],
            watchlist=loans.flag["watchlist"],
            collective=loans.flag["collective"],
            previous_stage=np.where(np.isnan(loans.num["previous_stage"]), 1.0, loans.num["previous_stage"]),
            periods_without_trigger=np.where(
                np.isnan(loans.num["periods_without_trigger"]), np.inf, loans.num["periods_without_trigger"]
            ),
        )
        return res, now, orig, annual_now, annual_orig

    if config.scenario_basis == "weighted":
        basis_life, basis_12 = weights @ pd_life, weights @ pd_12
    else:
        basis_life, basis_12 = pd_life[config.base_index], pd_12[config.base_index]
    res, now, orig, annual_now, annual_orig = stage_with(basis_life, basis_12)
    res, group, texts = _apply_profile(config, loans, res, limits, policy)

    scen_stage = np.empty((n_scen, n), dtype=np.int8)
    for k in range(n_scen):
        rs = stage_with(pd_life[k], pd_12[k])[0]
        scen_stage[k] = _apply_profile(config, loans, rs, limits, policy)[0].stage

    for note in policy.warnings():
        warnings.append(f"staging policy: {note}")
    if config.pd_measure == "twelve_month":
        bullets = (loans.text["amortization"] == "bullet") & (rem > MONTHS) & (loans.text["product"] == "term")
        if np.any(bullets):
            warnings.append(
                f"12-month PD is used to test for a significant increase, but {int(bullets.sum())} loans repay "
                "at maturity beyond 12 months; IFRS 9 B5.5.14 asks for a lifetime assessment there"
            )
    if n_scen == 1:
        warnings.append(
            "a single scenario is used: the result is probability-weighted only if losses are linear in "
            "the economy (IFRS 9 5.5.17(a))"
        )
    if np.any(np.isnan(orig_life)):
        warnings.append(f"{int(np.isnan(orig_life).sum())} loans have no origination PD or grade")

    stage = res.stage
    ecl = np.where(stage == 1, ecl12_w, np.where(stage == 2, life_w, ecl_default))
    return PortfolioResult(
        loans=loans,
        config=config,
        policy=policy,
        stage=stage,
        reasons=res.reasons,
        primary=res.primary,
        pd_now=now,
        pd_origination=orig,
        annual_pd_now=annual_now,
        annual_pd_origination=annual_orig,
        exposure=exposure,
        ecl=np.asarray(ecl, dtype=np.float64),
        ecl_12m=ecl12_w,
        ecl_lifetime=life_w,
        ecl_default=ecl_default,
        scenario_ecl_12m=ecl12,
        scenario_ecl_lifetime=life,
        scenario_pd_lifetime=pd_life,
        scenario_pd_12m=pd_12,
        scenario_stage=scen_stage,
        group=group,
        profile_reasons=texts,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _origination_pds(loans: Loans, prep: _Prepared, config: Config) -> tuple[Array, Array]:
    """PD expected at origination for today's remaining window: lifetime, and its first 12 months."""
    n = loans.n
    rem = prep.rem.astype(np.float64)
    window = np.minimum(rem, MONTHS)
    life = np.full(n, np.nan)
    first = np.full(n, np.nan)

    given = loans.num["pd_origination_remaining"]
    a = ~np.isnan(given)
    life[a] = given[a]
    first[a] = 1.0 - (1.0 - given[a]) ** (window[a] / rem[a])

    p12 = loans.num["pd_12m_origination"]
    b = ~a & ~np.isnan(p12)
    life[b] = 1.0 - (1.0 - p12[b]) ** (rem[b] / MONTHS)
    first[b] = 1.0 - (1.0 - p12[b]) ** (window[b] / MONTHS)

    c = ~a & ~b & (prep.grade_origination >= 0)
    for si, seg in enumerate(prep.segments):
        rows = np.nonzero(c & (prep.seg_index == si))[0]
        if rows.size == 0 or seg.matrix is None:
            continue
        horizon = int((prep.mob[rows] + prep.rem[rows]).max())
        monthly = transitions.generator(seg.matrix, seg.generator, tol=max(seg.matrix_tolerance, 1e-9)).matrix(
            1.0 / MONTHS
        )
        table = np.concatenate((np.zeros((monthly.shape[0], 1)), _matrix_table([monthly], horizon)), axis=1)
        g = prep.grade_origination[rows]
        at_start = table[g, prep.mob[rows]]
        at_end = table[g, prep.mob[rows] + prep.rem[rows]]
        at_12 = table[g, prep.mob[rows] + np.minimum(prep.rem[rows], MONTHS)]
        alive = 1.0 - at_start
        ok = alive > 0.0
        life[rows[ok]] = (at_end[ok] - at_start[ok]) / alive[ok]
        first[rows[ok]] = (at_12[ok] - at_start[ok]) / alive[ok]
    return life, first


def _effective_policy(config: Config, warnings: list[str]) -> tuple[StagingPolicy, tr_bddk.DayLimits | None]:
    if config.profile != "tr_bddk":
        return config.policy, None
    as_of = config.profile_options.get("as_of")
    try:
        date = dt.date.fromisoformat(as_of) if as_of else None
    except (TypeError, ValueError) as exc:
        raise InputError("profile_options.as_of must be a date like 2026-12-31") from exc
    limits = tr_bddk.day_limits(date, int(config.profile_options.get("days_in_year", 365)))
    p = config.policy
    if (p.days_past_due_backstop, p.default_days_past_due) != (limits.group2, limits.group3):
        warnings.append(
            f"profile tr_bddk: days past due limits set to {limits.group2} and {limits.group3} days "
            "(BDDK provisioning regulation, Article 4)"
        )
        p = p.with_changes(days_past_due_backstop=limits.group2, default_days_past_due=limits.group3)
    return p, limits


def _apply_profile(
    config: Config, loans: Loans, res: StagingResult, limits: tr_bddk.DayLimits | None, policy: StagingPolicy
) -> tuple[StagingResult, NDArray[np.int8] | None, tuple[tuple[str, ...], ...] | None]:
    if config.profile != "tr_bddk" or limits is None:
        return res, None, None
    judgement = loans.num["judgement_group"]
    cls = tr_bddk.classify(
        loans.num["dpd"],
        borrower=loans.text["borrower_id"].tolist(),
        consumer_loan=loans.flag["consumer"],
        significant_increase=res.stage == 2,
        unlikely_to_pay=loans.flag["unlikely_to_pay"] | (res.stage == 3),
        restructured=loans.flag["forborne"],
        in_monitoring_year=loans.flag["in_monitoring_year"],
        restructured_again=loans.flag["restructured_again"],
        judgement_group=None
        if np.all(np.isnan(judgement))
        else np.where(np.isnan(judgement), 1, judgement).astype(int),
        limits=limits,
        consumer_loans_at_loan_level=bool(config.profile_options.get("consumer_loans_at_loan_level", True)),
    )
    reasons = res.reasons.copy()
    reasons[(cls.stage == 3) & (res.stage < 3)] |= int(Reason.PROFILE_DEFAULT)
    reasons[(cls.stage == 2) & (res.stage < 2)] |= int(Reason.PROFILE)
    out = finish(reasons, res.pd_ratio, res.pd_change, policy.precedence)
    group = np.where((out.stage == 3) & (cls.group < 3), 3, cls.group).astype(np.int8)
    return out, group, cls.reasons


# ---- one loan, recomputed with the scalar building blocks --------------------------------------


def loan_curve(loans: Loans, config: Config, i: int, scenario: Scenario) -> PDCurve:
    """Monthly PD curve of loan ``i`` under a scenario, built with the public scalar functions."""
    prep = _prepare(loans.take(np.array([i])), config)
    seg = prep.segments[0]
    rem = int(prep.rem[0])
    n_years = -(-rem // MONTHS)
    z, w = scenario_years(config, scenario, n_years)
    if prep.use_matrix[0]:
        monthly = _monthly_matrices(seg, z, w)
        path = [monthly[t // MONTHS] for t in range(rem)]
        cum = transitions.cumulative_pd_path(path, tol=1e-9)
        return PDCurve(cum[int(prep.grade[0])], MONTHS)
    if scenario.z is None:
        return PDCurve.flat(float(loans.num["pd_12m"][i]), rem, MONTHS)
    return cycle.pit_curve(PDCurve.flat(float(prep.pd_long_run[0]), rem, MONTHS), seg.rho, z, w)


def loan_schedule(loans: Loans, config: Config, i: int) -> schedule.Schedule:
    one = loans.take(np.array([i]))
    prep = _prepare(one, config)
    rem = int(prep.rem[0])
    bal = float(one.num["balance"][0])
    rate = float(one.num["eir"][0] if np.isnan(one.num["rate"][0]) else one.num["rate"][0])
    kind = str(one.text["amortization"][0])
    if str(one.text["product"][0]) == "revolving":
        return schedule.Schedule(
            np.full(rem, bal), np.full(rem, bal * float(prep.rate_month[0])), np.zeros(rem), MONTHS
        )
    builder = {"bullet": schedule.bullet, "annuity": schedule.annuity, "equal_principal": schedule.equal_principal}[
        kind
    ]
    sched = builder(bal, rate, rem, MONTHS, config.rate_convention)
    if prep.cpr[0] > 0.0:
        sched = schedule.with_prepayment(sched, float(prep.cpr[0]))
    return sched


def loan_detail(result: PortfolioResult, loan_id: str, periods: bool = False) -> dict[str, Any]:
    """Everything behind one loan's stage and ECL, recomputed step by step."""
    loans, config = result.loans, result.config
    i = loans.index_of(loan_id)
    one = loans.take(np.array([i]))
    prep = _prepare(one, config)
    sched = loan_schedule(loans, config, i)
    undrawn, ccf = float(one.num["undrawn"][0]), float(one.num["ccf"][0])
    ead_drawn = sched.ead(config.ead_basis)
    lgd, eir = float(one.num["lgd"][0]), float(one.num["eir"][0])
    per_scenario = []
    for k, s in enumerate(config.scenarios):
        curve = loan_curve(loans, config, i, s)
        life = expected_credit_loss(curve, ead_drawn + ccf * undrawn, lgd, eir, timing=config.timing)
        first = expected_credit_loss(
            curve, ead_drawn + float(prep.ccf_12m[0]) * undrawn, lgd, eir, timing=config.timing
        )
        item: dict[str, Any] = {
            "name": s.name,
            "weight": s.weight,
            "twelve_month_pd": life.twelve_month_pd,
            "lifetime_pd": life.lifetime_pd,
            "twelve_month_ecl": first.twelve_month,
            "lifetime_ecl": life.lifetime,
            "stage_if_this_scenario_alone": int(result.scenario_stage[k, i]),
            "by_year": _by_year(life),
        }
        if periods:
            item["periods"] = life.to_dict()["periods"]
        per_scenario.append(item)
    w = config.weights
    stage = int(result.stage[i])
    ecl_12m = float(sum(wk * s["twelve_month_ecl"] for wk, s in zip(w, per_scenario, strict=True)))
    ecl_life = float(sum(wk * s["lifetime_ecl"] for wk, s in zip(w, per_scenario, strict=True)))
    exposure = float(one.num["balance"][0]) + ccf * undrawn
    ecl_default = exposure * float(prep.lgd_default[0])
    measured = {1: ecl_12m, 2: ecl_life, 3: ecl_default}[stage]

    def num(v: float) -> float | None:
        return None if np.isnan(v) else float(v)

    from .staging import names

    out: dict[str, Any] = {
        "loan_id": loan_id,
        "inputs": loans.row(i),
        "pd_source": "transition matrix of the segment" if prep.use_matrix[0] else "pd_12m as a constant hazard",
        "stage": stage,
        "reasons": names(int(result.reasons[i])),
        "primary_reason": Reason(int(result.primary[i])).name.lower() if result.primary[i] else "none",  # type: ignore[union-attr]
        "staging_test": {
            "pd_measure": config.pd_measure,
            "scenario_basis": config.scenario_basis,
            "pd_now": float(result.pd_now[i]),
            "pd_at_origination_for_the_remaining_window": num(float(result.pd_origination[i])),
            "ratio": num(float(result.pd_now[i] / result.pd_origination[i]))
            if result.pd_origination[i] and not np.isnan(result.pd_origination[i])
            else None,
            "annualized_lifetime_pd_now": float(result.annual_pd_now[i]),
            "annualized_lifetime_pd_at_origination": num(float(result.annual_pd_origination[i])),
            "days_past_due": float(one.num["dpd"][0]),
        },
        "exposure": exposure,
        "scenarios": per_scenario,
        "weighted": {"twelve_month_ecl": ecl_12m, "lifetime_ecl": ecl_life},
        "credit_impaired_ecl": ecl_default,
        "ecl": measured,
        "measurement": {1: "12-month ECL", 2: "lifetime ECL", 3: "exposure x loss rate in default"}[stage],
        "coverage": measured / exposure if exposure > 0 else 0.0,
    }
    if result.group is not None and result.profile_reasons is not None:
        out["regulatory_group"] = int(result.group[i])
        out["regulatory_reasons"] = list(result.profile_reasons[i])
    return out


def _by_year(res: Any) -> list[dict[str, float]]:
    out = []
    for y in range(-(-res.n_periods // MONTHS)):
        sl = slice(y * MONTHS, (y + 1) * MONTHS)
        out.append(
            {
                "year": y + 1,
                "pd": float(res.marginal_pd[sl].sum()),
                "average_ead": float(res.ead[sl].mean()),
                "ecl": float(res.contribution[sl].sum()),
            }
        )
    return out
