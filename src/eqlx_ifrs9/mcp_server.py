"""MCP server: lets an AI agent run IFRS 9 expected credit loss calculations on files in one folder.

Read-only by construction. The tools read loan files (CSV), settings and case files (JSON) inside the
``--root`` folder given at start-up; paths are resolved, so ``..`` and symlinks cannot leave it.
Nothing is written and nothing is sent anywhere; results go back to the agent that asked. The
calculator tools take plain numbers and touch no file at all.

Start it with ``ifrs9 mcp --root /path/to/folder`` (needs ``pip install 'eqlx-ifrs9[mcp]'``).
"""

from __future__ import annotations

import functools
import json
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, ParamSpec, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import __version__, calc, config, data, engine, individual, period, report, review
from .errors import InputError
from .profiles import tr_bddk

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
CACHE_SIZE = 4
MAX_SETTINGS_BYTES = 5_000_000


class OutsideRootError(ValueError):
    pass


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _told(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """Refused input comes back to the agent with its reason instead of a bare failure."""

    @functools.wraps(fn)
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return fn(*args, **kwargs)
        except FileNotFoundError as exc:
            raise ToolError(f"file not found: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise ToolError("the file is not UTF-8 text") from exc
        except (InputError, OutsideRootError, IsADirectoryError, NotADirectoryError, RecursionError) as exc:
            raise ToolError(str(exc) or type(exc).__name__) from exc

    return inner


def _resolve(root: Path, path: str) -> Path:
    p = (root / path).resolve()
    if p != root and root not in p.parents:
        raise OutsideRootError(f"{path!r} is outside the allowed folder")
    if not p.exists():
        raise FileNotFoundError(path)
    return p


class _Runs:
    """Keeps the last few loan files and runs in memory so follow-up questions are fast."""

    def __init__(self, root: Path, max_rows: int) -> None:
        self.root = root
        self.max_rows = max_rows
        self.loans: OrderedDict[tuple[str, int, int], data.Loans] = OrderedDict()
        self.runs: OrderedDict[tuple[Any, ...], engine.PortfolioResult] = OrderedDict()

    def file(self, path: str, suffix: str) -> Path:
        p = _resolve(self.root, path)
        if not p.is_file():
            raise InputError(f"{path!r} is not a file")
        if p.suffix.lower() != suffix:
            raise InputError(f"{path!r} is not a {suffix} file")
        return p

    def get_loans(self, path: str) -> tuple[tuple[str, int, int], data.Loans]:
        p = self.file(path, ".csv")
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
        if key not in self.loans:
            self.loans[key] = data.read_csv(p, max_rows=self.max_rows)
            while len(self.loans) > CACHE_SIZE:
                self.loans.popitem(last=False)
        self.loans.move_to_end(key)
        return key, self.loans[key]

    def settings(self, path: str | None, overrides: dict[str, Any] | None) -> dict[str, Any]:
        raw: dict[str, Any] = {}
        if path:
            p = self.file(path, ".json")
            if p.stat().st_size > MAX_SETTINGS_BYTES:
                raise InputError("the settings file is too large")
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise InputError(f"{path}: not valid JSON ({exc})") from exc
            if not isinstance(raw, dict):
                raise InputError(f"{path}: the settings must be a JSON object")
        return config.with_overrides(raw, overrides) if overrides else raw

    def run(self, loans: str, settings: str | None, overrides: dict[str, Any] | None) -> engine.PortfolioResult:
        key, lo = self.get_loans(loans)
        raw = self.settings(settings, overrides)
        run_key = (*key, json.dumps(raw, sort_keys=True))
        if run_key not in self.runs:
            self.runs[run_key] = engine.run(lo, config.from_dict(raw))
            while len(self.runs) > CACHE_SIZE:
                self.runs.popitem(last=False)
        self.runs.move_to_end(run_key)
        return self.runs[run_key]


def build_server(root: Path, max_rows: int = 5_000_000) -> MCPServer:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    runs = _Runs(root, max_rows)
    server = MCPServer(
        "eqlx-ifrs9",
        version=__version__,
        instructions=(
            "Computes IFRS 9 expected credit loss (ECL) and stage allocation for loan files inside one local "
            "folder. Paths are relative to that folder. A run needs a loan file (CSV) and usually a settings "
            "file (JSON) with the staging thresholds and scenarios. Start with `list_files` and "
            "`describe_loans`, then `run_portfolio`; use `list_loans` and `explain_loan` to look at individual "
            "loans and `compare_settings` to see what a different threshold or weight would change. "
            "To review a lender's own figures (columns reported_stage and reported_ecl) use `reconcile_reported`; "
            "`scenario_response` shows how the allowance reacts to the economy. With the loan file of an earlier "
            "date as well, `allowance_movement` explains why the allowance changed, `staging_quality` shows how "
            "the earlier stages turned out, `backtest_defaults` compares expected with actual defaults and "
            "`challenger_matrix` recomputes the allowance with the rating migration observed between the files. "
            "`calculate_ecl`, `pd_term_structure` and `provision_matrix` are calculators that need no file. "
            "For one credit-impaired debtor assessed individually, `assess_individual` reads a case file (JSON: "
            "debtor, exposures, other claims, collateral, scenarios with probabilities) and returns the "
            "allowance per exposure with every step; `collateral_recovery` is the calculator for one item. "
            "Everything is deterministic arithmetic on the inputs: the tool does not estimate PDs or forecast "
            "the economy, and thresholds are the user's policy. Quote the numbers the tools return, say which "
            "settings produced them, and pass on the warnings."
        ),
    )

    @server.tool(annotations=READ_ONLY)
    @_told
    def list_files(path: str = ".") -> list[dict[str, Any]]:
        """List the loan files (.csv) and settings files (.json) under a path in the allowed folder."""
        p = _resolve(root, path)
        files = [p] if p.is_file() else sorted(p.rglob("*"))
        out = []
        for f in files:
            if f.suffix.lower() not in (".csv", ".json") or not f.is_file():
                continue
            real = f.resolve()
            if root in real.parents:
                out.append({"path": str(real.relative_to(root)), "bytes": real.stat().st_size})
        return out

    @server.tool(annotations=READ_ONLY)
    @_told
    def describe_loans(loans: str) -> dict[str, Any]:
        """What a loan file contains before any calculation: number of loans and borrowers, columns,
        totals, segments, days-past-due buckets, and gaps such as loans without origination risk."""
        return report.describe(runs.get_loans(loans)[1])

    @server.tool(annotations=READ_ONLY)
    @_told
    def run_portfolio(
        loans: str, settings: str | None = None, overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Stage and ECL for every loan in a file; returns portfolio tables, not the loans.

        `settings` is a JSON file with staging thresholds, scenarios and segment parameters.
        `overrides` replaces single settings by dotted key, e.g. {"staging.relative_threshold": 2.5}.
        The result has totals, the split by stage and by segment, stage 2 by primary reason, the ECL under
        each scenario alone, supervisory checks (for example stage 1 loans whose PD has tripled), warnings
        about the policy, and the settings that were applied.
        """
        return report.summary(runs.run(loans, settings, overrides))

    @server.tool(annotations=READ_ONLY)
    @_told
    def list_loans(
        loans: str,
        settings: str | None = None,
        overrides: dict[str, Any] | None = None,
        stage: Literal[1, 2, 3] | None = None,
        reason: str | None = None,
        segment: str | None = None,
        borrower_id: str | None = None,
        sort: Literal["ecl", "exposure", "coverage", "pd_ratio", "dpd"] = "ecl",
        limit: int = 20,
    ) -> dict[str, Any]:
        """Loans matching a filter, largest first: stage, every reason that fired, exposure, ECL and PDs.

        `reason` is one of pd_relative, pd_absolute, pd_backstop, rating_notches, grade_threshold, forborne,
        watchlist, collective, no_origination_pd, days_past_due_backstop, probation, credit_impaired,
        default_days_past_due, cure_period, profile, profile_default, low_credit_risk.
        """
        return report.loan_rows(
            runs.run(loans, settings, overrides),
            stage=stage,
            reason=reason,
            segment=segment,
            borrower_id=borrower_id,
            sort=sort,
            limit=limit,
        )

    @server.tool(annotations=READ_ONLY)
    @_told
    def explain_loan(
        loans: str,
        loan_id: str,
        settings: str | None = None,
        overrides: dict[str, Any] | None = None,
        periods: bool = False,
    ) -> dict[str, Any]:
        """Everything behind one loan: its inputs, the staging test with the numbers compared, the PD and
        ECL under each scenario year by year, the weighted figures and which one is reported.
        `periods` adds the month-by-month table (marginal PD, exposure, LGD, discount factor, ECL)."""
        return engine.loan_detail(runs.run(loans, settings, overrides), loan_id, periods=periods)

    @server.tool(annotations=READ_ONLY)
    @_told
    def compare_settings(
        loans: str, changes: dict[str, Any], settings: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """What a change of settings would do: total ECL before and after, loans moving between stages,
        and the loans with the largest change. `changes` uses dotted keys, e.g.
        {"staging.relative_threshold": 3.0} or {"scenarios": [...]}. Nothing is saved."""
        if not changes:
            raise InputError("give at least one change, e.g. {'staging.relative_threshold': 3.0}")
        before = runs.run(loans, settings, None)
        after = runs.run(loans, settings, changes)
        return {"changed_settings": changes, **report.compare(before, after, limit)}

    @server.tool(annotations=READ_ONLY)
    @_told
    def calculate_ecl(
        balance: float,
        lgd: float,
        eir: float,
        remaining_months: int,
        pd_12m: float | None = None,
        cumulative_pd: list[float] | None = None,
        cumulative_pd_frequency: int = 12,
        amortization: Literal["bullet", "annuity", "equal_principal"] = "bullet",
        rate: float | None = None,
        undrawn: float = 0.0,
        ccf: float = 0.0,
        cpr: float = 0.0,
        timing: Literal["end", "mid"] = "end",
        z_by_year: list[float] | None = None,
        rho: float = 0.0,
        reversion_years: int = 0,
        periods: bool = False,
    ) -> dict[str, Any]:
        """12-month and lifetime ECL of one instrument from plain numbers (no file).

        Rates and PDs are fractions (0.05 = 5%). Give `pd_12m` (constant hazard) or `cumulative_pd` (cumulative
        PD at the end of each period, `cumulative_pd_frequency` periods a year). `eir` is the effective
        interest rate used for discounting; `rate` the contract rate of the repayment schedule (defaults to
        eir). `z_by_year` with `rho` shifts a long-run PD with the one-factor model (z > 0 good year); after
        the last year given, the PD reverts to the long-run PD over `reversion_years` transitional years.
        """
        return calc.instrument_ecl(
            balance=balance,
            lgd=lgd,
            eir=eir,
            remaining_months=remaining_months,
            pd_12m=pd_12m,
            cumulative_pd=cumulative_pd,
            cumulative_pd_frequency=cumulative_pd_frequency,
            amortization=amortization,
            rate=rate,
            undrawn=undrawn,
            ccf=ccf,
            cpr=cpr,
            timing=timing,
            z_by_year=z_by_year,
            rho=rho,
            reversion_years=reversion_years,
            periods=periods,
        )

    @server.tool(annotations=READ_ONLY)
    @_told
    def pd_term_structure(
        matrix: list[list[float]],
        years: int,
        grades: list[str] | None = None,
        monthly: bool = False,
        z_by_year: list[float] | None = None,
        rho: float = 0.0,
        tolerance: float = 1e-6,
    ) -> dict[str, Any]:
        """Cumulative PD by rating grade from a one-year transition matrix (rows = from, last state =
        default). Returns annual points, or monthly ones through a fitted generator, plus diagnostics on
        whether the matrix can come from a continuous-time chain. `z_by_year` shifts the matrix per year."""
        return calc.pd_term_structure(
            matrix, years, grades=grades, monthly=monthly, z_by_year=z_by_year, rho=rho, tolerance=tolerance
        )

    @server.tool(annotations=READ_ONLY)
    @_told
    def provision_matrix(
        gross_carrying_amount: list[float], loss_rate: list[float], buckets: list[str] | None = None
    ) -> dict[str, Any]:
        """Simplified approach for trade receivables: allowance per days-past-due bucket as gross carrying
        amount times lifetime loss rate, with totals and coverage."""
        return calc.provision_matrix(gross_carrying_amount, loss_rate, buckets)

    @server.tool(annotations=READ_ONLY)
    @_told
    def reconcile_reported(
        loans: str, settings: str | None = None, overrides: dict[str, Any] | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """The lender's own figures against the recalculation. Needs the columns `reported_stage` and/or
        `reported_ecl` in the loan file. Returns the stage table (rows reported, columns recalculated), the
        loans reported in a lower stage with the rule that puts them higher, the allowance gap in total, by
        segment and by stage with a verdict (recalculation above the reported figure by less than 5%:
        immaterial; 5% to 10%: explain; more: investigate), and the gap split into the part caused by staging
        and the part caused by measurement. `difference` is recalculated minus reported."""
        return review.reconcile(runs.run(loans, settings, overrides), limit=limit)

    @server.tool(annotations=READ_ONLY)
    @_told
    def scenario_response(
        loans: str,
        settings: str | None = None,
        overrides: dict[str, Any] | None = None,
        shifts: list[float] | None = None,
        scenario: str | None = None,
    ) -> dict[str, Any]:
        """How the allowance reacts to the economy. The base scenario's path (z, standard deviations of the
        economy, negative is worse) is moved by each shift and the portfolio is computed with that single
        scenario: once with the stages held, once with the loans staged again. Also returns the share of
        stage 2 per shift and how much more a step down costs than a step up saves."""
        lo = runs.get_loans(loans)[1]
        cfg = config.from_dict(runs.settings(settings, overrides))
        return review.response(lo, cfg, shifts=shifts if shifts is not None else review.SHIFTS, scenario=scenario)

    def two_dates(
        opening_loans: str,
        closing_loans: str,
        closing_settings: str | None,
        opening_settings: str | None,
        write_offs: str | None,
    ) -> tuple[engine.PortfolioResult, engine.PortfolioResult, list[str]]:
        closing = runs.run(closing_loans, closing_settings, None)
        opening = runs.run(opening_loans, opening_settings or closing_settings, None)
        gone = period.read_write_offs(runs.file(write_offs, ".csv")) if write_offs else []
        return opening, closing, gone

    @server.tool(annotations=READ_ONLY)
    @_told
    def allowance_movement(
        opening_loans: str,
        closing_loans: str,
        closing_settings: str | None = None,
        opening_settings: str | None = None,
        write_offs: str | None = None,
        order: list[str] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Why the allowance changed between two reporting dates. Loans are matched by loan_id. Returns the
        reconciliation of the loss allowance and of the gross carrying amount by stage (the layout of the
        IFRS 7 35H table): transfers between stages, remeasurement on transfer, new loans, derecognised loans,
        write-offs, and the change of continuing loans split into drivers (settings and scenarios, exposure
        and time, PD, LGD, other) by replacing one group of inputs at a time in `order`. The rows add up to
        the closing balance. `largest_changes` lists the loans behind the change with their drivers.
        `write_offs` is a CSV with a loan_id column; `opening_settings` defaults to `closing_settings`."""
        opening, closing, gone = two_dates(opening_loans, closing_loans, closing_settings, opening_settings, write_offs)
        return period.movement(opening, closing, write_offs=gone, order=order or period.DRIVERS, limit=limit)

    @server.tool(annotations=READ_ONLY)
    @_told
    def staging_quality(
        opening_loans: str,
        closing_loans: str,
        closing_settings: str | None = None,
        opening_settings: str | None = None,
        write_offs: str | None = None,
    ) -> dict[str, Any]:
        """How the stages of the earlier date turned out: the stage transition table, where stage 3 came from
        (share that skipped stage 2), default rate of stage 1 against stage 2, the inflow into stage 2 by
        reason and what became of the loans that were in stage 2. Warns when stage 2 was not riskier than
        stage 1."""
        opening, closing, gone = two_dates(opening_loans, closing_loans, closing_settings, opening_settings, write_offs)
        return period.staging_quality(opening, closing, write_offs=gone)

    @server.tool(annotations=READ_ONLY)
    @_told
    def backtest_defaults(
        opening_loans: str,
        closing_loans: str,
        closing_settings: str | None = None,
        opening_settings: str | None = None,
        write_offs: str | None = None,
        months: int = 12,
        realised_z: float | None = None,
    ) -> dict[str, Any]:
        """Defaults expected at the earlier date against defaults that happened (stage 3 at the later date, or
        written off), in total, by stage and by segment. With `realised_z`, the state of the economy that
        actually occurred (z > 0 good), the difference is split into `forecast_error` (what the scenarios got
        wrong about the economy) and `model_error` (what remains once the economy is known)."""
        opening, closing, gone = two_dates(opening_loans, closing_loans, closing_settings, opening_settings, write_offs)
        return period.backtest(opening, closing, write_offs=gone, months=months, realised_z=realised_z)

    @server.tool(annotations=READ_ONLY)
    @_told
    def challenger_matrix(
        opening_loans: str,
        closing_loans: str,
        closing_settings: str | None = None,
        opening_settings: str | None = None,
        write_offs: str | None = None,
        months: int = 12,
    ) -> dict[str, Any]:
        """Challenger for graded segments: counts the moves between rating grades from the earlier to the
        later file, computes the later portfolio again with the observed matrix instead of the settings'
        matrix, and compares the allowance with the reported one (or with the run on the settings' matrix),
        with the 5% / 10% verdict. Returns the counts and the observed one-year PD per grade; one period is a
        small sample, so read the counts before the result."""
        opening, closing, gone = two_dates(opening_loans, closing_loans, closing_settings, opening_settings, write_offs)
        return review.challenger(opening, closing, write_offs=gone, months=months)

    @server.tool(annotations=READ_ONLY)
    @_told
    def assess_individual(case: str) -> dict[str, Any]:
        """Allowance of one credit-impaired debtor assessed individually, from a case file (.json) in the
        allowed folder. The case lists the debtor, the lender's exposures (gross carrying amount, effective
        interest rate, seniority rank), the other creditors' claims, the collateral with its sale assumptions,
        and scenarios with probabilities: `gone_concern` (collateral is sold), `steady_state` (one sustainable
        year of cash flow times a multiple), `two_step` (yearly projection plus terminal value) or
        `market_price`. Returns, per exposure and scenario, the cash flow value, collateral value, recoverable
        amount and shortfall, the probability-weighted allowance, and warnings where the inputs depart from
        published supervisory expectations. `ifrs9 sample` writes two example case files."""
        p = runs.file(case, ".json")
        if p.stat().st_size > MAX_SETTINGS_BYTES:
            raise InputError("the case file is too large")
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InputError(f"{case}: not valid JSON ({exc})") from exc
        return individual.assess(raw)

    @server.tool(annotations=READ_ONLY)
    @_told
    def collateral_recovery(
        value: float,
        eir: float,
        years_to_sale: float,
        haircut: float,
        value_change: float = 0.0,
        selling_cost_rate: float = 0.0,
        annual_cost_rate: float = 0.0,
        costs: list[dict[str, float]] | None = None,
        income: list[dict[str, float]] | None = None,
        prior_claims: float = 0.0,
        share: float = 1.0,
    ) -> dict[str, Any]:
        """Present value of selling one collateral item and the part that belongs to the lender. The item is
        sold `years_to_sale` years from now at `value * (1 + value_change)`, less the market price discount
        (`haircut`), selling costs (rate on the discounted price) and a yearly holding cost (rate on `value`);
        `costs` and `income` are lists of {"year", "amount"}. Flows are discounted at `eir`, the loan's
        effective interest rate. Claims ranking before the lender (`prior_claims`) are served first; `share`
        is the lender's part of what is left. Rates are fractions (0.10 for 10%)."""
        return individual.collateral_recovery(
            value,
            eir,
            years_to_sale=years_to_sale,
            haircut=haircut,
            value_change=value_change,
            selling_cost_rate=selling_cost_rate,
            annual_cost_rate=annual_cost_rate,
            costs=costs or [],
            income=income or [],
            prior_claims=prior_claims,
            share=share,
        )

    @server.tool(annotations=READ_ONLY)
    @_told
    def turkey_reclassification_check(
        check: Literal["performing_again", "restructured_npl_to_group_2", "exit_restructured_status"],
        facts: dict[str, Any],
    ) -> dict[str, Any]:
        """Turkey (BDDK provisioning regulation): may a loan move back up? Returns eligible yes/no and the
        unmet conditions with article numbers.

        `performing_again` (Art. 6): arrears_collected_without_collateral, no_amount_past_due,
        last_two_payments_on_time, meets_group_1_or_2, and optionally partially_written_off,
        collateral_taken_over, repaid_in_kind.
        `restructured_npl_to_group_2` (Art. 7(5)): months_as_non_performing_since_restructuring,
        non_performing_conditions_gone, no_delay_and_no_doubt, arrears_and_written_off_principal_collected.
        `exit_restructured_status` (Art. 7(6), 7(8)): months_monitored_in_group_2,
        share_of_remaining_debt_repaid, max_days_past_due_any_loan, financial_difficulty_ended, and
        optionally applies_tfrs9, came_from_non_performing.
        """
        try:
            if check == "performing_again":
                verdict = tr_bddk.performing_again(**facts)
            elif check == "restructured_npl_to_group_2":
                verdict = tr_bddk.restructured_npl_to_group_2(**facts)
            else:
                verdict = tr_bddk.exit_restructured_status(**facts)
        except TypeError as exc:
            raise InputError(f"{check}: {exc}") from exc
        return verdict.to_dict()

    return server
