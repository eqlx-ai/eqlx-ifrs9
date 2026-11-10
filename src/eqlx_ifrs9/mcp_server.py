"""MCP server: lets an AI agent run IFRS 9 expected credit loss calculations on files in one folder.

Read-only by construction. The tools read loan files (CSV) and settings files (JSON) inside the
``--root`` folder given at start-up; paths are resolved, so ``..`` and symlinks cannot leave it.
Nothing is written and nothing is sent anywhere; results go back to the agent that asked. The
calculator tools take plain numbers and touch no file at all.

Start it with ``ifrs9 mcp --root /path/to/folder`` (needs ``pip install 'eqlx-ifrs9[mcp]'``).
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import __version__, calc, config, data, engine, report
from .errors import InputError
from .profiles import tr_bddk

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
CACHE_SIZE = 4
MAX_SETTINGS_BYTES = 5_000_000


class OutsideRootError(ValueError):
    pass


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
            "`calculate_ecl`, `pd_term_structure` and `provision_matrix` are calculators that need no file. "
            "Everything is deterministic arithmetic on the inputs: the tool does not estimate PDs or forecast "
            "the economy, and thresholds are the user's policy. Quote the numbers the tools return, say which "
            "settings produced them, and pass on the warnings."
        ),
    )

    @server.tool(annotations=READ_ONLY)
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
    def describe_loans(loans: str) -> dict[str, Any]:
        """What a loan file contains before any calculation: number of loans and borrowers, columns,
        totals, segments, days-past-due buckets, and gaps such as loans without origination risk."""
        return report.describe(runs.get_loans(loans)[1])

    @server.tool(annotations=READ_ONLY)
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
    def provision_matrix(
        gross_carrying_amount: list[float], loss_rate: list[float], buckets: list[str] | None = None
    ) -> dict[str, Any]:
        """Simplified approach for trade receivables: allowance per days-past-due bucket as gross carrying
        amount times lifetime loss rate, with totals and coverage."""
        return calc.provision_matrix(gross_carrying_amount, loss_rate, buckets)

    @server.tool(annotations=READ_ONLY)
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
