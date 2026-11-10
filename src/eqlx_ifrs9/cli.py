"""Command line: ``ifrs9 run | loan | list | compare | describe | ecl | sample | mcp``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__, calc, config, data, engine, report, synthetic
from .errors import InputError
from .staging import Reason, names


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """``staging.relative_threshold=2.5`` -> ``{"staging.relative_threshold": 2.5}`` (values are JSON)."""
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise InputError(f"--set expects key=value, got {item!r}")
        try:
            out[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            out[key.strip()] = value
    return out


def load_run(loans_path: str, config_path: str | None, overrides: dict[str, Any]) -> engine.PortfolioResult:
    raw: dict[str, Any] = {}
    if config_path:
        try:
            raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InputError(f"{config_path}: not valid JSON ({exc})") from exc
    cfg = config.from_dict(config.with_overrides(raw, overrides) if overrides else raw)
    return engine.run(data.read_csv(loans_path), cfg)


def write_results(r: engine.PortfolioResult, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        head = ["loan_id", "stage", "primary_reason", "reasons", "exposure", "ecl", "ecl_12m", "ecl_lifetime"]
        head += ["pd_now", "pd_origination"]
        if r.group is not None:
            head.append("regulatory_group")
        w.writerow(head)
        for i in range(r.n):
            row: list[Any] = [
                r.loans.text["loan_id"][i],
                int(r.stage[i]),
                Reason(int(r.primary[i])).name.lower() if r.primary[i] else "none",  # type: ignore[union-attr]
                "|".join(names(int(r.reasons[i]))),
                repr(float(r.exposure[i])),
                repr(float(r.ecl[i])),
                repr(float(r.ecl_12m[i])),
                repr(float(r.ecl_lifetime[i])),
                repr(float(r.pd_now[i])),
                "" if r.pd_origination[i] != r.pd_origination[i] else repr(float(r.pd_origination[i])),
            ]
            if r.group is not None:
                row.append(int(r.group[i]))
            w.writerow(row)


def _emit(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ifrs9", description="Deterministic IFRS 9 expected credit loss calculations.")
    p.add_argument("--version", action="version", version=f"eqlx-ifrs9 {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def portfolio(name: str, help_text: str) -> argparse.ArgumentParser:
        s = sub.add_parser(name, help=help_text)
        s.add_argument("loans", help="loan file (CSV)")
        s.add_argument("--config", help="settings file (JSON)")
        s.add_argument(
            "--set",
            action="append",
            metavar="KEY=VALUE",
            help="override a setting, e.g. staging.relative_threshold=2.5",
        )
        return s

    s = portfolio("run", "stage and ECL for a loan file")
    s.add_argument("--json", action="store_true", help="print the full summary as JSON")
    s.add_argument("--out", help="write one row per loan to this CSV file")

    s = portfolio("loan", "everything behind one loan's stage and ECL")
    s.add_argument("loan_id")
    s.add_argument("--periods", action="store_true", help="include the month-by-month table")

    s = portfolio("list", "loans matching a filter")
    s.add_argument("--stage", type=int, choices=(1, 2, 3))
    s.add_argument("--reason")
    s.add_argument("--segment")
    s.add_argument("--borrower")
    s.add_argument("--sort", default="ecl")
    s.add_argument("--limit", type=int, default=20)

    s = portfolio("compare", "effect of changing settings: --set gives the changed run")
    s.add_argument("--limit", type=int, default=20)

    s = sub.add_parser("describe", help="what a loan file contains, before any calculation")
    s.add_argument("loans")

    s = sub.add_parser("ecl", help="12-month and lifetime ECL of one instrument")
    s.add_argument("--balance", type=float, required=True)
    s.add_argument("--pd", type=float, required=True, help="12-month PD")
    s.add_argument("--lgd", type=float, required=True)
    s.add_argument("--eir", type=float, required=True)
    s.add_argument("--months", type=int, required=True)
    s.add_argument("--amortization", default="bullet", choices=("bullet", "annuity", "equal_principal"))
    s.add_argument("--rate", type=float)
    s.add_argument("--timing", default="end", choices=("end", "mid"))
    s.add_argument("--periods", action="store_true")

    s = sub.add_parser("sample", help="write a made-up portfolio with known answers")
    s.add_argument("directory")
    s.add_argument("--loans", type=int, default=2000)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--no-profile", action="store_true", help="leave out the Turkey (BDDK) profile")

    s = sub.add_parser("mcp", help="start the read-only MCP server for AI agents")
    s.add_argument("--root", required=True, help="the only folder the server may read")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (InputError, FileNotFoundError, NotADirectoryError, IsADirectoryError, PermissionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    cmd = args.command
    if cmd == "mcp":
        from .mcp_server import build_server

        build_server(Path(args.root)).run()
        return 0
    if cmd == "describe":
        _emit(report.describe(data.read_csv(args.loans)))
        return 0
    if cmd == "ecl":
        _emit(
            calc.instrument_ecl(
                balance=args.balance,
                pd_12m=args.pd,
                lgd=args.lgd,
                eir=args.eir,
                remaining_months=args.months,
                amortization=args.amortization,
                rate=args.rate,
                timing=args.timing,
                periods=args.periods,
            )
        )
        return 0
    if cmd == "sample":
        out = Path(args.directory)
        out.mkdir(parents=True, exist_ok=True)
        sample = synthetic.generate(args.loans, args.seed, None if args.no_profile else "tr_bddk")
        (out / "loans.csv").write_text(sample.csv, encoding="utf-8")
        (out / "settings.json").write_text(json.dumps(sample.settings, indent=2) + "\n", encoding="utf-8")
        (out / "planted_cases.json").write_text(
            json.dumps([p.__dict__ for p in sample.planted], indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote loans.csv, settings.json and planted_cases.json to {out}")
        return 0

    overrides = parse_overrides(args.set)
    if cmd == "compare":
        if not overrides:
            raise InputError("compare needs at least one --set KEY=VALUE for the changed run")
        before = load_run(args.loans, args.config, {})
        after = load_run(args.loans, args.config, overrides)
        _emit({"changed_settings": overrides, **report.compare(before, after, args.limit)})
        return 0
    result = load_run(args.loans, args.config, overrides)
    if cmd == "run":
        table = report.summary(result)
        print(json.dumps(table, indent=2, ensure_ascii=False) if args.json else report.text_summary(table))
        if args.out:
            write_results(result, args.out)
        return 0
    if cmd == "loan":
        _emit(engine.loan_detail(result, args.loan_id, periods=args.periods))
        return 0
    if cmd == "list":
        _emit(
            report.loan_rows(
                result,
                stage=args.stage,
                reason=args.reason,
                segment=args.segment,
                borrower_id=args.borrower,
                sort=args.sort,
                limit=args.limit,
            )
        )
        return 0
    raise InputError(f"unknown command {cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
