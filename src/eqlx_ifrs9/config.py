"""Run settings for a portfolio calculation, read from a JSON object.

Everything the standard leaves to the entity lives here and nowhere else: staging thresholds,
scenarios and their weights, the forecast horizon and reversion, segment parameters and the
calculation conventions. A run is fully described by the loan file plus this object.

```json
{
  "timing": "end",
  "ead_basis": "start",
  "rate_convention": "nominal",
  "pd_basis": "long_run",
  "staging": {"relative_threshold": 2.0, "pd_measure": "annualized_lifetime", "scenario_basis": "weighted"},
  "profile": "tr_bddk",
  "scenarios": [
    {"name": "base", "weight": 0.6, "z": [0.0, 0.1]},
    {"name": "down", "weight": 0.2, "z": [-1.5, -1.0]},
    {"name": "up", "weight": 0.2, "z": [1.0, 0.5]}
  ],
  "base_scenario": "base",
  "forecast": {"reversion_years": 2, "method": "linear"},
  "segments": {"*": {"rho": 0.12}, "corporate": {"rho": 0.2, "grades": ["A", "B"], "matrix": [[...]]}}
}
```
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import transitions
from ._checks import Array
from .errors import InputError
from .scenarios import check_weights
from .staging import StagingPolicy

PdMeasure = Literal["annualized_lifetime", "lifetime", "twelve_month"]
ScenarioBasis = Literal["weighted", "base"]

_POLICY_KEYS = set(StagingPolicy.__dataclass_fields__) - {"precedence"}
_TOP_KEYS = {
    "timing",
    "ead_basis",
    "rate_convention",
    "staging",
    "profile",
    "profile_options",
    "scenarios",
    "base_scenario",
    "forecast",
    "segments",
    "pd_basis",
    "current_z",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    weight: float
    z: tuple[float, ...] | None
    """State of the economy per forecast year (z > 0 good). ``None``: PDs are used as given."""


@dataclass(frozen=True, eq=False)
class Segment:
    name: str
    rho: float = 0.0
    cpr: float = 0.0
    grades: tuple[str, ...] = ()
    matrix: Array | None = None
    generator: transitions.GeneratorMethod = "wa"
    matrix_tolerance: float = 1e-6

    def grade_index(self, label: str) -> int:
        try:
            return self.grades.index(label)
        except ValueError:
            raise InputError(f"grade {label!r} is not defined for segment {self.name!r}") from None


@dataclass(frozen=True, eq=False)
class Config:
    policy: StagingPolicy = field(default_factory=StagingPolicy)
    pd_measure: PdMeasure = "annualized_lifetime"
    scenario_basis: ScenarioBasis = "weighted"
    scenarios: tuple[Scenario, ...] = (Scenario("base", 1.0, None),)
    base_scenario: str = "base"
    reversion_years: int = 0
    reversion_method: Literal["linear", "exponential"] = "linear"
    reversion_decay: float = 0.5
    segments: dict[str, Segment] = field(default_factory=dict)
    timing: Literal["end", "mid"] = "end"
    ead_basis: Literal["start", "start_plus_interest", "mid"] = "start"
    rate_convention: Literal["nominal", "effective"] = "nominal"
    profile: str | None = None
    profile_options: dict[str, Any] = field(default_factory=dict)
    pd_basis: Literal["long_run", "current"] | None = None
    """What ``pd_12m`` in the loan file is: the long-run average PD, or the PD in today's economy."""
    current_z: float = 0.0
    """State of the economy today, needed to read a ``current`` PD (z > 0 good)."""
    source: dict[str, Any] = field(default_factory=dict)

    def segment(self, name: str) -> Segment:
        if name in self.segments:
            return self.segments[name]
        if "*" in self.segments:
            return self.segments["*"]
        return Segment(name)

    @property
    def weights(self) -> Array:
        return np.array([s.weight for s in self.scenarios])

    @property
    def base_index(self) -> int:
        return [s.name for s in self.scenarios].index(self.base_scenario)

    def to_dict(self) -> dict[str, Any]:
        """The settings as they were read, with defaults filled in for the top-level choices."""
        out = dict(self.source)
        out.setdefault("timing", self.timing)
        out.setdefault("ead_basis", self.ead_basis)
        out.setdefault("rate_convention", self.rate_convention)
        out["staging"] = {
            **self.policy.to_dict(),
            "pd_measure": self.pd_measure,
            "scenario_basis": self.scenario_basis,
        }
        out["scenarios"] = [
            {"name": s.name, "weight": s.weight, "z": None if s.z is None else list(s.z)} for s in self.scenarios
        ]
        out["base_scenario"] = self.base_scenario
        return out


def _choice(obj: dict[str, Any], key: str, allowed: tuple[str, ...], default: str) -> Any:
    v = obj.get(key, default)
    if v not in allowed:
        raise InputError(f"{key} must be one of {', '.join(allowed)}; got {v!r}")
    return v


def _object(obj: dict[str, Any], key: str) -> dict[str, Any]:
    v = obj.get(key)
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise InputError(f"{key} must be an object")
    return dict(v)


def _unknown(obj: dict[str, Any], allowed: set[str], where: str) -> None:
    extra = sorted(set(obj) - allowed)
    if extra:
        raise InputError(f"unknown setting(s) in {where}: {', '.join(extra)}")


def _number(obj: dict[str, Any], key: str, default: float, lo: float, hi: float, where: str) -> float:
    v = obj.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= float(v) <= hi:
        raise InputError(f"{where}.{key} must be a number in [{lo:g}, {hi:g}]; got {v!r}")
    return float(v)


def _segment(name: str, raw: Any) -> Segment:
    if not isinstance(raw, dict):
        raise InputError(f"segment {name!r} must be an object")
    _unknown(raw, {"rho", "cpr", "grades", "matrix", "generator", "matrix_tolerance"}, f"segment {name!r}")
    where = f"segments.{name}"
    rho = _number(raw, "rho", 0.0, 0.0, 0.999999, where)
    cpr = _number(raw, "cpr", 0.0, 0.0, 1.0, where)
    tol = _number(raw, "matrix_tolerance", 1e-6, 0.0, 0.01, where)
    method = _choice(raw, "generator", ("wa", "da", "jlt", "log"), "wa")
    grades = raw.get("grades", [])
    matrix = raw.get("matrix")
    if (matrix is None) != (not grades):
        raise InputError(f"{where}: 'grades' and 'matrix' must be given together")
    m: Array | None = None
    if matrix is not None:
        if not isinstance(grades, list) or not all(isinstance(g, str) and g for g in grades):
            raise InputError(f"{where}.grades must be a list of names, best first")
        if len(set(grades)) != len(grades):
            raise InputError(f"{where}.grades contains duplicates")
        m = transitions.check_matrix(matrix, tol=tol)
        if m.shape[0] != len(grades) + 1:
            raise InputError(
                f"{where}: the matrix has {m.shape[0]} states, so it needs {m.shape[0] - 1} grade names "
                "(default is the last state and is not named)"
            )
        if method == "log":
            raise InputError(f"{where}: generator 'log' may give negative probabilities; use 'wa', 'da' or 'jlt'")
    return Segment(name, rho, cpr, tuple(grades), m, method, tol)


def from_dict(raw: dict[str, Any] | None) -> Config:
    """Build a :class:`Config`, rejecting anything misspelled or out of range."""
    if raw is not None and not isinstance(raw, dict):
        raise InputError("the settings must be a JSON object")
    obj = dict(raw or {})
    _unknown(obj, _TOP_KEYS, "the settings")

    st = _object(obj, "staging")
    _unknown(st, _POLICY_KEYS | {"pd_measure", "scenario_basis"}, "staging")
    pd_measure = _choice(st, "pd_measure", ("annualized_lifetime", "lifetime", "twelve_month"), "annualized_lifetime")
    scenario_basis = _choice(st, "scenario_basis", ("weighted", "base"), "weighted")
    kw = {k: v for k, v in st.items() if k in _POLICY_KEYS}
    if "relative_bands" in kw:
        try:
            kw["relative_bands"] = tuple((float(a), float(b)) for a, b in kw["relative_bands"])
        except (TypeError, ValueError) as exc:
            raise InputError("staging.relative_bands must be a list of [upper origination PD, ratio]") from exc
    try:
        policy = StagingPolicy(**kw)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, InputError):
            raise InputError(f"staging: {exc}") from None
        raise InputError(f"staging: {exc}") from exc

    raw_scen = obj.get("scenarios")
    if raw_scen is None:
        scenarios: tuple[Scenario, ...] = (Scenario("base", 1.0, None),)
    else:
        if not isinstance(raw_scen, list) or not raw_scen:
            raise InputError("scenarios must be a non-empty list")
        built = []
        for s in raw_scen:
            if not isinstance(s, dict):
                raise InputError("each scenario must be an object")
            _unknown(s, {"name", "weight", "z"}, "a scenario")
            name = s.get("name")
            if not isinstance(name, str) or not name:
                raise InputError("each scenario needs a name")
            z = s.get("z")
            if z is not None:
                if (
                    not isinstance(z, list)
                    or not z
                    or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in z)
                ):
                    raise InputError(f"scenario {name!r}: z must be a non-empty list of numbers, one per forecast year")
                z = tuple(float(v) for v in z)
            built.append(Scenario(name, _number(s, "weight", -1.0, 0.0, 1.0, f"scenario {name!r}"), z))
        scenarios = tuple(built)
        if len({s.name for s in scenarios}) != len(scenarios):
            raise InputError("scenario names must be unique")
        check_weights([s.weight for s in scenarios])
    base = obj.get("base_scenario", scenarios[0].name if len(scenarios) == 1 else None)
    names = [s.name for s in scenarios]
    if base is None:
        base = "base" if "base" in names else None
    if base not in names:
        raise InputError(f"base_scenario must name one of the scenarios ({', '.join(names)})")

    fc = _object(obj, "forecast")
    _unknown(fc, {"reversion_years", "method", "decay"}, "forecast")
    rev = fc.get("reversion_years", 0)
    if isinstance(rev, bool) or not isinstance(rev, int) or rev < 0:
        raise InputError("forecast.reversion_years must be a non-negative whole number")

    seg_raw = obj.get("segments")
    seg_raw = {} if seg_raw is None else seg_raw
    if not isinstance(seg_raw, dict):
        raise InputError("segments must be an object keyed by segment name ('*' for all others)")
    segments = {name: _segment(name, s) for name, s in seg_raw.items()}

    profile = obj.get("profile")
    if profile not in (None, "tr_bddk"):
        raise InputError(f"unknown profile {profile!r}; available: tr_bddk")
    options = _object(obj, "profile_options")
    _unknown(options, {"days_in_year", "consumer_loans_at_loan_level", "as_of"}, "profile_options")
    if options.get("days_in_year", 365) not in (360, 365, 366) or isinstance(options.get("days_in_year"), bool):
        raise InputError("profile_options.days_in_year must be 360, 365 or 366")
    if not isinstance(options.get("consumer_loans_at_loan_level", True), bool):
        raise InputError("profile_options.consumer_loans_at_loan_level must be true or false")
    if not isinstance(options.get("as_of", ""), str):
        raise InputError("profile_options.as_of must be a date like 2026-12-31")

    pd_basis = obj.get("pd_basis")
    if pd_basis not in (None, "long_run", "current"):
        raise InputError("pd_basis must be 'long_run' or 'current'")
    if pd_basis is None and any(s.z is not None for s in scenarios):
        raise InputError(
            "scenarios with a z path need 'pd_basis': say whether pd_12m in the loan file is the long-run "
            "average PD ('long_run') or the PD in today's economy ('current', together with 'current_z')"
        )
    if pd_basis == "current" and "current_z" not in obj:
        raise InputError("pd_basis 'current' needs 'current_z', the state of the economy today")
    if pd_basis != "current" and "current_z" in obj:
        raise InputError("'current_z' is only used with pd_basis 'current'")
    current_z = _number(obj, "current_z", 0.0, -10.0, 10.0, "settings")

    return Config(
        pd_basis=pd_basis,
        current_z=current_z,
        policy=policy,
        pd_measure=pd_measure,
        scenario_basis=scenario_basis,
        scenarios=scenarios,
        base_scenario=base,
        reversion_years=rev,
        reversion_method=_choice(fc, "method", ("linear", "exponential"), "linear"),
        reversion_decay=_number(fc, "decay", 0.5, 0.0, 1.0, "forecast"),
        segments=segments,
        timing=_choice(obj, "timing", ("end", "mid"), "end"),
        ead_basis=_choice(obj, "ead_basis", ("start", "start_plus_interest", "mid"), "start"),
        rate_convention=_choice(obj, "rate_convention", ("nominal", "effective"), "nominal"),
        profile=profile,
        profile_options=options,
        source=obj,
    )


def load(path: str | Path) -> Config:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"{path}: not valid JSON ({exc})") from exc
    return from_dict(raw)


def with_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Copy of the settings with dotted keys replaced, e.g. ``{"staging.relative_threshold": 2.5}``."""
    out: dict[str, Any] = json.loads(json.dumps(raw))
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node = out
        for p in parts[:-1]:
            nxt = node.get(p)
            if nxt is None:
                nxt = {}
                node[p] = nxt
            if not isinstance(nxt, dict):
                raise InputError(f"cannot set {dotted!r}: {p!r} is not an object")
            node = nxt
        node[parts[-1]] = value
    return out
