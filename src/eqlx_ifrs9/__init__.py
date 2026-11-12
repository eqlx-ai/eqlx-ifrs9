"""Deterministic IFRS 9 expected credit loss calculations.

The library computes; it does not estimate models or forecast the economy. PDs, LGDs, exposures,
scenario paths and policy thresholds are inputs, and every convention the standard leaves open
(period length, timing of default, staging thresholds) is an explicit parameter.
"""

from . import (
    calc,
    config,
    cycle,
    data,
    engine,
    individual,
    lgd,
    report,
    scenarios,
    schedule,
    simplified,
    staging,
    transitions,
)
from .curves import PDCurve
from .ecl import ECLResult, discount_factors, expected_credit_loss, loss_rate_ecl
from .errors import InputError
from .schedule import Schedule

__version__ = "0.1.0"

__all__ = [
    "ECLResult",
    "InputError",
    "PDCurve",
    "Schedule",
    "__version__",
    "calc",
    "config",
    "cycle",
    "data",
    "discount_factors",
    "engine",
    "expected_credit_loss",
    "individual",
    "lgd",
    "loss_rate_ecl",
    "report",
    "scenarios",
    "schedule",
    "simplified",
    "staging",
    "transitions",
]
