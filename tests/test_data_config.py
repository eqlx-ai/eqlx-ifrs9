import json

import numpy as np
import pytest

from eqlx_ifrs9 import InputError, config, data, report

HEAD = "loan_id,balance,eir,remaining_months,lgd,pd_12m"


def test_minimal_file_and_defaults():
    loans = data.from_text(f"{HEAD}\nA,1000,0.06,36,0.45,0.02\n\nB,500,0.1,12,0.5,0.01\n")
    assert loans.n == 2 and loans.columns == tuple(HEAD.split(","))
    assert loans.text["borrower_id"].tolist() == ["A", "B"]
    assert loans.text["segment"].tolist() == ["all", "all"] and loans.text["product"].tolist() == ["term", "term"]
    assert loans.num["undrawn"].tolist() == [0, 0] and np.isnan(loans.num["rate"]).all()
    assert not loans.flag["forborne"].any()
    assert loans.index_of("B") == 1 and loans.row(0)["remaining_months"] == 36
    assert loans.take(np.array([1])).text["loan_id"].tolist() == ["B"]
    with pytest.raises(InputError):
        loans.index_of("C")


def test_flags_text_case_and_byte_order_mark(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text(
        "﻿Loan_ID,Balance,EIR,remaining_months,lgd,pd_12m,forborne,product,amortization\nA,1,0.1,3,0.5,0.1,YES,Revolving,Bullet\n",
        encoding="utf-8",
    )
    loans = data.read_csv(p)
    assert loans.flag["forborne"].tolist() == [True] and loans.text["product"].tolist() == ["revolving"]
    assert report.describe(loans)["flags"] == {"forborne": 1}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        (f"{HEAD}\n", "no rows"),
        (f"{HEAD},mood\nA,1,0.1,3,0.5,0.1,ok\n", "unknown column"),
        ("loan_id,balance,eir,lgd,pd_12m\nA,1,0.1,0.5,0.1\n", "missing required"),
        ("loan_id,balance,eir,remaining_months,lgd\nA,1,0.1,3,0.5\n", "pd_12m or a grade"),
        (f"{HEAD}\nA,1,0.1,3,0.5\n", "row 2: 5 cells"),
        (f"{HEAD}\nA,1,0.1,3,0.5,0.1\nA,1,0.1,3,0.5,0.1\n", "more than once"),
        (f"{HEAD}\n,1,0.1,3,0.5,0.1\n", "row 2, column loan_id"),
        (f"{HEAD}\nA,abc,0.1,3,0.5,0.1\n", "row 2, column balance: 'abc' is not a number"),
        (f"{HEAD}\nA,1,0.1,3,45,0.1\n", "row 2, column lgd: 45 is outside"),
        (f"{HEAD}\nA,1,0.1,3.5,0.5,0.1\n", "whole number"),
        (f"{HEAD}\nA,1,0.1,0,0.5,0.1\n", "remaining_months"),
        (f"{HEAD}\nA,,0.1,3,0.5,0.1\n", "column balance: empty"),
        (f"{HEAD},grade\nA,1,0.1,3,0.5,,\n", "neither pd_12m nor grade"),
        (f"{HEAD},forborne\nA,1,0.1,3,0.5,0.1,maybe\n", "yes/no"),
        (f"{HEAD},product\nA,1,0.1,3,0.5,0.1,lease\n", "not one of term"),
        (f"{HEAD},balance\nA,1,0.1,3,0.5,0.1,2\n", "duplicate column"),
        (f"{HEAD}\nA,1,0.1,3,0.5,nan\n", "'nan' is not a number"),
        (f"{HEAD}\nA,inf,0.1,3,0.5,0.1\n", "'inf' is not a number"),
        (f"{HEAD}\nA,1e400,0.1,3,0.5,0.1\n", "too large"),
        (f"{HEAD}\nA,1_000,0.1,3,0.5,0.1\n", "'1_000' is not a number"),
        (f"{HEAD}\nA,\u0661\u0660,0.1,3,0.5,0.1\n", "is not a number"),
        (f"{HEAD},dpd\nA,1,0.1,3,0.5,0.1,30.9\n", "whole number"),
        (f"{HEAD},undrawn\nA,1,0.1,3,0.5,0.1,nan\n", "is not a number"),
        (f"{HEAD}\n\n  , , , , , \nA,1,0.1,3,0.5,abc\n", "row 4, column pd_12m"),
    ],
)
def test_bad_files_stop_with_the_row(text, message):
    with pytest.raises(InputError, match=message):
        data.from_text(text)


def test_scientific_notation_quotes_and_windows_line_endings():
    loans = data.from_text(f'{HEAD}\r\n"A",1e3,6E-2,36,.45,2e-2\r\n')
    assert loans.num["balance"].tolist() == [1000.0] and loans.num["lgd"].tolist() == [0.45]
    assert data.from_text(f"\ufeff{HEAD}\nA,1,0.1,3,0.5,0.1\n").n == 1


def test_file_that_is_not_utf8(tmp_path):
    p = tmp_path / "latin.csv"
    p.write_bytes((HEAD + "\n\xe7,1,0.1,3,0.5,0.1\n").encode("latin-1"))
    with pytest.raises(InputError, match="not UTF-8"):
        data.read_csv(p)


def test_row_limit():
    with pytest.raises(InputError, match="more than 1 rows"):
        data.from_text(f"{HEAD}\nA,1,0.1,3,0.5,0.1\nB,1,0.1,3,0.5,0.1\n", max_rows=1)


def test_describe():
    text = f"{HEAD},dpd,pd_12m_origination,segment,previous_stage\nA,1000,0.06,36,0.45,0.02,0,0.01,x,1\nB,500,0.1,12,0.5,0.01,45,,y,\nC,1,0.1,6,0.5,0.01,120,,y,2\n"
    d = report.describe(data.from_text(text))
    assert d["loans"] == 3 and d["gross_carrying_amount"] == 1501 and d["segments"] == {"x": 1, "y": 2}
    assert d["days_past_due"] == {"0": 1, "1-30": 0, "31-90": 1, "over 90": 1}
    assert d["origination_risk_known"] == 1 and d["origination_risk_missing"] == 2
    assert d["previous_stage_given"] == 2 and d["remaining_months"] == {"min": 6, "median": 12.0, "max": 36}
    json.dumps(d)


def test_default_settings():
    cfg = config.from_dict(None)
    assert [s.name for s in cfg.scenarios] == ["base"] and cfg.weights.tolist() == [1.0]
    assert cfg.pd_measure == "annualized_lifetime" and cfg.timing == "end" and cfg.profile is None
    assert cfg.segment("anything").rho == 0.0
    d = cfg.to_dict()
    assert d["staging"]["backstop_ratio"] == 3.0 and d["scenarios"][0]["z"] is None
    json.dumps(d)


def test_segments_fall_back_to_star():
    cfg = config.from_dict({"segments": {"*": {"rho": 0.1}, "a": {"rho": 0.3, "cpr": 0.05}}})
    assert cfg.segment("a").rho == 0.3 and cfg.segment("b").rho == 0.1 and cfg.segment("b").name == "*"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"stagin": {}}, "unknown setting"),
        ({"staging": {"relative_treshold": 2}}, "unknown setting"),
        ({"staging": {"relative_threshold": 0.5}}, "ratio"),
        ({"staging": {"pd_measure": "annual"}}, "pd_measure"),
        ({"staging": {"relative_bands": [[0.01]]}}, "relative_bands"),
        ({"scenarios": []}, "non-empty"),
        ({"scenarios": [{"name": "a", "weight": 0.5}, {"name": "b", "weight": 0.4}]}, "add up to 1"),
        ({"scenarios": [{"name": "a", "weight": 0.5}, {"name": "a", "weight": 0.5}]}, "unique"),
        ({"scenarios": [{"name": "a", "weight": 1.0, "z": []}]}, "z must be"),
        ({"scenarios": [{"name": "a", "weight": 1.0, "z": [0.0]}]}, "pd_basis"),
        ({"scenarios": [{"name": "a", "weight": 0.5}, {"name": "b", "weight": 0.5}]}, "base_scenario"),
        ({"pd_basis": "current"}, "current_z"),
        ({"pd_basis": "long_run", "current_z": 0.5}, "only used"),
        ({"pd_basis": "ttc"}, "pd_basis"),
        ({"timing": "start"}, "timing"),
        ({"profile": "eu_eba"}, "unknown profile"),
        ({"profile_options": {"days": 360}}, "unknown setting"),
        ({"forecast": {"reversion_years": 1.5}}, "whole number"),
        ({"segments": {"a": {"rho": 1.0}}}, "rho"),
        ({"segments": {"a": {"grades": ["A"]}}}, "together"),
        ({"segments": {"a": {"grades": ["A", "B"], "matrix": [[0.9, 0.1], [0, 1]]}}}, "1 grade names"),
        ({"segments": {"a": {"grades": ["A"], "matrix": [[0.9, 0.2], [0, 1]]}}}, "adds up to"),
        (
            {"segments": {"a": {"grades": ["A"], "matrix": [[0.9, 0.1], [0, 1]], "generator": "log"}}},
            "negative probabilities",
        ),
        ({"segments": []}, "segments must be"),
        ({"staging": {"forborne_is_stage2": "False"}}, "true or false"),
        ({"staging": {"backstop_ratio": "3"}}, "must be a number"),
        ({"staging": {"relative_threshold": True}}, "must be a number"),
        ({"staging": {"probation_periods": 1.5}}, "whole number"),
        ({"staging": {"notch_threshold": 0}}, "at least 1"),
        ({"staging": 5}, "staging must be an object"),
        ({"forecast": [1]}, "forecast must be an object"),
        ({"profile_options": {"days_in_year": "abc"}}, "days_in_year"),
        ({"profile_options": {"days_in_year": 365.9}}, "days_in_year"),
        ({"profile_options": {"consumer_loans_at_loan_level": "false"}}, "true or false"),
        ([1, 2], "JSON object"),
    ],
)
def test_bad_settings_are_rejected(raw, message):
    with pytest.raises(InputError, match=message):
        config.from_dict(raw)


def test_overrides_by_dotted_key():
    raw = {"staging": {"relative_threshold": 2.0}}
    changed = config.with_overrides(raw, {"staging.relative_threshold": 3.0, "forecast.reversion_years": 2})
    assert raw["staging"]["relative_threshold"] == 2.0  # the original is not touched
    cfg = config.from_dict(changed)
    assert cfg.policy.relative_threshold == 3.0 and cfg.reversion_years == 2
    with pytest.raises(InputError):
        config.with_overrides({"staging": 1}, {"staging.relative_threshold": 3.0})


def test_load_from_file(tmp_path):
    p = tmp_path / "s.json"
    p.write_text('{"timing": "mid"}')
    assert config.load(p).timing == "mid"
    p.write_text("{not json")
    with pytest.raises(InputError, match="not valid JSON"):
        config.load(p)
