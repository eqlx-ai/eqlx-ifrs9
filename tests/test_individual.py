"""Individual assessment of credit-impaired exposures.

The published figures come from the ECB Asset Quality Review Phase 2 Manual (June 2018), section 4.6,
and the ECB Guidance to banks on non-performing loans (March 2017), section 7.4.3. Where the source
prints a rounded figure, the test rounds the same way.
"""

import copy
import json
import random

import pytest

from eqlx_ifrs9 import individual as ind
from eqlx_ifrs9 import lgd, synthetic
from eqlx_ifrs9.cli import main
from eqlx_ifrs9.errors import InputError


def gone(exposure=100.0, eir=0.05, **collateral):
    item = {"id": "C", "value": 240.0, "secures": "L", "central_to_cash_flows": False, **collateral}
    return {
        "debtor": {"id": "D"},
        "exposures": [{"id": "L", "gross_carrying_amount": exposure, "eir": eir}],
        "collateral": [item],
        "scenarios": [{"name": "sale", "probability": 1.0, "approach": "gone_concern"}],
    }


# --- published examples ----------------------------------------------------------------------


def test_aqr_gone_concern_example():
    """AQR manual 4.6.3: exposure 100, appraisal 240, costs of 5% in year 1 and 10% in year 2, sale after
    two years, first lien pari passu with claims of 150, EIR 5%: proceeds 73.80, ECL 26.20."""
    case = gone(
        pari_passu_claims=150.0,
        sale={"years": 2, "haircut": 0.0, "costs": [{"year": 1, "amount": 12.0}, {"year": 2, "amount": 24.0}]},
    )
    r = ind.assess(case)
    sc = r["exposures"][0]["scenarios"][0]
    assert r["scenarios"][0]["collateral_sold"][0]["share"] == pytest.approx(0.40)
    assert round(sc["recoverable"], 2) == 73.80
    assert round(r["total"]["ecl"], 2) == 26.20
    by_hand = 0.4 * (240 / 1.05**2 - 12 / 1.05 - 24 / 1.05**2)
    assert sc["recoverable"] == pytest.approx(by_hand, rel=1e-12)


def test_npl_guidance_collateral_example():
    """NPL guidance 7.4.3: loan 300, open market value 200, market price discount 10%, selling costs 10%,
    maintenance 5% a year, sale after five years, EIR 5%: flows -10, -10, -10, -10, 152; present value 84;
    impairment 216."""
    c = ind.collateral_recovery(
        200.0, 0.05, years_to_sale=5, haircut=0.10, selling_cost_rate=0.10, annual_cost_rate=0.05
    )
    by_year = {}
    for f in c["flows"]:
        by_year[f["year"]] = by_year.get(f["year"], 0.0) + f["amount"]
    assert by_year == {1.0: -10.0, 2.0: -10.0, 3.0: -10.0, 4.0: -10.0, 5.0: 152.0}
    assert round(c["present_value"]) == 84
    assert round(300.0 - c["present_value"]) == 216


AQR_STEADY_STATE = {
    "revenue": 103_095,
    "cost_of_sales": 72_959,
    "distribution_costs": 10_172,
    "administrative_expenses": 4_307,
    "other_gains_losses": -493,
    "capex": 4_617,
    "sustainability_deductions": 205 + 155,
}


def test_aqr_steady_state_example():
    """AQR manual 4.6.6.1: EBITDA 15,164; cash flow adjustment -7,058; sustainability adjustment -360;
    sustainable cash flow 7,746; times 6 = 46,476."""
    s = ind.sustainable_cash_flow(**AQR_STEADY_STATE, income_tax=2_441)
    assert s["ebitda"] == 15_164
    assert s["cash_flow_adjustment"] == -7_058
    assert s["sustainability_adjustment"] == -360
    assert s["sustainable_cash_flow"] == 7_746
    assert s["sustainable_cash_flow"] * 6 == 46_476
    # The same with the tax derived: profit before tax 10,170 (EBITDA less 4,994 of interest and
    # depreciation) at 24.0% is 2,440.8, which the manual prints as 2,441.
    t = ind.sustainable_cash_flow(**AQR_STEADY_STATE, tax_rate=0.24, net_interest_expense=3_000, depreciation=1_994)
    assert t["profit_before_tax"] == 10_170
    assert round(t["income_tax"]) == 2_441
    assert round(t["sustainable_cash_flow"]) == 7_746


def test_aqr_seniority_example():
    """AQR manual 4.6.6.2: 35 to allocate; 25 more senior, exposure 15 pari passu with 15, 45 junior;
    cash flow value 5; pledged government bonds worth 3; recoverable 8; impairment 7."""
    assert ind.allocate(35.0, 15.0, preferred_claims=25.0, pari_passu_claims=15.0) == 5.0
    assert ind.waterfall(35.0, [(2, 15.0), (1, 25.0), (2, 15.0), (3, 45.0)]) == [5.0, 25.0, 5.0, 0.0]
    r = ind.assess(
        {
            "debtor": {"id": "D"},
            "exposures": [{"id": "i", "gross_carrying_amount": 15.0, "eir": 0.05, "rank": 2}],
            "other_claims": [
                {"name": "senior", "rank": 1, "amount": 25.0},
                {"name": "pari passu", "rank": 2, "amount": 15.0},
                {"name": "junior", "rank": 3, "amount": 45.0},
            ],
            "collateral": [
                {
                    "id": "bonds",
                    "value": 3.0,
                    "secures": "i",
                    "central_to_cash_flows": False,
                    "sale": {"years": 0, "haircut": 0.0},
                }
            ],
            "scenarios": [
                {
                    "name": "going",
                    "probability": 1.0,
                    "approach": "steady_state",
                    "sustainable_cash_flow": 35.0 / 6,
                    "multiple": 6,
                }
            ],
        }
    )
    sc = r["exposures"][0]["scenarios"][0]
    assert (sc["cash_flow_value"], sc["collateral_value"]) == (pytest.approx(5.0), 3.0)
    assert sc["recoverable"] == pytest.approx(8.0)
    assert r["total"]["ecl"] == pytest.approx(7.0)


TWO_STEP = {
    "debtor": {"id": "X"},
    "exposures": [
        {
            "id": "A",
            "gross_carrying_amount": 200.0,
            "eir": 0.07,
            "rank": 2,
            "expected_draws": [{"year": 1, "amount": 25.0}, {"year": 2, "amount": 25.0}],
        }
    ],
    "other_claims": [
        {"name": "senior creditors", "rank": 1, "amount": 200.0},
        {"name": "other syndicate banks", "rank": 2, "amount": 750.0},
    ],
    "scenarios": [
        {
            "name": "updated projection",
            "probability": 1.0,
            "approach": "two_step",
            "cash_flows": [-100, -100, 100, 100, 100, 100, 100, 100, 100, 100],
            "terminal": {"liquidation_value": 400},
        }
    ],
}


def test_aqr_two_step_example():
    """AQR manual 4.6.7: syndicated project loan, bank share 200 of 800 plus 50 of a 200 line drawn over
    two years, EIR 7%, liquidation value 400 in year 10. Flows to the bank 25 in years 5 to 9 and 125 in
    year 10; recoverable 141.70; exposure 245.20; ECL 103.50, of which 84.40 on the loan and 19.10 on the
    commitment. The manual rounds to the nearest 0.10."""
    r = ind.assess(TWO_STEP)
    e = r["exposures"][0]
    to_bank = [y["to_lender"]["A"] for y in r["scenarios"][0]["detail"]["years"]]
    assert to_bank == [0, 0, 0, 0, 25, 25, 25, 25, 25, 125]
    assert [y["to_other_creditors"] for y in r["scenarios"][0]["detail"]["years"]][2:4] == [100, 100]
    assert e["scenarios"][0]["recoverable"] == pytest.approx(141.70, abs=0.05)
    assert e["exposure"] == pytest.approx(245.20, abs=0.005)
    assert e["ecl"] == pytest.approx(103.50, abs=0.05)
    assert e["ecl_on_loan"] == pytest.approx(84.40, abs=0.05)
    assert e["ecl_on_commitment"] == pytest.approx(19.10, abs=0.05)


@pytest.mark.parametrize(("recoverable", "ecl"), [(80.0, 20.0), (120.0, 0.0)])
def test_aqr_cases_no_negative_allowance(recoverable, ecl):
    """AQR manual 4.6: exposure 100, recoverable 80 -> ECL 20; recoverable 120 -> ECL 0, not -20."""
    r = ind.assess(gone(sale={"years": 0, "haircut": 0.0}, value=recoverable))
    assert r["total"]["ecl"] == pytest.approx(ecl)
    assert r["exposures"][0]["scenarios"][0]["capped_at_exposure"] is (recoverable > 100.0)


# --- independent checks ----------------------------------------------------------------------


def test_waterfall_equals_the_published_formula():
    rng = random.Random(11)
    for _ in range(300):
        claims = [(rng.randint(1, 4), rng.choice([0.0, rng.uniform(1, 500)])) for _ in range(rng.randint(1, 8))]
        amount = rng.uniform(0, 1500)
        paid = ind.waterfall(amount, claims)
        assert sum(paid) == pytest.approx(min(amount, sum(c for _, c in claims)))
        for i, (rank, claim) in enumerate(claims):
            preferred = sum(c for r, c in claims if r < rank)
            pari = sum(c for j, (r, c) in enumerate(claims) if r == rank and j != i)
            assert paid[i] == pytest.approx(ind.allocate(amount, claim, preferred, pari), abs=1e-9)
            assert 0.0 <= paid[i] <= claim + 1e-9


def test_gone_concern_equals_the_credit_impaired_allowance():
    """The same flows through lgd.credit_impaired_allowance, which shares no code with this module."""
    sale = {"years": 3, "haircut": 0.2, "selling_cost_rate": 0.05, "annual_cost_rate": 0.02}
    r = ind.assess(gone(exposure=500.0, eir=0.31, value=600.0, sale=sale))
    price = 600.0 * 0.8
    flows = [-12.0, -12.0, price * 0.95 - 12.0]
    assert r["total"]["ecl"] == pytest.approx(lgd.credit_impaired_allowance(500.0, flows, [1, 2, 3], 0.31))


def test_part_year_holding_cost_and_income():
    c = ind.collateral_recovery(
        100.0, 0.10, years_to_sale=1.5, haircut=0.0, annual_cost_rate=0.04, income=[(0.5, 3.0), (1.5, 3.0)]
    )
    expected = 100 / 1.1**1.5 - 4 / 1.1 - 2 / 1.1**1.5 + 3 / 1.1**0.5 + 3 / 1.1**1.5
    assert c["present_value"] == pytest.approx(expected)


def test_prior_claims_are_served_first():
    c = ind.collateral_recovery(100.0, 0.0, years_to_sale=0, haircut=0.0, prior_claims=70.0, share=0.5)
    assert c["recoverable"] == 15.0
    assert ind.collateral_recovery(100.0, 0.0, years_to_sale=0, haircut=0.0, prior_claims=130.0)["recoverable"] == 0.0


def cases():
    return [copy.deepcopy(c) for c in synthetic.individual_cases().values()]


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["debtor"]["id"])
def test_probability_weighting_is_the_weighted_sum_of_single_scenarios(case):
    whole = ind.assess(case)
    for k, e in enumerate(whole["exposures"]):
        total = 0.0
        for sc in case["scenarios"]:
            alone = copy.deepcopy(case)
            alone["scenarios"] = [{**sc, "probability": 1.0}]
            total += sc["probability"] * ind.assess(alone)["exposures"][k]["ecl"]
        assert e["ecl"] == pytest.approx(total)
        assert 0.0 <= e["ecl"] <= e["exposure"]
        assert e["ecl_on_loan"] + e["ecl_on_commitment"] == pytest.approx(e["ecl"])
    assert whole["total"]["ecl"] == pytest.approx(sum(e["ecl"] for e in whole["exposures"]))


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["debtor"]["id"])
def test_order_of_the_input_does_not_matter(case):
    base = {e["id"]: e["ecl"] for e in ind.assess(case)["exposures"]}
    rng = random.Random(3)
    for _ in range(5):
        shuffled = copy.deepcopy(case)
        for key in ("exposures", "other_claims", "collateral", "scenarios"):
            rng.shuffle(shuffled.get(key, []))
        again = {e["id"]: e["ecl"] for e in ind.assess(shuffled)["exposures"]}
        assert again == pytest.approx(base)


def test_worse_assumptions_never_recover_more():
    def recoverable(**sale):
        base = {"years": 2, "haircut": 0.1, "selling_cost_rate": 0.03}
        return ind.assess(gone(exposure=1000.0, eir=0.3, sale={**base, **sale}))["exposures"][0]["scenarios"][0][
            "recoverable"
        ]

    for key, values in (
        ("haircut", [0.1, 0.2, 0.4, 1.0]),
        ("years", [0, 1, 2.5, 6]),
        ("selling_cost_rate", [0.0, 0.05, 0.5]),
        ("value_change", [0.0, -0.1, -0.6]),
    ):
        results = [recoverable(**{key: v}) for v in values]
        assert results == sorted(results, reverse=True) and len(set(results)) == len(results)
    assert recoverable(haircut=1.0) == 0.0


def test_collateral_central_to_cash_flows_is_not_sold_while_the_business_runs():
    case = cases()[0]
    r = ind.assess(case)
    going, liquidation = r["scenarios"]
    assert [c["id"] for c in going["collateral_sold"]] == ["PLEDGED-DEPOSIT"]
    assert [c["id"] for c in liquidation["collateral_sold"]] == ["FACTORY", "PLEDGED-DEPOSIT"]
    assert (
        liquidation["collateral_sold"][0]["haircut"] == 0.25 and liquidation["collateral_sold"][0]["years_to_sale"] == 3
    )


def test_collateral_shared_between_two_exposures_is_discounted_at_each_rate():
    case = {
        "debtor": {"id": "D"},
        "exposures": [
            {"id": "a", "gross_carrying_amount": 100.0, "eir": 0.10},
            {"id": "b", "gross_carrying_amount": 100.0, "eir": 0.50},
        ],
        "collateral": [
            {
                "id": "C",
                "value": 100.0,
                "secures": {"a": 0.25, "b": 0.75},
                "central_to_cash_flows": False,
                "sale": {"years": 2, "haircut": 0.2},
            }
        ],
        "scenarios": [{"name": "s", "probability": 1.0, "approach": "gone_concern"}],
    }
    a, b = (e["scenarios"][0]["recoverable"] for e in ind.assess(case)["exposures"])
    assert a == pytest.approx(25 * 0.8 / 1.1**2) and b == pytest.approx(75 * 0.8 / 1.5**2)


def test_market_price_scenario_and_comparison():
    case = copy.deepcopy(TWO_STEP)
    case["exposures"][0]["market_price"] = 0.5
    case["scenarios"][0]["probability"] = 0.7
    case["scenarios"].append({"name": "sale of the loan", "probability": 0.3, "approach": "market_price", "price": 0.5})
    r = ind.assess(case)
    e = r["exposures"][0]
    assert e["scenarios"][1]["recoverable"] == pytest.approx(0.5 * e["exposure"])  # price on loan and commitment
    assert e["ecl"] == pytest.approx(0.7 * (e["exposure"] - 141.7441863) + 0.3 * 0.5 * e["exposure"])
    assert [w["code"] for w in r["warnings"]] == ["above_market_price"]


def test_a_loan_priced_at_par_loses_nothing_even_with_an_undrawn_line():
    case = {
        "debtor": {"id": "D"},
        "exposures": [
            {"id": "L", "gross_carrying_amount": 100.0, "eir": 0.0, "expected_draws": [{"year": 1, "amount": 50.0}]}
        ],
        "scenarios": [{"name": "s", "probability": 1.0, "approach": "market_price", "price": 1.0}],
    }
    assert ind.assess(case)["total"] == {"exposure": 150.0, "ecl": 0.0, "coverage": 0.0}


def test_collateral_share_counts_expected_drawings():
    def case(gross, draw, pari=None, secures="L"):
        item = {"id": "C", "value": 100.0, "secures": secures, "central_to_cash_flows": False}
        item["sale"] = {"years": 0, "haircut": 0.0}
        if pari is not None:
            item["pari_passu_claims"] = pari
        loans = [
            {"id": "L", "gross_carrying_amount": gross, "eir": 0.0, "expected_draws": [{"year": 0, "amount": draw}]}
        ]
        loans.append({"id": "M", "gross_carrying_amount": 400.0, "eir": 0.0})
        return {
            "debtor": {"id": "D"},
            "exposures": loans,
            "collateral": [item],
            "scenarios": [{"name": "s", "probability": 1.0, "approach": "gone_concern"}],
        }

    def collected(c):
        return ind.assess(c)["exposures"][0]["scenarios"][0]["collateral_value"]

    assert collected(case(0.0, 100.0)) == 100.0  # nothing drawn yet: the collateral still belongs to the lender
    assert collected(case(50.0, 50.0, pari=100.0)) == 50.0  # 100 of exposure next to 100 of other claims
    assert collected(case(100.0, 0.0, pari=100.0, secures={"L": 1.0, "M": 0.0})) == 50.0  # M has no part in it


# --- warnings --------------------------------------------------------------------------------


def codes(case):
    return {w["code"] for w in ind.assess(case)["warnings"]}


def test_warnings_follow_the_published_expectations():
    sale = {"years": 2, "haircut": 0.05, "auction": True}
    assert codes(gone(sale=sale)) == {"low_haircut", "single_scenario"}
    assert "low_haircut" not in codes(gone(sale={**sale, "haircut": 0.10}))
    assert "low_haircut" in codes(gone(sale={"years": 1, "haircut": 0.0, "foreclosed_months_ago": 24}))
    assert "low_haircut" in codes(gone(sale={"years": 1, "haircut": 0.09, "vendor_finance": True}))
    assert "no_haircut" in codes(gone(sale={"years": 1, "haircut": 0.0}))
    assert "value_increase_assumed" in codes(gone(sale={"years": 1, "haircut": 0.2, "value_change": 0.05}))
    assert "stale_valuation" in codes(gone(sale={"years": 1, "haircut": 0.2}, valuation_age_months=13))
    assert "stale_valuation" not in codes(gone(sale={"years": 1, "haircut": 0.2}, valuation_age_months=12))

    going = cases()[0]
    assert codes(going) == {"stale_valuation", "no_haircut"}
    late = copy.deepcopy(going)
    late["debtor"]["months_past_due"] = 19
    late["debtor"]["ebitda_history"] = [-1.0, -2.0]
    late["scenarios"][0]["multiple"] = 6.5
    assert codes(late) == {
        "stale_valuation",
        "no_haircut",
        "gone_concern_presumed",
        "negative_ebitda",
        "multiple_above_benchmark",
    }
    late["debtor"]["sector"] = "utilities"
    assert "multiple_above_benchmark" not in codes(late)

    short = copy.deepcopy(TWO_STEP)
    short["scenarios"][0]["cash_flows"] = [100, 100]
    short["exposures"][0]["expected_draws"] = []
    assert "projection_length" in codes(short)
    for w in ind.assess(late)["warnings"]:
        assert w["source"] and w["message"] and w["where"]


def test_gone_concern_past_18_months_is_what_is_expected():
    assert codes(cases()[1]) == set()


def test_a_scenario_with_no_weight_is_not_a_second_scenario():
    case = gone(sale={"years": 1, "haircut": 0.2})
    case["scenarios"].append({"name": "never", "probability": 0.0, "approach": "gone_concern"})
    assert codes(case) == {"single_scenario"}


# --- input -----------------------------------------------------------------------------------


def broken(change):
    case = cases()[0]
    change(case)
    return case


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(notes="x"),
        lambda c: c["debtor"].update(sector="shipping"),
        lambda c: c["exposures"][0].update(gross_carrying_amount=True),
        lambda c: c["exposures"][0].update(eir=-1.0),
        lambda c: c["exposures"][0].pop("rank"),
        lambda c: c["exposures"].append(dict(c["exposures"][0])),
        lambda c: c["exposures"][0].update(rank=0),
        lambda c: c["collateral"][0].update(secures="NO-SUCH-LOAN"),
        lambda c: c["collateral"][0].update(secures={"TERM-LOAN": 0.5, "REVOLVING": 0.4}),
        lambda c: c["collateral"][0].update(share=0.5, pari_passu_claims=10.0),
        lambda c: c["collateral"][0].pop("central_to_cash_flows"),
        lambda c: c["collateral"][0]["sale"].pop("haircut"),
        lambda c: c["collateral"][0]["sale"].update(haircut=1.2),
        lambda c: c["collateral"][0]["sale"].update(costs=[{"year": 9, "amount": 1.0}]),
        lambda c: c["collateral"][0]["sale"].update(discount=0.1),
        lambda c: c["scenarios"][0].update(probability=0.5),
        lambda c: c["scenarios"][0].update(approach="hope"),
        lambda c: c["scenarios"][0].pop("multiple"),
        lambda c: c["scenarios"][0].update(sustainable_cash_flow=1.0),
        lambda c: c["scenarios"][0]["cash_flow"].update(ebitda=5.0),
        lambda c: c["scenarios"][0]["cash_flow"].pop("tax_rate"),
        lambda c: c["scenarios"][1].update(collateral={"WAREHOUSE": {"haircut": 0.5}}),
        lambda c: c["scenarios"][1].update(name="restructuring holds"),
        lambda c: c["scenarios"][1].update(cash_flows=[1, 2, 3]),
        lambda c: c.update(scenarios=[]),
        lambda c: c.update(exposures=[]),
        lambda c: c["debtor"].update(sector=[]),
        lambda c: c["debtor"].update(months_past_due="abc"),
        lambda c: c["debtor"].update(ebitda_history="abc"),
        lambda c: c["exposures"][0].update(gross_carrying_amount=10**400),
        lambda c: c["exposures"][0].update(gross_carrying_amount=1e19),
        lambda c: c["collateral"][0]["sale"].update(years=400),
        lambda c: c["collateral"][0]["sale"].update(value_change=1e6),
    ],
)
def test_bad_input_is_refused(change):
    with pytest.raises(InputError):
        ind.assess(broken(change))


def test_debtor_fields_are_checked_in_every_case():
    case = cases()[1]  # gone concern only: these fields are not used, and still checked
    case["debtor"]["months_past_due"] = "abc"
    with pytest.raises(InputError, match="months_past_due"):
        ind.assess(case)


def test_figures_that_overflow_are_refused_not_crashed():
    with pytest.raises(InputError):
        ind.collateral_recovery(100.0, -0.9, years_to_sale=400, haircut=0.1)
    with pytest.raises(InputError):
        ind.collateral_recovery(1e308, 0.05, years_to_sale=1, haircut=0.1, value_change=5)
    with pytest.raises(InputError, match="pairs"):
        ind.waterfall(10, [(1, 5), (1,)])
    flat = ind.sustainable_cash_flow(ebitda=10.0, income_tax=0.0)
    assert str(flat["cash_flow_adjustment"]) == "0.0" and str(flat["sustainability_adjustment"]) == "0.0"


def test_draws_after_the_projection_are_refused():
    case = copy.deepcopy(TWO_STEP)
    case["exposures"][0]["expected_draws"] = [{"year": 11, "amount": 5.0}]
    with pytest.raises(InputError, match="after the last projected year"):
        ind.assess(case)


def test_input_is_not_modified():
    case = cases()[0]
    before = copy.deepcopy(case)
    ind.assess(case)
    assert case == before


# --- command line and sample files -----------------------------------------------------------


def test_command_line(tmp_path, capsys):
    assert main(["sample", str(tmp_path), "--loans", "20"]) == 0
    capsys.readouterr()
    for name, case in synthetic.individual_cases().items():
        assert json.loads((tmp_path / name).read_text()) == case
        assert main(["individual", str(tmp_path / name)]) == 0
        assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(ind.assess(case)))
    (tmp_path / "bad.json").write_text("[1, 2")
    assert main(["individual", str(tmp_path / "bad.json")]) == 2
    (tmp_path / "list.json").write_text("[1, 2]")
    assert main(["individual", str(tmp_path / "list.json")]) == 2
    assert main(["individual", str(tmp_path / "missing.json")]) == 2
    (tmp_path / "latin.json").write_bytes(b'\xff\xfe{"a": 1}')
    assert main(["individual", str(tmp_path / "latin.json")]) == 2
    (tmp_path / "deep.json").write_text("[" * 5000 + "]" * 5000)
    assert main(["individual", str(tmp_path / "deep.json")]) == 2
    assert "error:" in capsys.readouterr().err
