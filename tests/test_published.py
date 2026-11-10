"""Figures printed in public documents, reproduced exactly.

IE = IFRS 9 Illustrative Examples (IFRS Foundation), paragraph numbers given per test.
ITG = IFRS Transition Resource Group for Impairment of Financial Instruments, meeting summary 11 December 2015.
IG = IFRS 7 Implementation Guidance.
IRW = Israel, Rosenthal, Wei (2001), "Finding generators for Markov chains via empirical transition
matrices, with applications to credit ratings", Mathematical Finance 11(2).
"""

import numpy as np
import pytest

from eqlx_ifrs9 import PDCurve, expected_credit_loss, loss_rate_ecl, schedule, simplified, transitions
from eqlx_ifrs9.scenarios import weighted_ecl


def test_ie_example_8_explicit_pd_12_month_ecl():
    # IE49-IE52: CU1,000,000, 12-month PD 0.5%, LGD 25% -> CU1,250. The example ignores discounting.
    res = expected_credit_loss(PDCurve.from_cumulative([0.005], periods_per_year=1), [1_000_000], 0.25, 0.0)
    assert res.twelve_month == pytest.approx(1250.0, abs=1e-9)
    assert res.twelve_month_pd == 0.005


def test_ie_example_9_loss_rate_approach():
    # IE53-IE57: group X 1,000 loans of CU200, group Y 1,000 loans of CU300.
    # Historical: 4 and 2 defaults a year, present value of observed loss CU600 and CU450.
    gca = np.array([200_000.0, 300_000.0])
    historical_loss = np.array([600.0, 450.0])
    historical_defaults = np.array([4.0, 2.0])
    assert historical_loss / gca == pytest.approx([0.003, 0.0015])
    # Forward-looking: 5 and 3 defaults expected, loss per defaulting client unchanged.
    expected_loss = historical_loss / historical_defaults * np.array([5.0, 3.0])
    assert expected_loss == pytest.approx([750.0, 675.0])
    rates = expected_loss / gca
    assert rates == pytest.approx([0.00375, 0.00225])
    assert loss_rate_ecl(gca, rates) == pytest.approx([750.0, 675.0])


def test_ie_example_10_revolving_exposure():
    # IE58-IE65: drawn CU60,000, undrawn CU40,000. A quarter of the portfolio has a significant
    # increase in credit risk: drawn CU20,000, undrawn CU10,000, further drawdowns CU5,000 expected
    # over the remaining life. The rest: CU5,000 expected to be drawn within 12 months.
    lifetime = schedule.commitment_ead(20_000, 10_000, 5_000 / 10_000)
    twelve_month = schedule.commitment_ead(40_000, 30_000, 5_000 / 30_000)
    assert lifetime == pytest.approx([25_000.0])
    assert twelve_month == pytest.approx([45_000.0])


def test_ie_example_12_provision_matrix():
    # IE74-IE77
    pm = simplified.provision_matrix(
        [15_000_000, 7_500_000, 4_000_000, 2_500_000, 1_000_000],
        [0.003, 0.016, 0.036, 0.066, 0.106],
        ["current", "1-30", "31-60", "61-90", "over 90"],
    )
    assert pm.allowance == pytest.approx([45_000, 120_000, 144_000, 165_000, 106_000])
    assert pm.total_gross_carrying_amount == 30_000_000
    assert pm.total_allowance == pytest.approx(580_000)


@pytest.mark.parametrize(
    ("amounts", "rates", "printed", "total"),
    [
        ([20_777, 1_416, 673, 235], [0.001, 0.02, 0.05, 0.13], [21, 28, 34, 31], 114),
        ([19_222, 2_010, 301, 154], [0.002, 0.03, 0.08, 0.15], [38, 60, 24, 23], 145),
    ],
)
def test_ifrs7_ig20d_provision_matrices(amounts, rates, printed, total):
    # IG20D, CU'000: dealer financing and customer financing. Printed cells are rounded to whole thousands.
    pm = simplified.provision_matrix(amounts, rates)
    assert [round(float(a)) for a in pm.allowance] == printed
    assert sum(printed) == total
    assert pm.total_allowance == pytest.approx(total, abs=1.0)


def test_itg_probability_weighted_scenarios():
    # ITG 11 Dec 2015, paragraphs 50-51: unemployment 4% / 5% / 6% with likelihood 20% / 50% / 30%
    # gives ECL CU30 / CU70 / CU170. Weighted CU92, against CU70 for the most likely scenario alone.
    res = weighted_ecl([30, 70, 170], [0.2, 0.5, 0.3], ["4%", "5%", "6%"], base="5%")
    assert res.weighted == pytest.approx(92.0)
    assert res.base_ecl == 70.0
    assert res.weighted_to_base == pytest.approx(92 / 70)


# JLT (1997) Table 3 as reproduced in IRW (2001), section 4. States AAA, AA, A, BBB, BB, B, CCC, D.
JLT = np.array(
    [
        [0.8910, 0.0963, 0.0078, 0.0019, 0.0030, 0.0000, 0.0000, 0.0000],
        [0.0086, 0.9010, 0.0747, 0.0099, 0.0029, 0.0029, 0.0000, 0.0000],
        [0.0009, 0.0291, 0.8894, 0.0649, 0.0101, 0.0045, 0.0000, 0.0009],
        [0.0006, 0.0043, 0.0656, 0.8427, 0.0644, 0.0160, 0.0018, 0.0045],
        [0.0004, 0.0022, 0.0079, 0.0719, 0.7764, 0.1043, 0.0127, 0.0241],
        [0.0000, 0.0019, 0.0031, 0.0066, 0.0517, 0.8246, 0.0435, 0.0685],
        [0.0000, 0.0000, 0.0116, 0.0116, 0.0203, 0.0754, 0.6493, 0.2319],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
    ]
)
ROUNDED = 5e-4  # the published rows add up to 0.9998..1.0001


def test_irw_principal_logarithm_first_row():
    g = transitions.generator(JLT, "log", tol=ROUNDED)
    printed = [-0.115931, 0.107466, 0.004208, 0.001334, 0.003372, -0.000409, -0.000014, -0.000025]
    assert g.q[0] == pytest.approx(printed, abs=1e-6)
    assert g.negative_mass > 0 and g.warnings


def test_irw_generator_accuracy():
    # IRW report the summed absolute difference between P and exp(Q): 0.002736 for the diagonal
    # adjustment and 0.002686 for the weighted adjustment, against 0.116900 for the JLT approximation.
    assert transitions.generator(JLT, "da", tol=ROUNDED).l1_error == pytest.approx(0.002736, abs=5e-7)
    assert transitions.generator(JLT, "wa", tol=ROUNDED).l1_error == pytest.approx(0.002686, abs=5e-7)
    jlt = transitions.generator(JLT, "jlt", tol=ROUNDED)
    assert jlt.q[0, :5] == pytest.approx([-0.1154, 0.1019, 0.0083, 0.0020, 0.0031], abs=1e-4)
    # The paper's 0.116900 is reproduced to three decimals only (0.1165 here; it appears to have
    # been computed from the generator rounded to four decimals).
    assert jlt.l1_error == pytest.approx(0.1169, abs=5e-4)
    assert -np.log(np.linalg.det(JLT)) == pytest.approx(1.417263, abs=1e-6)


def test_irw_matrix_has_no_exact_generator():
    d = transitions.embedding_diagnostics(JLT, tol=ROUNDED)
    # AAA cannot move to B, CCC or D in one year, yet reaches them over several years.
    assert (0, 7) in d["reachable_but_zero"]
    assert d["may_have_generator"] is False
