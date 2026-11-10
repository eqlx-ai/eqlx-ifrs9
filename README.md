# eqlx-ifrs9

Deterministic IFRS 9 expected credit loss: PD term structures, stage allocation, 12-month and
lifetime ECL, scenario weighting, the provision matrix, and a rule profile for Turkey (BDDK).
A Python library, a command line tool and a read-only MCP server for AI agents.

It computes. It does not estimate PD or LGD models and it does not forecast the economy: PDs, LGDs,
scenario paths and thresholds are inputs. Same input, same output, and every figure can be traced
back to the loan, the rule and the setting that produced it.

- **Explicit choices.** IFRS 9 leaves a lot to the entity: what "significant" means, how many
  scenarios, when inside a period default happens. Each of those is a named setting, never a hidden
  default, and settings that supervisors have publicly criticised produce a warning.
- **Checked against published numbers.** The IASB's own illustrative examples, the transition
  resource group's scenario example and a published rating matrix are reproduced in the test-suite;
  everything else is checked against closed forms, direct enumeration and simulation.
- **Built for agents.** The agent decides what to look at; the tool returns the number and the
  reason. The MCP server only reads files in one folder and writes nothing.

## Who it is for

Credit risk and finance teams who want an independent recalculation of their allowance, auditors and
model validators who need a transparent benchmark, teams in smaller lenders who do not run a vendor
engine, and people building AI agents for credit work.

## Install

```bash
pip install eqlx-ifrs9            # library and command line
pip install "eqlx-ifrs9[mcp]"     # with the MCP server
```

Python 3.10 or later. Dependencies: NumPy and SciPy.

## A first look

```bash
ifrs9 sample demo                 # a made-up portfolio of 2,011 loans, settings, and known answers
ifrs9 run demo/loans.csv --config demo/settings.json
```

```text
loans 2,011   exposure 2,828,567,386.34   ECL 102,727,651.27   coverage 3.63%

stage      loans            exposure    share               ECL  coverage
1          1,602    2,237,464,647.42    79.1%     12,357,793.30     0.55%
2            353      453,760,033.72    16.0%     13,361,070.87     2.94%
3             56      137,342,705.20     4.9%     77,008,787.11    56.07%

stage 2 by primary reason
  pd_relative                    241      325,253,083.84     11,191,309.94
  days_past_due_backstop          55       64,521,505.67      1,441,479.13
  forborne                        29       37,901,342.97        536,260.82
  watchlist                       27       25,984,101.23        190,225.11
  probation                        1          100,000.00          1,795.86

scenarios (ECL if one scenario alone had full weight)
  base          weight   60%  same stages    94,513,659.98  restaged    92,832,581.87  (50 loans change stage)
  downside      weight   25%  same stages   132,726,339.74  restaged   142,055,113.01  (330 loans change stage)
  upside        weight   15%  same stages    85,585,802.36  restaged    83,836,400.27  (151 loans change stage)
  weighted / base: 1.107

checks
  stage 1 loans whose PD has tripled since origination: 0
  stage 2 only because of days past due: 55 (14.2% of stage 2 exposure)
  stage 1 kept by the low credit risk exemption: 1
  loans without origination PD: 0
```

Then ask about one loan, or about a different policy:

```bash
ifrs9 loan demo/loans.csv CASE-GROUP-X-CURRENT --config demo/settings.json
ifrs9 compare demo/loans.csv --config demo/settings.json --set staging.relative_threshold=3
ifrs9 list demo/loans.csv --config demo/settings.json --stage 2 --reason days_past_due_backstop
```

The first command explains a loan that is not past due itself and still sits in stage 3:

```json
{
  "loan_id": "CASE-GROUP-X-CURRENT",
  "stage": 3,
  "reasons": ["profile_default"],
  "regulatory_group": 4,
  "regulatory_reasons": [
    "another loan of the borrower is non-performing: same group 4 (Art. 5(5))"
  ],
  "exposure": 5000000.0,
  "ecl": 2750000.0,
  "measurement": "exposure x loss rate in default"
}
```

## As a library

```python
from eqlx_ifrs9 import PDCurve, cycle, expected_credit_loss, schedule
from eqlx_ifrs9.staging import StagingPolicy, assess

# A 36-month instalment loan, 12-month PD 3%, LGD 60%, effective interest rate 31%.
loan = schedule.annuity(100_000, 0.30, 36)
curve = PDCurve.flat(0.03, 36)
res = expected_credit_loss(curve, loan.ead(), lgd=0.60, effective_interest_rate=0.31)
res.twelve_month, res.lifetime  # 1406.99, 2351.36

# The same loan in a downturn: long-run PD shifted year by year (z < 0 is a bad year).
down = cycle.pit_curve(curve, rho=0.06, z_by_year=[-1.6, -1.0, -0.4])
expected_credit_loss(down, loan.ead(), 0.60, 0.31).lifetime  # 4226.37

# Has credit risk increased significantly? Compare with the PD that was expected at origination
# for the SAME remaining window: two years into a five-year loan, that is years 3 to 5.
at_origination = PDCurve.from_cumulative([0.01, 0.025, 0.045, 0.07, 0.10], periods_per_year=1)
expected_then = at_origination.forward(2).pd_over()  # 0.0769, not the five-year 0.10
r = assess(StagingPolicy(relative_threshold=2.0), pd_now=[0.16], pd_origination=[expected_then])
r.stage[0], r.reason_names(0), r.pd_ratio[0]  # 2, ['pd_relative'], 2.08
```

Modules: `curves` (PD term structures), `transitions` (rating matrices, generators, cycle shift),
`cycle` (one-factor point-in-time adjustment), `schedule` (repayment and exposure), `lgd`
(collateral, workout, credit-impaired assets), `ecl`, `staging`, `scenarios`, `simplified`
(provision matrix), `profiles.tr_bddk`, and for whole portfolios `data`, `config`, `engine`,
`report`.

## For AI agents (MCP)

```bash
ifrs9 mcp --root /path/to/folder
```

Claude Code, Claude Desktop or any other MCP client can then use these tools. All are read-only and
confined to that folder; nothing is written and nothing leaves the machine.

| Tool | What it returns |
|---|---|
| `list_files` | loan files (.csv) and settings files (.json) in the folder |
| `describe_loans` | what a loan file contains, and what is missing, before any calculation |
| `run_portfolio` | stage and ECL for every loan: tables by stage, segment, reason and scenario, checks, warnings |
| `list_loans` | loans matching a filter, with every rule that fired |
| `explain_loan` | one loan step by step: staging test, PD and ECL per scenario and year |
| `compare_settings` | what a different threshold, weight or scenario would change |
| `calculate_ecl` | 12-month and lifetime ECL of one instrument from plain numbers |
| `pd_term_structure` | cumulative PD by grade from a transition matrix, with generator diagnostics |
| `provision_matrix` | simplified approach for trade receivables |
| `turkey_reclassification_check` | BDDK conditions for moving a loan back up, with article numbers |

## Input

**Loan file** (CSV, one row per instrument). Required: `loan_id`, `balance`, `eir`,
`remaining_months`, `lgd`, and `pd_12m` or `grade`. Optional columns cover the borrower, segment,
repayment type, undrawn limits, origination risk, days past due and status flags; the full list is
in [`eqlx_ifrs9/data.py`](src/eqlx_ifrs9/data.py). Rates and PDs are fractions (`0.05` for 5%).

**Settings** (JSON). Staging thresholds, scenarios with weights and a z path per forecast year,
reversion after the forecast horizon, segment parameters (asset correlation, prepayment, a
transition matrix), calculation conventions and an optional profile. `ifrs9 sample` writes a
complete example; [`eqlx_ifrs9/config.py`](src/eqlx_ifrs9/config.py) documents every key.
Unknown keys and out-of-range values are rejected.

## How it is verified

| What | Checked against |
|---|---|
| 12-month ECL, explicit PD | IFRS 9 Illustrative Example 8: CU1,250 |
| Loss rate approach | Illustrative Example 9: CU750 and CU675 |
| Exposure of revolving facilities | Illustrative Example 10: CU25,000 and CU45,000 |
| Provision matrix | Illustrative Example 12: CU580,000; IFRS 7 IG20D: CU114 and CU145 thousand |
| Probability weighting | Transition resource group, December 2015: CU92 against CU70 for one scenario |
| Generators of rating matrices | Israel, Rosenthal and Wei (2001): error norms 0.002736 and 0.002686 |
| Lifetime ECL with discounting | closed form for a bullet loan; continuous-time limit; direct enumeration |
| The same, amortising loan | simulation of two million default times |
| One-factor shift | the average over the cycle returns the long-run PD and matrix (numerical integration) |
| Portfolio run | every loan recomputed with the scalar functions; independence of row order and chunking |
| Turkey profile | article by article against the regulation text; contagion against direct enumeration |

The illustrative examples do not cover multi-period discounting, several scenarios or a numeric
staging threshold. Those parts rest on the closed forms, the enumeration and the simulation.
Tests also pin down frequent mistakes: summing conditional PDs without survival weighting, using a
12-month PD on a six-month loan, dividing an annual PD by twelve, giving a bullet loan no exposure
in its last period, and comparing today's remaining-life PD with the full-life PD at origination.

## Limits

This is a calculation tool, not an accounting opinion and not a validated model. It covers the
impairment arithmetic of IFRS 9 Section 5.5 for loans, undrawn commitments and trade receivables.
It does not cover classification and measurement, hedge accounting, purchased or originated
credit-impaired assets, or the allowance movement table. See [LIMITS.md](LIMITS.md) before relying
on a number.

## References

- IFRS 9 *Financial Instruments*, Section 5.5 and Application Guidance B5.5; Illustrative Examples
  (IFRS Foundation). TFRS 9 is the Turkish equivalent with the same paragraph numbers (KGK).
- IFRS 7 *Financial Instruments: Disclosures*, paragraphs 35A-35N and Implementation Guidance IG20D.
- IFRS Transition Resource Group for Impairment of Financial Instruments, meeting summaries 2015.
- Basel Committee on Banking Supervision (2015), *Guidance on credit risk and accounting for expected
  credit losses*.
- European Banking Authority (2021, 2023), *IFRS 9 implementation by EU institutions*, monitoring
  reports; ECB Banking Supervision, letter of 4 December 2020 on credit risk identification.
- BDDK, *Kredilerin Sınıflandırılması ve Bunlar İçin Ayrılacak Karşılıklara İlişkin Usul ve Esaslar
  Hakkında Yönetmelik*, Official Gazette 22 June 2016, no. 29750, as amended.
- Vasicek, O. (2002), "The distribution of loan portfolio value", *Risk*.
- Belkin, B., Suchower, S. and Forest, L. (1998), "A one-parameter representation of credit risk and
  transition matrices", *CreditMetrics Monitor*.
- Israel, R., Rosenthal, J. and Wei, J. (2001), "Finding generators for Markov chains via empirical
  transition matrices, with applications to credit ratings", *Mathematical Finance* 11(2).
- Jarrow, R., Lando, D. and Turnbull, S. (1997), "A Markov model for the term structure of credit
  risk spreads", *Review of Financial Studies* 10(2).

## License

Apache-2.0. Part of [eqlx.org](https://eqlx.org): deterministic tools for AI agents in credit and
finance. Questions: research@eqlx.ai.
