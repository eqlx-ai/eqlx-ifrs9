# Limits

Read this before relying on a number.

## What the tool is

Arithmetic on your inputs. It does not estimate PDs, LGDs or correlations, does not forecast the
economy and does not choose thresholds. A result is only as good as the parameters and the policy
you give it, and it is not an accounting opinion, an audit or a validated model.

## Not covered

- **Classification and measurement** (business model, SPPI test) and **hedge accounting**.
- **Purchased or originated credit-impaired assets** (credit-adjusted effective interest rate,
  allowance as the cumulative change in lifetime ECL).
- **Allowance movement table** (IFRS 7 35H-35I) and interest revenue by stage. Planned.
- **Modification gains and losses.** A restructured loan is staged through its flags; the change in
  gross carrying amount is not computed.
- **Financial guarantees and lease receivables** as instruments of their own. A guarantee can be
  entered as an undrawn exposure with a drawdown share; nothing lease-specific exists.
- **Foreign currency.** A loan file is treated as one currency.
- **Collective assessment by proportion.** A `collective` flag per loan is supported; moving a share
  of a segment to lifetime ECL without naming the loans is not.

## Conventions you should know

- **Monthly periods** in the portfolio run. The scalar functions accept any period length.
- **Default timing** is the end of the month by default (`timing: "mid"` for mid-month). **Exposure**
  is the balance before the month's payment by default. Both are settings; the standard prescribes
  neither.
- **Discounting** is at the effective interest rate you supply, compounded as
  `(1 + rate) ** (months / 12)`, to the reporting date.
- **Contract rate.** `rate` drives the repayment schedule and is read as a nominal annual rate
  (divided by 12) unless `rate_convention` says otherwise. Without a `rate` column the schedule is
  built from `eir` as an effective rate, so that schedule and discounting agree.
- **`ead_basis: "mid"`** averages the balance before and after the month's payment. For a bullet
  loan that halves the exposure of the final month.
- **LGD** is taken as given, measured at the default date, constant over the life of the loan in the
  portfolio run. Use the `lgd` module for collateral-based term structures in your own code. Do not
  feed regulatory downturn LGDs: IFRS 9 asks for a best estimate.
- **Stage 3** is measured as exposure times `lgd_default`. It is not scenario-weighted and not
  discounted again; put the present value into the rate.
- **Constant hazard.** A loan with only `pd_12m` gets a flat curve. An origination PD given as
  `pd_12m_origination` is extended the same way, so for those loans the comparison reduces to a
  comparison of annual PDs. Supply `pd_origination_remaining` or grades with a matrix for a real term
  structure.
- **Origination PD from a grade** uses the segment's long-run matrix, not the forward-looking view
  that existed at origination.

## Scenarios and the one-factor model

- A scenario is a path of z, the standardised state of the economy, one value per forecast year
  (z > 0 is a good year). Where z comes from is outside this tool.
- **z = 0 is the median year, not the average.** At z = 0 the conditional PD is below the long-run
  PD. A base scenario of zeros lowers PDs.
- You must say whether `pd_12m` is the long-run average (`pd_basis: "long_run"`) or the PD in
  today's economy (`"current"`, with `current_z`). The tool refuses to guess.
- After the forecast horizon the PD reverts to the long-run PD: immediately with
  `reversion_years: 0`, or through that many transitional years (with 1, the first year after the
  horizon is half-way). The last z is held while the weight of the shifted PD falls.
- A segment that appears in the loan file needs an entry under `segments` (or a `*` entry) as soon
  as a scenario has a z path; the run stops otherwise instead of assuming a correlation of zero.
- The shift is applied to the annualised PD of each 12-month block and the hazard inside the block
  is scaled. Asset correlation is one number per segment.
- Transition matrices are shifted year by year and turned into monthly matrices through a fitted
  generator. Most published matrices have no exact generator; the tool says so and reports the
  error of the approximation.
- Staging uses the scenario-weighted PD by default. One stage per loan is assigned and the ECL of
  that stage is weighted. Assigning a stage per scenario and weighting afterwards gives a different
  number; the per-scenario table shows it.

## Staging

- Thresholds are yours. The warnings repeat points that supervisors have made in public reports;
  they are not rules of the standard.
- The low credit risk exemption switches off the PD and grade tests only. Days past due and the
  forborne, watch-list and collective flags still move a loan to stage 2.
- Without an origination PD (or origination grade, for a notch rule) there is nothing to compare
  with: the loan goes to stage 2 unless it is low credit risk (IFRS 9 7.2.20), or the run stops,
  or the tests are skipped, depending on `missing_origination_pd`.
- Probation and cure periods need `previous_stage` and `periods_without_trigger` from your own
  history; the tool does not keep state between runs, and it warns when the history is missing.
  A loan leaving stage 3 passes through stage 2 for the probation period.
- "Primary reason" depends on the order of precedence, which is a setting.

## Individual assessment

- **Judgement stays with you.** Whether the business survives, how likely each scenario is, what the
  collateral is worth, how long a sale takes and how the claims rank in law are inputs. The tool does
  the arithmetic and says where an input departs from a published expectation.
- **The warnings are ECB expectations**, from the asset quality review manual (2018) and the guidance
  on non-performing loans (2017): multiples of 6, 10 and 12, a discount of at least 10% in an auction,
  18 months past due, valuations within 12 months. They are not rules of IFRS 9 and not rules of any
  other supervisor. They never change a result.
- **Not linked to the portfolio run.** `ifrs9 run` still measures stage 3 as exposure times
  `lgd_default`. Take the allowance from the case and put it into the rate, or report it separately.
- **Annual steps** in the two-step approach; cash is assumed to arrive at the end of each year.
  Collateral sales and other dated flows can fall at any fraction of a year.
- **Claims are nominal.** Other creditors' claims and the lender's exposures do not accrue interest
  while cash is shared; a negative cash flow year pays nobody and is not carried forward. Expected
  drawings of undrawn lines are given by you, with their year.
- **Steady state** treats the multiple as the present value: the amount is shared at once and not
  discounted again. A negative sustainable cash flow counts as zero.
- **Collateral and the waterfall are separate.** Proceeds from collateral are added to an exposure's
  share of the cash flows; they do not reduce the claim that takes part in the sharing, and collateral
  worth more than its exposure does not flow to other exposures. The total is capped at the exposure.
- **Prior claims on collateral** are deducted from the present value of the net proceeds, then the
  rest is shared with equal-ranking claims in proportion to the amounts owed (or by the `share` you
  give).
- **One collateral item securing several exposures** is split by the fractions you give, and each part
  is discounted at that exposure's own effective interest rate, not at a weighted average rate.
- **The allowance is split** between the loan and the undrawn commitment in proportion to the gross
  carrying amount and the present value of expected drawings.
- Groups of connected debtors, several currencies, guarantees as a separate source of recovery and
  purchased or originated credit-impaired assets are not covered.

## Turkey profile

- It encodes the mechanical part of the BDDK provisioning regulation as amended up to 24 May 2022:
  day counts, restructuring, borrower-level contagion and the conditions for reclassification.
  Check for later amendments and Board decisions before relying on it.
- Most criteria of Article 4 are judgements about the borrower. They enter as flags
  (`unlikely_to_pay`, `judgement_group`). A classification from day counts alone is a floor.
- "One year" is 365 days unless you set `days_in_year`.
- The small-amount exemption of Article 5(7) is not applied: its thresholds are indexed every year.
- Contagion for a borrower who has both consumer and other loans follows one reading of Article
  5(5): a non-performing consumer loan moves the borrower's other loans to at least group 2, not
  into default. See the docstring of `tr_bddk.classify`.
- A loan held in stage 3 by a cure period counts as non-performing for contagion.
- The three-month exit from restructured status (Article 7(8)) is applied only to loans restructured
  while performing, which is a literal reading of the text.
- The 2020-2021 relief (90 and 180 days) is available through `as_of`; the grandfathering of loans
  already past due on 1 October 2021 is not modelled, and neither is the suspension of Article
  4(1)(c)5 (re-default of restructured loans) that applied until 30 June 2021.
- Fixed provision rates and collateral consideration rates concern banks that do not apply TFRS 9.
  The exclusions of Article 10(4) are not modelled. Leasing, factoring and financing companies have
  their own regulation, which is not encoded.

## Data

Never put real customer data into an issue, a test or an example. Use `ifrs9 sample`.
