# Changelog

## Unreleased

- Individual assessment of credit-impaired debtors (`individual`, `ifrs9 individual`, MCP tools
  `assess_individual` and `collateral_recovery`): gone concern, steady-state and two-step going concern,
  market price; sharing of cash by seniority; probability-weighted scenarios; undrawn commitments;
  warnings against published ECB expectations. Checked against the worked examples of the ECB asset
  quality review manual and the guidance on non-performing loans.
- `ifrs9 sample` also writes two made-up case files.

## 0.1.0

First release.

- PD term structures: cumulative, marginal and conditional PD, forward curves for a remaining
  window, constant hazard, Weibull, refinement from annual to monthly.
- Rating transition matrices: multi-period PD, time-varying paths, generators (principal logarithm,
  diagonal and weighted adjustment, JLT), embedding diagnostics, one-factor cycle shift.
- Point-in-time adjustment of PD curves with reversion to the long-run PD.
- Repayment schedules (bullet, annuity, equal principal), prepayment, undrawn commitments.
- LGD from collateral and workout cash flows; allowance of credit-impaired assets.
- 12-month and lifetime ECL with explicit timing and discounting conventions.
- Stage allocation with relative, absolute and backstop PD tests, grade rules, days past due,
  qualitative flags, low credit risk exemption, probation and cure periods; every rule that fired is
  recorded; warnings for policies that supervisors have criticised.
- Probability weighting across scenarios, with per-scenario sensitivity and re-staging.
- Provision matrix, loss rates from history and from flow rates.
- Turkey (BDDK) profile: five groups, restructuring, contagion, reclassification conditions, fixed
  rates for banks not applying TFRS 9.
- Portfolio run on a loan file with settings in JSON; tables by stage, segment, reason and
  scenario; supervisory checks; comparison of two settings; step-by-step explanation of one loan.
- Command line `ifrs9` and a read-only MCP server with ten tools.
