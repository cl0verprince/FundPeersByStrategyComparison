# step9_fees_turnover — design

## Purpose
Add **point-in-time expense ratios and portfolio turnover** to the unified model from the
SEC's structured prospectus data, build the **expense-rank naive baseline** the external
review asked for, add the **fund-disjoint split check** (adopted critique item B4), and
re-evaluate. This is the project's most promising route past step7's central finding —
the model ties the best naive rule (mean reversion) at AUC ~0.717 — because fees and
turnover are the two classic underperformance predictors the model has never seen.
Reported honestly whichever way it comes out: "fees lift the edge" and "even fees don't
beat mean reversion at a quarterly horizon on this panel" are both publishable findings.

## Origin — the external-review arc (recorded in decisions.json, 2026-07-11)
The reviewer said the model should include expense ratio and turnover and be benchmarked
against an expense-rank rule. First response: impossible — N-PORT carries no fee fields,
and yfinance exposes only today's value (look-ahead). The user then asked what lifting the
no-scraping rule would cost; re-checking surfaced the **SEC DERA Mutual Fund Prospectus
Risk/Return Summary Data Sets** — quarterly bulk ZIPs (same DERA family as our N-PORT
source) of flattened XBRL extracted from prospectus risk/return summaries, explicitly
including the **fee table** (net + gross expense ratios) and **portfolio turnover**,
as-filed and dated. Point-in-time by construction, free, structured, no scraping. Source:
https://www.sec.gov/dera/data/mutual-fund-prospectus-risk-return-summary-data-sets

## Design decisions

### 1. Data acquisition: RR quarterly ZIPs, 2021q1–2024q4
Download the Risk/Return data sets covering 2021 through 2024 — 2021 included because the
fee "in effect" at 2022q1 comes from a prospectus filed BEFORE it (annual prospectus
cadence means a fund's most recent filing can be up to ~15 months old). Cached under
`data/raw/` like the N-PORT ZIPs; requires the same SEC User-Agent header; idempotent
re-download skip.

**Schema discovery is the first implementation task, not an assumption.** The exact file
layout (sub/num/txt-style flattened tables, tag names such as net/gross
expenses-over-assets and portfolio turnover in the `rr` taxonomy, and how series/class
identifiers appear) is verified by downloading ONE quarter and inspecting it before any
join code is written; the verified schema is recorded in this design as an amendment.

### 2. Point-in-time join rule
For each (series_id, quarter Q): take the fee/turnover values from the **most recent
filing dated on or before Q's end**, joined via EDGAR series identifiers (the same
`series_id` keys the whole pipeline uses). No value → NaN (never forward-filled from a
LATER filing — that would be the exact look-ahead this step exists to avoid). Each row
retains its `source_filing_date` for auditability.

### 3. Class→series aggregation
Expense ratios are per share class; our unit is the series. Reuse step1's existing
class→series policy (largest/representative class as resolved in step1's design) so fees
and returns describe the same share class. **Net** expense ratio (after waivers — what an
investor actually pays) is the feature; gross is stored alongside for the record, not fed
to the model.

### 4. Coverage gate before modeling
RR tagging is mandatory but joins can fail (identifier drift, missing filings). First
deliverable is a **coverage report**: share of the 2,087 strategy funds (and of the
20,838 labeled panel rows) with resolved point-in-time fees. Decision rule, honest by
construction:
- Coverage ≥ 80% of labeled rows → single model comparison: unified features vs unified +
  fees/turnover, both evaluated on the **common covered subset** so the comparison is
  apples-to-apples (plus the original full-panel model unchanged as reference).
- Coverage < 80% → dual-model framing (with-fees model on the covered subset only),
  reported with the attrition stated plainly.

### 5. New features and the new baseline
- Features added to the panel: `expense_ratio_net`, `portfolio_turnover` (both point-in-
  time as of Q). Rows missing them are dropped from the with-fees variant only (the
  existing drop-not-impute policy).
- **Expense-rank baseline**: score = expense_ratio_net (higher fee → predicted
  underperformer), AUC on the same test rows. This is the reviewer's "dumb thing a
  practitioner would do" — now honestly constructible.
- Reported side by side: random 0.5, persistence (and its honest reversed reading),
  expense-rank, unified model, unified+fees model — pooled AUC with the same
  fund-clustered bootstrap CI machinery from step7 (`fund_clustered_bootstrap` reused),
  plus per-quarter spread.

### 6. Fund-disjoint split check (adopted critique item B4)
In addition to the chronological split, evaluate a **doubly-disjoint** split: train on
2022–2023 quarters of a seeded 80% of strategy funds, test on the 2024 quarters of the
held-out 20% of funds (never seen in training, in a never-trained-on period; ~1,250 test
rows). Run for BOTH the unified model and the with-fees variant. This speaks directly to
whether the ~0.002 residual edge over mean reversion is real signal or fund
fingerprinting. Seeded, deterministic.

### 7. Persistence/naming
New tables: `rr_fees` (series_id, quarter, expense_ratio_net, expense_ratio_gross,
portfolio_turnover, source_filing_date), `fees_model_eval` (long format like
`unified_model_eval`). Model bundle `unified_fees_rf_model.joblib` saved ONLY if the
with-fees model is worth keeping (a reported decision, not automatic). step7's tables and
model are untouched. Not wired into `conductor.py` (same rationale as steps 6–8); run via
`python -m steps.step9_fees_turnover.build`.

## Config additions (`config.json`)
```
"fees": {
  "rr_years": [2021, 2022, 2023, 2024],
  "coverage_gate": 0.80,
  "fund_disjoint_test_share": 0.20
}
```

## Out of scope
- Scraping individual 485BPOS/N-1A filings (not needed — structured data exists).
- Historical categories or survivorship-bias repair (documented limitations, separate
  concern).
- Re-clustering or label changes — step7's label and clusters are frozen here.

## UAT (acceptance for this step)
- RR ZIPs for all configured years downloaded, cached, and parsed; verified schema
  recorded in this design as an amendment.
- `rr_fees` has one row per (series_id, quarter) with no future-dated source filings
  (assert `source_filing_date` <= quarter end for every row).
- Coverage report printed and recorded: % of strategy funds and % of labeled panel rows
  with resolved fees; the ≥/< 80% branch taken is stated.
- Expense-rank baseline AUC reported with CI on the same test rows as the models.
- Unified vs unified+fees comparison on the common covered subset: pooled AUC + CI +
  per-quarter spread for both; stated plainly whether fees/turnover move the edge over
  reversed persistence beyond noise.
- Fund-disjoint split AUCs reported for both models.
- Feature importances of the with-fees model reported (where do fees rank?).
- All numbers recorded in this design's UAT-results section, honest whichever way.
