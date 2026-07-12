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

Note (2026-07-12): `rr_years` was extended to `[2021, 2022, 2023, 2024, 2025, 2026]` for
the step10 universe expansion — the labeled panel now runs to 2026q1, and the fee in effect
at any 2022q1–2026q1 quarter needs prospectuses filed as early as 2021. Only 2026q1 is
published so far; `acquire.download_all` skips-with-log any configured quarter the SEC
landing page hasn't published yet, so a partially-published trailing year is fine.

## Schema amendment (verified 2026-07-12)
Verified by downloading a real quarter (`2024q4_rr1.zip`, 39.2 MB, HTTP 200) and inspecting
its members and actual row values. This amendment is the contract Tasks 2–3 code to.

### URL pattern (verified: landing page hrefs + a real 200 download)
- Landing page: `https://www.sec.gov/dera/data/mutual-fund-prospectus-risk-return-summary-data-sets`
- Quarterly ZIP: `https://www.sec.gov/files/dera/data/mutual-fund-prospectus-risk/return-summary-data-sets/{year}q{q}_rr1.zip`
  - Note the path really splits "risk" / "return" with a slash, and the file suffix is
    `_rr1` (there is exactly one ZIP per quarter, not `_rr1`/`_rr2` shards).
- Same requests + SEC `User-Agent` (from `.env`) pattern as the N-PORT downloads. WebFetch
  is blocked by SEC (403); the landing page and ZIPs must be fetched with the SEC UA.
- Published range as of 2026-07-12: **2010q4 through 2026q1**, contiguous. All 21 quarters
  the config needs (2021q1–2026q1) exist; 2026q2–q4 are not published yet (skip-with-log).

### ZIP members (tab-delimited `.tsv`, SEC financial-statement-data-set layout)
`sub.tsv` (submissions/filings), `num.tsv` (numeric facts — **this is the one that matters**),
`tag.tsv` (taxonomy tag dictionary), `lab.tsv`, `cal.tsv`, `txt.tsv` (text facts),
`readme.htm`, `rr1-metadata.json`. Delimiter is `\t`; read with `dtype=str`.

### Where identifiers live
- `num.tsv` columns: `adsh, tag, version, ddate, uom, series, class, measure, document,
  otherdims, iprx, value, footnote, footlen, dimn, dcml`.
- **`series`** = EDGAR series id (`S000######`) and **`class`** = EDGAR class id
  (`C000######`) live **directly in `num.tsv`** — no join needed to reach them. These are
  the same `series_id` keys the whole pipeline uses (step1's N-PORT `SERIES_ID`), so the
  point-in-time join keys straight onto `series`.
- One filing (`adsh`) spans an entire fund family: a single `adsh` carried **51 distinct
  series / 249 distinct classes** in the sample. So the class→series work is per-`adsh`.

### Where the dates live (point-in-time anchor)
- `sub.tsv` columns include `adsh, cik, name, ..., pdate, effdate, form, filed, accepted`.
- **`filed`** (YYYYMMDD) is populated for **every** row (0/1460 null) — this is the
  look-ahead-safe filing-date anchor for the point-in-time rule and the UAT's
  `source_filing_date <= quarter end` assert. Join `num.tsv.adsh → sub.tsv.adsh` to get it.
- `effdate` (prospectus effective date) is the more precise "in effect" date but is sparse
  (**408/1460 null**) — use only as an optional refinement, never as the join key.
- Each quarterly ZIP holds only **that quarter's** filings (sample `filed` range
  20241001–20241231) — which is exactly why the join must sweep back to 2021 to find the
  most-recent-on-or-before-Q filing for a fund that files annually.
- `form` values in sample: `485BPOS` (786), `497` (668), `485APOS` (6).

### The exact tags (rr taxonomy) and fraction-vs-percent (from ACTUAL values)
All three are `uom == "pure"` and stored as **fractions, not percents** (verified from real
rows — an equity fund's fee reads ~0.005–0.015):

| Concept | tag (`num.tsv.tag`) | datatype | sample values | scale |
|---|---|---|---|---|
| Net expense ratio | `NetExpensesOverAssets` | nonNegativePure4 | 0.0054, 0.0175, 0.0051; median 0.009, max 0.0598 | **fraction** (0.0054 = 0.54%) |
| Gross expense ratio | `ExpensesOverAssets` | nonNegativePure4 | 0.0050, 0.0085, 0.0175; median 0.009 | **fraction** |
| Portfolio turnover | `PortfolioTurnoverRate` | pure | 0.55, 0.30, 0.87; median 0.32 | **fraction** (0.55 = 55% turnover) |

- **Net-vs-gross fallback is mandatory for the coverage gate.** In the sample, `Net…` covers
  1,501 series but `Expenses…` (gross) covers 3,448 series; **2,890 of 7,034** distinct
  (series,class) keys report gross-only (net-only is just 17). Keying `expense_ratio_net`
  on `NetExpensesOverAssets` alone would silently drop >half the funds. Task 2 rule:
  **`expense_ratio_net = NetExpensesOverAssets if present else ExpensesOverAssets`**, and
  keep gross (`ExpensesOverAssets`) alongside for the record. (`TotalAnnualFundOperatingExpenses`
  is a rare custom tag — 2 rows — ignore it.)
- **Scale hygiene — validate BOTH turnover and (gross-sourced) expenses.**
  - Turnover: ~25/3701 `PortfolioTurnoverRate` rows are mis-scaled garbage (max 44577.0 —
    filers who entered a percent or a raw count). Cap/drop above a sane bound (~30 = 3000%).
  - Expenses: **`NetExpensesOverAssets` is clean** (max 0.0598, 0 rows >10%), **but
    `ExpensesOverAssets` (gross) is NOT** — 49/8915 rows >0.10 and 7 rows >1.0 (max 6.3041 =
    630%). This matters because the mandatory net-else-gross fallback feeds **gross into
    `expense_ratio_net` for the ~2,890 gross-only funds**, so a 6.3 would flow straight into
    the model feature. Task 2 must therefore **sanity-bound the expense feature too** (e.g.
    upper-bound the gross-sourced value, ~0.10 = 10% is generous for a US equity fund), not
    just turnover.

### Granularity differs: expenses per-class, turnover per-series
- `NetExpensesOverAssets` / `ExpensesOverAssets` rows have **both `series` and `class`
  populated** → per share class. Task 2 must collapse class→series — but this is a
  **deliberate Task-2 choice, not settled here.** Note step1 only uses lowest-`classId` for
  *ticker lookup*; its per-series *returns* are the **mean across classes**
  (`_aggregate_class_returns_to_series`). Since the panel predicts per-series mean returns, a
  **mean-across-classes** expense collapse is the more consistent default; lowest-`classId`
  representative-class is the alternative. Task 2 picks and documents one.
- `PortfolioTurnoverRate` rows have **`series` populated but `class` == NaN** → reported at
  the **series** level already; no class→series step needed for turnover.

### Deduplication (deterministic, for Task 2)
Within a `(adsh, series, class)` group the fee tag can repeat (480/4965 net rows):
- `iprx` (presentation index) is almost always `0` (4964/4965) — not the main cause.
- The real cause is multiple `ddate` periods (current + prior fiscal year) reported in one
  filing, plus occasional same-`ddate` near-duplicates (e.g. 0.0092 vs 0.0091).
- Task 2 rule: **keep the row with the max `ddate`** per (adsh, series, class); if still
  tied, break deterministically (e.g. max `value`) so the pipeline is reproducible.

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
