# step10_full_universe — design

## Purpose
The data foundation for the project's second wave, in one re-ingestion: (1) **exhaust the
candidate pool** (user: "4,000 isn't a lot" — we ingest every remaining candidate);
(2) **extend the data through 2026q1** (N-PORT coverage limit as of 2026-07); (3) **relax
the complete-panel rule** so extension enlarges rather than shrinks the universe — and
materially reduces the survivorship bias documented as a limitation; (4) **retrain with a
2025–26 test period** and (5) **score the frozen step7 model and the published 2024q4
forward predictions against what actually happened** — true out-of-time validation.

## Why relaxed completeness (supersedes the all-quarters rule)
The original sampler required a fund to file in ALL 12 quarters (2022q1–2024q4). Extending
to 17 quarters under that rule would SHRINK the pool and worsen survivorship (only funds
alive the whole 2022–2026 span). The completeness rule was a step1 simplification, not a
methodological need: trailing features only need a ≥2-month history, labels only need Q+1,
peers are per-quarter. So step10 replaces it:

**Eligibility: a fund is included for every quarter it filed, provided it has ≥6
consecutive quarters of returns somewhere in the window** (enough for a meaningful trailing
window plus a label). Funds that launched mid-window enter when they appear; funds that
**died mid-window stay in until they die** — their final quarters (often the worst) now
appear in training and evaluation, directly shrinking the survivor tilt. The remaining
survivorship (funds dead before 2022, and the current-ticker-map filter) stays documented.

## Design decisions

### 1. Quarters and sources
`data.quarters` extended to 2022q1–2026q1 (17 quarters). New N-PORT ZIPs: 2025q1–2026q1
(5 quarters, cached under data/raw like the rest). 2026q2 is not yet published (filing +
publication lag) — the automation step (step13) will pick it up later.

### 2. Universe: everything eligible, one final namespace `_full`
Fresh ingestion into `funds_full`/`holdings_full`/`monthly_returns_full`. The candidate
pool is re-derived under the relaxed rule (expected: > the old 9,445, since partial-panel
funds now qualify; the exact number is a Task-1 discovery output, recorded here as an
amendment). Every candidate is attempted — after this run the window's universe is
complete, a claim the README gets to make. Efficiency: ticker/category metadata for the
~6,000 already-resolved series is REUSED from existing tables (no repeat Yahoo hits);
only new candidates get lookups. Expected new-candidate volume: several thousand; Yahoo
attrition ~45% as before. The old `_all` and per-batch tables remain untouched (history).

### 3. Segmentation, clustering, label, features: step7's method, re-fit
Same segment rule (Target-Date/Allocation out), same embedding pipeline, same kNN-peer
label (top-10, min 5 valid), same feature set. k=30 is re-checked (one sweep iteration at
k∈{30, 40, 50} on the final strategy-segment size — the universe may grow ~2x; the sweep
protocol from step7 §3 applies) and the choice recorded. Per-quarter clustering now covers
17 quarters. Funds with partial histories flow through naturally (rows lacking features or
labels drop per the existing policy — but the fund's valid quarters stay).

### 4. Training split: train 2022–2024, test 2025–2026
Transitions (Q, Q+1) for Q in 2022q1..2025q4 (16 labeled transitions). Train = Q ≤ 2024q3
(11 transitions); test = Q in 2024q4..2025q4 (5 transitions, labels realized through
2026q1). Same seeded RF, same MC uncertainty layer (fund-clustered bootstrap CI, paired
persistence significance, label-stability study), same honest yardsticks — reversed
persistence above all. Fund-disjoint split check included (from the adopted critique item).

### 5. Out-of-time validation of the FROZEN step7 model (the headline)
Two scores, before any retraining is reported:
- **The published forward predictions vs reality**: the dashboard's 2,086 predictions
  (2024q4 → 2025q1) scored against realized 2025q1 returns and realized peer medians.
  One number: did the model's genuine, committed-before-the-fact predictions work?
- **The frozen unified model rolled forward**: scored on every new labeled transition
  (2025q1→q2 ... 2025q4→2026q1) using `_full` data restricted to the funds/peers it knows.
Attrition honestly counted: how many of the 2,086 funds actually filed in 2025q1; missing
= died/merged/late — reported, not ignored.

### 6. What is deliberately NOT here
Fees/turnover (step9 runs after this, on `_full`), dashboard redesign (step12), quarterly
automation (step13 — but §1's ingestion is written as an idempotent "ensure quarters
present" so step13 can reuse it as the advance-quarter mode), README/push (after step9's
numbers).

## Config changes
- `data.quarters`: extended to 17 quarters (2022q1–2026q1).
- New `full`: `{ "min_consecutive_quarters": 6, "namespace": "_full" }`.
- `unified.*` unchanged (k re-check may amend `n_clusters` for `_full` — recorded here).

## UAT (acceptance)
- New pool size under the relaxed rule recorded; 100% of candidates attempted; resolution,
  equity-flag, and segment counts reported; metadata reuse confirmed (zero repeat Yahoo
  hits for known series).
- Dead-fund inclusion verified: count of funds whose last filing is mid-window (> 0, else
  the relaxed rule isn't biting); their final-quarter rows present in the panel.
- 17 quarters clustered; k choice re-validated and recorded; purity/ARI per quarter.
- Frozen-model out-of-time scores reported (both variants of §5) with attrition counts.
- Retrained model: pooled + per-quarter AUC + CI on the 2025–26 test, vs random,
  persistence, reversed persistence; fund-disjoint check; stability study on `_full`.
- All numbers recorded in a UAT-results section, honest whichever way.
