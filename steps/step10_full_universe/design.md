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
- `unified.*` unchanged. The k re-check (below) chose **k=40** for `_full`, recorded as
  `full.n_clusters: 40`; the `_full` clustering run passes it explicitly.

## Pool amendment (2026-07-11)
The `_full` ingestion ran to completion (attempt 2, 4,247s ≈ 71 min; resolution phase
3,735s). An earlier identical attempt was externally killed at 12,702/12,871 candidates
before anything was saved (no traceback — the harness stopped the background task); the
rerun was launched as a detached process and, being idempotent, reproduced the identical
candidate order (same seed). Log: `.superpowers/sdd/step10-ingest.log`.

- **Relaxed pool: 12,871 series** (≥6 consecutive calendar quarters within 2022q1–2026q1)
  — vs 9,445 under the old all-12-quarters rule, confirming the relaxed rule enlarges.
- **Attempted: 12,871 (100% — uncapped, every candidate).**
- **Resolved: 8,766 (68.1%)** — 6,000 via metadata reuse + 2,766 new via Yahoo (of the
  6,871 new candidates; the ~60% new-candidate attrition = missing current-ticker-map
  entries + Yahoo misses; 536 unique tickers had failed lookups, all permanent-miss or
  transient-exhausted).
- **Equity-flagged: 5,203** by Yahoo asset class; **3,231 final `is_us_equity`** after the
  holdings US-share ≥ 0.70 geography check.
- **Segment split (per `assign_segment` on yahoo_category):** all resolved: 8,062 strategy
  / 704 allocation; within US-equity: 3,035 strategy / 196 allocation.
- **Dead funds: 611** series whose last filed quarter < 2026q1 — the relaxed rule is
  biting (these were impossible under the strict rule); their final quarters are in the
  panel.
- **Metadata-reuse hit-rate: 100%.** All 6,000 series known from `funds`/`funds_oos`/
  `funds_oos2` appear in `funds_full` with reused Yahoo fields and made ZERO Yahoo calls
  (verified: the failed-lookup ticker set from the log has empty intersection with the
  6,000 known tickers, and the 62-min resolution wall time is only consistent with ~6.9k
  network lookups, not 12.9k).
- **Tables saved:** `funds_full` 137,544 rows; `holdings_full` 46,405,028 rows;
  `monthly_returns_full` 412,632 rows (17 quarters, 2022q1–2026q1; per-quarter filings
  ranged 11,470–12,553).

## k re-check (2026-07-11)
Re-ran the step7 §3 sweep protocol on the `_full` strategy segment at the latest quarter
(2026q1), the population the clustering actually serves: `is_us_equity & segment=='strategy'`,
EC-only holdings, `embed_quarter_funds` (all-MiniLM-L6-v2, top-25), and `KMeans(random_state=42,
n_init=10)` — the exact params `similarity.run` uses, so these numbers preview the real run.
Universe: **2,856 funds clustered at 2026q1** (window strategy total 3,035; fewer file the
final quarter because the 611 dead funds' last filing is < 2026q1). That is ~37% larger than
step7's 2,086. Choice is made on 2026q1 and applied to all 17 quarters (as step7 chose on
2024q4). Scratch script: `scratchpad/k_recheck_full.py`.

| k  | silhouette | purity | ARI   | min | p10 | median | max | n<20 | distinct dominant categories |
|----|-----------|--------|-------|-----|-----|--------|-----|------|------------------------------|
| 30 | 0.069     | 0.513  | 0.258 | 11  | 24  | 79     | 244 | 1    | 19                           |
| 40 | 0.070     | 0.554  | 0.258 | 7   | 13  | 66     | 266 | 6    | 24                           |
| 50 | 0.068     | 0.558  | 0.229 | 3   | 11  | 47     | 192 | 13   | 27                           |

**Choice: k=40** (changed from step7's 30). By the step7 rule — near-peak ARI, best power
profile, meaningful category resolution — the ~37%-larger universe flips the decision that
step7 made at 2,086:
- **ARI no longer decays at 40.** k=40 ARI (0.2583) is the grid peak, tied with k=30
  (0.2583) — at step7's 2,086 funds, 30→40 cost ARI (0.265→0.244) because 40 sharded real
  categories; with +37% funds there is now enough population to support 40 clusters without
  that decay. k=50 does decay (0.229), so the peak has simply moved out to ~40.
- **k=40 buys genuine category resolution: 19→24 distinct dominant categories (+5)**, versus
  the mere +2 (21→23) that 40 bought over 30 at step7 — the extra clusters resolve new
  categories here rather than fragmenting existing ones.
- **Power stays healthy**: only 6 clusters under 20 members (right at step7's soft ~6 bar,
  and below the "no more than ~6" UAT line), median size 66. k=50 fails this — 13 small
  clusters and decayed ARI. k=30 is more powered (n<20=1) but leaves 5 real categories
  unresolved at no ARI benefit.

Purity rises with k mechanically (ignored as a tiebreaker). Config change: new
`full.n_clusters: 40` (recorded in Config changes above); the `_full` run passes
`n_clusters=40` explicitly.

Prerequisite fix, discovered here: the Task-2 ingestion never persisted a `segment` column on
`funds_full` (it computed the split on the fly for the pool-amendment numbers). Added via
`assign_segment` (the design-§3 rule) before anything read it; the resulting counts match the
pool amendment exactly (8,062/704 all-resolved, 3,035/196 US-equity), verified before saving.

### `_full` clustering + metrics run (2026-07-12; 17 quarters, k=40, top_n_peers=15)
`similarity.run(cfg, table_suffix="_full", n_clusters=40, top_n_peers=15,
require_segment="strategy", save_coords=True)` then `metrics.run(cfg, table_suffix="_full")`.
Log: `.superpowers/sdd/step10-cluster2.log`. Run history, honestly: the first detached attempt
died silently overnight after 3 quarters (machine sleep — same environmental class as the
Task-2 attempt-1 kill; no traceback). The rerun completed the whole similarity stage in
**826.8 s (~14 min)** — far under the 1.5–2 h estimate — then `metrics.run` crashed on a
latent shared-code bug: `compute_dominant_category_info` called `mode().iloc[0]` on a cluster
whose members ALL lack a `yahoo_category` (2022q3 cluster 37; 40 category-less funds exist
panel-wide, and at k=40 they concentrated into one all-NaN cluster for the first time). Fixed
in `fundspeers/category.py` by `fillna("Unknown")` — the same convention step2's purity truth
already uses — with a regression test in `tests/test_category.py`; suite green (96). Metrics
then completed in a foreground rerun.

Per-quarter clustering quality (strategy segment; funds per quarter 2,545 → 3,005, growing as
the relaxed rule admits launches; dead funds leave before 2026q1):

- **Purity range 0.508 (2022q1) – 0.554 (2026q1)**, 17-quarter mean **0.535**.
- **ARI range 0.201 (2022q1/2023q1) – 0.258 (2026q1)**, mean **0.231**.
- Both drift UP over the window — later quarters are bigger and better-resolved, consistent
  with the k=40 sweep having been run on 2026q1.
- vs step7 `_all` at k=30 (purity 0.504–0.544 / ARI 0.229–0.297): purity comparable-to-better,
  ARI somewhat lower — expected, since `_full` adds partial-history and dead funds whose
  categories are noisier, and ARI is chance-corrected against a finer 40-way partition.
- Tier breakdown (pooled): Large purity 0.546 / Mid 0.447 / Small 0.451 / Sector-Other 0.239 —
  the same shape as step7 (style-box tiers cluster well; Sector/Other is the hard residual).
- `metrics.run`: 3,217 fund-quarters have no resolved cluster (allocation-segment funds plus
  step2's zero-EC-holdings edge case) — expected and warned, cluster-relative metrics NaN.
  Saved: `fund_metrics_overall_full` (3,231 funds), `fund_metrics_quarterly_full` (51,199
  fund-quarters), `cluster_definitions_full` (680 = 17×40 quarter-cluster combos).
- Similarity-stage tables: `fund_clusters_full` (47,982 fund-quarters), `fund_peers_full`,
  `cluster_validation_full`, `cluster_map_coords_full`; plot
  `reports/cluster_map_2026q1_full.png`.

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
