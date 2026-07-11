# step7_unified_universe — design

## Purpose
Rebuild the modeling method on **one merged universe** of all 2,243 ingested equity funds
(original 363 + oos 391 + oos2 1,489), fixing the two problems diagnosed when the model was
evaluated against larger populations, and evaluate the result against **strong baselines**
rather than against the old task's AUC numbers (the label changes, so 0.710/0.725/0.680
become historical context, not a yardstick).

## Background — what the step6/oos2 investigation found (2026-07-11)
Evaluating on and retraining with the oos2 batch (1,489 equity funds) produced:
frozen 754-fund model on oos2 AUC=0.701 (generalizes fine), but naive 3-way retrain
AUC=0.680 vs the official model's own 0.725. Per-origin breakdown showed this is a
**composition effect**: the retrained model scores 0.708/0.700 on original/oos-style test
funds (unchanged grip) but 0.668 on oos2, and oos2 is 66% of the blended test set. The
frozen model scores 0.691 on the same oos2 slice — so oos2 is intrinsically harder with the
current method, not corrupted by retraining. Root causes diagnosed:

1. **Peer granularity degrades with population size.** `n_clusters=15` is fixed, so
   "underperform your cluster median" meant ~12 close peers in the original batch but a
   median of 62 (max 327) heterogeneous funds in oos2 — the label and the model's #1 feature
   (`return_vs_cluster_median_q`, importance 0.22) both got mushier. The task itself silently
   changed with scale.
2. **Holdings barely enter the features.** The pipeline's thesis is holdings-based
   similarity, yet the model's features are almost entirely trailing-return stats; holdings
   contribute only 4 tier dummies (combined importance 0.036) and the cluster median.
3. **Batch namespaces are an ingestion artifact.** An oos2 fund's peers could only ever be
   other oos2 funds. A real deployment computes peers over everything it knows.

## Design decisions

### 1. Merged `_all` table namespace
Concatenate `funds`/`holdings`/`monthly_returns` across the three batch namespaces into
`funds_all`, `holdings_all`, `monthly_returns_all` (dedup-asserted on `series_id` — the
batches were already verified pairwise disjoint). Per-batch tables are never touched.
Same 12 quarters (2022q1–2024q4). No new network work — this is a pure re-combination of
already-ingested data. Steps 2–3 logic runs against `_all` via the existing `table_suffix`
mechanism (with the segmentation + k changes below).

### 2. Segmentation: target-date / allocation funds leave the strategy clustering
Scratch analysis on the merged 2024q4 universe showed TDF clusters are **fund families,
not strategies**: every TDF-dominated cluster mixes all vintages (2040→2065+) because a
provider's whole suite holds the same underlying index funds — nearly identical holdings
text embeds together as a family suite. They are in the "equity" universe at all because
filers report underlying fund shares as `EC` holdings (a fund-of-funds classification
quirk). At every useful k they consume 3–10 clusters that all mean "some provider's
glide-path suite" — redundant and misleading for the human audience.

Rule: funds whose `yahoo_category` starts with `Target-Date` or `Allocation` form a
separate **allocation segment** (`segment='allocation'` on `funds_all`; ~134 TDFs at
2024q4), excluded from the strategy clustering and presented by vintage in the dashboard
(step8). Everything else is `segment='strategy'` (~2,100 funds). A build-time diagnostic
logs how many strategy-segment funds still have fund-of-funds-looking top holdings
(issuer names matching FUND/PORTFOLIO/INDEX/ETF patterns), so the rule's coverage is
verified each run, not assumed. Funds with `Unknown` category stay in the strategy segment
(clustering is holdings-based; categories are only used for naming/validation).

### 3. Clustering: k=30, chosen from a measured sweep
K-sweep on the real merged 2024q4 universe (silhouette, purity, ARI vs Yahoo categories,
size distribution, TDF spread):

| k   | silhouette | purity | ARI   | median size | clusters <20 funds |
|-----|-----------|--------|-------|-------------|--------------------|
| 15  | 0.072     | 0.427  | 0.307 | 88          | 0                  |
| 20  | 0.081     | 0.469  | 0.326 | 67          | 3                  |
| 30  | 0.089     | 0.514  | 0.316 | 34          | 5                  |
| 40  | 0.084     | 0.523  | 0.263 | 40          | 10                 |
| 50  | 0.085     | 0.531  | 0.237 | 27          | 19                 |
| 100 | 0.109     | 0.569  | 0.210 | 11          | 58                 |

Purity rises with k mechanically; **ARI (chance-corrected) peaks at k=20–30 and decays
beyond** — past ~30 we shard real categories — and the under-20-members count explodes past
k=40.

A second sweep was then run on the **segmented** population (the ~2,086 strategy funds the
clustering will actually serve, TDF/Allocation excluded), including k=35 on request, to
check whether 35 buys extra category resolution between 30 and 40:

| k  | silhouette | purity | ARI   | median size | clusters <20 funds | distinct dominant categories |
|----|-----------|--------|-------|-------------|--------------------|------------------------------|
| 25 | 0.073     | 0.528  | 0.323 | 64          | 5                  | 17                           |
| 30 | 0.073     | 0.544  | 0.265 | 55          | 4                  | 21                           |
| 35 | 0.065     | 0.562  | 0.264 | 41          | 10                 | 23                           |
| 40 | 0.074     | 0.554  | 0.244 | 41          | 9                  | 23                           |

k=35 vs k=30: only +2 distinct dominant categories (21→23) and flat ARI, while the
under-20-members cluster count jumps 4→10 and silhouette dips — i.e. the extra 5 clusters
mostly shard existing groups into underpowered fragments rather than resolving new
categories. **k=30 confirmed**: on the segmented population it has the best power profile
of the whole grid (only 4 small clusters, median 55) while keeping near-best category
resolution. Config: `similarity.n_clusters` stays 15 for the legacy per-batch runs; new
`unified.n_clusters: 30` applies to `_all`. The step's UAT re-checks purity/ARI/size
distribution on the full segmented run across all quarters.

### 4. Label redefinition: kNN-peer median, not cluster median
"Underperform" becomes: **a fund's next-quarter return is below the median of its own
top-N most-similar funds' next-quarter returns** (N=10 by cosine similarity on the holdings
embeddings — the existing `fund_peers` computation, now over the merged strategy segment).
Peer groups are bespoke and constant-size by construction, immune to universe growth —
this permanently fixes diagnosis #1 instead of re-tuning `n_clusters` per population.
Peers are as-of quarter Q; the label compares Q+1 returns. Rows with fewer than 5 peers
having a valid Q+1 return are dropped (logged). Clusters remain for human-facing grouping,
visualization, and narration — they no longer define the label.

### 5. Features: holdings finally enter the model
Per fund-quarter, keeping the trailing-return stats (trailing return / volatility /
Sharpe / max drawdown) and `net_assets`, replacing `return_vs_cluster_median_q` with
`return_vs_peer_median_q` (same idea, kNN-peer version, consistent with the label), and
adding holdings-derived features:
- **mean cosine similarity to its top-10 peers** — how "typical" the fund is of its niche;
- **holdings concentration**: HHI and top-10 weight share of the equity sleeve;
- **number of EC holdings**;
- **tier dummies** (kept) — dominant-category tier as before;
- **net-asset momentum**: QoQ change in `net_assets` (NaN for a fund's first quarter —
  those rows keep the existing dropped-if-missing handling).

### 6. Model and evaluation
Same seeded `RandomForestClassifier` construction and hyperparameters as step4 (config
`model.rf`), same time-based split: train 2022–2023 transitions, test 2024 transitions.
Success bar (agreed): **clearly beat both baselines** on the held-out test —
- random (AUC 0.5), and
- **persistence**: score = last quarter's `-return_vs_peer_median_q` ("was below peer
  median at Q → predicted below at Q+1"), evaluated as an AUC on the same test rows.

Report pooled AUC **and per-quarter AUC spread** (the oos2 work showed per-quarter AUC
swings 0.588–0.819, so a pooled number alone over-states precision), plus feature
importances. One-time sanity check, recorded here for the design record: the old
cluster-median label with scaled clusters (approach B) is run once on the merged universe
to confirm granularity was really the driver, then not maintained.

### 6b. Monte Carlo uncertainty layer (not a simulator of fund futures)
Considered and deliberately scoped: Monte Carlo is NOT used to simulate fund returns —
that would require assuming a return-generating process (plus peer correlations, since the
label is cross-sectional), replacing measured patterns with assumed ones. It IS used as
the honesty machinery around the evaluation, three ways, all seeded and deterministic:

1. **Fund-clustered bootstrap CI on the pooled test AUC**: resample test-set *funds* with
   replacement (keeping each drawn fund's quarters together, since a fund's rows
   correlate), recompute AUC per resample (no refitting), report the 95% interval
   alongside the point estimate. With only 3 test quarters, a point AUC alone overstates
   precision.
2. **Paired bootstrap significance of the edge over persistence**: on the *same* resamples,
   compute (model AUC − persistence AUC); report the CI of the difference and the fraction
   of resamples where the edge ≤ 0 (a one-sided p-value). Directly answers the skeptic's
   question: is the model's edge over the naive rule statistically real?
3. **Label-stability perturbation study**: the label depends on the top-10 peer choice, so
   for each fund-quarter, draw many perturbed peer sets (random 8 of the top 12 by cosine)
   and measure how often the underperform label flips. The flip rate estimates the label's
   intrinsic noise floor — context for how much AUC is achievable at all, and a guard
   against chasing model improvements into pure label noise.

Outputs land in `unified_model_eval` (AUCs, CI bounds, edge-vs-persistence p) and
`unified_label_stability` (flip-rate distribution summary). Config: `unified.bootstrap_iterations`
(2000) and `unified.label_stability_draws` (100).

### 7. Persistence and naming
The new model is saved as `models/unified_rf_model.joblib` (bundle: model + feature_cols
+ label definition metadata). `models/random_forest_model.joblib` (the promoted 754-fund,
old-label model) is untouched — different task, kept as history. New tables:
`fund_clusters_all`, `fund_peers_all`, `cluster_validation_all`, `fund_metrics_*_all`,
`cluster_definitions_all`, `unified_panel`, `unified_predictions`,
`unified_feature_importances`, `unified_model_eval` (AUCs incl. baselines, per quarter).

### 8. Not wired into `conductor.py` (same rationale as step6)
Post-completion exploratory scope; run on demand:
`python -m steps.step7_unified_universe.build`. If this method graduates to the official
pipeline later, wiring it into the conductor is its own explicit decision.

## Config additions (`config.json`)
```
"unified": {
  "n_clusters": 30,
  "peer_label_top_n": 10,
  "min_valid_peers_for_label": 5,
  "bootstrap_iterations": 2000,
  "label_stability_draws": 100
}
```

## UAT (acceptance for this step)
- `funds_all` has exactly 2,243 unique equity `series_id`s; zero duplicates; segment split
  logged (≈134 allocation / ≈2,109 strategy at 2024q4).
- Strategy-segment clustering at k=30: purity/ARI vs Yahoo categories reported per quarter;
  no quarter with more than ~6 clusters under 20 members; zero TDF-dominated clusters.
- Every strategy fund-quarter has 10 peers with cosine similarities; label coverage
  reported (rows dropped for <5 valid-peer returns are logged and <2% of the panel).
- The unified model beats BOTH baselines: pooled test AUC above 0.5 and above the
  persistence baseline's AUC, AND above persistence in a majority of individual test
  quarters (so one lucky quarter can't carry the claim); per-quarter AUCs reported with
  the spread stated honestly.
- Uncertainty layer reported honestly whichever way it comes out: 95% bootstrap CI on the
  pooled test AUC, CI + one-sided p for the edge over persistence, and the label
  flip-rate summary from the peer-perturbation study. (These are reporting criteria, not
  pass/fail thresholds — a not-significant edge is a finding, not a UAT failure.)
- Model round-trip: reloaded `unified_rf_model.joblib` reproduces in-memory predictions.
- `unified_predictions` includes a genuine forward prediction set (2024q4 features →
  2025q1, unlabeled) for the dashboard.
