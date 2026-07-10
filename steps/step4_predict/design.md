# step4_predict — design

## Purpose (traces to Required Output)
Assemble a pooled, time-safe supervised panel ("features known at quarter Q -> did the fund
underperform its peer cluster's median return in quarter Q+1"), fit a decision tree then a
random forest, and confirm the model beats a naive baseline on a genuinely held-out,
later-in-time test set.

## Two real correctness risks, designed around up front (not discovered after the fact)

**1. Leakage via whole-period features.** `fund_metrics_overall` (step3) is computed from a
fund's **entire 36-month history** - using it as a feature "at quarter Q" would leak information
from quarters *after* Q into the prediction, since e.g. `cumulative_return` already reflects
Q+1..Q12. Features must instead be **trailing, point-in-time**: computed only from months up
to and including Q. This step introduces its own trailing-window feature functions rather than
reusing `fund_metrics_overall` for anything but the final report.

**2. Cluster ids are not comparable across quarters.** Step2's design explicitly notes KMeans
cluster labels are arbitrary per-quarter fits - `cluster_id=3` in 2022q1 has no relationship to
`cluster_id=3` in 2023q2. Pooling `cluster_id` as a one-hot categorical feature across quarters
would treat unrelated clusters as the same feature value - wrong. Instead: (a)
`return_vs_cluster_median` (already a *relative*, cluster-agnostic number) is used directly as a
feature, and (b) `yahoo_category`, bucketed into the same Large/Mid/Small/Sector-tier grouping
step2 used for validation, is used as the stable cross-quarter categorical feature - a fund's
category doesn't change quarter to quarter, unlike its arbitrary cluster label.

## Panel construction
- **Universe:** 363 US-equity funds, 12 quarters -> 11 chronological (Q, Q+1) transitions
  per fund (2022q1->2022q2, ..., 2023q4->2024q1... through 2024q3->2024q4).
- **Label** (at Q, predicting Q+1): `underperform_next_quarter = 1` if
  `fund_metrics_quarterly.return_vs_cluster_median` at Q+1 `< 0`, else `0`. Rows where Q+1's
  value is NaN (step3's documented missing-cluster or missing-month cases) are dropped from
  the panel - the label is unknown, not zero.
- **Features** (all computed using only data through Q, verified point-in-time-safe):
  - `trailing_return`, `trailing_volatility`, `trailing_sharpe`, `trailing_max_drawdown` -
    computed from `monthly_returns` over a trailing 12-month (4-quarter) window ending at Q
    (or fewer months if Q is early in the fund's history - not dropped, just a shorter window;
    documented as a simplification, consistent with this project's other trailing-data choices).
  - `return_vs_cluster_median_q` - the fund's own realized relative performance *at* Q itself
    (already known by the time Q ends - not a leak).
  - `net_assets_q` - fund size at Q, from `funds`.
  - `category_tier_q` - Large/Mid/Small/Sector-Other (reusing step2's bucketing logic), one-hot
    encoded. Stable across quarters, unlike `cluster_id`.
- Rows with any NaN feature (e.g. the 2 funds with one unparseable month from step3) are
  dropped, logged with a count - not imputed.

## Time-based split
11 chronological transitions pooled; **first 8 transitions -> train, last 3 -> test**
(`model.test_transitions_holdout: 3` in config) - never trains on a transition that ends later
in time than one held out for test, avoiding look-ahead bias in the split itself (not just
per-feature leakage).

## Models
1. **Decision tree** (`sklearn.tree.DecisionTreeClassifier`) - shallow and regularized
   (`model.tree.max_depth: 4`, `model.tree.min_samples_leaf: 20`) for interpretability; fit
   first, evaluated, and its structure exported (a simple to-read description of the top
   splits) as the "single tree you can read" baseline before the ensemble.
2. **Random forest** (`sklearn.ensemble.RandomForestClassifier`), `model.rf` params already in
   config (`n_estimators: 300, max_depth: 8, min_samples_leaf: 10`), seeded from `cfg["seed"]`.
- **Baseline:** a `DummyClassifier` (most-frequent-class) or literal AUC=0.5 reference - the
  Required Output only asks to beat this.
- **Metric:** ROC AUC on the held-out test transitions, plus accuracy for a plain-language
  reference. Feature importances reported from the random forest.

## Outputs
- **`fund_predictions.parquet`**: `series_id`, `quarter` (the Q the features were computed at),
  `predicted_probability`, `actual_label`, `split` (`train`/`test`).
- **`model_feature_importances.parquet`**: `feature`, `importance` (from the random forest).
- Console/log report: baseline AUC (0.5), decision tree AUC, random forest AUC, accuracy, and
  the tree's top-level split(s) in plain text.

## Determinism
`DecisionTreeClassifier(random_state=cfg["seed"])`, `RandomForestClassifier(random_state=cfg["seed"])`.
No other randomness (the time-based split is deterministic by construction, not sampled).

## Config additions (`config.json`)
- `model.test_transitions_holdout: 3`
- `model.tree.max_depth: 4`
- `model.tree.min_samples_leaf: 20`
(`model.rf` already exists from step0.)

## UAT (acceptance for this step) - all confirmed on real data
- Point-in-time safety: unit-tested directly (`test_trailing_features_are_point_in_time_safe`) -
  trailing features at Q are bit-identical whether or not later quarters exist in the input.
- Time-based split: confirmed by both a unit test and a runtime check in `run()` (raises, not a
  strippable `assert`) that `max(train_quarters) < min(test_quarters)`. Real run: train on
  2022q1-2023q4 (8 transitions, 2887 rows), test on 2024q1-2024q3 (3 transitions, 1088 rows).
- 18 of 3993 candidate panel rows dropped (missing feature or unknown label) - logged, not
  silently imputed.
- **Baseline AUC 0.500** (most-frequent-class dummy) -> **decision tree AUC 0.638** ->
  **random forest AUC 0.710** - comfortably clears the Required Output's ">0.5" bar, and shows
  the expected tree-to-ensemble improvement.
- Feature importances sum to 1.000 (sanity-checked, not assumed). Dominated by the fund's own
  recent performance (`return_vs_cluster_median_q` 0.21, trailing drawdown/Sharpe/return
  0.13-0.18 each) - `net_assets` modest (0.09), category tier dummies weak (0.006-0.012 each).
  A plausible pattern (recent underperformance/poor risk-adjusted returns predict continued
  underperformance), not a spurious one dominated by an unrelated column.
- Decision tree's top split is on `trailing_sharpe`, consistent with the importance ranking -
  human-readable and plausible.
