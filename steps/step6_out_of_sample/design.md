# step6_out_of_sample — design

## Purpose
Test whether the model trained in step4 actually **generalizes to funds it has never seen** -
a genuinely disjoint sample of 1000 new US funds, run through the identical pipeline (steps
1-3), evaluated two ways: (1) the **frozen** step4 model applied to the new funds as-is, and
(2) a **retrained** model fit on the combined original + new funds. These answer two different
questions and both are wanted: "does the existing model generalize?" (frozen) vs. "does more
data help?" (retrained).

## Sampling a disjoint fund set
Step1's complete 12-quarter panel pool has **9,445 candidate series**; only 1000 were used for
training. Re-run `steps/step1_ingest/ingest.py`'s iterative-fill sampling with the **same seed**
(so the deterministic shuffle order is unchanged) but pass the original 1000 `series_id`s as an
`exclude_series` set - the sampler skips them and keeps going through the same shuffled order
until 1000 *new* funds resolve. This guarantees disjointness while staying deterministic and
reusing the already-cached raw N-PORT ZIPs (no re-download - only new Yahoo lookups and holdings
extraction for the new candidate set).

## Table namespace: `_oos` suffix, not new table names or overwriting
`ingest.py`, `similarity.py`, and `metrics.py`'s `run()` functions gain two new parameters,
**both defaulting to preserve the exact original behavior**:
- `exclude_series: set | None = None` (ingest only)
- `table_suffix: str = ""`

All `save_table`/`load_table` calls use `f"{name}{table_suffix}"`. Calling `run(cfg)` as before
is unaffected; calling `run(cfg, exclude_series=original_ids, table_suffix="_oos")` produces a
fully parallel, non-overlapping table set: `funds_oos`, `holdings_oos`, `monthly_returns_oos`,
`fund_clusters_oos`, `fund_peers_oos`, `cluster_validation_oos`, `fund_metrics_overall_oos`,
`fund_metrics_quarterly_oos`, `cluster_definitions_oos`. The original tables are never touched.
Same 12 quarters (2022q1-2024q4) as the original run, for direct comparability - this step tests
generalization across **funds**, not across time (that was already step4's original time-split).

## Model persistence (new: step4 didn't save the fitted model before)
step4 fit the random forest in memory and only ever persisted its *predictions*, never the
model object itself - there was nothing to reuse for a frozen-model comparison. Add
`fundspeers.io.save_model`/`load_model` (via `joblib`, already an installed scikit-learn
dependency) and a new `paths.models` directory (config: `"models"`, gitignored like
`data/processed/` - models are regenerable, not source). step4's `run()` now saves the fitted
`RandomForestClassifier` as `random_forest_model.joblib`.

## Two evaluations
1. **Frozen-model, fund-level out-of-sample:** load the persisted step4 model (never refit),
   assemble the OOS panel (reusing step4's `compute_trailing_features`/`assemble_panel` against
   the `_oos` tables) - **no time-based split needed here**: none of these funds' data was used
   in training regardless of quarter, so the entire OOS panel is valid held-out data. Predict,
   compute AUC. This is the direct answer to "does the model generalize to new funds?"
2. **Retrain on the combined set:** pool the original panel (step4's already-assembled data) with
   the new OOS panel, apply the *same* time-based split (train on 2022-2023, test on 2024, now
   over ~2x as many funds per quarter), refit tree + forest, compute AUC. Answers "does more
   data improve the model?" - a different question, reported separately, not blended with (1).

Both AUCs are reported alongside the original step4 test AUC (0.710) for a direct three-way
comparison - genuinely reported whichever way it comes out, not assumed to improve.

## Config additions (`config.json`)
- `paths.models: "models"`
- `data.oos_max_funds: 1000` (mirrors `data.max_funds`, reused for the disjoint sample size)

## Not wired into `conductor.py`'s default pipeline (deliberate)
Unlike steps 1-5, `step6_out_of_sample` is **not** added to `conductor.py`'s `build_pipeline()`.
It's a ~15-minute, network-heavy exploration beyond `plan.md`'s original Required Output scope
(requested separately, after the five required steps were already complete) - every
`python conductor.py` run reproducing the core deliverable shouldn't also re-sample 1000 new
funds and re-hit SEC/Yahoo every time. Run it on demand: `python -m steps.step6_out_of_sample.evaluate`
(or call `run(cfg)` directly), same pattern as calling `narrate_fund()` ad hoc for a specific
fund rather than narrating the whole panel.

## Determinism
Same seed reused for the OOS sampling (shuffle order identical, only the exclusion set differs).
Retraining uses the same seeded `RandomForestClassifier`/`DecisionTreeClassifier` construction
as step4.

## Follow-up (2026-07-11): OOS clustering is an independent re-fit, and the promoted model
Two clarifications requested after reviewing the original vs. OOS cluster-map PNGs:

**Cluster colors/labels don't match between the two maps - is that a bug?** No. Confirmed
directly from the code (`similarity.py`'s per-quarter loop): `KMeans(...).fit_predict(...)`
and `PCA(...).fit_transform(...)` both run as a **fresh, independent fit** every time `run()`
is called - the OOS run creates its own new `KMeans`/`PCA` instances on the OOS embeddings,
never reusing the original run's fitted centroids/transform. This is the same label-switching
behavior step2's design already documents *across quarters* (`cluster_id=3` in one quarter has
no relationship to `cluster_id=3` in another), just extended *across runs*. Since the two runs
also use fully disjoint fund sets, ARI/AMI between the two clusterings isn't even computable
(no shared population to compare labels on) - the right structural check is each run's *own*
validation quality (purity/ARI vs. Yahoo categories, both reported below), not cross-run label
agreement. With `short_title` now on the cluster-map legend, a real structural consistency
check *is* visible: the separated left-island cluster in both plots is dominated by Target-Date
allocation funds, even though the numeric `cluster_id` differs between runs.

**Was the retrained-on-combined model persisted?** Originally no - `evaluate_retrained_on_combined`
only returned an AUC, the fitted model was discarded after computing it. On request, added
`promote_retrained_model()`: refits on the combined panel (via the shared `fit_retrained_on_combined`
helper), backs up the previous `random_forest_model.joblib` to
`random_forest_model_original_363funds.joblib` (not silently overwritten), saves the new model
as the official one, and **regenerates `fund_predictions`/`model_feature_importances`** so the
persisted tables stay consistent with whichever model is actually saved as official - promoting
just the raw model file while leaving stale predictions behind would have left the database
internally inconsistent. This is a deliberate, explicit action (not called from `run()`
automatically) - confirmed with the user before executing. After promotion: 754 funds, test
AUC=0.725, `fund_predictions` regenerated to 8266 rows (6005 train + 2261 test).

## UAT (acceptance for this step) - all confirmed on real data
- OOS fund set confirmed fully disjoint: 1000 original + 1000 new series, **zero overlap**.
- OOS panel completeness matched the original quality bar exactly: 391 equity funds
  (of 1000 sampled), 12/12 quarters and 36/36 return-rows per fund.
- OOS clustering quality was consistent with (in fact slightly better than) the original:
  purity=0.459/ARI=0.291 vs. the original 0.409/0.249 - reassurance the embedding approach
  wasn't overfit to the specific original sample.
- Model round-trip verified: reloaded `.joblib` bundle (`model` + its exact `feature_cols`)
  reproduced the same predictions as the in-memory model.
- **Frozen model on 1000 new, never-trained-on funds: AUC=0.693** (vs. 0.710 on the original
  held-out time period) - only a modest drop, meaning the model generalizes to genuinely new
  funds and wasn't just capturing quirks of the original 363.
- **Retrained on the combined 754 equity funds (363 original + 391 new): AUC=0.725** - a modest
  improvement over both prior numbers, consistent with "more diverse training data helps a
  little," a different and complementary finding to the frozen-model result, not a replacement
  for it.
