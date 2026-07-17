"""step10_full_universe/build.py - orchestrated final run: pipeline repair, out-of-time
validation of the FROZEN step7 model (locked BEFORE the retrained model exists), retrain
on the `_full` universe (train 2022-2024 / test 2025-2026), a fund-disjoint split check,
and the label-stability study.

Clustering + metrics were already run in Task 4 (design.md `### _full clustering + metrics
run`); this module deliberately does NOT re-cluster. Run on demand (NOT wired into
conductor.py):

    python -m steps.step10_full_universe.build

Ordering matters: validation runs FIRST so the frozen-model / published-prediction numbers
are committed to a table before the retrained model's numbers exist - no peeking incentive.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from fundspeers.io import load_table, save_table
from steps.step4_predict.predict import time_based_split
from steps.step7_unified_universe.model import train_and_evaluate
from steps.step7_unified_universe.panel import assemble_unified_panel
from steps.step7_unified_universe.stability import run_stability
from steps.step10_full_universe.validate import (
    score_frozen_model_rolled_forward, score_published_forward_predictions)

log = logging.getLogger(__name__)

HOLDOUT_TRANSITIONS = 5  # test = 2024q4..2025q4 (labels realized through 2026q1)


def ensure_funds_full_segment(cfg: dict) -> pd.DataFrame:
    """Reviewer-mandated pipeline repair: guarantee `funds_full` carries a `segment` column.

    It currently exists only via an ad-hoc DB mutation (Task 4). This makes it reproducible:
    a fresh ingestion would land segment-less, and everything downstream
    (assemble_unified_panel, run_stability) filters on segment=='strategy'. Idempotent - if
    the column is already present it is left untouched (never recomputed/overwritten)."""
    funds_full = load_table("funds_full", cfg)
    if "segment" in funds_full.columns:
        log.info("segment column already present on funds_full - leaving untouched")
        return funds_full

    from steps.step7_unified_universe.merge import assign_segment
    funds_full["segment"] = funds_full["yahoo_category"].map(assign_segment)
    save_table(funds_full, "funds_full", cfg)
    per_series = funds_full.drop_duplicates("series_id")
    n_strategy = int((per_series["segment"] == "strategy").sum())
    n_allocation = int((per_series["segment"] == "allocation").sum())
    log.info(f"segment column added to funds_full: {n_strategy} strategy / "
             f"{n_allocation} allocation ({len(per_series)} series)")
    return funds_full


def _write_oot_validation(published: dict, frozen: dict, cfg: dict) -> None:
    """Long-format `oot_validation` table (columns: metric, quarter, value, source).

    published (single 2024q4->2025q1 score) is written first via CREATE OR REPLACE; the
    frozen rolled-forward rows (pooled + per-quarter) are then appended. Idempotent across a
    full re-run: published recreates the table fresh, frozen appends onto that fresh copy."""
    pub_rows = [
        {"metric": "auc", "quarter": "", "value": published["auc"], "source": "published_forward"},
        {"metric": "n_forward_total", "quarter": "",
         "value": float(published["n_scored"] + published["n_missing_own_return"]
                        + published["n_insufficient_peers"]), "source": "published_forward"},
        {"metric": "n_scored", "quarter": "", "value": float(published["n_scored"]),
         "source": "published_forward"},
        {"metric": "n_missing_own_return", "quarter": "",
         "value": float(published["n_missing_own_return"]), "source": "published_forward"},
        {"metric": "n_insufficient_peers", "quarter": "",
         "value": float(published["n_insufficient_peers"]), "source": "published_forward"},
        {"metric": "base_rate", "quarter": "", "value": published["base_rate"],
         "source": "published_forward"},
    ]
    save_table(pd.DataFrame(pub_rows), "oot_validation", cfg)

    frozen_rows = [
        {"metric": "auc_pooled", "quarter": "", "value": frozen["auc_pooled"],
         "source": "frozen_rolled_forward"},
        {"metric": "n_rows", "quarter": "", "value": float(frozen["n_rows"]),
         "source": "frozen_rolled_forward"},
    ]
    frozen_rows += [{"metric": "auc", "quarter": q, "value": v, "source": "frozen_rolled_forward"}
                    for q, v in frozen["per_quarter"].items()]
    existing = load_table("oot_validation", cfg)
    save_table(pd.concat([existing, pd.DataFrame(frozen_rows)], ignore_index=True),
               "oot_validation", cfg)


def fund_disjoint_auc(cfg: dict) -> float:
    """Fund-disjoint split check (adopted critique item; `evaluate.fund_disjoint_split` was
    only ever planned, never built - implemented inline here). A seeded 80/20 split of the
    labeled `_full` funds: train on the 80%'s train-quarter rows, test on the 20%'s
    test-quarter rows - so no fund appears in both train and test. Same seeded RF, same
    2022-2024 / 2025-2026 quarter boundary as the main retrain."""
    labeled, forward, feature_cols = assemble_unified_panel(cfg, table_suffix="_full")
    quarters_ordered = sorted(set(labeled["quarter"]) | set(forward["quarter"]))
    _train, _test, train_q, test_q = time_based_split(labeled, quarters_ordered, HOLDOUT_TRANSITIONS)

    rng = np.random.default_rng(cfg["seed"])
    funds = np.array(sorted(labeled["series_id"].unique()))
    perm = rng.permutation(len(funds))
    n_train = int(len(funds) * 0.8)
    train_funds, test_funds = set(funds[perm[:n_train]]), set(funds[perm[n_train:]])

    train = labeled[labeled["quarter"].isin(train_q) & labeled["series_id"].isin(train_funds)]
    test = labeled[labeled["quarter"].isin(test_q) & labeled["series_id"].isin(test_funds)]
    rf = RandomForestClassifier(
        n_estimators=cfg["model"]["rf"]["n_estimators"],
        max_depth=cfg["model"]["rf"]["max_depth"],
        min_samples_leaf=cfg["model"]["rf"]["min_samples_leaf"],
        random_state=cfg["seed"],
    ).fit(train[feature_cols], train["underperform_next_quarter"].astype(int))
    auc = float(roc_auc_score(test["underperform_next_quarter"].astype(int),
                              rf.predict_proba(test[feature_cols])[:, 1]))
    log.info(f"fund-disjoint split: {len(train_funds)} train funds / {len(test_funds)} test "
             f"funds ({len(train)} train / {len(test)} test rows), AUC={auc:.3f}")

    ev = load_table("full_model_eval", cfg)
    ev = pd.concat([ev, pd.DataFrame([{"metric": "auc_fund_disjoint", "quarter": "", "value": auc}])],
                   ignore_index=True)
    save_table(ev, "full_model_eval", cfg)
    return auc


def run_retired(cfg: dict) -> None:
    """Retired-model refresh (design step16): keep the falsifiable record growing -
    segment repair, frozen out-of-time scoring, label-stability - and nothing that
    trains or emits new predictions."""
    log.info("=== step10 (RETIRED): pipeline repair (funds_full.segment) ===")
    ensure_funds_full_segment(cfg)
    log.info("=== step10 (RETIRED): out-of-time scoring of the frozen model ===")
    published = score_published_forward_predictions(cfg)
    frozen = score_frozen_model_rolled_forward(cfg)
    _write_oot_validation(published, frozen, cfg)
    log.info("=== step10 (RETIRED): label-stability study ===")
    run_stability(cfg, table_suffix="_full", output_table="full_label_stability")
    log.info("frozen record updated; no retraining, no new forward predictions (model retired)")


def run(cfg: dict) -> None:
    # 0. Pipeline repair: ensure funds_full carries the segment column (reproducibly).
    log.info("=== step10: pipeline repair (funds_full.segment) ===")
    ensure_funds_full_segment(cfg)

    # 1. OUT-OF-TIME VALIDATION FIRST - frozen numbers locked before the retrained model exists.
    log.info("=== step10: out-of-time validation (frozen step7 model) ===")
    published = score_published_forward_predictions(cfg)
    frozen = score_frozen_model_rolled_forward(cfg)
    _write_oot_validation(published, frozen, cfg)

    # 2. Retrain on the full universe (train 2022-2024, test 2025-2026).
    log.info("=== step10: retrain + Monte Carlo evaluation (_full) ===")
    results = train_and_evaluate(cfg, table_suffix="_full", holdout_transitions=HOLDOUT_TRANSITIONS,
                                 output_prefix="full")

    # 2b. Fund-disjoint split check.
    log.info("=== step10: fund-disjoint split check ===")
    fd_auc = fund_disjoint_auc(cfg)

    # 3. Label-stability study on the full universe.
    log.info("=== step10: label-stability study (_full) ===")
    stability = run_stability(cfg, table_suffix="_full", output_table="full_label_stability")

    log.info(
        "SUMMARY | published forward predictions vs reality: AUC=%.3f (n_scored=%d, "
        "n_missing_own=%d, n_insufficient_peers=%d, base_rate=%.3f) | frozen rolled forward: "
        "pooled AUC=%.3f (n=%d, per_quarter=%s) | retrained: pooled AUC=%.3f [%.3f, %.3f] vs "
        "persistence %.3f (reversed %.3f); edge CI [%.3f, %.3f], p(edge<=0)=%.4f; model beat "
        "persistence in %d/%d comparable quarters | fund-disjoint AUC=%.3f | stability: mean "
        "flip=%.3f, share>10%%=%.3f (n=%d)",
        published["auc"], published["n_scored"], published["n_missing_own_return"],
        published["n_insufficient_peers"], published["base_rate"],
        frozen["auc_pooled"], frozen["n_rows"], frozen["per_quarter"],
        results["auc"], results["auc_ci_low"], results["auc_ci_high"],
        results["persistence_auc"], 1.0 - results["persistence_auc"],
        results["edge_ci_low"], results["edge_ci_high"], results["p_edge_le_zero"],
        results["quarters_model_wins"], results["n_quarters_comparable"],
        fd_auc, stability["mean_flip_rate"], stability["share_flip_gt_10pct"],
        stability["n_evaluated"],
    )


if __name__ == "__main__":
    from fundspeers.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(load_config())
