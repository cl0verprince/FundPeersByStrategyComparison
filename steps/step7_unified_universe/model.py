"""step7_unified_universe/model.py - fit the unified RF, evaluate vs baselines, and run
the Monte Carlo uncertainty layer (see design.md section 6b: MC quantifies evaluation
uncertainty; it never simulates fund returns).
"""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from fundspeers.io import save_model, save_table
from steps.step4_predict.predict import time_based_split
from steps.step7_unified_universe.label import LABEL_DEFINITION
from steps.step7_unified_universe.panel import assemble_unified_panel

log = logging.getLogger(__name__)


def fund_clustered_bootstrap(test: pd.DataFrame, y_col: str, model_score_col: str,
                             persistence_score_col: str, iterations: int, seed: int) -> dict:
    """Resample test-set FUNDS with replacement (a fund's quarters stay together - rows
    within a fund correlate, so row-level resampling would understate variance). Returns
    95% CIs for model AUC, persistence AUC, their difference, and the one-sided
    p(edge <= 0). Paired: both AUCs computed on the SAME resample each iteration."""
    rng = np.random.default_rng(seed)
    groups = {s: g for s, g in test.groupby("series_id")}
    series = np.array(sorted(groups))
    model_aucs, persist_aucs = [], []
    for _ in range(iterations):
        draw = rng.choice(series, size=len(series), replace=True)
        sample = pd.concat([groups[s] for s in draw], ignore_index=True)
        if sample[y_col].nunique() < 2:
            continue
        model_aucs.append(roc_auc_score(sample[y_col], sample[model_score_col]))
        persist_aucs.append(roc_auc_score(sample[y_col], sample[persistence_score_col]))
    model_aucs, persist_aucs = np.array(model_aucs), np.array(persist_aucs)
    edge = model_aucs - persist_aucs
    return {
        "auc_ci_low": float(np.percentile(model_aucs, 2.5)),
        "auc_ci_high": float(np.percentile(model_aucs, 97.5)),
        "persistence_ci_low": float(np.percentile(persist_aucs, 2.5)),
        "persistence_ci_high": float(np.percentile(persist_aucs, 97.5)),
        "edge_ci_low": float(np.percentile(edge, 2.5)),
        "edge_ci_high": float(np.percentile(edge, 97.5)),
        "p_edge_le_zero": float((edge <= 0).mean()),
    }


def train_and_evaluate(cfg: dict) -> dict:
    labeled, forward, feature_cols = assemble_unified_panel(cfg)
    quarters_ordered = sorted(set(labeled["quarter"]) | set(forward["quarter"]))
    train, test, train_q, test_q = time_based_split(
        labeled, quarters_ordered, cfg["model"]["test_transitions_holdout"])
    if max(train_q) >= min(test_q):
        raise RuntimeError(f"split leaks: train up to {max(train_q)}, test from {min(test_q)}")

    x_train, y_train = train[feature_cols], train["underperform_next_quarter"].astype(int)
    x_test, y_test = test[feature_cols], test["underperform_next_quarter"].astype(int)

    rf = RandomForestClassifier(
        n_estimators=cfg["model"]["rf"]["n_estimators"],
        max_depth=cfg["model"]["rf"]["max_depth"],
        min_samples_leaf=cfg["model"]["rf"]["min_samples_leaf"],
        random_state=cfg["seed"],
    ).fit(x_train, y_train)

    test = test.assign(proba=rf.predict_proba(x_test)[:, 1],
                       persist=-test["return_vs_peer_median_q"])
    pooled_auc = roc_auc_score(y_test, test["proba"])
    persistence_auc = roc_auc_score(y_test, test["persist"])

    eval_rows = [
        {"metric": "auc_pooled", "quarter": "", "value": pooled_auc},
        {"metric": "auc_persistence_baseline", "quarter": "", "value": persistence_auc},
        {"metric": "auc_random_baseline", "quarter": "", "value": 0.5},
    ]
    quarters_model_wins = 0
    for q, g in test.groupby("quarter"):
        yq = g["underperform_next_quarter"].astype(int)
        if yq.nunique() < 2:
            continue
        auc_q = roc_auc_score(yq, g["proba"])
        persist_q = roc_auc_score(yq, g["persist"])
        quarters_model_wins += int(auc_q > persist_q)
        eval_rows.append({"metric": "auc_pooled", "quarter": q, "value": auc_q})
        eval_rows.append({"metric": "auc_persistence_baseline", "quarter": q, "value": persist_q})

    boot = fund_clustered_bootstrap(
        test, "underperform_next_quarter", "proba", "persist",
        iterations=cfg["unified"]["bootstrap_iterations"], seed=cfg["seed"])
    eval_rows += [{"metric": k, "quarter": "", "value": v} for k, v in boot.items()]

    label_definition = LABEL_DEFINITION.format(top_n=cfg["unified"]["peer_label_top_n"])
    save_model({"model": rf, "feature_cols": feature_cols,
                "label_definition": label_definition}, "unified_rf_model", cfg)

    predictions = pd.concat([
        train.assign(split="train", predicted_probability=rf.predict_proba(x_train)[:, 1]),
        test.assign(split="test", predicted_probability=test["proba"]),
        forward.assign(split="forward",
                       predicted_probability=rf.predict_proba(forward[feature_cols])[:, 1]),
    ])[["series_id", "quarter", "predicted_probability", "underperform_next_quarter", "split"]]
    predictions = predictions.rename(columns={"underperform_next_quarter": "actual_label"})
    predictions["actual_label"] = predictions["actual_label"].astype("float")  # NA-safe for duckdb

    importances = pd.DataFrame({"feature": feature_cols,
                                "importance": rf.feature_importances_}
                               ).sort_values("importance", ascending=False)

    save_table(pd.concat([labeled, forward], ignore_index=True), "unified_panel", cfg)
    save_table(predictions, "unified_predictions", cfg)
    save_table(importances, "unified_feature_importances", cfg)
    save_table(pd.DataFrame(eval_rows), "unified_model_eval", cfg)

    log.info(f"unified RF: pooled test AUC={pooled_auc:.3f} "
             f"[{boot['auc_ci_low']:.3f}, {boot['auc_ci_high']:.3f}] vs persistence "
             f"{persistence_auc:.3f}; edge CI [{boot['edge_ci_low']:.3f}, "
             f"{boot['edge_ci_high']:.3f}], p(edge<=0)={boot['p_edge_le_zero']:.4f}; "
             f"model beat persistence in {quarters_model_wins}/{len(test_q)} test quarters")
    return {"auc": pooled_auc, "persistence_auc": persistence_auc,
            "quarters_model_wins": quarters_model_wins, "n_test_quarters": len(test_q), **boot}
