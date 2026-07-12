"""step9_fees_turnover.evaluate — does adding point-in-time expense ratio + portfolio
turnover move the model's edge past the mean-reversion tie?

Design decisions 4-6 made concrete. Everything runs on the COMMON COVERED SUBSET (labeled
rows that carry BOTH fee features) so the no-fees vs with-fees comparison is apples-to-apples
— the only difference between the two arms is the two extra columns. Four fits:
{no-fees, +fees} x {chronological split, fund-disjoint split}. Baselines on the chronological
test rows: random 0.5, persistence (raw AND its reversed mean-reversion reading), expense-rank.

Baselines re-based for the `_full` universe (step10): reversed persistence ~0.604, the
step10 no-fees full-panel model ~0.614. holdout_transitions=5 (step10's boundary: test =
2024q4..2025q4 transitions) so the test window matches. fund_clustered_bootstrap is reused
verbatim from step7 (generic in its two score columns).
"""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from fundspeers.io import load_table, save_table
from steps.step4_predict.predict import time_based_split
from steps.step7_unified_universe.model import fund_clustered_bootstrap
from steps.step7_unified_universe.panel import assemble_unified_panel

log = logging.getLogger(__name__)

HOLDOUT_TRANSITIONS = 5  # match step10: test = 2024q4..2025q4 transitions
FEE_FEATURES = ["expense_ratio_net", "portfolio_turnover"]


def fund_disjoint_split(panel, quarters_ordered, holdout_transitions, test_share, seed):
    """Doubly-disjoint split: train on train-quarter rows of an 80% fund sample, test on
    test-quarter rows of the held-out 20% (funds never seen, in a period never trained on).
    Pure/deterministic. Mirrors step10's inline fund-disjoint pattern, parameterized."""
    transition_quarters = quarters_ordered[:-1]
    test_q = set(transition_quarters[-holdout_transitions:])
    train_q = set(transition_quarters[:-holdout_transitions])
    rng = np.random.default_rng(seed)
    funds = np.array(sorted(panel["series_id"].unique()))
    perm = rng.permutation(len(funds))
    n_test = int(np.floor(test_share * len(funds)))
    test_funds = set(funds[perm[:n_test]])
    train_funds = set(funds[perm[n_test:]])
    train = panel[panel["quarter"].isin(train_q) & panel["series_id"].isin(train_funds)]
    test = panel[panel["quarter"].isin(test_q) & panel["series_id"].isin(test_funds)]
    return train, test


def expense_rank_auc(df, y_col="underperform_next_quarter", score_col="expense_ratio_net"):
    """The reviewer's naive baseline: rank funds by net expense (higher fee -> predicted
    underperformer) and score AUC. A pure helper so it is unit-testable."""
    return float(roc_auc_score(df[y_col].astype(int), df[score_col]))


def _fit_score(train, test, feature_cols, cfg):
    rf = RandomForestClassifier(
        n_estimators=cfg["model"]["rf"]["n_estimators"],
        max_depth=cfg["model"]["rf"]["max_depth"],
        min_samples_leaf=cfg["model"]["rf"]["min_samples_leaf"],
        random_state=cfg["seed"],
    ).fit(train[feature_cols], train["underperform_next_quarter"].astype(int))
    proba = rf.predict_proba(test[feature_cols])[:, 1]
    auc = float(roc_auc_score(test["underperform_next_quarter"].astype(int), proba))
    return rf, proba, auc


def run_evaluation(cfg: dict) -> dict:
    labeled, forward, base_features = assemble_unified_panel(cfg, table_suffix="_full")
    quarters_ordered = sorted(set(labeled["quarter"]) | set(forward["quarter"]))

    rr_fees = load_table("rr_fees", cfg)[["series_id", "quarter", *FEE_FEATURES]]
    labeled = labeled.merge(rr_fees, on=["series_id", "quarter"], how="left")

    # Common covered subset: rows carrying BOTH fee features (apples-to-apples).
    covered = labeled.dropna(subset=FEE_FEATURES).reset_index(drop=True)
    fees_features = base_features + FEE_FEATURES
    log.info(f"covered subset: {len(covered)}/{len(labeled)} labeled rows carry both fees")

    # ---- Chronological split ----
    train, test, train_q, test_q = time_based_split(covered, quarters_ordered, HOLDOUT_TRANSITIONS)
    if max(train_q) >= min(test_q):
        raise RuntimeError(f"split leaks: train up to {max(train_q)}, test from {min(test_q)}")
    _rf_nf, proba_nf, auc_nf = _fit_score(train, test, base_features, cfg)
    rf_fe, proba_fe, auc_fe = _fit_score(train, test, fees_features, cfg)

    y = test["underperform_next_quarter"].astype(int)
    test = test.assign(
        proba_nofees=proba_nf, proba_fees=proba_fe,
        persist_raw=-test["return_vs_peer_median_q"],       # step7 orientation (~0.396)
        persist_reversed=test["return_vs_peer_median_q"],   # mean-reversion reading (~0.604)
        expense_rank=test["expense_ratio_net"])
    auc_random = 0.5
    auc_persist_raw = float(roc_auc_score(y, test["persist_raw"]))
    auc_persist_rev = float(roc_auc_score(y, test["persist_reversed"]))
    auc_exprank = expense_rank_auc(test)

    it, seed = cfg["unified"]["bootstrap_iterations"], cfg["seed"]
    boot_fees_vs_rev = fund_clustered_bootstrap(test, "underperform_next_quarter",
                                                "proba_fees", "persist_reversed", it, seed)
    boot_fees_vs_exp = fund_clustered_bootstrap(test, "underperform_next_quarter",
                                                "proba_fees", "expense_rank", it, seed)
    boot_fees_vs_nf = fund_clustered_bootstrap(test, "underperform_next_quarter",
                                               "proba_fees", "proba_nofees", it, seed)
    boot_nf_vs_rev = fund_clustered_bootstrap(test, "underperform_next_quarter",
                                              "proba_nofees", "persist_reversed", it, seed)

    # per-quarter AUCs (both models)
    pq_rows = []
    for q, g in test.groupby("quarter"):
        yq = g["underperform_next_quarter"].astype(int)
        if yq.nunique() < 2:
            continue
        pq_rows.append({"metric": "auc_pooled", "quarter": q,
                        "value": float(roc_auc_score(yq, g["proba_nofees"])), "variant": "nofees_chrono"})
        pq_rows.append({"metric": "auc_pooled", "quarter": q,
                        "value": float(roc_auc_score(yq, g["proba_fees"])), "variant": "fees_chrono"})

    # ---- Fund-disjoint split (both arms) ----
    fd_train, fd_test = fund_disjoint_split(
        covered, quarters_ordered, HOLDOUT_TRANSITIONS,
        cfg["fees"]["fund_disjoint_test_share"], seed)
    _rf, _p, auc_nf_fd = _fit_score(fd_train, fd_test, base_features, cfg)
    _rf, _p, auc_fe_fd = _fit_score(fd_train, fd_test, fees_features, cfg)

    # ---- Feature importances (fees model) ----
    importances = pd.DataFrame({"feature": fees_features,
                                "importance": rf_fe.feature_importances_}
                               ).sort_values("importance", ascending=False).reset_index(drop=True)
    save_table(importances, "fees_feature_importances", cfg)
    fee_ranks = {f: int(importances.index[importances["feature"] == f][0]) + 1 for f in FEE_FEATURES}

    # ---- Persist eval table (long format: metric, quarter, value, variant) ----
    rows = [
        {"metric": "auc_pooled", "quarter": "", "value": auc_nf, "variant": "nofees_chrono"},
        {"metric": "auc_pooled", "quarter": "", "value": auc_fe, "variant": "fees_chrono"},
        {"metric": "auc_pooled", "quarter": "", "value": auc_nf_fd, "variant": "nofees_fund_disjoint"},
        {"metric": "auc_pooled", "quarter": "", "value": auc_fe_fd, "variant": "fees_fund_disjoint"},
        {"metric": "auc_random", "quarter": "", "value": auc_random, "variant": "baseline_chrono"},
        {"metric": "auc_persistence_raw", "quarter": "", "value": auc_persist_raw, "variant": "baseline_chrono"},
        {"metric": "auc_persistence_reversed", "quarter": "", "value": auc_persist_rev, "variant": "baseline_chrono"},
        {"metric": "auc_expense_rank", "quarter": "", "value": auc_exprank, "variant": "baseline_chrono"},
        {"metric": "n_test_rows", "quarter": "", "value": float(len(test)), "variant": "chrono"},
        {"metric": "n_test_rows", "quarter": "", "value": float(len(fd_test)), "variant": "fund_disjoint"},
        {"metric": "n_covered_labeled", "quarter": "", "value": float(len(covered)), "variant": "chrono"},
    ]
    for name, b in [("fees_vs_reversed", boot_fees_vs_rev), ("fees_vs_exprank", boot_fees_vs_exp),
                    ("fees_vs_nofees", boot_fees_vs_nf), ("nofees_vs_reversed", boot_nf_vs_rev)]:
        rows += [{"metric": k, "quarter": "", "value": v, "variant": name} for k, v in b.items()]
    rows += pq_rows
    save_table(pd.DataFrame(rows), "fees_model_eval", cfg)

    result = {
        "auc_nofees_chrono": auc_nf, "auc_fees_chrono": auc_fe,
        "auc_nofees_fund_disjoint": auc_nf_fd, "auc_fees_fund_disjoint": auc_fe_fd,
        "auc_persist_raw": auc_persist_raw, "auc_persist_reversed": auc_persist_rev,
        "auc_expense_rank": auc_exprank,
        "fees_auc_ci": [boot_fees_vs_rev["auc_ci_low"], boot_fees_vs_rev["auc_ci_high"]],
        "nofees_auc_ci": [boot_nf_vs_rev["auc_ci_low"], boot_nf_vs_rev["auc_ci_high"]],
        "edge_fees_vs_reversed": [boot_fees_vs_rev["edge_ci_low"], boot_fees_vs_rev["edge_ci_high"]],
        "p_fees_le_reversed": boot_fees_vs_rev["p_edge_le_zero"],
        "edge_fees_vs_exprank": [boot_fees_vs_exp["edge_ci_low"], boot_fees_vs_exp["edge_ci_high"]],
        "p_fees_le_exprank": boot_fees_vs_exp["p_edge_le_zero"],
        "edge_fees_vs_nofees": [boot_fees_vs_nf["edge_ci_low"], boot_fees_vs_nf["edge_ci_high"]],
        "p_fees_le_nofees": boot_fees_vs_nf["p_edge_le_zero"],
        "fee_importance_ranks": fee_ranks, "n_features": len(fees_features),
        "n_covered_labeled": len(covered), "n_test_rows": len(test), "n_fd_test_rows": len(fd_test),
    }
    log.info(
        "step9 eval | chrono: no-fees AUC=%.3f [%.3f,%.3f], +fees AUC=%.3f [%.3f,%.3f] | "
        "bars: reversed-persistence=%.3f, expense-rank=%.3f, random=0.5 | edge +fees vs reversed "
        "CI [%.3f,%.3f] p=%.3f; +fees vs no-fees CI [%.3f,%.3f] p=%.3f; +fees vs expense-rank CI "
        "[%.3f,%.3f] p=%.3f | fund-disjoint: no-fees=%.3f, +fees=%.3f | fee ranks (of %d): %s",
        auc_nf, *result["nofees_auc_ci"], auc_fe, *result["fees_auc_ci"],
        auc_persist_rev, auc_exprank,
        *result["edge_fees_vs_reversed"], result["p_fees_le_reversed"],
        *result["edge_fees_vs_nofees"], result["p_fees_le_nofees"],
        *result["edge_fees_vs_exprank"], result["p_fees_le_exprank"],
        auc_nf_fd, auc_fe_fd, len(fees_features), fee_ranks)
    return result
