"""step10 out-of-time validation, pure-logic parts on synthetic temp-DB fixtures.

Two scorers:
- score_published_forward_predictions: the committed 2024q4->2025q1 predictions scored
  against realized 2025q1 returns + realized 2024q4-peer medians (realized-label
  correctness, min-valid-peers and dead-fund attrition buckets, AUC sanity).
- score_frozen_model_rolled_forward: feature alignment to a frozen bundle's columns
  (missing tier col -> zeros, extra panel col dropped) + pooled/per-quarter AUC helper.
"""
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from fundspeers.io import load_model, save_model, save_table
from steps.step10_full_universe.validate import (
    align_features, score_published_forward_predictions, _pooled_and_per_quarter_auc)


@pytest.fixture
def cfg(tmp_path):
    return {
        "seed": 42,
        "paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "processed"),
                  "reports": str(tmp_path / "reports"), "models": str(tmp_path / "models")},
        "metrics": {"risk_free_annual": 0.02},
        "unified": {"peer_label_top_n": 10, "min_valid_peers_for_label": 5},
    }


def _months(sid, quarter, ret):
    return [{"series_id": sid, "quarter": quarter, "month_in_quarter": m, "total_return": ret}
            for m in (1, 2, 3)]


@pytest.fixture
def forward_tables(cfg):
    """One forward-predicted fund per realized-label outcome:
      F_below  own 2025q1 return < peer median, 10 valid peers -> label 1, scored
      F_above  own 2025q1 return > peer median, 10 valid peers -> label 0, scored
      F_4peers own return present but only 4 valid peers        -> n_insufficient_peers
      F_noown  10 valid peers but no realized 2025q1 return      -> n_missing_own_return
    """
    valid_peers = [f"pV{i}" for i in range(1, 11)]      # all have 2025q1 returns
    w_peers = [f"pW{i}" for i in range(1, 11)]          # only pW1..pW4 have 2025q1 returns

    monthly = []
    for p in valid_peers:
        monthly += _months(p, "2025q1", 2.0)
    for p in w_peers[:4]:
        monthly += _months(p, "2025q1", 2.0)
    monthly += _months("F_below", "2025q1", 1.0)        # below peers -> label 1
    monthly += _months("F_above", "2025q1", 3.0)        # above peers -> label 0
    monthly += _months("F_4peers", "2025q1", 1.0)       # own present, peers insufficient
    # F_noown: deliberately no 2025q1 rows -> attrition
    save_table(pd.DataFrame(monthly), "monthly_returns_full", cfg)

    peer_rows = []
    for fund, pool in [("F_below", valid_peers), ("F_above", valid_peers),
                       ("F_noown", valid_peers), ("F_4peers", w_peers)]:
        for rank, peer in enumerate(pool, start=1):
            peer_rows.append({"series_id": fund, "quarter": "2024q4", "peer_rank": rank,
                              "peer_series_id": peer, "cosine_similarity": 0.9})
    # a stale earlier-quarter peer row that must be ignored (only 2024q4 counts):
    peer_rows.append({"series_id": "F_below", "quarter": "2024q3", "peer_rank": 1,
                      "peer_series_id": "pV1", "cosine_similarity": 0.9})
    save_table(pd.DataFrame(peer_rows), "fund_peers_all", cfg)

    preds = [{"series_id": s, "quarter": "2024q4", "predicted_probability": p,
              "actual_label": None, "split": "forward"}
             for s, p in [("F_below", 0.9), ("F_above", 0.1),
                          ("F_4peers", 0.5), ("F_noown", 0.5)]]
    # a non-forward row that must be excluded from scoring:
    preds.append({"series_id": "F_below", "quarter": "2024q3", "predicted_probability": 0.2,
                  "actual_label": 1.0, "split": "test"})
    save_table(pd.DataFrame(preds), "unified_predictions", cfg)


def test_realized_label_buckets_partition_forward_set(cfg, forward_tables):
    out = score_published_forward_predictions(cfg)
    assert out["n_scored"] == 2
    assert out["n_missing_own_return"] == 1        # F_noown
    assert out["n_insufficient_peers"] == 1        # F_4peers, only 4 valid peers
    # strong invariant: the three buckets partition the 4 forward-predicted funds exactly
    assert (out["n_scored"] + out["n_missing_own_return"]
            + out["n_insufficient_peers"]) == 4
    # F_below (below its peer median) -> label 1, F_above -> label 0 -> base rate 0.5
    assert out["base_rate"] == 0.5


def test_auc_perfect_ranking_is_one(cfg, forward_tables):
    # F_below has label 1 and the higher probability (0.9); F_above label 0, prob 0.1.
    # A perfect ranking of the realized labels -> AUC 1.0 (and this confirms below->1,
    # above->0: a swapped label assignment would give AUC 0.0).
    out = score_published_forward_predictions(cfg)
    assert out["auc"] == 1.0


def test_pooled_and_per_quarter_auc(cfg):
    # perfect ranking within each quarter -> pooled and per-quarter AUC all 1.0;
    # a single-class quarter is skipped (undefined AUC), not crashed on.
    frame = pd.DataFrame([
        {"quarter": "2025q1", "y": 1, "proba": 0.9},
        {"quarter": "2025q1", "y": 0, "proba": 0.1},
        {"quarter": "2025q2", "y": 1, "proba": 0.8},
        {"quarter": "2025q2", "y": 0, "proba": 0.2},
        {"quarter": "2025q3", "y": 1, "proba": 0.5},   # single-class quarter -> skipped
        {"quarter": "2025q3", "y": 1, "proba": 0.6},
    ])
    pooled, per_quarter = _pooled_and_per_quarter_auc(frame)
    assert pooled == 1.0
    assert per_quarter == {"2025q1": 1.0, "2025q2": 1.0}
    assert "2025q3" not in per_quarter


def test_feature_alignment_missing_col_zeroed_extra_dropped(cfg):
    # A frozen bundle trained on 2 features, one of which (tier_X) is a tier dummy absent
    # from the new panel; the panel also carries an extra column the bundle never saw.
    feature_cols = ["feat_a", "tier_X"]
    x_train = pd.DataFrame({"feat_a": [0.0, 1.0, 0.0, 1.0], "tier_X": [0, 0, 1, 1]})
    y_train = [0, 1, 0, 1]
    clf = DecisionTreeClassifier(random_state=0).fit(x_train, y_train)
    save_model({"model": clf, "feature_cols": feature_cols,
                "label_definition": "d"}, "unified_rf_model", cfg)

    bundle = load_model("unified_rf_model", cfg)
    panel = pd.DataFrame({"feat_a": [0.5, 1.5],
                          "feat_extra": [9.0, 9.0],        # extra -> dropped by reindex
                          "underperform_next_quarter": [0, 1]})
    aligned = align_features(panel, bundle["feature_cols"])
    assert list(aligned.columns) == feature_cols           # exact columns, exact order
    assert (aligned["tier_X"] == 0).all()                  # missing tier col -> zeros
    assert "feat_extra" not in aligned.columns             # extra col dropped
    # the frozen model predicts on the aligned frame without a column-mismatch error:
    proba = bundle["model"].predict_proba(aligned)[:, 1]
    assert len(proba) == 2
