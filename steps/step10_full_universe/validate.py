"""step10_full_universe/validate.py - out-of-time validation of the FROZEN step7 model.

Two scores, computed before any retraining is reported (design.md section 5):

1. `score_published_forward_predictions` - the dashboard's genuine, committed-before-the-
   fact 2024q4 -> 2025q1 predictions, scored against what actually happened. Each forward
   fund's REALIZED label is: own realized 2025q1 quarterly return < the median of its
   2024q4 top-N cosine-similarity peers' realized 2025q1 returns (>= min_valid_peers peers
   with a valid realized return required, AND a valid own return). Funds failing either gate
   are UNSCORED and counted honestly - never imputed:
     - no realized 2025q1 return (died/merged/late filer) -> n_missing_own_return
     - too few peers with a realized return                -> n_insufficient_peers
   The realized label is exactly `compute_peer_labels`' output at quarter 2024q4 when called
   with quarters_ordered=["2024q4","2025q1"] and fund_peers filtered to 2024q4 (that row's
   `underperform_next_quarter` = own 2025q1 return < median of top-N peers' 2025q1 returns,
   gated on own-present AND n_valid_peers >= min). The cross-namespace join is deliberate:
   predictions + peers come from `_all` (as-of 2024q4); realized returns from
   `monthly_returns_full` - a 2024q4 fund/peer absent from the `_full` data is attrition,
   which is precisely what we are measuring.

2. `score_frozen_model_rolled_forward` - the frozen unified RF bundle scored on every NEW
   labeled transition (Q >= 2025q1) in the `_full` panel. "Restricted to the funds/peers it
   knows" (design.md section 5) is satisfied by temporal disjointness (the frozen model
   trained on Q <= 2024q3) plus feature alignment to the bundle's exact training columns -
   not a fund filter. Q >= 2025q1 leaves the 2024q4 -> 2025q1 transition to scorer 1, so the
   two scores do not overlap.
"""
import logging

import pandas as pd
from sklearn.metrics import roc_auc_score

from fundspeers.io import load_model, load_table
# The reindex-to-frozen-columns alignment is identical to step6's out-of-sample pattern:
# a tier dummy absent from the new panel becomes an all-zero column (correct - no fund here
# belongs to that tier), and any extra panel column is dropped. Reused, not re-derived.
from steps.step6_out_of_sample.evaluate import align_features
from steps.step7_unified_universe.label import compute_peer_labels
from steps.step7_unified_universe.panel import (
    _quarterly_returns_from_monthly, assemble_unified_panel)

log = logging.getLogger(__name__)

FORWARD_QUARTER = "2024q4"
REALIZED_QUARTER = "2025q1"


def _score_forward(forward: pd.DataFrame, peers_q: pd.DataFrame, quarterly_returns: pd.DataFrame,
                   top_n: int, min_valid_peers: int) -> dict:
    """Pure scorer behind `score_published_forward_predictions`. `forward` = the committed
    2024q4 predictions (series_id, predicted_probability); `peers_q` = fund_peers as-of
    2024q4; `quarterly_returns` = realized quarterly returns (>= 2025q1)."""
    labels = compute_peer_labels(
        peers_q, quarterly_returns, quarters_ordered=[FORWARD_QUARTER, REALIZED_QUARTER],
        top_n=top_n, min_valid_peers=min_valid_peers)
    realized = labels.loc[labels["quarter"] == FORWARD_QUARTER,
                          ["series_id", "n_valid_peers_next", "underperform_next_quarter"]]

    # `compute_peer_labels` does not surface own_return_next, so we recover realized own
    # returns separately to split the two unscored reasons apart (label is NA iff own-missing
    # OR peers < min; own present distinguishes insufficient-peers from missing).
    own_next = quarterly_returns.loc[
        quarterly_returns["quarter"] == REALIZED_QUARTER, ["series_id", "quarterly_return"]
    ].rename(columns={"quarterly_return": "own_return_next"})

    df = (forward[["series_id", "predicted_probability"]].drop_duplicates("series_id")
          .merge(realized, on="series_id", how="left")
          .merge(own_next, on="series_id", how="left"))

    has_own = df["own_return_next"].notna()
    scored_mask = df["underperform_next_quarter"].notna()
    # The three buckets partition the forward set exactly: scored => own present, so
    # {scored} u {~own} u {own & ~scored} is a disjoint cover of every forward fund.
    n_missing = int((~has_own).sum())
    n_insufficient = int((has_own & ~scored_mask).sum())

    scored = df[scored_mask]
    y = scored["underperform_next_quarter"].astype(int)
    auc = float(roc_auc_score(y, scored["predicted_probability"])) if y.nunique() >= 2 else float("nan")
    base_rate = float(y.mean()) if len(y) else float("nan")

    log.info(f"published forward predictions: {len(scored)} scored (AUC={auc:.3f}, "
             f"base rate={base_rate:.3f}), {n_missing} missing own 2025q1 return, "
             f"{n_insufficient} with too few valid peers")
    return {"auc": auc, "n_scored": int(len(scored)),
            "n_missing_own_return": n_missing, "n_insufficient_peers": n_insufficient,
            "base_rate": base_rate}


def score_published_forward_predictions(cfg: dict) -> dict:
    """Did the model's genuine, committed-before-the-fact 2024q4 -> 2025q1 predictions work?
    Returns {auc, n_scored, n_missing_own_return, n_insufficient_peers, base_rate}."""
    predictions = load_table("unified_predictions", cfg)
    fund_peers = load_table("fund_peers_all", cfg)
    monthly = load_table("monthly_returns_full", cfg)

    forward = predictions[predictions["split"] == "forward"]
    peers_q = fund_peers[fund_peers["quarter"] == FORWARD_QUARTER]
    quarterly_returns = _quarterly_returns_from_monthly(monthly)

    return _score_forward(
        forward, peers_q, quarterly_returns,
        top_n=cfg["unified"]["peer_label_top_n"],
        min_valid_peers=cfg["unified"]["min_valid_peers_for_label"])


def _pooled_and_per_quarter_auc(frame: pd.DataFrame):
    """`frame` has columns quarter, y (0/1), proba. Returns (pooled_auc, {quarter: auc}).
    A single-class quarter (AUC undefined) is skipped, not an error."""
    pooled = float(roc_auc_score(frame["y"], frame["proba"]))
    per_quarter = {}
    for q, g in frame.groupby("quarter"):
        if g["y"].nunique() >= 2:
            per_quarter[str(q)] = float(roc_auc_score(g["y"], g["proba"]))
    return pooled, per_quarter


def score_frozen_model_rolled_forward(cfg: dict) -> dict:
    """The frozen unified RF rolled onto every new labeled transition (Q >= 2025q1) in the
    `_full` panel. Returns {auc_pooled, per_quarter: {q: auc}, n_rows}."""
    bundle = load_model("unified_rf_model", cfg)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    labeled, _forward, _panel_feature_cols = assemble_unified_panel(cfg, table_suffix="_full")
    # String compare works for the yyyy'q'q quarter format; leaves 2024q4->2025q1 to scorer 1.
    rolled = labeled[labeled["quarter"] >= REALIZED_QUARTER].copy()

    # Align to the FROZEN bundle's columns (not the _full panel's own feature_cols).
    x = align_features(rolled, feature_cols)
    proba = model.predict_proba(x)[:, 1]

    frame = pd.DataFrame({
        "quarter": rolled["quarter"].to_numpy(),
        "y": rolled["underperform_next_quarter"].astype(int).to_numpy(),
        "proba": proba,
    })
    pooled, per_quarter = _pooled_and_per_quarter_auc(frame)

    log.info(f"frozen model rolled forward on {len(rolled)} labeled Q>=2025q1 rows: "
             f"pooled AUC={pooled:.3f}; per-quarter {per_quarter}")
    return {"auc_pooled": pooled, "per_quarter": per_quarter, "n_rows": int(len(rolled))}
