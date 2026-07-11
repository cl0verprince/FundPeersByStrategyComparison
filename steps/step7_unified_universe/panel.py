"""step7_unified_universe/panel.py - assemble the unified feature/label panel.

One row per (strategy-segment series_id, quarter): trailing return stats (point-in-time
safe, reused from step4), holdings features, peer features, the kNN-peer label. Rows with
any missing feature are dropped, not imputed; last-quarter rows with complete features
form the FORWARD set (label unknowable yet - the dashboard's genuine forward predictions).
"""
import logging

import pandas as pd

from fundspeers.category import category_tier
from fundspeers.io import load_table
from steps.step4_predict.predict import compute_trailing_features
from steps.step7_unified_universe.features import (
    compute_holdings_features, compute_net_asset_momentum, compute_peer_similarity_feature)
from steps.step7_unified_universe.label import compute_peer_labels

log = logging.getLogger(__name__)

BASE_FEATURE_COLS = [
    "trailing_return", "trailing_volatility", "trailing_sharpe", "trailing_max_drawdown",
    "return_vs_peer_median_q", "net_assets", "net_assets_qoq", "mean_peer_similarity",
    "hhi", "top10_weight", "n_holdings",
]


def _quarterly_returns_from_monthly(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """Compound each (series_id, quarter)'s 3 monthly returns; .values so a missing month
    correctly propagates NaN (the step3 np.prod-skipna bug, not repeated here)."""
    import numpy as np
    df = monthly_returns.copy()
    df["r"] = df["total_return"] / 100.0
    df = df.sort_values(["series_id", "quarter", "month_in_quarter"])
    grouped = df.groupby(["series_id", "quarter"])["r"].apply(
        lambda s: np.prod(1 + s.values) - 1)
    return grouped.reset_index().rename(columns={"r": "quarterly_return"})


def assemble_unified_panel(cfg: dict):
    funds = load_table("funds_all", cfg)
    monthly_returns = load_table("monthly_returns_all", cfg)
    holdings = load_table("holdings_all", cfg)
    fund_peers = load_table("fund_peers_all", cfg)

    strategy = funds[funds["is_us_equity"] & (funds["segment"] == "strategy")]
    strategy_series = set(strategy["series_id"].unique())
    quarters_ordered = sorted(strategy["quarter"].unique())
    returns = monthly_returns[monthly_returns["series_id"].isin(strategy_series)]

    trailing = compute_trailing_features(returns, cfg["metrics"]["risk_free_annual"])
    quarterly_returns = _quarterly_returns_from_monthly(returns)
    labels = compute_peer_labels(
        fund_peers, quarterly_returns, quarters_ordered,
        top_n=cfg["unified"]["peer_label_top_n"],
        min_valid_peers=cfg["unified"]["min_valid_peers_for_label"])

    acc_lookup = strategy[["accession_number", "series_id"]].drop_duplicates()
    ec = holdings[(holdings["asset_cat"] == "EC")
                  & holdings["accession_number"].isin(set(acc_lookup["accession_number"]))]
    ec = ec.merge(acc_lookup, on="accession_number", how="inner")
    holdings_feats = compute_holdings_features(ec[["series_id", "quarter", "currency_value"]])

    peer_sim = compute_peer_similarity_feature(fund_peers, cfg["unified"]["peer_label_top_n"])
    momentum = compute_net_asset_momentum(strategy)

    fund_static = strategy.drop_duplicates("series_id")[["series_id", "yahoo_category"]].copy()
    fund_static["category_tier"] = fund_static["yahoo_category"].map(category_tier)
    funds_at_q = strategy[["series_id", "quarter", "net_assets"]].drop_duplicates(
        ["series_id", "quarter"])

    panel = (trailing
             .merge(labels, on=["series_id", "quarter"], how="inner")
             .merge(holdings_feats, on=["series_id", "quarter"], how="left")
             .merge(peer_sim, on=["series_id", "quarter"], how="left")
             .merge(momentum, on=["series_id", "quarter"], how="left")
             .merge(funds_at_q, on=["series_id", "quarter"], how="left")
             .merge(fund_static, on="series_id", how="left"))

    tier_dummies = pd.get_dummies(panel["category_tier"], prefix="tier")
    panel = pd.concat([panel.reset_index(drop=True), tier_dummies.reset_index(drop=True)], axis=1)
    tier_cols = sorted(c for c in panel.columns if c.startswith("tier_"))
    feature_cols = BASE_FEATURE_COLS + tier_cols

    complete = panel.dropna(subset=feature_cols)
    n_dropped = len(panel) - len(complete)
    if n_dropped:
        log.info(f"dropped {n_dropped} of {len(panel)} rows with a missing feature")

    last_quarter = quarters_ordered[-1]
    forward = complete[(complete["quarter"] == last_quarter)
                       & complete["underperform_next_quarter"].isna()]
    labeled = complete[complete["underperform_next_quarter"].notna()]
    log.info(f"unified panel: {len(labeled)} labeled rows, {len(forward)} forward rows "
             f"({last_quarter} -> next), {len(feature_cols)} features")
    return labeled, forward, feature_cols
