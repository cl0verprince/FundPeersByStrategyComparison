"""step7_unified_universe/stability.py - Monte Carlo label-stability study.

The label depends on WHICH top-10 peers the similarity step surfaced. Perturb each fund's
peer set (draw_size of the top-pool_size peers, many seeded draws), recompute the label,
and measure the flip rate - an estimate of the label's intrinsic noise floor. High flip
rates would mean AUC gains are being chased into benchmark noise (design.md section 6b).
"""
import logging

import numpy as np
import pandas as pd

from fundspeers.io import load_table, save_table
from steps.step7_unified_universe.panel import _quarterly_returns_from_monthly

log = logging.getLogger(__name__)


def compute_label_flip_rates(fund_peers: pd.DataFrame, quarterly_returns: pd.DataFrame,
                             quarters_ordered: list, top_n: int, pool_size: int,
                             draw_size: int, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quarter_to_next = dict(zip(quarters_ordered[:-1], quarters_ordered[1:]))

    pool = fund_peers[fund_peers["peer_rank"] <= pool_size].copy()
    pool["next_quarter"] = pool["quarter"].map(quarter_to_next)
    ret_next = quarterly_returns.rename(
        columns={"series_id": "peer_series_id", "quarter": "next_quarter",
                 "quarterly_return": "peer_return_next"})
    pool = pool.merge(ret_next, on=["peer_series_id", "next_quarter"], how="left")
    own_next = quarterly_returns.rename(
        columns={"quarter": "next_quarter", "quarterly_return": "own_return_next"})

    rows = []
    for (series_id, quarter), g in pool.groupby(["series_id", "quarter"]):
        g = g.sort_values("peer_rank")
        valid = g["peer_return_next"].dropna().values
        nq = quarter_to_next.get(quarter)
        if nq is None or len(valid) < pool_size:
            continue
        own = quarterly_returns[(quarterly_returns["series_id"] == series_id)
                                & (quarterly_returns["quarter"] == nq)]["quarterly_return"]
        if own.empty or pd.isna(own.iloc[0]):
            continue
        own_r = own.iloc[0]
        base_pool = g["peer_return_next"].values[:top_n]
        base_label = own_r < np.nanmedian(base_pool)
        # all draws at once: (draws, draw_size) index matrix into the top-pool_size returns
        idx = np.array([rng.choice(pool_size, size=draw_size, replace=False)
                        for _ in range(draws)])
        medians = np.median(valid[:pool_size][idx], axis=1)
        flip_rate = float(((own_r < medians) != base_label).mean())
        rows.append({"series_id": series_id, "quarter": quarter, "flip_rate": flip_rate})
    return pd.DataFrame(rows, columns=["series_id", "quarter", "flip_rate"])


def run_stability(cfg: dict) -> dict:
    funds = load_table("funds_all", cfg)
    strategy_series = set(funds.loc[funds["is_us_equity"]
                                    & (funds["segment"] == "strategy"), "series_id"])
    monthly = load_table("monthly_returns_all", cfg)
    quarterly_returns = _quarterly_returns_from_monthly(
        monthly[monthly["series_id"].isin(strategy_series)])
    fund_peers = load_table("fund_peers_all", cfg)
    quarters_ordered = sorted(funds.loc[funds["series_id"].isin(strategy_series), "quarter"].unique())

    flips = compute_label_flip_rates(
        fund_peers, quarterly_returns, quarters_ordered,
        top_n=cfg["unified"]["peer_label_top_n"], pool_size=12, draw_size=8,
        draws=cfg["unified"]["label_stability_draws"], seed=cfg["seed"])
    save_table(flips, "unified_label_stability", cfg)
    summary = {
        "mean_flip_rate": float(flips["flip_rate"].mean()),
        "share_flip_gt_10pct": float((flips["flip_rate"] > 0.10).mean()),
        "n_evaluated": int(len(flips)),
    }
    log.info(f"label stability: mean flip rate {summary['mean_flip_rate']:.3f}, "
             f"{summary['share_flip_gt_10pct']:.1%} of fund-quarters flip >10% of draws "
             f"({summary['n_evaluated']} evaluated)")
    return summary
