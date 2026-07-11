"""step7_unified_universe/label.py - the kNN-peer-relative underperformance label.

'Underperform' = the fund's next-quarter return is below the MEDIAN of its own top-N
most-similar funds' next-quarter returns (peers as-of Q, by cosine similarity on holdings
embeddings). Bespoke, constant-size peer groups by construction - immune to universe
growth, unlike the old cluster-median label (see design.md, diagnosis #1).
"""
import logging

import pandas as pd

log = logging.getLogger(__name__)

LABEL_DEFINITION = ("next-quarter return below the median of the fund's top-{top_n} "
                    "cosine-similarity peers' next-quarter returns")


def compute_peer_labels(fund_peers: pd.DataFrame, quarterly_returns: pd.DataFrame,
                        quarters_ordered: list, top_n: int, min_valid_peers: int) -> pd.DataFrame:
    """One row per (series_id, quarter) present in fund_peers. Label is pd.NA when the
    fund's own Q+1 return is missing, fewer than min_valid_peers peers have a valid Q+1
    return, or Q is the last quarter (those rows are the forward-prediction set - kept)."""
    n_dup = int(quarterly_returns.duplicated(["series_id", "quarter"]).sum())
    if n_dup:
        raise ValueError(
            f"quarterly_returns must be unique on (series_id, quarter); "
            f"found {n_dup} duplicate row(s)")
    quarter_to_next = dict(zip(quarters_ordered[:-1], quarters_ordered[1:]))
    peers = fund_peers[fund_peers["peer_rank"] <= top_n].copy()
    peers["next_quarter"] = peers["quarter"].map(quarter_to_next)

    peer_ret_q = quarterly_returns.rename(
        columns={"series_id": "peer_series_id", "quarterly_return": "peer_return_q"})
    peers = peers.merge(peer_ret_q, on=["peer_series_id", "quarter"], how="left")
    peer_ret_next = quarterly_returns.rename(
        columns={"series_id": "peer_series_id", "quarter": "next_quarter",
                 "quarterly_return": "peer_return_next"})
    peers = peers.merge(peer_ret_next, on=["peer_series_id", "next_quarter"], how="left")

    agg = peers.groupby(["series_id", "quarter"]).agg(
        peer_median_return_q=("peer_return_q", "median"),
        peer_median_return_next=("peer_return_next", "median"),
        n_valid_peers_next=("peer_return_next", "count"),
    ).reset_index()

    agg = agg.merge(quarterly_returns, on=["series_id", "quarter"], how="left")
    agg["next_quarter"] = agg["quarter"].map(quarter_to_next)
    own_next = quarterly_returns.rename(
        columns={"quarter": "next_quarter", "quarterly_return": "own_return_next"})
    agg = agg.merge(own_next, on=["series_id", "next_quarter"], how="left")

    agg["return_vs_peer_median_q"] = agg["quarterly_return"] - agg["peer_median_return_q"]

    valid = agg["own_return_next"].notna() & (agg["n_valid_peers_next"] >= min_valid_peers)
    agg["underperform_next_quarter"] = pd.Series(pd.NA, index=agg.index, dtype="Int64")
    agg.loc[valid, "underperform_next_quarter"] = (
        agg.loc[valid, "own_return_next"] < agg.loc[valid, "peer_median_return_next"]
    ).astype("Int64")

    n_dropped_label = int((~valid & agg["next_quarter"].notna()).sum())
    if n_dropped_label:
        log.info(f"{n_dropped_label} non-final fund-quarters have no label "
                 f"(own Q+1 return missing or <{min_valid_peers} valid peers)")
    return agg[["series_id", "quarter", "peer_median_return_q", "return_vs_peer_median_q",
                "peer_median_return_next", "n_valid_peers_next", "underperform_next_quarter"]]
