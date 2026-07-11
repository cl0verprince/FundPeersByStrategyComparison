"""step7_unified_universe/features.py - holdings-derived model features.

The pipeline's thesis is holdings-based similarity, yet the step4 model's features were
almost entirely trailing-return stats (see design.md, diagnosis #2). These features put
the holdings data into the model: concentration, breadth, peer-typicality, and asset flow.
"""
import pandas as pd


def compute_holdings_features(equity_holdings: pd.DataFrame) -> pd.DataFrame:
    """Per (series_id, quarter), from the fund's EC sleeve: HHI and top-10 weight share of
    sleeve-renormalized weights (consistent with step2's build_holdings_description
    renormalization), plus the raw count of EC holdings."""
    rows = []
    for (series_id, quarter), g in equity_holdings.groupby(["series_id", "quarter"]):
        total = g["currency_value"].sum()
        if total <= 0:
            continue
        w = (g["currency_value"] / total).sort_values(ascending=False)
        rows.append({
            "series_id": series_id, "quarter": quarter,
            "hhi": float((w ** 2).sum()),
            "top10_weight": float(w.head(10).sum()),
            "n_holdings": int(len(w)),
        })
    return pd.DataFrame(rows, columns=["series_id", "quarter", "hhi", "top10_weight", "n_holdings"])


def compute_peer_similarity_feature(fund_peers: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Mean cosine similarity to the fund's top-N peers - how 'typical' the fund is of its
    own niche. A low value = the fund has no close peers; its peer benchmark is weaker."""
    top = fund_peers[fund_peers["peer_rank"] <= top_n]
    return (top.groupby(["series_id", "quarter"])["cosine_similarity"].mean()
            .rename("mean_peer_similarity").reset_index())


def compute_net_asset_momentum(funds: pd.DataFrame) -> pd.DataFrame:
    """Quarter-over-quarter change in net assets (a flow/size-trend proxy). NaN for each
    fund's first observed quarter - those rows keep the panel's dropped-if-missing handling."""
    df = (funds[["series_id", "quarter", "net_assets"]]
          .drop_duplicates(["series_id", "quarter"])
          .sort_values(["series_id", "quarter"]).copy())
    df["net_assets_qoq"] = df.groupby("series_id")["net_assets"].pct_change()
    return df[["series_id", "quarter", "net_assets_qoq"]]
