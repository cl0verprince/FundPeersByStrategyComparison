"""step8_dashboard/data.py - assemble the dashboard's single JSON payload from the step7
tables. Pure read-and-shape: every number shown in the dashboard is computed upstream and
verified there; this module only selects, joins, and orders (deterministically - every
list sorted) for display.
"""
import pandas as pd

from fundspeers.io import load_table, table_exists

DISCLAIMER = (
    "Educational and informational purposes only. Not investment advice or a "
    "recommendation. Predictions are statistical estimates that may be highly inaccurate. "
    "No liability or responsibility is assumed for decisions made based on this material. "
    "Past performance does not guarantee future results.")


def _none_if_na(x):
    return None if x is None or pd.isna(x) else float(x)


def _member_row(fund_row, metrics_row, probability=None, include_probability=True):
    m = {
        "sid": fund_row["series_id"], "name": fund_row["series_name"],
        "ticker": fund_row["ticker"], "net_assets": _none_if_na(fund_row["net_assets"]),
        "sharpe": _none_if_na(metrics_row.get("sharpe_ratio")),
        "volatility": _none_if_na(metrics_row.get("annualized_volatility")),
        "max_drawdown": _none_if_na(metrics_row.get("max_drawdown")),
        "cumulative_return": _none_if_na(metrics_row.get("cumulative_return")),
    }
    if include_probability:
        m["probability"] = probability
    return m


def _top_holdings_for(members, latest_quarter, funds, holdings, top_n=10):
    """Sleeve-renormalize each member's EC holdings, then average weights across members."""
    accs = funds[(funds["series_id"].isin(members)) & (funds["quarter"] == latest_quarter)]
    acc_to_sid = dict(zip(accs["accession_number"], accs["series_id"]))
    ec = holdings[(holdings["quarter"] == latest_quarter) & (holdings["asset_cat"] == "EC")
                  & holdings["accession_number"].isin(acc_to_sid)].copy()
    if ec.empty:
        return []
    ec["sid"] = ec["accession_number"].map(acc_to_sid)
    ec["w"] = ec.groupby("sid")["currency_value"].transform(lambda v: v / v.sum())
    avg = (ec.groupby("issuer_name")["w"].sum() / len(members)).sort_index()
    avg = avg.sort_values(ascending=False, kind="stable")
    return [{"issuer": issuer, "weight": round(float(w), 6)}
            for issuer, w in avg.head(top_n).items()]


def build_payload(cfg: dict, narratives: dict) -> dict:
    funds = load_table("funds_all", cfg)
    metrics_overall = load_table("fund_metrics_overall_all", cfg).set_index("series_id")
    clusters_tbl = load_table("fund_clusters_all", cfg)
    definitions = load_table("cluster_definitions_all", cfg)
    holdings = load_table("holdings_all", cfg)
    model_eval = load_table("unified_model_eval", cfg)
    coords = load_table("cluster_map_coords_all", cfg)
    predictions = load_table("unified_predictions", cfg)
    stability = (load_table("unified_label_stability", cfg)
                 if table_exists("unified_label_stability", cfg) else pd.DataFrame(
                     columns=["flip_rate"]))

    equity = funds[funds["is_us_equity"]]
    per_series = equity.drop_duplicates("series_id")
    quarters = sorted(equity["quarter"].unique())
    latest = quarters[-1]
    latest_funds = equity[equity["quarter"] == latest].drop_duplicates("series_id")

    forward = predictions[predictions["split"] == "forward"].set_index("series_id")

    def pooled(metric):
        row = model_eval[(model_eval["metric"] == metric) & (model_eval["quarter"] == "")]
        return float(row["value"].iloc[0]) if len(row) else None

    per_quarter = []
    q_auc = model_eval[(model_eval["metric"] == "auc_pooled") & (model_eval["quarter"] != "")]
    q_persist = model_eval[(model_eval["metric"] == "auc_persistence_baseline")
                           & (model_eval["quarter"] != "")].set_index("quarter")
    for _, row in q_auc.sort_values("quarter").iterrows():
        per_quarter.append({"quarter": row["quarter"], "auc": float(row["value"]),
                            "persistence_auc": _none_if_na(
                                q_persist["value"].get(row["quarter"]))})

    latest_clusters = clusters_tbl[clusters_tbl["quarter"] == latest]
    latest_defs = definitions[definitions["quarter"] == latest].set_index("cluster_id")

    cluster_payloads = []
    for cluster_id in sorted(latest_defs.index):
        d = latest_defs.loc[cluster_id]
        member_ids = sorted(latest_clusters.loc[
            latest_clusters["cluster_id"] == cluster_id, "series_id"])
        members = []
        present_member_ids = []
        for sid in member_ids:
            fund_row = latest_funds[latest_funds["series_id"] == sid]
            if fund_row.empty:
                continue
            metrics_row = (metrics_overall.loc[sid].to_dict()
                           if sid in metrics_overall.index else {})
            prob = (_none_if_na(forward.loc[sid, "predicted_probability"])
                    if sid in forward.index else None)
            members.append(_member_row(fund_row.iloc[0], metrics_row, probability=prob))
            present_member_ids.append(sid)
        member_metrics = pd.DataFrame(members)
        cluster_payloads.append({
            "cluster_id": int(cluster_id),
            "short_title": d["short_title"],
            "dominant_category": d["dominant_category"],
            "dominant_share": float(d["dominant_category_share"]),
            "member_count": len(members),
            "avg_sharpe": _none_if_na(member_metrics["sharpe"].mean()) if len(members) else None,
            "avg_volatility": _none_if_na(member_metrics["volatility"].mean()) if len(members) else None,
            "avg_max_drawdown": _none_if_na(member_metrics["max_drawdown"].mean()) if len(members) else None,
            "median_net_assets": _none_if_na(member_metrics["net_assets"].median()) if len(members) else None,
            "top_holdings": _top_holdings_for(set(present_member_ids), latest, equity, holdings),
            "narrative": narratives.get(int(cluster_id), ""),
            "members": members,
        })

    allocation_payloads = []
    alloc_funds = latest_funds[latest_funds["series_id"].isin(
        per_series.loc[per_series["segment"] == "allocation", "series_id"])]
    for vintage in sorted(alloc_funds["yahoo_category"].dropna().unique()):
        vintage_members = []
        for _, fund_row in alloc_funds[alloc_funds["yahoo_category"] == vintage].sort_values(
                "series_id").iterrows():
            metrics_row = (metrics_overall.loc[fund_row["series_id"]].to_dict()
                           if fund_row["series_id"] in metrics_overall.index else {})
            vintage_members.append(_member_row(fund_row, metrics_row,
                                               include_probability=False))
        allocation_payloads.append({"vintage": vintage, "members": vintage_members})

    return {
        "universe": {
            "n_funds": int(per_series.shape[0]),
            "n_strategy": int((per_series["segment"] == "strategy").sum()),
            "n_allocation": int((per_series["segment"] == "allocation").sum()),
            "quarters": quarters, "latest_quarter": latest,
        },
        "scorecard": {
            "auc": pooled("auc_pooled"),
            "auc_ci": [pooled("auc_ci_low"), pooled("auc_ci_high")],
            "persistence_auc": pooled("auc_persistence_baseline"),
            "p_edge_le_zero": pooled("p_edge_le_zero"),
            "per_quarter": per_quarter,
            "mean_flip_rate": _none_if_na(stability["flip_rate"].mean())
                              if len(stability) else None,
        },
        "coords": [{"sid": r["series_id"], "x": round(float(r["x"]), 4),
                    "y": round(float(r["y"]), 4), "cluster": int(r["cluster_id"])}
                   for _, r in coords.sort_values("series_id").iterrows()],
        "clusters": cluster_payloads,
        "allocation": allocation_payloads,
        "disclaimer": DISCLAIMER,
    }
