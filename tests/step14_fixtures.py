"""tests/step14_fixtures.py — shared synthetic source DB + extract builder for step14 tests.

`make_synthetic_source(path)` is the union of the two synthetic-source fixtures that used
to live inline in tests/test_step14_extract.py (the fund-view `src` fixture and the
model-view `model_src` fixture), plus the extra tables `build_cluster_views` needs
(`cluster_map_coords_full`, `dashboard_narratives`) and an empty `full_panel` (only read
when a real model bundle/cfg is supplied — never in tests).

Tickers: AAAAX (alive, big), BBBBX (alive, small cluster -> percentiles suppressed),
DDDDX (dead, last_quarter 2026q1). AAAAX and BBBBX both have a forward prediction
(0.41 and 0.55); DDDDX has none — it dropped out of the universe before 2026q2.

`build_synthetic_extract(out_path)` runs the same three view-builders `extract.run` runs,
against `make_synthetic_source`, and writes them with `extract.write_extract` — i.e. it
produces exactly the kind of extract.duckdb the real pipeline would produce, just tiny.
"""
from pathlib import Path

import duckdb
import pandas as pd


def make_synthetic_source(path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(path))

    con.register("funds", pd.DataFrame([
        # series_id, series_name, quarter, net_assets, ticker, yahoo_category, segment, accession_number
        ("A", "Alpha Large Blend Fund", "2026q1", 5e9, "AAAAX", "Large Blend", "strategy", "acc-A1"),
        ("A", "Alpha Large Blend Fund", "2026q2", 6e9, "AAAAX", "Large Blend", "strategy", "acc-A2"),
        ("B", "Beta Small Value Fund", "2026q1", 1e8, "BBBBX", "Small Value", "strategy", "acc-B1"),
        ("B", "Beta Small Value Fund", "2026q2", 1e8, "BBBBX", "Small Value", "strategy", "acc-B2"),
        ("D", "Delta Dead Fund", "2026q1", 2e8, "DDDDX", "Large Blend", "strategy", "acc-D1"),
    ], columns=["series_id", "series_name", "quarter", "net_assets", "ticker",
                "yahoo_category", "segment", "accession_number"]))
    con.execute("CREATE TABLE funds_full AS SELECT * FROM funds")

    con.register("clusters", pd.DataFrame([
        ("A", "2026q1", 1), ("A", "2026q2", 1), ("B", "2026q1", 2), ("B", "2026q2", 2),
        ("D", "2026q1", 1),
    ], columns=["series_id", "quarter", "cluster_id"]))
    con.execute("CREATE TABLE fund_clusters_full AS SELECT * FROM clusters")

    con.register("cdefs", pd.DataFrame([
        ("2026q2", 1, 20, "Large Blend", 0.8, 0.15, 0.5, "Leaning Large Blend"),
        ("2026q2", 2, 5, "Small Value", 0.6, 0.20, 0.3, "Tiny Small Value"),
        ("2026q1", 1, 21, "Large Blend", 0.8, 0.15, 0.5, "Leaning Large Blend"),
        ("2026q1", 2, 5, "Small Value", 0.6, 0.20, 0.3, "Tiny Small Value"),
    ], columns=["quarter", "cluster_id", "member_count", "dominant_category",
                "dominant_category_share", "avg_volatility", "avg_sharpe", "short_title"]))
    con.execute("CREATE TABLE cluster_definitions_full AS SELECT * FROM cdefs")

    con.register("mq", pd.DataFrame([
        ("A", "2026q1", 0.04, 1, 0.03, 0.01), ("A", "2026q2", 0.02, 1, 0.03, -0.01),
        ("B", "2026q1", 0.05, 2, 0.05, 0.00), ("B", "2026q2", 0.01, 2, 0.02, -0.01),
        ("D", "2026q1", -0.02, 1, 0.03, -0.05),
    ], columns=["series_id", "quarter", "quarterly_return", "cluster_id",
                "cluster_median_return", "return_vs_cluster_median"]))
    con.execute("CREATE TABLE fund_metrics_quarterly_full AS SELECT * FROM mq")

    con.register("mo", pd.DataFrame([
        ("A", 0.30, 0.14, 0.52, -0.08), ("B", 0.10, 0.22, 0.20, -0.30),
        ("D", -0.05, 0.18, -0.10, -0.25),
    ], columns=["series_id", "cumulative_return", "annualized_volatility",
                "sharpe_ratio", "max_drawdown"]))
    con.execute("CREATE TABLE fund_metrics_overall_full AS SELECT * FROM mo")

    con.register("peers", pd.DataFrame([
        ("A", "2026q2", 1, "B", 0.91), ("A", "2026q2", 2, "D", 0.80),
        ("B", "2026q2", 1, "A", 0.91),
    ], columns=["series_id", "quarter", "peer_rank", "peer_series_id", "cosine_similarity"]))
    con.execute("CREATE TABLE fund_peers_full AS SELECT * FROM peers")

    con.register("fees", pd.DataFrame([
        ("A", "2026q2", 0.0004, 0.0005, 0.05), ("B", "2026q2", 0.0110, 0.0120, 1.20),
    ], columns=["series_id", "quarter", "expense_ratio_net", "expense_ratio_gross",
                "portfolio_turnover"]))
    con.execute("CREATE TABLE rr_fees AS SELECT * FROM fees")

    con.register("hold", pd.DataFrame([
        ("acc-A2", "MICROSOFT CORP", 6.0, "2026q2"), ("acc-A2", "APPLE INC", 5.0, "2026q2"),
        ("acc-B2", "SOME SMALL CO", 3.0, "2026q2"),
    ], columns=["accession_number", "issuer_name", "percentage", "quarter"]))
    con.execute("CREATE TABLE holdings_full AS SELECT * FROM hold")

    con.register("preds", pd.DataFrame([
        ("A", "2026q2", 0.41, None, "forward"),
        ("A", "2026q1", 0.70, 1.0, "test"), ("A", "2025q4", 0.30, 0.0, "test"),
        ("B", "2026q2", 0.55, None, "forward"),
        ("B", "2026q1", 0.20, 1.0, "test"),
    ], columns=["series_id", "quarter", "predicted_probability", "actual_label", "split"]))
    con.execute("CREATE TABLE full_predictions AS SELECT * FROM preds")

    con.register("ev", pd.DataFrame([
        ("auc_pooled", "", 0.578), ("auc_pooled", "2025q4", 0.457), ("auc_pooled", "2026q1", 0.427),
        ("auc_persistence_baseline", "2025q4", 0.552), ("auc_persistence_baseline", "2026q1", 0.569),
    ], columns=["metric", "quarter", "value"]))
    con.execute("CREATE TABLE full_model_eval AS SELECT * FROM ev")

    con.register("oot", pd.DataFrame([
        ("auc", "", 0.574, "published_forward"), ("base_rate", "", 0.532, "published_forward"),
        ("n_scored", "", 2057.0, "published_forward"),
        ("auc", "2026q1", 0.418, "frozen_rolled_forward"),
    ], columns=["metric", "quarter", "value", "source"]))
    con.execute("CREATE TABLE oot_validation AS SELECT * FROM oot")

    con.register("stab", pd.DataFrame([
        ("A", "2026q1", 0.05), ("B", "2026q1", 0.30),
    ], columns=["series_id", "quarter", "flip_rate"]))
    con.execute("CREATE TABLE full_label_stability AS SELECT * FROM stab")

    con.register("rlog", pd.DataFrame([("2026q2", "2026-07-16T06:45:21+00:00")],
                                      columns=["quarter", "refreshed_at"]))
    con.execute("CREATE TABLE refresh_log AS SELECT * FROM rlog")

    # v_cluster_map (build_cluster_views) needs series coordinates for the current snapshot.
    con.register("coords", pd.DataFrame([
        ("A", 0.1, 0.2, 1), ("B", -0.3, 0.5, 2), ("D", 0.4, -0.1, 1),
    ], columns=["series_id", "x", "y", "cluster_id"]))
    con.execute("CREATE TABLE cluster_map_coords_full AS SELECT * FROM coords")

    # v_cluster_summary LEFT JOINs narratives; empty is a legal (undescribed-yet) state.
    con.execute(
        "CREATE TABLE dashboard_narratives ("
        "cluster_id INTEGER, quarter VARCHAR, narrative VARCHAR)")

    # Only read by extract._prediction_intervals when a real cfg/model bundle is supplied;
    # cfg=None (every test) skips it entirely, so an empty table is enough to satisfy schema.
    con.execute("CREATE TABLE full_panel (series_id VARCHAR, quarter VARCHAR)")

    return con


def build_synthetic_extract(out_path, retired=False) -> Path:
    """Build fund/model/cluster views from a fresh synthetic source and write them to
    out_path exactly as steps.step14_webapp.extract.run does (cfg=None -> NULL intervals).

    retired=True passes a minimal retired cfg into build_model_views (interval lookup
    fails harmlessly -> NULL intervals, existing behavior)."""
    from steps.step14_webapp.extract import (
        build_cluster_views, build_fund_views, build_model_views, write_extract)

    out_path = Path(out_path)
    src_path = out_path.with_name(f"_src_{out_path.stem}.duckdb")
    con = make_synthetic_source(src_path)
    cfg = {"model": {"retirement": {
        "as_of": "2026q1", "statement": "Retired for the synthetic record."}}} if retired else None
    try:
        views = {}
        views.update(build_fund_views(con))
        views.update(build_model_views(con, cfg=cfg))
        views.update(build_cluster_views(con))
    finally:
        con.close()

    write_extract(views, out_path)
    return out_path
