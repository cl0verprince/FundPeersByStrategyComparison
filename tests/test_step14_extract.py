"""tests/test_step14_extract.py — extract builder: views exist, honest rules enforced.

All tests run against a tiny synthetic source DB built in tmp_path — never the real 5.6 GB DB.
"""
import duckdb
import pandas as pd
import pytest

from steps.step14_webapp.extract import (
    build_fund_views, latest_quarter, normalize_name)


@pytest.fixture
def src(tmp_path):
    """Synthetic source DB: 3 funds (A alive+big, B alive+small-cluster, D dead), 2 quarters."""
    con = duckdb.connect(str(tmp_path / "src.duckdb"))
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
    yield con
    con.close()


def test_latest_quarter(src):
    assert latest_quarter(src) == "2026q2"


def test_normalize_name():
    assert normalize_name("Vanguard 500 Index — Admiral™!") == "vanguard 500 index admiral"


def test_fund_views_exist_and_nonempty(src):
    views = build_fund_views(src)
    for name in ["v_fund_header", "v_fund_search", "v_fund_peer_relative_ts",
                 "v_fund_cluster_percentiles", "v_peer_display", "v_top_holdings"]:
        assert name in views and len(views[name]) > 0, name


def test_fund_header_activity_flag(src):
    header = build_fund_views(src)["v_fund_header"].set_index("series_id")
    assert bool(header.loc["A", "is_active"]) is True
    assert bool(header.loc["D", "is_active"]) is False
    assert header.loc["D", "last_quarter"] == "2026q1"
    assert header.loc["A", "cluster_name"] == "Leaning Large Blend"


def test_percentiles_suppressed_for_small_clusters(src):
    pct = build_fund_views(src)["v_fund_cluster_percentiles"].set_index("series_id")
    # cluster 1 has member_count 20 -> percentiles present; cluster 2 has 5 -> suppressed
    assert pd.notna(pct.loc["A", "pctile_volatility"])
    assert pd.isna(pct.loc["B", "pctile_volatility"])


def test_peer_display_joins_names_and_fees(src):
    peers = build_fund_views(src)["v_peer_display"]
    row = peers[(peers.series_id == "A") & (peers.peer_rank == 1)].iloc[0]
    assert row["peer_ticker"] == "BBBBX"
    assert row["peer_name"] == "Beta Small Value Fund"
    assert row["peer_expense_net"] == pytest.approx(0.0110)
