"""tests/test_step14_extract.py — extract builder: views exist, honest rules enforced.

All tests run against a tiny synthetic source DB built in tmp_path — never the real 5.6 GB DB.
"""
import pandas as pd
import pytest

from steps.step14_webapp.extract import (
    build_fund_views, latest_quarter, normalize_name)
from tests.step14_fixtures import make_synthetic_source


@pytest.fixture
def src(tmp_path):
    """Synthetic source DB: 3 funds (A alive+big, B alive+small-cluster, D dead), 2 quarters."""
    con = make_synthetic_source(tmp_path / "src.duckdb")
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


from steps.step14_webapp.extract import compute_health_state


def test_health_state_rule_truth_table():
    # oldest-first [(auc, persistence_auc), ...] for the last two realized quarters
    assert compute_health_state([(0.60, 0.55), (0.62, 0.55)])[0] == "healthy"
    assert compute_health_state([(0.55, 0.57), (0.56, 0.57)])[0] == "weak"     # above 0.5, below baseline
    assert compute_health_state([(0.55, 0.50), (0.42, 0.57)])[0] == "degraded" # one below coin-flip
    assert compute_health_state([(0.42, 0.57), (0.41, 0.57)])[0] == "degraded"
    state, rule_text = compute_health_state([(0.42, 0.57), (0.41, 0.57)])
    assert "0.5" in rule_text  # the rule is disclosed, not proprietary


def test_model_views_from_synthetic(model_src):
    from steps.step14_webapp.extract import build_model_views
    views = build_model_views(model_src, cfg=None)  # cfg only needed for tree intervals; None skips
    cur = views["v_model_health_current"]
    assert len(cur) == 1
    assert cur.iloc[0]["health_state"] in ("healthy", "weak", "degraded")
    pred = views["v_fund_prediction_current"].set_index("series_id")
    assert "A" in pred.index and 0.0 <= pred.loc["A", "predicted_probability"] <= 1.0
    hist = views["v_fund_prediction_history"]
    assert set(hist["split"]) <= {"test", "train"}
    calib = views["v_calibration_bins"]
    assert (calib["n"] > 0).all()
    # v_data_provenance's "extract built" stamp is the build time, not the refresh_log row
    # (that row feeds v_model_health_current.refreshed_at instead) - see step14 final review.
    prov = views["v_data_provenance"]
    built_at = prov.iloc[0]["refreshed_at"]
    assert built_at is not None
    pd.Timestamp(built_at)  # parses as ISO timestamp
    assert built_at != "2026-07-16T06:45:21+00:00"  # the synthetic refresh_log value


@pytest.fixture
def model_src(tmp_path):
    con = make_synthetic_source(tmp_path / "msrc.duckdb")
    yield con
    con.close()
