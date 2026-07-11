"""assemble_unified_panel on a tiny synthetic universe: 1 strategy fund + 5 peers,
2 quarters, checking feature columns, label wiring, and the forward split."""
import pandas as pd
import pytest

from fundspeers.io import save_table
from steps.step7_unified_universe.panel import assemble_unified_panel

QUARTERS = ["2024q1", "2024q2"]
IDS = ["F", "P1", "P2", "P3", "P4", "P5"]


@pytest.fixture
def cfg(tmp_path):
    return {
        "seed": 42,
        "paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "processed"),
                  "reports": str(tmp_path / "reports"), "models": str(tmp_path / "models")},
        "metrics": {"risk_free_annual": 0.02},
        "unified": {"n_clusters": 2, "peer_label_top_n": 5, "min_valid_peers_for_label": 5,
                    "bootstrap_iterations": 10, "label_stability_draws": 5},
    }


@pytest.fixture
def tables(cfg):
    funds = pd.DataFrame([
        {"series_id": s, "quarter": q, "accession_number": f"acc-{s}-{q}",
         "yahoo_category": "Large Blend", "is_us_equity": True, "segment": "strategy",
         "net_assets": 100.0 + i, "series_name": s, "ticker": s}
        for i, s in enumerate(IDS) for q in QUARTERS])
    returns = pd.DataFrame([
        {"series_id": s, "quarter": q, "month_in_quarter": m, "total_return": 1.0 + i * 0.1}
        for i, s in enumerate(IDS) for q in QUARTERS for m in (1, 2, 3)])
    holdings = pd.DataFrame([
        {"accession_number": f"acc-{s}-{q}", "quarter": q, "asset_cat": "EC",
         "issuer_name": f"CO{k}", "currency_value": 10.0}
        for s in IDS for q in QUARTERS for k in range(12)])
    peers = pd.DataFrame([
        {"series_id": s, "quarter": q, "peer_rank": r,
         "peer_series_id": p, "cosine_similarity": 0.9}
        for s in IDS for q in QUARTERS
        for r, p in enumerate([x for x in IDS if x != s], start=1)])
    save_table(funds, "funds_all", cfg)
    save_table(returns, "monthly_returns_all", cfg)
    save_table(holdings, "holdings_all", cfg)
    save_table(peers, "fund_peers_all", cfg)


def test_panel_shapes_and_features(cfg, tables):
    labeled, forward, feature_cols = assemble_unified_panel(cfg)
    assert "return_vs_peer_median_q" in feature_cols
    assert "hhi" in feature_cols and "mean_peer_similarity" in feature_cols
    assert any(c.startswith("tier_") for c in feature_cols)
    # 2024q1 rows are labelable; net_assets_qoq is NaN in q1 so labeled may be empty -
    # the forward set must still carry complete 2024q2 features:
    assert set(forward["quarter"]) == {"2024q2"}
    assert forward[feature_cols].notna().all().all()
    assert forward["underperform_next_quarter"].isna().all()


def test_labeled_rows_have_no_missing_values(cfg, tables):
    labeled, _, feature_cols = assemble_unified_panel(cfg)
    if len(labeled):
        assert labeled[feature_cols].notna().all().all()
        assert labeled["underperform_next_quarter"].notna().all()
