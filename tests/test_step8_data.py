"""build_payload: schema completeness, deterministic ordering, allocation separation."""
import pandas as pd
import pytest

from fundspeers.io import save_table
from steps.step8_dashboard.data import DISCLAIMER, build_payload


@pytest.fixture
def cfg(tmp_path):
    return {"paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "processed"),
                      "reports": str(tmp_path / "reports"), "models": str(tmp_path / "models")}}


@pytest.fixture
def tables(cfg):
    q = "2024q4"
    save_table(pd.DataFrame([
        {"series_id": "F1", "quarter": q, "accession_number": "a1", "series_name": "Fund One",
         "ticker": "ONE", "yahoo_category": "Large Blend", "is_us_equity": True,
         "segment": "strategy", "net_assets": 100.0},
        {"series_id": "F2", "quarter": q, "accession_number": "a2", "series_name": "Fund Two",
         "ticker": "TWO", "yahoo_category": "Large Blend", "is_us_equity": True,
         "segment": "strategy", "net_assets": 200.0},
        {"series_id": "T1", "quarter": q, "accession_number": "a3", "series_name": "Target Fund",
         "ticker": "TGT", "yahoo_category": "Target-Date 2050", "is_us_equity": True,
         "segment": "allocation", "net_assets": 50.0}]), "funds_all", cfg)
    save_table(pd.DataFrame([{
        "quarter": q, "cluster_id": 0, "member_count": 2, "dominant_category": "Large Blend",
        "dominant_category_share": 1.0, "dominant_tier": "Large", "avg_volatility": 0.15,
        "avg_sharpe": 0.5, "title": "t", "short_title": "Leaning Large Blend"}]),
        "cluster_definitions_all", cfg)
    save_table(pd.DataFrame([
        {"series_id": s, "cumulative_return": 0.2, "annualized_volatility": 0.15,
         "sharpe_ratio": 0.5, "max_drawdown": -0.1} for s in ["F1", "F2", "T1"]]),
        "fund_metrics_overall_all", cfg)
    save_table(pd.DataFrame([
        {"series_id": s, "quarter": q, "cluster_id": 0} for s in ["F1", "F2"]]),
        "fund_clusters_all", cfg)
    save_table(pd.DataFrame([
        {"series_id": "F1", "quarter": q, "predicted_probability": 0.7,
         "actual_label": None, "split": "forward"}]), "unified_predictions", cfg)
    save_table(pd.DataFrame([
        {"metric": "auc_pooled", "quarter": "", "value": 0.7},
        {"metric": "auc_persistence_baseline", "quarter": "", "value": 0.6},
        {"metric": "auc_pooled", "quarter": "2024q1", "value": 0.72},
        {"metric": "auc_persistence_baseline", "quarter": "2024q1", "value": 0.61},
        {"metric": "auc_ci_low", "quarter": "", "value": 0.65},
        {"metric": "auc_ci_high", "quarter": "", "value": 0.75},
        {"metric": "p_edge_le_zero", "quarter": "", "value": 0.02}]),
        "unified_model_eval", cfg)
    save_table(pd.DataFrame([{"series_id": "F1", "quarter": q, "flip_rate": 0.05}]),
               "unified_label_stability", cfg)
    save_table(pd.DataFrame([
        {"series_id": s, "x": 0.1, "y": 0.2, "cluster_id": 0} for s in ["F1", "F2"]]),
        "cluster_map_coords_all", cfg)
    save_table(pd.DataFrame([
        {"accession_number": a, "quarter": q, "asset_cat": "EC", "issuer_name": "ACME CORP",
         "currency_value": 100.0} for a in ["a1", "a2", "a3"]]), "holdings_all", cfg)


def test_payload_schema_and_separation(cfg, tables):
    payload = build_payload(cfg, narratives={0: "A calm cluster."})
    assert payload["universe"]["n_funds"] == 3
    assert payload["universe"]["latest_quarter"] == "2024q4"
    assert len(payload["clusters"]) == 1
    cluster = payload["clusters"][0]
    assert cluster["short_title"] == "Leaning Large Blend"
    assert cluster["narrative"] == "A calm cluster."
    members = {m["sid"]: m for m in cluster["members"]}
    assert members["F1"]["probability"] == 0.7
    assert members["F2"]["probability"] is None
    assert payload["allocation"][0]["vintage"] == "Target-Date 2050"
    assert "probability" not in payload["allocation"][0]["members"][0]
    assert payload["scorecard"]["auc"] == 0.7
    assert payload["scorecard"]["auc_ci"] == [0.65, 0.75]
    assert payload["disclaimer"] == DISCLAIMER


def test_payload_is_deterministic(cfg, tables):
    import json
    a = json.dumps(build_payload(cfg, narratives={}), sort_keys=True)
    b = json.dumps(build_payload(cfg, narratives={}), sort_keys=True)
    assert a == b
