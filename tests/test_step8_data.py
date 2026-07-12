"""build_payload: schema completeness, deterministic ordering, allocation separation."""
import pandas as pd
import pytest

from fundspeers.io import save_table
from steps.step8_dashboard.data import DISCLAIMER, build_payload


@pytest.fixture
def cfg(tmp_path):
    return {"paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "processed"),
                      "reports": str(tmp_path / "reports"), "models": str(tmp_path / "models")}}


def _write_universe_tables(cfg, suffix="_all", predictions_table="unified_predictions",
                           eval_table="unified_model_eval",
                           stability_table="unified_label_stability"):
    """Write a minimal 3-fund universe under `suffix` (funds{suffix}, ...) plus the three
    non-suffixed unified_* tables the payload reads by name. Shared by the default-path,
    suffix-path, and oot tests."""
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
         "segment": "allocation", "net_assets": 50.0}]), f"funds{suffix}", cfg)
    save_table(pd.DataFrame([{
        "quarter": q, "cluster_id": 0, "member_count": 2, "dominant_category": "Large Blend",
        "dominant_category_share": 1.0, "dominant_tier": "Large", "avg_volatility": 0.15,
        "avg_sharpe": 0.5, "title": "t", "short_title": "Leaning Large Blend"}]),
        f"cluster_definitions{suffix}", cfg)
    save_table(pd.DataFrame([
        {"series_id": s, "cumulative_return": 0.2, "annualized_volatility": 0.15,
         "sharpe_ratio": 0.5, "max_drawdown": -0.1} for s in ["F1", "F2", "T1"]]),
        f"fund_metrics_overall{suffix}", cfg)
    save_table(pd.DataFrame([
        {"series_id": s, "quarter": q, "cluster_id": 0} for s in ["F1", "F2", "F3"]]),
        f"fund_clusters{suffix}", cfg)
    save_table(pd.DataFrame([
        {"series_id": "F1", "quarter": q, "predicted_probability": 0.7,
         "actual_label": None, "split": "forward"}]), predictions_table, cfg)
    save_table(pd.DataFrame([
        {"metric": "auc_pooled", "quarter": "", "value": 0.7},
        {"metric": "auc_persistence_baseline", "quarter": "", "value": 0.6},
        {"metric": "auc_pooled", "quarter": "2024q1", "value": 0.72},
        {"metric": "auc_persistence_baseline", "quarter": "2024q1", "value": 0.61},
        {"metric": "auc_ci_low", "quarter": "", "value": 0.65},
        {"metric": "auc_ci_high", "quarter": "", "value": 0.75},
        {"metric": "p_edge_le_zero", "quarter": "", "value": 0.02}]),
        eval_table, cfg)
    save_table(pd.DataFrame([{"series_id": "F1", "quarter": q, "flip_rate": 0.05}]),
               stability_table, cfg)
    save_table(pd.DataFrame([
        {"series_id": s, "x": 0.1, "y": 0.2, "cluster_id": 0} for s in ["F1", "F2"]]),
        f"cluster_map_coords{suffix}", cfg)
    save_table(pd.DataFrame([
        {"accession_number": a, "quarter": q, "asset_cat": "EC", "issuer_name": "ACME CORP",
         "currency_value": 100.0} for a in ["a1", "a2", "a3"]]), f"holdings{suffix}", cfg)


@pytest.fixture
def tables(cfg):
    _write_universe_tables(cfg, suffix="_all")


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


def test_top_holdings_averaged_over_filtered_members_only(cfg, tables):
    # F3 is a cluster member (fund_clusters_all) with no funds_all row, so it's absent from
    # latest_funds and gets skipped via `continue` when building `members`. top_holdings must
    # average over the same 2 filtered members the payload reports, not the unfiltered 3.
    payload = build_payload(cfg, narratives={})
    cluster = payload["clusters"][0]
    assert cluster["member_count"] == 2
    holdings = {h["issuer"]: h["weight"] for h in cluster["top_holdings"]}
    # Both real members hold 100% ACME CORP after sleeve-renormalization, so the average
    # weight over the correct (filtered) denominator of 2 is 1.0 - not 2/3 (~0.667), which
    # is what the old buggy denominator (unfiltered count of 3) would have produced.
    assert holdings["ACME CORP"] == 1.0


def test_oot_fields_absent_are_present_and_none(cfg, tables):
    # No oot_validation table written -> every oot key present with None, so the payload
    # schema is identical whether or not step10 has run (renderer gates on the value).
    sc = build_payload(cfg, narratives={})["scorecard"]
    for k in ["oot_published_auc", "oot_published_n_scored", "oot_published_base_rate",
              "oot_frozen_pooled_auc", "oot_frozen_per_quarter"]:
        assert k in sc
        assert sc[k] is None


def test_oot_fields_read_from_oot_validation(cfg, tables):
    # Mirror steps/step10_full_universe/build.py::_write_oot_validation exactly, including
    # the metric="auc" collision across the two sources that the reader must disambiguate.
    save_table(pd.DataFrame([
        {"metric": "auc", "quarter": "", "value": 0.54, "source": "published_forward"},
        {"metric": "n_forward_total", "quarter": "", "value": 500.0, "source": "published_forward"},
        {"metric": "n_scored", "quarter": "", "value": 421.0, "source": "published_forward"},
        {"metric": "n_missing_own_return", "quarter": "", "value": 40.0, "source": "published_forward"},
        {"metric": "n_insufficient_peers", "quarter": "", "value": 39.0, "source": "published_forward"},
        {"metric": "base_rate", "quarter": "", "value": 0.48, "source": "published_forward"},
        {"metric": "auc_pooled", "quarter": "", "value": 0.52, "source": "frozen_rolled_forward"},
        {"metric": "n_rows", "quarter": "", "value": 1500.0, "source": "frozen_rolled_forward"},
        {"metric": "auc", "quarter": "2025q1", "value": 0.61, "source": "frozen_rolled_forward"},
        {"metric": "auc", "quarter": "2024q4", "value": 0.47, "source": "frozen_rolled_forward"},
    ]), "oot_validation", cfg)
    sc = build_payload(cfg, narratives={})["scorecard"]
    assert sc["oot_published_auc"] == 0.54          # published_forward auc, NOT the frozen auc rows
    assert sc["oot_published_n_scored"] == 421       # coerced to int
    assert sc["oot_published_base_rate"] == 0.48
    assert sc["oot_frozen_pooled_auc"] == 0.52       # distinct metric name auc_pooled
    # per-quarter frozen aucs, sorted by quarter ascending, the below-0.5 quarter preserved
    assert sc["oot_frozen_per_quarter"] == [
        {"quarter": "2024q4", "auc": 0.47}, {"quarter": "2025q1", "auc": 0.61}]


def test_suffix_path_reads_full_universe_tables(cfg):
    # The _full generalization: same builder, tables under a different suffix, unified_*
    # tables passed by name. Uses distinct sentinel table names to prove nothing falls back
    # to the _all defaults.
    _write_universe_tables(cfg, suffix="_full", predictions_table="full_predictions",
                           eval_table="full_model_eval", stability_table="full_label_stability")
    payload = build_payload(cfg, narratives={0: "full-universe cluster"},
                            table_suffix="_full", predictions_table="full_predictions",
                            eval_table="full_model_eval", stability_table="full_label_stability")
    assert payload["universe"]["n_funds"] == 3
    assert payload["clusters"][0]["short_title"] == "Leaning Large Blend"
    assert payload["clusters"][0]["narrative"] == "full-universe cluster"
    assert payload["scorecard"]["auc"] == 0.7
    # oot table still absent on this path -> keys present with None
    assert payload["scorecard"]["oot_published_auc"] is None
