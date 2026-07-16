"""Narrative caching: cached mode never calls the LLM for cached clusters; the prompt is
grounded (contains only the cluster's own numbers); skip mode returns empty."""
import pandas as pd
import pytest

from fundspeers.io import load_table, save_table, table_exists
from steps.step8_dashboard.narratives import (
    PLACEHOLDER_NARRATIVE, build_cluster_prompt, get_narratives)

CLUSTER = {
    "cluster_id": 3, "short_title": "Leaning Large Blend", "dominant_category": "Large Blend",
    "dominant_share": 0.62, "member_count": 40, "avg_sharpe": 0.51, "avg_volatility": 0.16,
    "avg_max_drawdown": -0.22,
    "top_holdings": [{"issuer": "APPLE INC", "weight": 0.06}],
}


@pytest.fixture
def cfg(tmp_path):
    return {"paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "processed"),
                      "reports": str(tmp_path / "reports"), "models": str(tmp_path / "models")},
            "llm": {"model_name": "phi-4", "temperature": 0, "top_peers_to_narrate": 5}}


def test_prompt_contains_only_cluster_facts():
    prompt = build_cluster_prompt(CLUSTER)
    assert "Leaning Large Blend" in prompt
    assert "APPLE INC" in prompt
    assert "62%" in prompt or "0.62" in prompt


def test_skip_mode_returns_empty(cfg):
    assert get_narratives(cfg, [CLUSTER], mode="skip") == {}


def test_cached_mode_uses_cache_without_llm(cfg):
    save_table(pd.DataFrame([{"cluster_id": 3, "quarter": "2024q4",
                              "narrative": "From cache."}]), "dashboard_narratives", cfg)
    # No LM Studio running in tests: if this tried the LLM it would raise.
    out = get_narratives(cfg, [CLUSTER], mode="cached")
    assert out == {3: "From cache."}


def test_broken_llm_yields_placeholder_not_failure(cfg, monkeypatch):
    # step13 contract: an unreachable/broken LM Studio must never fail a dashboard build -
    # the uncached cluster gets a placeholder, and the placeholder is NOT persisted, so a
    # later build with LM Studio healthy regenerates it. (Replaces the pre-step13 fail-loud
    # contract this test used to pin.)
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1/v1")
    save_table(pd.DataFrame([{"cluster_id": 99, "quarter": "2024q4",
                              "narrative": "other"}]), "dashboard_narratives", cfg)
    out = get_narratives(cfg, [CLUSTER], mode="cached")
    assert out[3] == PLACEHOLDER_NARRATIVE

    tbl = load_table("dashboard_narratives", cfg)
    assert 3 not in tbl["cluster_id"].tolist()  # placeholder never cached


def test_cache_save_preserves_other_quarters(cfg, monkeypatch):
    # Seed a prior quarter's row plus a full cache hit for the current quarter/cluster so no
    # LLM call is needed for the "cached" mode call below.
    save_table(pd.DataFrame([
        {"cluster_id": 7, "quarter": "2024q3", "narrative": "Q3 narrative."},
        {"cluster_id": 3, "quarter": "2024q4", "narrative": "From cache."},
    ]), "dashboard_narratives", cfg)

    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1/v1")
    out = get_narratives(cfg, [CLUSTER], mode="cached", quarter="2024q4")
    assert out == {3: "From cache."}

    tbl = load_table("dashboard_narratives", cfg)
    q3_rows = tbl[tbl["quarter"] == "2024q3"]
    assert len(q3_rows) == 1
    assert q3_rows.iloc[0]["cluster_id"] == 7
    assert q3_rows.iloc[0]["narrative"] == "Q3 narrative."

    # Now exercise the actual save path (a cache miss triggers generation + save) without
    # touching the LLM, and confirm the 2024q3 row still survives.
    monkeypatch.setattr("steps.step8_dashboard.narratives.generate_one",
                        lambda client, cfg, cluster: "Regenerated.")
    out = get_narratives(cfg, [CLUSTER], mode="regenerate", quarter="2024q4")
    assert out == {3: "Regenerated."}

    tbl = load_table("dashboard_narratives", cfg)
    q3_rows = tbl[tbl["quarter"] == "2024q3"]
    assert len(q3_rows) == 1
    assert q3_rows.iloc[0]["narrative"] == "Q3 narrative."
    q4_rows = tbl[tbl["quarter"] == "2024q4"]
    assert len(q4_rows) == 1
    assert q4_rows.iloc[0]["narrative"] == "Regenerated."


def test_regenerate_mode_overwrites_cache(cfg, monkeypatch):
    save_table(pd.DataFrame([{"cluster_id": 3, "quarter": "2024q4",
                              "narrative": "OLD"}]), "dashboard_narratives", cfg)
    monkeypatch.setattr("steps.step8_dashboard.narratives.generate_one",
                        lambda client, cfg, cluster: "NEW")

    out = get_narratives(cfg, [CLUSTER], mode="regenerate", quarter="2024q4")
    assert out == {3: "NEW"}

    tbl = load_table("dashboard_narratives", cfg)
    q4_rows = tbl[tbl["quarter"] == "2024q4"]
    assert len(q4_rows) == 1
    assert q4_rows.iloc[0]["narrative"] == "NEW"
