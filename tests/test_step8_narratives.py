"""Narrative caching: cached mode never calls the LLM for cached clusters; the prompt is
grounded (contains only the cluster's own numbers); skip mode returns empty."""
import pandas as pd
import pytest

from fundspeers.io import load_table, save_table, table_exists
from steps.step8_dashboard.narratives import build_cluster_prompt, get_narratives

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


def test_cached_mode_fails_loudly_for_missing_cluster_without_llm(cfg, monkeypatch):
    # Force an unreachable endpoint so this test exercises the loud-failure path regardless
    # of whether LM Studio happens to be running on the host (task: tests must not depend on it).
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1/v1")
    save_table(pd.DataFrame([{"cluster_id": 99, "quarter": "2024q4",
                              "narrative": "other"}]), "dashboard_narratives", cfg)
    with pytest.raises(Exception):   # cluster 3 uncached -> attempts LLM -> connection error
        get_narratives(cfg, [CLUSTER], mode="cached")
