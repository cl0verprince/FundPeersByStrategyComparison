"""step10 build orchestrator - the segment-repair pipeline fix, on synthetic temp-DB data.

The repair guarantees `funds_full` carries a `segment` column reproducibly (a fresh
ingestion lands segment-less; everything downstream filters on segment=='strategy'). It
must ADD the column with the correct assign_segment values when absent, and leave a table
that ALREADY has segment completely untouched (never recompute/overwrite).
"""
import pandas as pd
import pytest

from fundspeers.io import load_table, save_table
from steps.step10_full_universe import build as full_build
from steps.step10_full_universe.build import ensure_funds_full_segment


@pytest.fixture
def cfg(tmp_path):
    return {"paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "processed"),
                      "reports": str(tmp_path / "reports"), "models": str(tmp_path / "models")}}


def test_segment_column_added_with_correct_values(cfg):
    # A segment-less funds_full (as a fresh ingestion would produce).
    funds = pd.DataFrame({
        "series_id": ["A", "B", "C", "D"],
        "yahoo_category": ["Large Blend", "Target-Date 2050",
                           "Allocation--50% to 70% Equity", None],
        "is_us_equity": [True, True, True, True],
    })
    save_table(funds, "funds_full", cfg)
    assert "segment" not in load_table("funds_full", cfg).columns

    ensure_funds_full_segment(cfg)

    out = load_table("funds_full", cfg).set_index("series_id")
    assert "segment" in out.columns
    assert out["segment"].to_dict() == {
        "A": "strategy",       # Large Blend
        "B": "allocation",     # Target-Date
        "C": "allocation",     # Allocation--...
        "D": "strategy",       # None -> strategy
    }


def test_table_with_segment_is_left_untouched(cfg):
    # Sentinel segment values that assign_segment would NEVER produce - if the repair
    # recomputed them, these would be overwritten to strategy/allocation.
    funds = pd.DataFrame({
        "series_id": ["A", "B"],
        "yahoo_category": ["Large Blend", "Target-Date 2050"],
        "segment": ["SENTINEL_X", "SENTINEL_Y"],
    })
    save_table(funds, "funds_full", cfg)

    ensure_funds_full_segment(cfg)

    out = load_table("funds_full", cfg).set_index("series_id")
    assert out["segment"].to_dict() == {"A": "SENTINEL_X", "B": "SENTINEL_Y"}


# --- run_retired (design step16): score-frozen-only, no retrain/no fund-disjoint check ---


def _boom(*_a, **_k):
    raise AssertionError("must not be called on the retired path")


def test_run_retired_calls_all_five_stages_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(full_build, "ensure_funds_full_segment",
                        lambda cfg: calls.append("ensure_funds_full_segment"))
    monkeypatch.setattr(full_build, "score_published_forward_predictions",
                        lambda cfg: calls.append("score_published_forward_predictions")
                        or {"auc": 0.5})
    monkeypatch.setattr(full_build, "score_frozen_model_rolled_forward",
                        lambda cfg: calls.append("score_frozen_model_rolled_forward")
                        or {"auc_pooled": 0.5})
    monkeypatch.setattr(full_build, "_write_oot_validation",
                        lambda published, frozen, cfg: calls.append("_write_oot_validation"))
    monkeypatch.setattr(full_build, "run_stability",
                        lambda cfg, table_suffix, output_table: calls.append("run_stability"))
    # Trip-wires: the retired path must never train or emit new predictions.
    monkeypatch.setattr(full_build, "train_and_evaluate", _boom)
    monkeypatch.setattr(full_build, "fund_disjoint_auc", _boom)

    full_build.run_retired({"seed": 42})

    assert calls == [
        "ensure_funds_full_segment",
        "score_published_forward_predictions",
        "score_frozen_model_rolled_forward",
        "_write_oot_validation",
        "run_stability",
    ]


def test_run_retired_passes_published_and_frozen_scores_to_oot_writer(monkeypatch):
    published = {"auc": 0.457}
    frozen = {"auc_pooled": 0.427}
    written = {}
    monkeypatch.setattr(full_build, "ensure_funds_full_segment", lambda cfg: None)
    monkeypatch.setattr(full_build, "score_published_forward_predictions", lambda cfg: published)
    monkeypatch.setattr(full_build, "score_frozen_model_rolled_forward", lambda cfg: frozen)
    monkeypatch.setattr(full_build, "_write_oot_validation",
                        lambda pub, frz, cfg: written.update(published=pub, frozen=frz))
    monkeypatch.setattr(full_build, "run_stability", lambda cfg, table_suffix, output_table: None)
    monkeypatch.setattr(full_build, "train_and_evaluate", _boom)
    monkeypatch.setattr(full_build, "fund_disjoint_auc", _boom)

    full_build.run_retired({"seed": 42})

    assert written == {"published": published, "frozen": frozen}


def test_run_retired_runs_stability_on_full_universe(monkeypatch):
    stability_calls = []
    monkeypatch.setattr(full_build, "ensure_funds_full_segment", lambda cfg: None)
    monkeypatch.setattr(full_build, "score_published_forward_predictions", lambda cfg: {"auc": 0.5})
    monkeypatch.setattr(full_build, "score_frozen_model_rolled_forward",
                        lambda cfg: {"auc_pooled": 0.5})
    monkeypatch.setattr(full_build, "_write_oot_validation", lambda pub, frz, cfg: None)
    monkeypatch.setattr(full_build, "run_stability",
                        lambda cfg, table_suffix, output_table: stability_calls.append(
                            (table_suffix, output_table)))
    monkeypatch.setattr(full_build, "train_and_evaluate", _boom)
    monkeypatch.setattr(full_build, "fund_disjoint_auc", _boom)

    full_build.run_retired({"seed": 42})

    assert stability_calls == [("_full", "full_label_stability")]
