"""step10 Task 3: the step7 modules gain backward-compatible params (table_suffix,
holdout_transitions, output_prefix). Defaults must reproduce today's behavior exactly.
Signature/default assertions follow the tests/test_step7_similarity_ext.py pattern; one
temp-DB test proves the suffix is honored; plus the label.py dedup guard."""
import inspect

import pandas as pd
import pytest

from fundspeers.io import save_table
from steps.step7_unified_universe.panel import assemble_unified_panel
from steps.step7_unified_universe.model import train_and_evaluate
from steps.step7_unified_universe.stability import run_stability
from steps.step7_unified_universe.label import compute_peer_labels

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


# --- signature / default assertions -----------------------------------------

def test_assemble_unified_panel_signature_backward_compatible():
    sig = inspect.signature(assemble_unified_panel)
    assert list(sig.parameters) == ["cfg", "table_suffix"]
    assert sig.parameters["table_suffix"].default == "_all"


def test_train_and_evaluate_signature_backward_compatible():
    sig = inspect.signature(train_and_evaluate)
    assert list(sig.parameters) == [
        "cfg", "table_suffix", "holdout_transitions", "output_prefix"]
    assert sig.parameters["table_suffix"].default == "_all"
    assert sig.parameters["holdout_transitions"].default is None
    assert sig.parameters["output_prefix"].default == "unified"


def test_run_stability_signature_backward_compatible():
    sig = inspect.signature(run_stability)
    assert list(sig.parameters) == ["cfg", "table_suffix", "output_table"]
    assert sig.parameters["table_suffix"].default == "_all"
    assert sig.parameters["output_table"].default is None


# --- temp-DB: the suffix is honored -----------------------------------------

def _save_universe(cfg, suffix):
    """Adapted from tests/test_step7_panel.py's `tables` fixture, but the four tables are
    saved under an arbitrary `suffix` so we can prove assemble_unified_panel reads them."""
    funds = pd.DataFrame([
        {"series_id": s, "quarter": q, "accession_number": f"acc-{s}-{q}",
         "yahoo_category": "Large Blend", "is_us_equity": True, "segment": "strategy",
         "net_assets": 100.0 + i, "series_name": s, "ticker": s}
        for i, s in enumerate(IDS) for q in QUARTERS])
    returns = pd.DataFrame([
        {"series_id": s, "quarter": q, "month_in_quarter": m,
         "total_return": 1.0 + i * 0.1 + m * 0.5 + (0.3 if q == "2024q2" else 0.0)}
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
    save_table(funds, f"funds{suffix}", cfg)
    save_table(returns, f"monthly_returns{suffix}", cfg)
    save_table(holdings, f"holdings{suffix}", cfg)
    save_table(peers, f"fund_peers{suffix}", cfg)


def test_table_suffix_reads_suffixed_tables(cfg):
    # Only the `_x` tables exist; no `_all` tables. If the suffix weren't honored the
    # load_table calls would raise. The panel must come back with the expected forward set.
    _save_universe(cfg, "_x")
    labeled, forward, feature_cols = assemble_unified_panel(cfg, table_suffix="_x")
    assert set(forward["quarter"]) == {"2024q2"}
    assert len(forward) == 6
    assert "return_vs_peer_median_q" in feature_cols
    assert forward[feature_cols].notna().all().all()


# --- label.py dedup guard ----------------------------------------------------

def test_compute_peer_labels_rejects_duplicate_quarterly_returns():
    fund_peers = pd.DataFrame([
        {"series_id": "F", "quarter": "2024q1", "peer_rank": 1, "peer_series_id": "P1"},
    ])
    dup_returns = pd.DataFrame([
        {"series_id": "F", "quarter": "2024q1", "quarterly_return": 0.1},
        {"series_id": "F", "quarter": "2024q1", "quarterly_return": 0.2},  # duplicate key
    ])
    with pytest.raises(ValueError, match="unique"):
        compute_peer_labels(fund_peers, dup_returns, ["2024q1", "2024q2"],
                            top_n=5, min_valid_peers=1)


def test_compute_peer_labels_accepts_unique_quarterly_returns():
    fund_peers = pd.DataFrame([
        {"series_id": "F", "quarter": "2024q1", "peer_rank": 1, "peer_series_id": "P1"},
    ])
    returns = pd.DataFrame([
        {"series_id": "F", "quarter": "2024q1", "quarterly_return": 0.1},
        {"series_id": "P1", "quarter": "2024q1", "quarterly_return": 0.2},
    ])
    out = compute_peer_labels(fund_peers, returns, ["2024q1", "2024q2"],
                              top_n=5, min_valid_peers=1)
    assert list(out["series_id"]) == ["F"]
