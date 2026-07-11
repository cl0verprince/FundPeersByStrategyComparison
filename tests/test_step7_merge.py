"""Unit tests for step7's batch merge - synthetic data, temp DB, never the real one."""
import pandas as pd
import pytest

from steps.step7_unified_universe.merge import assign_segment, build_unified_tables
from fundspeers.io import load_table, save_table


@pytest.fixture
def cfg(tmp_path):
    return {"paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "processed"),
                      "reports": str(tmp_path / "reports"), "models": str(tmp_path / "models")}}


def _make_batch(series_ids, categories):
    funds = pd.DataFrame({
        "series_id": series_ids, "quarter": "2024q4",
        "accession_number": [f"acc-{s}" for s in series_ids],
        "yahoo_category": categories, "is_us_equity": True, "net_assets": 1.0,
        "series_name": series_ids, "ticker": series_ids,
    })
    holdings = pd.DataFrame({
        "accession_number": [f"acc-{s}" for s in series_ids], "quarter": "2024q4",
        "asset_cat": "EC", "issuer_name": "ACME CORP", "currency_value": 100.0,
    })
    returns = pd.DataFrame({
        "series_id": series_ids, "quarter": "2024q4", "month_in_quarter": 1,
        "total_return": 1.0,
    })
    return funds, holdings, returns


def test_assign_segment():
    assert assign_segment("Target-Date 2045") == "allocation"
    assert assign_segment("Allocation--50% to 70% Equity") == "allocation"
    assert assign_segment("Large Blend") == "strategy"
    assert assign_segment(None) == "strategy"
    assert assign_segment(float("nan")) == "strategy"


def test_merge_concatenates_and_segments(cfg):
    for suffix, ids, cats in [("", ["A"], ["Large Blend"]),
                              ("_oos", ["B"], ["Target-Date 2050"]),
                              ("_oos2", ["C"], ["Small Value"])]:
        funds, holdings, returns = _make_batch(ids, cats)
        save_table(funds, f"funds{suffix}", cfg)
        save_table(holdings, f"holdings{suffix}", cfg)
        save_table(returns, f"monthly_returns{suffix}", cfg)
    counts = build_unified_tables(cfg)
    assert counts == {"n_funds": 3, "n_strategy": 2, "n_allocation": 1}
    funds_all = load_table("funds_all", cfg)
    assert set(funds_all["series_id"]) == {"A", "B", "C"}
    assert funds_all.set_index("series_id")["segment"].to_dict() == {
        "A": "strategy", "B": "allocation", "C": "strategy"}
    assert len(load_table("holdings_all", cfg)) == 3
    assert len(load_table("monthly_returns_all", cfg)) == 3


def test_merge_raises_on_overlapping_batches(cfg):
    for suffix in ["", "_oos", "_oos2"]:
        funds, holdings, returns = _make_batch(["DUP"], ["Large Blend"])
        save_table(funds, f"funds{suffix}", cfg)
        save_table(holdings, f"holdings{suffix}", cfg)
        save_table(returns, f"monthly_returns{suffix}", cfg)
    with pytest.raises(RuntimeError, match="overlap"):
        build_unified_tables(cfg)
