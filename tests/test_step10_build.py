"""step10 build orchestrator - the segment-repair pipeline fix, on synthetic temp-DB data.

The repair guarantees `funds_full` carries a `segment` column reproducibly (a fresh
ingestion lands segment-less; everything downstream filters on segment=='strategy'). It
must ADD the column with the correct assign_segment values when absent, and leave a table
that ALREADY has segment completely untouched (never recompute/overwrite).
"""
import pandas as pd
import pytest

from fundspeers.io import load_table, save_table
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
