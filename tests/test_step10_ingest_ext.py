"""Unit tests for step10's ingestion extensions: relaxed-pool eligibility (calendar-
consecutive quarters), Yahoo-metadata reuse (first table wins), and the pure helpers
behind exhaust mode. No network, small synthetic frames."""
import numpy as np
import pandas as pd

from steps.step1_ingest.ingest import (
    _build_metadata_reuse_map,
    derive_relaxed_pool,
)


def _presence(rows):
    return pd.DataFrame(rows, columns=["series_id", "quarter"])


def _funds_frame(rows):
    return pd.DataFrame(
        rows, columns=["series_id", "ticker", "yahoo_category", "yahoo_stock_position"]
    )


# --- derive_relaxed_pool ---------------------------------------------------


def test_relaxed_pool_six_consecutive_qualifies():
    rows = [("F6", q) for q in
            ["2022q1", "2022q2", "2022q3", "2022q4", "2023q1", "2023q2"]]
    assert derive_relaxed_pool(_presence(rows), 6) == {"F6"}


def test_relaxed_pool_five_consecutive_excluded():
    rows = [("F5", q) for q in ["2022q1", "2022q2", "2022q3", "2022q4", "2023q1"]]
    assert derive_relaxed_pool(_presence(rows), 6) == set()


def test_relaxed_pool_gap_breaks_run_even_when_fund_is_alone():
    # 3 + 3 with a gap at 2022q4; this fund is the ONLY fund in the frame, so the gapped
    # quarter is absent from every derived list. A positional (index-adjacency) impl would
    # collapse the gap and wrongly qualify it; calendar-ordinal adjacency must not.
    rows = [("Fgap", q) for q in
            ["2022q1", "2022q2", "2022q3", "2023q1", "2023q2", "2023q3"]]
    assert derive_relaxed_pool(_presence(rows), 6) == set()


def test_relaxed_pool_all_seventeen_qualifies():
    quarters = [f"{y}q{q}" for y in (2022, 2023, 2024, 2025) for q in (1, 2, 3, 4)]
    quarters += ["2026q1"]
    rows = [("F17", q) for q in quarters]
    assert derive_relaxed_pool(_presence(rows), 6) == {"F17"}


def test_relaxed_pool_dead_fund_with_six_run_mid_window_qualifies():
    # Fdead lived q1..q6 then died; Falive appears only for a 5-quarter tail. The dead fund
    # must qualify (relaxed pool's whole point), the 5-run fund must not.
    rows = [("Fdead", q) for q in
            ["2022q1", "2022q2", "2022q3", "2022q4", "2023q1", "2023q2"]]
    rows += [("Falive", q) for q in
             ["2025q1", "2025q2", "2025q3", "2025q4", "2026q1"]]
    pool = derive_relaxed_pool(_presence(rows), 6)
    assert "Fdead" in pool
    assert "Falive" not in pool


def test_relaxed_pool_uses_longest_run_not_total_count():
    # 5 consecutive + a gap + 5 consecutive = 10 quarters total but longest run is 5.
    rows = [("Fmany", q) for q in
            ["2022q1", "2022q2", "2022q3", "2022q4", "2023q1",   # run of 5
             "2023q3", "2023q4", "2024q1", "2024q2", "2024q3"]]  # gap at 2023q2, run of 5
    assert derive_relaxed_pool(_presence(rows), 6) == set()


# --- _build_metadata_reuse_map --------------------------------------------


def test_reuse_map_builds_and_dedupes_series():
    f = _funds_frame([
        ("S1", "AAA", "Large Blend", 0.98),
        ("S1", "AAA", "Large Blend", 0.98),   # per-quarter duplicate row
        ("S2", "BBB", "Mid-Cap Growth", 0.90),
    ])
    m = _build_metadata_reuse_map([f])
    assert set(m) == {"S1", "S2"}
    assert m["S1"] == {"ticker": "AAA", "yahoo_category": "Large Blend",
                       "yahoo_stock_position": 0.98}


def test_reuse_map_first_table_wins_on_conflict():
    f1 = _funds_frame([("S1", "AAA", "Large Blend", 0.98)])
    f2 = _funds_frame([
        ("S1", "ZZZ", "Small Cap", 0.50),   # conflicts with f1 - f1 wins
        ("S3", "CCC", "Foreign", 0.70),     # new series still gets added
    ])
    m = _build_metadata_reuse_map([f1, f2])
    assert m["S1"]["ticker"] == "AAA"
    assert m["S1"]["yahoo_category"] == "Large Blend"
    assert m["S3"]["ticker"] == "CCC"


def test_reuse_map_nan_category_is_not_a_false_conflict():
    # A missing yahoo_category (NaN) appearing in both tables must not register as a conflict
    # (NaN != NaN naively) nor crash; the series is reused normally.
    f1 = _funds_frame([("S1", "AAA", np.nan, 0.98)])
    f2 = _funds_frame([("S1", "AAA", np.nan, 0.98)])
    m = _build_metadata_reuse_map([f1, f2])
    assert m["S1"]["ticker"] == "AAA"
    assert pd.isna(m["S1"]["yahoo_category"])


def test_reuse_map_skips_empty_frames():
    empty = _funds_frame([]).astype({"yahoo_stock_position": float})
    f = _funds_frame([("S1", "AAA", "Large Blend", 0.98)])
    m = _build_metadata_reuse_map([empty, f])
    assert set(m) == {"S1"}
