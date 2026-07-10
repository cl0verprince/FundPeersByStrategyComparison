"""Unit tests for step4_predict's pure logic, focused on the two correctness risks the
design is built around: point-in-time feature safety and a leak-free time-based split."""
import numpy as np
import pandas as pd
from pytest import approx

from steps.step4_predict.predict import (
    assemble_panel,
    category_tier,
    compute_trailing_features,
    time_based_split,
)


def _monthly_rows(series_id, quarters, pct_returns_per_quarter):
    rows = []
    for quarter, pcts in zip(quarters, pct_returns_per_quarter):
        for i, r in enumerate(pcts):
            rows.append({"series_id": series_id, "quarter": quarter,
                         "month_in_quarter": i + 1, "total_return": r})
    return rows


def test_category_tier_matches_step2s_bucketing():
    assert category_tier("Large Growth") == "Large"
    assert category_tier("Small Value") == "Small"
    assert category_tier("Foreign Large Blend") == "Sector/Other"


def test_trailing_features_are_point_in_time_safe():
    # 4 quarters of monthly returns; compare trailing features AT Q2 computed from a
    # history that stops at Q2 vs. one that continues through Q4 - must be identical.
    quarters = ["2024q1", "2024q2", "2024q3", "2024q4"]
    pcts = [[1.0, 1.0, 1.0], [2.0, -1.0, 0.5], [5.0, 5.0, 5.0], [-9.0, -9.0, -9.0]]

    full_history = pd.DataFrame(_monthly_rows("S1", quarters, pcts))
    truncated_at_q2 = pd.DataFrame(_monthly_rows("S1", quarters[:2], pcts[:2]))

    full_result = compute_trailing_features(full_history, risk_free_annual=0.02)
    truncated_result = compute_trailing_features(truncated_at_q2, risk_free_annual=0.02)

    full_q2 = full_result[full_result["quarter"] == "2024q2"].iloc[0]
    truncated_q2 = truncated_result[truncated_result["quarter"] == "2024q2"].iloc[0]

    assert full_q2["trailing_return"] == approx(truncated_q2["trailing_return"])
    assert full_q2["trailing_volatility"] == approx(truncated_q2["trailing_volatility"])
    assert full_q2["trailing_max_drawdown"] == approx(truncated_q2["trailing_max_drawdown"])


def test_trailing_return_matches_hand_calculation_for_short_window():
    # Only one quarter of history - trailing return should be that quarter's compounded return.
    rows = _monthly_rows("S1", ["2024q1"], [[10.0, 10.0, -10.0]])
    result = compute_trailing_features(pd.DataFrame(rows), risk_free_annual=0.02)
    expected = 1.1 * 1.1 * 0.9 - 1
    assert result.iloc[0]["trailing_return"] == approx(expected)


def test_time_based_split_never_puts_a_later_quarter_in_train():
    quarters_ordered = [f"2024q{i}" for i in range(1, 5)] + [f"2025q{i}" for i in range(1, 5)]
    panel = pd.DataFrame({
        "series_id": ["S1"] * 7,
        "quarter": quarters_ordered[:7],  # last quarter has no Q+1, excluded upstream
        "underperform_next_quarter": [0, 1, 0, 1, 0, 1, 0],
    })
    train, test, train_q, test_q = time_based_split(panel, quarters_ordered, holdout_transitions=2)
    assert max(train_q) < min(test_q)
    assert set(train_q) | set(test_q) == set(panel["quarter"])
    assert len(train) + len(test) == len(panel)


def test_assemble_panel_drops_rows_with_missing_label_and_encodes_tier():
    funds = pd.DataFrame([
        {"series_id": "S1", "quarter": "2024q1", "net_assets": 100.0, "yahoo_category": "Large Blend"},
        {"series_id": "S1", "quarter": "2024q2", "net_assets": 110.0, "yahoo_category": "Large Blend"},
    ])
    trailing_features = pd.DataFrame([
        {"series_id": "S1", "quarter": "2024q1", "trailing_return": 0.05, "trailing_volatility": 0.1,
         "trailing_sharpe": 0.5, "trailing_max_drawdown": -0.02},
    ])
    fund_metrics_quarterly = pd.DataFrame([
        {"series_id": "S1", "quarter": "2024q1", "return_vs_cluster_median": 0.01},
        # 2024q2's return_vs_cluster_median is missing entirely -> label for 2024q1 row is unknown
    ])
    panel = assemble_panel(funds, trailing_features, fund_metrics_quarterly,
                            quarters_ordered=["2024q1", "2024q2"])
    assert len(panel) == 0  # the only candidate row has an unknown label, correctly dropped


def test_assemble_panel_labels_and_encodes_tier_when_data_is_complete():
    funds = pd.DataFrame([
        {"series_id": "S1", "quarter": "2024q1", "net_assets": 100.0, "yahoo_category": "Small Value"},
        {"series_id": "S1", "quarter": "2024q2", "net_assets": 110.0, "yahoo_category": "Small Value"},
    ])
    trailing_features = pd.DataFrame([
        {"series_id": "S1", "quarter": "2024q1", "trailing_return": 0.05, "trailing_volatility": 0.1,
         "trailing_sharpe": 0.5, "trailing_max_drawdown": -0.02},
    ])
    fund_metrics_quarterly = pd.DataFrame([
        {"series_id": "S1", "quarter": "2024q1", "return_vs_cluster_median": 0.01},
        {"series_id": "S1", "quarter": "2024q2", "return_vs_cluster_median": -0.03},  # Q+1's label source
    ])
    panel = assemble_panel(funds, trailing_features, fund_metrics_quarterly,
                            quarters_ordered=["2024q1", "2024q2"])
    assert len(panel) == 1
    row = panel.iloc[0]
    assert row["underperform_next_quarter"] == 1  # Q+1's return_vs_cluster_median was negative
    assert row["tier_Small"] == 1
    assert row.get("tier_Large", 0) == 0
