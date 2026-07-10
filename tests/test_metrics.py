"""Unit tests for step3_metrics's pure logic, using small hand-computed synthetic data."""
import numpy as np
import pandas as pd
from pytest import approx

from steps.step3_metrics.metrics import (
    _concentration_word,
    compute_cluster_definitions,
    compute_cluster_relative_metrics,
    compute_overall_metrics,
    compute_quarterly_returns,
)


def _monthly_returns(series_id, quarter, pct_returns):
    return [
        {"series_id": series_id, "quarter": quarter, "month_in_quarter": i + 1, "total_return": r}
        for i, r in enumerate(pct_returns)
    ]


def test_compute_quarterly_returns_compounds_the_three_months():
    # +10%, +10%, -10% compounds to 1.1*1.1*0.9 - 1 = 0.089
    rows = _monthly_returns("S1", "2024q1", [10.0, 10.0, -10.0])
    result = compute_quarterly_returns(pd.DataFrame(rows))
    assert len(result) == 1
    assert result["quarterly_return"].iloc[0] == approx(0.089)


def test_compute_overall_metrics_matches_hand_calculation():
    # Two quarters of steady +1% monthly returns: 6 months of +1%.
    rows = _monthly_returns("S1", "2024q1", [1.0, 1.0, 1.0]) + _monthly_returns(
        "S1", "2024q2", [1.0, 1.0, 1.0]
    )
    result = compute_overall_metrics(pd.DataFrame(rows), risk_free_annual=0.02)
    row = result.iloc[0]
    expected_cumulative = 1.01 ** 6 - 1
    assert row["cumulative_return"] == approx(expected_cumulative)
    assert row["max_drawdown"] == approx(0.0, abs=1e-9)  # monotonically up, no drawdown
    assert row["annualized_volatility"] == approx(0.0, abs=1e-9)  # constant returns, zero std


def test_compute_overall_metrics_max_drawdown_matches_hand_calculation():
    # +10%, then -20%, then +5%: wealth = 1.10, 0.88, 0.924
    # drawdown at step 2 = (0.88 - 1.10)/1.10 = -0.2; that's the worst point
    rows = _monthly_returns("S1", "2024q1", [10.0, -20.0, 5.0])
    result = compute_overall_metrics(pd.DataFrame(rows), risk_free_annual=0.02)
    assert result.iloc[0]["max_drawdown"] == approx(-0.2)


def test_quarterly_return_with_a_missing_month_is_nan_not_silently_computed_from_two():
    # Regression test: np.prod() on a pandas Series (vs. a numpy array) silently skips
    # NaN via Series.prod(skipna=True) - caught on real data where a fund had one
    # unparseable monthly return; the quarter's return must be NaN, not a 2-of-3 product.
    rows = _monthly_returns("S1", "2024q1", [10.0, np.nan, -5.0])
    result = compute_quarterly_returns(pd.DataFrame(rows))
    assert pd.isna(result["quarterly_return"].iloc[0])


def test_cluster_relative_metrics_excludes_missing_cluster_rows_not_group_them_together():
    quarterly_returns = pd.DataFrame([
        {"series_id": "S1", "quarter": "2024q1", "quarterly_return": 0.10},
        {"series_id": "S2", "quarter": "2024q1", "quarterly_return": 0.20},
        {"series_id": "S3", "quarter": "2024q1", "quarterly_return": 0.90},  # no cluster resolved
        {"series_id": "S4", "quarter": "2024q1", "quarterly_return": -0.50},  # no cluster resolved
    ])
    fund_clusters = pd.DataFrame([
        {"series_id": "S1", "quarter": "2024q1", "cluster_id": 0},
        {"series_id": "S2", "quarter": "2024q1", "cluster_id": 0},
        # S3, S4 absent - simulates step2's zero-EC-holdings exclusion
    ])
    result = compute_cluster_relative_metrics(quarterly_returns, fund_clusters)

    cluster0_median = result.set_index("series_id").loc["S1", "cluster_median_return"]
    assert cluster0_median == approx(0.15)  # median(0.10, 0.20)

    # S3/S4 must NOT be silently grouped together and given each other's median
    s3_row = result.set_index("series_id").loc["S3"]
    s4_row = result.set_index("series_id").loc["S4"]
    assert pd.isna(s3_row["cluster_median_return"])
    assert pd.isna(s4_row["cluster_median_return"])
    assert pd.isna(s3_row["return_vs_cluster_median"])


def test_compute_cluster_definitions_picks_dominant_category_and_averages_metrics():
    fund_clusters = pd.DataFrame([
        {"series_id": "S1", "quarter": "2024q1", "cluster_id": 0},
        {"series_id": "S2", "quarter": "2024q1", "cluster_id": 0},
        {"series_id": "S3", "quarter": "2024q1", "cluster_id": 0},
    ])
    funds = pd.DataFrame([
        {"series_id": "S1", "quarter": "2024q1", "yahoo_category": "Small Value"},
        {"series_id": "S2", "quarter": "2024q1", "yahoo_category": "Small Value"},
        {"series_id": "S3", "quarter": "2024q1", "yahoo_category": "Small Growth"},
    ])
    fund_metrics_overall = pd.DataFrame([
        {"series_id": "S1", "annualized_volatility": 0.20, "sharpe_ratio": 0.5,
         "cumulative_return": 0.1, "max_drawdown": -0.1},
        {"series_id": "S2", "annualized_volatility": 0.30, "sharpe_ratio": 0.3,
         "cumulative_return": 0.2, "max_drawdown": -0.2},
        {"series_id": "S3", "annualized_volatility": 0.25, "sharpe_ratio": 0.4,
         "cumulative_return": 0.15, "max_drawdown": -0.15},
    ])
    result = compute_cluster_definitions(fund_clusters, funds, fund_metrics_overall)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["member_count"] == 3
    assert row["dominant_category"] == "Small Value"  # 2 of 3
    assert row["dominant_category_share"] == approx(2 / 3)
    assert row["dominant_tier"] == "Small"
    assert row["avg_volatility"] == approx((0.20 + 0.30 + 0.25) / 3)
    assert row["avg_sharpe"] == approx((0.5 + 0.3 + 0.4) / 3)
    assert "Small tilt" in row["title"]
    assert "Small Value" in row["title"]
    assert row["short_title"] == "Leaning Small Value"  # share 2/3 ~ 0.67, in the 0.40-0.70 band


def test_concentration_word_thresholds():
    assert _concentration_word(0.86) == "Concentrated"
    assert _concentration_word(0.70) == "Concentrated"
    assert _concentration_word(0.69) == "Leaning"
    assert _concentration_word(0.40) == "Leaning"
    assert _concentration_word(0.39) == "Mixed"
    assert _concentration_word(0.0) == "Mixed"


def test_short_title_never_includes_a_performance_word():
    # Clusters are formed from holdings/allocation similarity, not performance - short_title
    # must never mention Sharpe, volatility, or "underperform" (which already has a specific,
    # different, cluster-relative meaning elsewhere in this pipeline).
    fund_clusters = pd.DataFrame([
        {"series_id": "S1", "quarter": "2024q1", "cluster_id": 0},
    ])
    funds = pd.DataFrame([
        {"series_id": "S1", "quarter": "2024q1", "yahoo_category": "Large Blend"},
    ])
    fund_metrics_overall = pd.DataFrame([
        {"series_id": "S1", "annualized_volatility": 0.20, "sharpe_ratio": -1.5,
         "cumulative_return": -0.3, "max_drawdown": -0.5},
    ])
    result = compute_cluster_definitions(fund_clusters, funds, fund_metrics_overall)
    short_title = result.iloc[0]["short_title"]
    for forbidden in ("sharpe", "volatility", "underperform"):
        assert forbidden not in short_title.lower()
    assert short_title == "Concentrated Large Blend"
