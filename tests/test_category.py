"""Unit tests for the shared fundspeers.category bucketing - used by step2, step3, and step4."""
import pandas as pd

from fundspeers.category import category_tier, compute_dominant_category_info, concentration_word


def test_category_tier_buckets_known_style_boxes_and_falls_back_to_other():
    assert category_tier("Large Blend") == "Large"
    assert category_tier("Small Growth") == "Small"
    assert category_tier("Mid-Cap Value") == "Mid"
    assert category_tier("Foreign Large Blend") == "Sector/Other"
    assert category_tier(None) == "Sector/Other"


def test_concentration_word_thresholds():
    assert concentration_word(0.86) == "Concentrated"
    assert concentration_word(0.70) == "Concentrated"
    assert concentration_word(0.69) == "Leaning"
    assert concentration_word(0.40) == "Leaning"
    assert concentration_word(0.39) == "Mixed"
    assert concentration_word(0.0) == "Mixed"


def test_compute_dominant_category_info_builds_allocation_only_short_title():
    cluster_assignments = pd.DataFrame([
        {"series_id": "S1", "cluster_id": 0},
        {"series_id": "S2", "cluster_id": 0},
        {"series_id": "S3", "cluster_id": 0},
    ])
    category_by_series = pd.Series(
        {"S1": "Small Value", "S2": "Small Value", "S3": "Small Growth"}
    )
    result = compute_dominant_category_info(cluster_assignments, category_by_series)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["dominant_category"] == "Small Value"
    assert row["dominant_category_share"] == 2 / 3
    assert row["short_title"] == "Leaning Small Value"
    for forbidden in ("sharpe", "volatility"):
        assert forbidden not in row["short_title"].lower()
