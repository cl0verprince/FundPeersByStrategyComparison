"""Unit tests for the shared fundspeers.category bucketing - used by step2, step3, and step4."""
from fundspeers.category import category_tier


def test_category_tier_buckets_known_style_boxes_and_falls_back_to_other():
    assert category_tier("Large Blend") == "Large"
    assert category_tier("Small Growth") == "Small"
    assert category_tier("Mid-Cap Value") == "Mid"
    assert category_tier("Foreign Large Blend") == "Sector/Other"
    assert category_tier(None) == "Sector/Other"
