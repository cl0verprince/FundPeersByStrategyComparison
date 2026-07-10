"""Unit tests for step6_out_of_sample's pure logic - column alignment between a frozen
model's training feature list and a differently-shaped panel."""
import pandas as pd

from steps.step6_out_of_sample.evaluate import align_features


def test_align_features_fills_missing_tier_column_with_zero():
    panel = pd.DataFrame({
        "trailing_return": [0.1, 0.2],
        "tier_Large": [1, 0],
        "tier_Small": [0, 1],
        # tier_Mid and tier_Sector/Other absent - this panel has no Mid or Sector/Other funds
    })
    feature_cols = ["trailing_return", "tier_Large", "tier_Mid", "tier_Sector/Other", "tier_Small"]
    aligned = align_features(panel, feature_cols)
    assert list(aligned.columns) == feature_cols
    assert (aligned["tier_Mid"] == 0).all()
    assert (aligned["tier_Sector/Other"] == 0).all()
    assert aligned["tier_Large"].tolist() == [1, 0]


def test_align_features_preserves_column_order_even_if_panel_order_differs():
    panel = pd.DataFrame({"tier_Small": [1], "trailing_return": [0.5], "tier_Large": [0]})
    feature_cols = ["trailing_return", "tier_Large", "tier_Small"]
    aligned = align_features(panel, feature_cols)
    assert list(aligned.columns) == feature_cols
