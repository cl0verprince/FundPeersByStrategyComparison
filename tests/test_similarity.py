"""Unit tests for step2_similarity's pure logic, using small synthetic data - no network,
no embedding-model download. The embedding model itself is exercised by running the real
step (steps/step2_similarity/design.md documents the empirical validation)."""
import pandas as pd

from steps.step2_similarity.similarity import (
    _category_tier,
    build_holdings_description,
    compute_purity,
    normalize_issuer_name,
)


def test_normalize_issuer_name_merges_casing_and_punctuation_variants():
    assert normalize_issuer_name("NVIDIA Corp") == normalize_issuer_name("NVIDIA CORP")
    assert normalize_issuer_name("Microsoft Corp.") == normalize_issuer_name("Microsoft Corp")
    assert normalize_issuer_name("Apple  Inc") == "APPLE INC"  # collapses double space
    assert normalize_issuer_name(None) == ""


def test_build_holdings_description_orders_by_weight_and_renormalizes():
    fund_holdings = pd.DataFrame({
        "issuer_norm": ["APPLE INC", "MICROSOFT CORP", "SMALL CO"],
        "currency_value": [60.0, 30.0, 10.0],
    })
    description = build_holdings_description(fund_holdings, top_n=2)
    assert description.startswith("Fund holdings: APPLE INC 60.0%, MICROSOFT CORP 30.0%")
    assert "SMALL CO" not in description  # top_n=2 truncates the smallest holding


def test_compute_purity_is_majority_fraction_weighted_by_cluster_size():
    # cluster 0: 2 of 3 are "A" (majority); cluster 1: 1 of 1 is "B"
    clusters = pd.Series([0, 0, 0, 1])
    truth = pd.Series(["A", "A", "B", "B"])
    purity = compute_purity(clusters, truth)
    assert purity == (2 + 1) / 4


def test_category_tier_buckets_known_style_boxes_and_falls_back_to_other():
    assert _category_tier("Large Blend") == "Large"
    assert _category_tier("Small Growth") == "Small"
    assert _category_tier("Mid-Cap Value") == "Mid"
    assert _category_tier("Foreign Large Blend") == "Sector/Other"
    assert _category_tier(None) == "Sector/Other"
