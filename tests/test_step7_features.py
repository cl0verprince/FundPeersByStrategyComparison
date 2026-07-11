import numpy as np
import pandas as pd

from steps.step7_unified_universe.features import (
    compute_holdings_features, compute_net_asset_momentum, compute_peer_similarity_feature)


def test_holdings_features_concentration():
    h = pd.DataFrame({
        "series_id": ["F"] * 2 + ["G"] * 4,
        "quarter": "2024q1",
        "currency_value": [50.0, 50.0, 25.0, 25.0, 25.0, 25.0],
    })
    out = compute_holdings_features(h).set_index("series_id")
    assert out.loc["F", "hhi"] == 0.5            # two equal holdings: 2*(0.5^2)
    assert out.loc["G", "hhi"] == 0.25           # four equal holdings: 4*(0.25^2)
    assert out.loc["F", "top10_weight"] == 1.0
    assert out.loc["F", "n_holdings"] == 2


def test_holdings_features_top10_weight_caps_at_ten():
    h = pd.DataFrame({"series_id": "F", "quarter": "2024q1",
                      "currency_value": [10.0] * 20})
    out = compute_holdings_features(h)
    assert np.isclose(out.iloc[0]["top10_weight"], 0.5)   # 10 of 20 equal holdings


def test_peer_similarity_feature_uses_top_n_only():
    peers = pd.DataFrame({
        "series_id": "F", "quarter": "2024q1", "peer_rank": [1, 2, 3],
        "peer_series_id": ["A", "B", "C"], "cosine_similarity": [0.9, 0.7, 0.1],
    })
    out = compute_peer_similarity_feature(peers, top_n=2)
    assert np.isclose(out.iloc[0]["mean_peer_similarity"], 0.8)


def test_net_asset_momentum_first_quarter_is_nan():
    funds = pd.DataFrame({
        "series_id": ["F", "F", "G"], "quarter": ["2024q1", "2024q2", "2024q1"],
        "net_assets": [100.0, 110.0, 5.0],
    })
    out = compute_net_asset_momentum(funds).set_index(["series_id", "quarter"])
    assert pd.isna(out.loc[("F", "2024q1"), "net_assets_qoq"])
    assert np.isclose(out.loc[("F", "2024q2"), "net_assets_qoq"], 0.10)
    assert pd.isna(out.loc[("G", "2024q1"), "net_assets_qoq"])


def test_net_asset_momentum_nan_mid_series_nulls_two_quarters():
    # pandas 2+ pct_change (default fill_method=None) computes diff()/shift(1): a NaN
    # value poisons both the quarter it's in (denominator NaN, from shift) and the
    # following quarter (numerator NaN, from diff) - on top of the first quarter always
    # being NaN. So all three quarters end up NaN here, not just the middle one.
    funds = pd.DataFrame({
        "series_id": ["F", "F", "F"], "quarter": ["2024q1", "2024q2", "2024q3"],
        "net_assets": [100.0, float("nan"), 120.0],
    })
    out = compute_net_asset_momentum(funds).set_index(["series_id", "quarter"])
    assert pd.isna(out.loc[("F", "2024q1"), "net_assets_qoq"])
    assert pd.isna(out.loc[("F", "2024q2"), "net_assets_qoq"])
    assert pd.isna(out.loc[("F", "2024q3"), "net_assets_qoq"])


def test_holdings_features_single_holding_hhi_is_1():
    h = pd.DataFrame({"series_id": ["F"], "quarter": ["2024q1"], "currency_value": [50.0]})
    out = compute_holdings_features(h)
    assert out.iloc[0]["hhi"] == 1.0
    assert out.iloc[0]["top10_weight"] == 1.0
    assert out.iloc[0]["n_holdings"] == 1


def test_empty_inputs_return_correct_columns():
    holdings_cols = ["series_id", "quarter", "currency_value"]
    out_h = compute_holdings_features(pd.DataFrame(columns=holdings_cols))
    assert out_h.empty
    assert out_h.columns.tolist() == ["series_id", "quarter", "hhi", "top10_weight", "n_holdings"]

    peers_cols = ["series_id", "quarter", "peer_rank", "peer_series_id", "cosine_similarity"]
    out_p = compute_peer_similarity_feature(pd.DataFrame(columns=peers_cols), top_n=2)
    assert out_p.empty
    assert out_p.columns.tolist() == ["series_id", "quarter", "mean_peer_similarity"]

    funds_cols = ["series_id", "quarter", "net_assets"]
    out_f = compute_net_asset_momentum(pd.DataFrame(columns=funds_cols))
    assert out_f.empty
    assert out_f.columns.tolist() == ["series_id", "quarter", "net_assets_qoq"]


def test_net_asset_momentum_duplicate_rows_keep_first():
    # drop_duplicates(["series_id", "quarter"]) defaults to keep="first", so the 999.0
    # second observation for 2024q1 is silently dropped and never affects the qoq calc.
    funds = pd.DataFrame({
        "series_id": ["F", "F", "F"], "quarter": ["2024q1", "2024q1", "2024q2"],
        "net_assets": [100.0, 999.0, 110.0],
    })
    out = compute_net_asset_momentum(funds).set_index(["series_id", "quarter"])
    assert np.isclose(out.loc[("F", "2024q2"), "net_assets_qoq"], 0.10)
