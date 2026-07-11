"""kNN-peer label: below the median of your own top-N peers' next-quarter returns."""
import pandas as pd

from steps.step7_unified_universe.label import compute_peer_labels

QUARTERS = ["2024q1", "2024q2"]


def _peers(series_id, peer_ids, quarter="2024q1"):
    return pd.DataFrame({
        "series_id": series_id, "quarter": quarter,
        "peer_rank": range(1, len(peer_ids) + 1),
        "peer_series_id": peer_ids, "cosine_similarity": 0.9,
    })


def _returns(mapping, quarter):
    return pd.DataFrame({"series_id": list(mapping), "quarter": quarter,
                         "quarterly_return": list(mapping.values())})


def test_label_below_peer_median_is_1():
    peers = _peers("F", ["P1", "P2", "P3", "P4", "P5"])
    returns = pd.concat([
        _returns({"F": 0.05, "P1": 0.0, "P2": 0.0, "P3": 0.0, "P4": 0.0, "P5": 0.0}, "2024q1"),
        _returns({"F": 0.01, "P1": 0.02, "P2": 0.03, "P3": 0.04, "P4": 0.05, "P5": 0.06}, "2024q2"),
    ])
    out = compute_peer_labels(peers, returns, QUARTERS, top_n=5, min_valid_peers=5)
    row = out[(out["series_id"] == "F") & (out["quarter"] == "2024q1")].iloc[0]
    assert row["underperform_next_quarter"] == 1          # 0.01 < median(0.02..0.06)=0.04
    assert row["peer_median_return_next"] == 0.04
    assert row["return_vs_peer_median_q"] == 0.05          # 0.05 - median(0.0)=0.0
    assert row["n_valid_peers_next"] == 5


def test_too_few_valid_peers_gives_na_label_but_keeps_features():
    peers = _peers("F", ["P1", "P2", "P3", "P4", "P5"])
    returns = pd.concat([
        _returns({"F": 0.05, "P1": 0.0, "P2": 0.0, "P3": 0.0, "P4": 0.0, "P5": 0.0}, "2024q1"),
        _returns({"F": 0.01, "P1": 0.02, "P2": 0.03}, "2024q2"),   # only 2 peers valid at Q+1
    ])
    out = compute_peer_labels(peers, returns, QUARTERS, top_n=5, min_valid_peers=5)
    row = out.iloc[0]
    assert pd.isna(row["underperform_next_quarter"])
    assert row["return_vs_peer_median_q"] == 0.05          # feature still computed


def test_last_quarter_rows_kept_with_na_label():
    peers = _peers("F", ["P1", "P2", "P3", "P4", "P5"], quarter="2024q2")
    returns = _returns({"F": 0.01, "P1": 0.0, "P2": 0.0, "P3": 0.0, "P4": 0.0, "P5": 0.0}, "2024q2")
    out = compute_peer_labels(peers, returns, QUARTERS, top_n=5, min_valid_peers=5)
    assert len(out) == 1                                    # forward row present
    assert pd.isna(out.iloc[0]["underperform_next_quarter"])


def test_only_top_n_peers_used():
    peers = _peers("F", ["P1", "P2", "P3", "P4", "P5", "OUTLIER"])   # rank 6 must be ignored
    returns = pd.concat([
        _returns({"F": 0.0, "P1": 0.0, "P2": 0.0, "P3": 0.0, "P4": 0.0, "P5": 0.0, "OUTLIER": 0.0}, "2024q1"),
        _returns({"F": 0.05, "P1": 0.01, "P2": 0.02, "P3": 0.03, "P4": 0.06, "P5": 0.07,
                  "OUTLIER": 99.0}, "2024q2"),
    ])
    out = compute_peer_labels(peers, returns, QUARTERS, top_n=5, min_valid_peers=5)
    assert out.iloc[0]["peer_median_return_next"] == 0.03   # median without OUTLIER


def test_label_above_peer_median_is_0():
    peers = _peers("F", ["P1", "P2", "P3", "P4", "P5"])
    returns = pd.concat([
        _returns({"F": 0.0, "P1": 0.0, "P2": 0.0, "P3": 0.0, "P4": 0.0, "P5": 0.0}, "2024q1"),
        _returns({"F": 0.10, "P1": 0.01, "P2": 0.02, "P3": 0.03, "P4": 0.04, "P5": 0.05}, "2024q2"),
    ])
    out = compute_peer_labels(peers, returns, QUARTERS, top_n=5, min_valid_peers=5)
    row = out[(out["series_id"] == "F") & (out["quarter"] == "2024q1")].iloc[0]
    assert row["underperform_next_quarter"] == 0            # 0.10 > median(0.01..0.05)=0.03


def test_missing_own_next_return_gives_na_label():
    peers = _peers("F", ["P1", "P2", "P3", "P4", "P5"])
    returns = pd.concat([
        _returns({"F": 0.05, "P1": 0.0, "P2": 0.0, "P3": 0.0, "P4": 0.0, "P5": 0.0}, "2024q1"),
        _returns({"P1": 0.01, "P2": 0.02, "P3": 0.03, "P4": 0.04, "P5": 0.05}, "2024q2"),  # F absent
    ])
    out = compute_peer_labels(peers, returns, QUARTERS, top_n=5, min_valid_peers=5)
    row = out[(out["series_id"] == "F") & (out["quarter"] == "2024q1")].iloc[0]
    assert pd.isna(row["underperform_next_quarter"])
    assert row["n_valid_peers_next"] == 5
