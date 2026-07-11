import numpy as np
import pandas as pd

from steps.step7_unified_universe.stability import compute_label_flip_rates

QUARTERS = ["2024q1", "2024q2"]


def _setup(peer_next_returns, own_next=0.0):
    ids = [f"P{i}" for i in range(len(peer_next_returns))]
    peers = pd.DataFrame({
        "series_id": "F", "quarter": "2024q1",
        "peer_rank": range(1, len(ids) + 1), "peer_series_id": ids,
        "cosine_similarity": 0.9})
    returns = pd.concat([
        pd.DataFrame({"series_id": ["F"] + ids, "quarter": "2024q1",
                      "quarterly_return": 0.0}),
        pd.DataFrame({"series_id": ["F"] + ids, "quarter": "2024q2",
                      "quarterly_return": [own_next] + peer_next_returns}),
    ])
    return peers, returns


def test_identical_peers_never_flip():
    peers, returns = _setup([0.05] * 15, own_next=0.0)   # fund clearly below every peer
    out = compute_label_flip_rates(peers, returns, QUARTERS,
                                   top_n=10, pool_size=12, draw_size=8, draws=50, seed=42)
    assert out.iloc[0]["flip_rate"] == 0.0


def test_knife_edge_peers_flip_sometimes():
    # 6 peers above the fund, 6 below: which 8 get drawn decides the median's side.
    peers, returns = _setup([0.1] * 6 + [-0.1] * 6 + [0.1] * 3, own_next=0.0)
    out = compute_label_flip_rates(peers, returns, QUARTERS,
                                   top_n=10, pool_size=12, draw_size=8, draws=200, seed=42)
    assert 0.0 < out.iloc[0]["flip_rate"] < 1.0


def test_deterministic_under_seed():
    peers, returns = _setup([0.1] * 6 + [-0.1] * 9, own_next=0.0)
    kw = dict(top_n=10, pool_size=12, draw_size=8, draws=100, seed=7)
    a = compute_label_flip_rates(peers, returns, QUARTERS, **kw)
    b = compute_label_flip_rates(peers, returns, QUARTERS, **kw)
    pd.testing.assert_frame_equal(a, b)
