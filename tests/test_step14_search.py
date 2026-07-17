"""tests/test_step14_search.py — omnibox scorer: tiers, fuzziness, dead-fund demotion."""
from webapp.data import score_query

INDEX = [
    {"series_id": "V1", "ticker": "VFIAX", "series_name": "Vanguard 500 Index Admiral",
     "name_normalized": "vanguard 500 index admiral", "net_assets": 4e11,
     "is_active": True, "last_quarter": "2026q2", "cluster_name": "Leaning Large Blend"},
    {"series_id": "V2", "ticker": "VFINX", "series_name": "Vanguard 500 Index Investor",
     "name_normalized": "vanguard 500 index investor", "net_assets": 1e10,
     "is_active": True, "last_quarter": "2026q2", "cluster_name": "Leaning Large Blend"},
    {"series_id": "F1", "ticker": "FMAGX", "series_name": "Fidelity Magellan",
     "name_normalized": "fidelity magellan", "net_assets": 2e10,
     "is_active": True, "last_quarter": "2026q2", "cluster_name": "Leaning Large Growth"},
    {"series_id": "D1", "ticker": "VDEAD", "series_name": "Vanguard Departed Fund",
     "name_normalized": "vanguard departed fund", "net_assets": 9e11,
     "is_active": False, "last_quarter": "2024q3", "cluster_name": "Concentrated Value"},
]


def test_exact_ticker_first():
    assert score_query("vfiax", INDEX)[0]["ticker"] == "VFIAX"


def test_ticker_prefix_beats_name_match():
    got = [r["ticker"] for r in score_query("VFI", INDEX)]
    assert got[:2] == ["VFIAX", "VFINX"]  # prefix tier, AUM desc within tier


def test_name_token_prefixes():
    assert score_query("vang adm", INDEX)[0]["ticker"] == "VFIAX"


def test_fuzzy_typo():
    assert score_query("vangard 500", INDEX)[0]["ticker"] == "VFIAX"


def test_dead_fund_demoted_despite_aum():
    got = [r["ticker"] for r in score_query("vanguard", INDEX)]
    assert got.index("VDEAD") > got.index("VFIAX")  # 9e11 AUM but dead -> below living


def test_no_match_empty():
    assert score_query("zzzzqq", INDEX) == []
