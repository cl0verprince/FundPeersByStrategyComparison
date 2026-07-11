"""Renderer: self-contained, deterministic, disclaimer everywhere, placeholders on skip."""
import re

from steps.step8_dashboard.data import DISCLAIMER
from steps.step8_dashboard.render import render_dashboard

PAYLOAD = {
    "universe": {"n_funds": 3, "n_strategy": 2, "n_allocation": 1,
                 "quarters": ["2024q4"], "latest_quarter": "2024q4"},
    "scorecard": {"auc": 0.7, "auc_ci": [0.65, 0.75], "persistence_auc": 0.6,
                  "p_edge_le_zero": 0.02, "per_quarter": [], "mean_flip_rate": 0.05},
    "coords": [{"sid": "F1", "x": 0.1, "y": 0.2, "cluster": 0}],
    "clusters": [{"cluster_id": 0, "short_title": "Leaning Large Blend",
                  "dominant_category": "Large Blend", "dominant_share": 1.0,
                  "member_count": 1, "avg_sharpe": 0.5, "avg_volatility": 0.15,
                  "avg_max_drawdown": -0.1, "median_net_assets": 100.0,
                  "top_holdings": [{"issuer": "ACME", "weight": 0.5}],
                  "narrative": "",
                  "members": [{"sid": "F1", "name": "Fund One", "ticker": "ONE",
                               "net_assets": 100.0, "sharpe": 0.5, "volatility": 0.15,
                               "max_drawdown": -0.1, "cumulative_return": 0.2,
                               "probability": 0.7}]}],
    "allocation": [{"vintage": "Target-Date 2050",
                    "members": [{"sid": "T1", "name": "Target Fund", "ticker": "TGT",
                                 "net_assets": 50.0, "sharpe": 0.5, "volatility": 0.15,
                                 "max_drawdown": -0.1, "cumulative_return": 0.2}]}],
    "disclaimer": DISCLAIMER,
}


def test_renders_and_contains_disclaimer():
    html = render_dashboard(PAYLOAD)
    assert DISCLAIMER in html
    assert "Leaning Large Blend" in html
    assert "Target-Date 2050" in html


def test_no_external_requests():
    html = render_dashboard(PAYLOAD)
    assert not re.search(r'<script[^>]+src=', html)
    assert not re.search(r'<link[^>]+href="https?://', html)
    assert "http://" not in html.replace("http://www.w3.org", "")   # SVG namespace ok
    assert "https://" not in html


def test_deterministic():
    assert render_dashboard(PAYLOAD) == render_dashboard(PAYLOAD)


def test_empty_narrative_shows_placeholder():
    html = render_dashboard(PAYLOAD)
    assert "narrative not generated" in html
