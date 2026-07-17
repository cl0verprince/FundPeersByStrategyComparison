"""tests/test_step14_charts.py — the honesty rules that live in chart options."""
import pandas as pd

from webapp.components.charts import (
    auc_by_quarter_option, diverging_delta_option, fund_vs_peers_option)


def test_fund_vs_peers_never_connects_nulls_and_solid_grid():
    opt = fund_vs_peers_option("light", ["2026q1", "2026q2"], [0.04, None], [0.03, 0.02], "VFIAX")
    for s in opt["series"]:
        assert s["connectNulls"] is False
    assert opt["yAxis"]["splitLine"]["lineStyle"]["type"] == "solid"
    assert len(opt["series"]) == 2  # fund + median, emphasis form


def test_fund_line_uses_s1_median_uses_demph():
    from webapp.theme import TOKENS
    opt = fund_vs_peers_option("dark", ["2026q1"], [0.01], [0.02], "X")
    assert opt["series"][0]["lineStyle"]["color"] == TOKENS["dark"]["s1"]
    assert opt["series"][1]["lineStyle"]["color"] == TOKENS["dark"]["demph"]


def test_auc_chart_has_labeled_coinflip_line_and_flags_below_chance():
    df = pd.DataFrame({"quarter": ["2025q4", "2026q1"], "auc": [0.457, 0.427],
                       "persistence_auc": [0.552, 0.569], "source": ["retrained"] * 2})
    opt = auc_by_quarter_option("light", df)
    markline = opt["series"][0]["markLine"]["data"][0]
    assert markline["yAxis"] == 0.5 and "coin flip" in markline["label"]["formatter"].lower()
    # both points below 0.5 wear the critical status color via itemStyle per-point
    from webapp.theme import TOKENS
    pts = opt["series"][0]["data"]
    assert all(p["itemStyle"]["color"] == TOKENS["light"]["critical"] for p in pts)


def test_diverging_uses_polarity_colors():
    from webapp.theme import TOKENS
    opt = diverging_delta_option("light", ["q1", "q2"], [0.01, -0.02])
    colors = [d["itemStyle"]["color"] for d in opt["series"][0]["data"]]
    assert colors == [TOKENS["light"]["div_pos"], TOKENS["light"]["div_neg"]]


def test_status_has_retired_state():
    from webapp.theme import STATUS, TOKENS
    icon, label, token = STATUS["retired"]
    assert (icon, label) == ("✕", "Signal retired")
    assert token in TOKENS["light"] and token not in ("critical", "warning", "serious", "good")
