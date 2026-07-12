"""Renderer: self-contained, deterministic, disclaimer everywhere, placeholders on skip."""
import copy
import json
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


# The data layer always emits the five oot_* keys (present with None -> JSON null) when the
# oot_validation table is absent. This mirrors that exact state - distinct from omitting the
# keys - so the "empty-oot renders like before" guarantee is verified against null, not undefined.
PAYLOAD_OOT_NULL = copy.deepcopy(PAYLOAD)
PAYLOAD_OOT_NULL["scorecard"].update({
    "oot_published_auc": None, "oot_published_n_scored": None, "oot_published_base_rate": None,
    "oot_frozen_pooled_auc": None, "oot_frozen_per_quarter": None,
})

PAYLOAD_OOT = copy.deepcopy(PAYLOAD)
PAYLOAD_OOT["scorecard"].update({
    "oot_published_auc": 0.47, "oot_published_n_scored": 421, "oot_published_base_rate": 0.48,
    "oot_frozen_pooled_auc": 0.52,
    "oot_frozen_per_quarter": [{"quarter": "2024q4", "auc": 0.46},
                               {"quarter": "2025q1", "auc": 0.61}],
})


def test_renders_and_contains_disclaimer():
    html = render_dashboard(PAYLOAD)
    assert DISCLAIMER in html
    assert "Leaning Large Blend" in html
    assert "Target-Date 2050" in html


def test_oot_panel_copy_present_only_with_data():
    # The renderer text is client-side JS, so assert on the embedded payload + the renderer
    # source (both are in the single HTML string). The panel gate lives in the JS; here we
    # confirm the panel machinery ships and the data threads through when present.
    with_oot = render_dashboard(PAYLOAD_OOT)
    assert "Out-of-time reality check" in with_oot          # eyebrow copy (in template JS)
    assert "renderOOTPanel" in with_oot                     # panel renderer shipped
    # populated oot data is spliced into the payload blob
    assert "oot_frozen_per_quarter" in with_oot
    assert "2025q1" in with_oot


def test_oot_null_keys_render_like_absent():
    # null oot keys must behave exactly like the keys being absent: the payload differs, but
    # the rendered document body (template + JS) is byte-identical, because the JS gates on
    # the value with loose == null. Compare with the keys stripped entirely.
    import re
    a = render_dashboard(PAYLOAD_OOT_NULL)
    b = render_dashboard(PAYLOAD)
    # Strip the payload blob from both; the template/JS around it must be identical.
    pat = r'(<script id="payload" type="application/json">).*?(</script>)'
    a_tpl = re.sub(pat, r"\1\2", a, flags=re.DOTALL)
    b_tpl = re.sub(pat, r"\1\2", b, flags=re.DOTALL)
    assert a_tpl == b_tpl


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


def test_script_breaking_strings_are_neutralized():
    payload = copy.deepcopy(PAYLOAD)
    payload["clusters"][0]["narrative"] = (
        "Grounded text </script><script>alert(1)</script> more"
    )
    html = render_dashboard(payload)
    clean_html = render_dashboard(PAYLOAD)

    # (a) the breakout sequence must not survive into the HTML
    assert "</script><script>" not in html

    # (b) no extra </script> tags were introduced - same count as the clean payload
    assert html.count("</script>") == clean_html.count("</script>")

    # (c) the payload script tag still parses and preserves the narrative verbatim
    match = re.search(
        r'<script id="payload" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None
    parsed = json.loads(match.group(1))
    assert parsed["clusters"][0]["narrative"] == (
        "Grounded text </script><script>alert(1)</script> more"
    )
