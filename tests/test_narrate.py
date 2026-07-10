"""Unit tests for step5_narrate's pure formatting logic - no network, no LM Studio call.
build_context/narrate_fund/run() need a live database and LLM server, exercised manually
per the design doc's UAT, not here."""
import numpy as np

from steps.step5_narrate.narrate import _fmt_num, _fmt_pct, format_context_as_text


def _base_context(**overrides):
    context = {
        "series_id": "S1", "quarter": "2024q4", "series_name": "Example Fund",
        "ticker": "EXFND", "yahoo_category": "Large Blend", "net_assets": 5_000_000.0,
        "cluster_id": 3, "cluster_purity": 0.41, "cluster_ari": 0.25,
        "peers": [{"rank": 1, "name": "Similar Fund", "category": "Large Blend", "similarity": 0.9}],
        "cumulative_return": 0.15, "annualized_volatility": 0.18, "sharpe_ratio": 0.3,
        "max_drawdown": -0.22, "quarterly_return": 0.02, "return_vs_cluster_median": -0.01,
        "predicted_probability": 0.65, "actual_label": 1,
    }
    context.update(overrides)
    return context


def test_fmt_helpers_handle_none_and_nan():
    assert _fmt_pct(None) == "unavailable"
    assert _fmt_pct(np.nan) == "unavailable"
    assert _fmt_pct(0.05) == "5.0%"
    assert _fmt_num(None) == "unavailable"
    assert _fmt_num(1.23456, digits=2) == "1.23"


def test_format_includes_key_retrieved_facts_verbatim():
    text = format_context_as_text(_base_context())
    assert "Example Fund" in text
    assert "EXFND" in text
    assert "cluster 3" in text
    assert "Similar Fund" in text
    assert "65.0%" in text  # predicted probability
    assert "did underperform" in text  # actual_label == 1


def test_format_reports_missing_cluster_instead_of_fabricating():
    text = format_context_as_text(_base_context(cluster_id=None))
    assert "unavailable" in text.lower()
    assert "Nearest peers" not in text  # no peer list without a resolved cluster


def test_format_reports_missing_prediction_instead_of_fabricating():
    text = format_context_as_text(_base_context(predicted_probability=None, actual_label=None))
    assert "unavailable (this fund-quarter is outside the model's evaluated panel)" in text
    assert "predicted probability of underperforming" not in text.lower()
