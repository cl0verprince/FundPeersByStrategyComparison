"""Unit tests for step1_ingest's pure logic, using small synthetic data - no network,
no real N-PORT zips. Covers the two real bugs found while building this step:
duplicate same-quarter filings, and the geography-blind equity flag."""
import pandas as pd

from steps.step1_ingest.ingest import (
    _aggregate_class_returns_to_series,
    _combine_us_equity_flag,
    _compute_holdings_us_equity_share,
    _dedupe_one_filing_per_series,
)


def test_dedupe_keeps_one_row_per_series_deterministically():
    frame = pd.DataFrame({
        "ACCESSION_NUMBER": ["0002", "0001", "0003"],
        "SERIES_ID": ["S1", "S1", "S2"],
    })
    result = _dedupe_one_filing_per_series(frame)
    assert sorted(result["SERIES_ID"]) == ["S1", "S2"]
    # lowest ACCESSION_NUMBER wins for the duplicated series
    assert result.loc[result["SERIES_ID"] == "S1", "ACCESSION_NUMBER"].iloc[0] == "0001"


def test_aggregate_class_returns_averages_across_classes():
    returns = pd.DataFrame({
        "SERIES_ID": ["S1", "S1"],
        "MONTHLY_TOTAL_RETURN1": ["1.0", "3.0"],
        "MONTHLY_TOTAL_RETURN2": ["2.0", "4.0"],
        "MONTHLY_TOTAL_RETURN3": ["0.0", "0.0"],
    })
    result = _aggregate_class_returns_to_series(returns, quarter="2024q1")
    assert len(result) == 3  # one row per month_in_quarter for the single series
    month1 = result[result["month_in_quarter"] == 1].iloc[0]
    month2 = result[result["month_in_quarter"] == 2].iloc[0]
    assert month1["total_return"] == 2.0  # mean(1.0, 3.0)
    assert month2["total_return"] == 3.0  # mean(2.0, 4.0)
    assert (result["quarter"] == "2024q1").all()


def test_holdings_us_equity_share_weights_by_value():
    holdings = pd.DataFrame({
        "accession_number": ["A1", "A1", "A1"],
        "asset_cat": ["EC", "EC", "DBT"],
        "investment_country": ["US", "FR", "US"],
        "currency_value": [70.0, 20.0, 10.0],
    })
    result = _compute_holdings_us_equity_share(holdings)
    # only the first row (US, EC, 70) counts as US equity value; denominator is all 100
    assert result.loc[result["accession_number"] == "A1", "holdings_us_equity_share"].iloc[0] == 0.70


def test_combine_us_equity_flag_requires_both_yahoo_and_holdings_signals():
    funds_static = pd.DataFrame({
        "series_id": ["S_domestic", "S_international", "S_bond"],
        "yahoo_is_equity": [True, True, False],
    })
    holdings = pd.DataFrame({
        "accession_number": ["A_dom", "A_intl", "A_bond"],
        "asset_cat": ["EC", "EC", "DBT"],
        "investment_country": ["US", "FR", "US"],
        "currency_value": [100.0, 100.0, 100.0],
    })
    accession_to_series = pd.DataFrame({
        "accession_number": ["A_dom", "A_intl", "A_bond"],
        "series_id": ["S_domestic", "S_international", "S_bond"],
    })
    result = _combine_us_equity_flag(funds_static, holdings, accession_to_series, us_holdings_share_min=0.70)
    flags = result.set_index("series_id")["is_us_equity"]
    assert flags["S_domestic"] is True or flags["S_domestic"] == True  # noqa: E712
    assert flags["S_international"] == False  # noqa: E712 - equity but not US-domiciled
    assert flags["S_bond"] == False  # noqa: E712 - US-domiciled but not equity
