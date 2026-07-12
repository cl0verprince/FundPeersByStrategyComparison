"""Unit tests for step9_fees_turnover.fees — the pure point-in-time join and the
class->series (mean-across-classes) collapse. No DB, no network: synthetic rr_raw frames.

The point-in-time rule (design decision 2) is the whole reason this step avoids look-ahead:
for (series, quarter) take the LAST filing dated <= quarter end, NEVER a later one.
"""
import numpy as np
import pandas as pd

from steps.step9_fees_turnover import fees


def _raw(rows):
    cols = ["series_id", "class_id", "filing_date", "expense_ratio_net",
            "expense_ratio_gross", "portfolio_turnover", "fee_source"]
    return pd.DataFrame(rows, columns=cols)


QUARTERS = ["2022q1", "2022q2", "2022q3", "2022q4", "2023q1", "2023q2"]


def _sample_raw():
    return _raw([
        # S1, filing 2022-02-15: two classes -> mean net = (0.01 + 0.03)/2 = 0.02
        ["S1", "C1", "2022-02-15", 0.01, 0.01, np.nan, "net"],
        ["S1", "C2", "2022-02-15", 0.03, 0.03, np.nan, "net"],
        ["S1", None, "2022-02-15", np.nan, np.nan, 0.40, None],
        # S1, later filing 2022-08-10: single class net 0.02
        ["S1", "C1", "2022-08-10", 0.02, 0.02, np.nan, "net"],
        ["S1", None, "2022-08-10", np.nan, np.nan, 0.60, None],
        # S2: only ever a 2023-05-01 filing
        ["S2", "C9", "2023-05-01", 0.015, 0.015, np.nan, "net"],
    ])


def test_quarter_end_mapping():
    assert fees.quarter_end("2022q1") == "2022-03-31"
    assert fees.quarter_end("2022q2") == "2022-06-30"
    assert fees.quarter_end("2022q3") == "2022-09-30"
    assert fees.quarter_end("2024q4") == "2024-12-31"


def test_picks_latest_filing_and_mean_collapse():
    out = fees.point_in_time_fees(_sample_raw(), ["S1", "S2"], QUARTERS)
    s1q1 = out[(out["series_id"] == "S1") & (out["quarter"] == "2022q1")].iloc[0]
    assert s1q1["expense_ratio_net"] == 0.02             # mean across C1,C2
    assert s1q1["source_filing_date"] == "2022-02-15"
    assert s1q1["portfolio_turnover"] == 0.40


def test_never_picks_a_later_filing():
    out = fees.point_in_time_fees(_sample_raw(), ["S1", "S2"], QUARTERS)
    # S2's only filing is 2023-05-01 -> NaN for everything up to and including 2023q1.
    early = out[(out["series_id"] == "S2")
                & out["quarter"].isin(["2022q1", "2022q2", "2022q3", "2022q4", "2023q1"])]
    assert early["expense_ratio_net"].isna().all()
    assert early["source_filing_date"].isna().all()
    s2q2 = out[(out["series_id"] == "S2") & (out["quarter"] == "2023q2")].iloc[0]
    assert s2q2["expense_ratio_net"] == 0.015
    assert s2q2["source_filing_date"] == "2023-05-01"


def test_source_filing_never_after_quarter_end():
    out = fees.point_in_time_fees(_sample_raw(), ["S1", "S2"], QUARTERS)
    got = out[out["source_filing_date"].notna()].copy()
    ends = got["quarter"].map(fees.quarter_end)
    assert (pd.to_datetime(got["source_filing_date"]) <= pd.to_datetime(ends)).all()


def test_carries_forward_until_superseded():
    out = fees.point_in_time_fees(_sample_raw(), ["S1", "S2"], QUARTERS).set_index("quarter")
    s1 = out[out["series_id"] == "S1"]
    # 2022q1 and 2022q2 both served by the 2022-02-15 filing (carried forward)
    assert s1.loc["2022q1", "source_filing_date"] == "2022-02-15"
    assert s1.loc["2022q2", "source_filing_date"] == "2022-02-15"
    # 2022q3 onward served by the newer 2022-08-10 filing
    assert s1.loc["2022q3", "source_filing_date"] == "2022-08-10"
    assert s1.loc["2022q3", "expense_ratio_net"] == 0.02
    assert s1.loc["2022q3", "portfolio_turnover"] == 0.60


def test_one_row_per_series_quarter():
    out = fees.point_in_time_fees(_sample_raw(), ["S1", "S2"], QUARTERS)
    assert len(out) == len(QUARTERS) * 2
    assert not out.duplicated(["series_id", "quarter"]).any()
