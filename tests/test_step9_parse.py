"""Unit tests for step9_fees_turnover.parse — mapping RR num.tsv/sub.tsv rows to the
rr_fees_raw contract. A synthetic fixture ZIP built in-test matches the AMENDED schema
(design.md "Schema amendment (verified 2026-07-12)"); no network, no real ZIP needed.

Covers: column mapping, net-else-gross fallback + fee_source, the at-least-one-value row
filter, (adsh,series,class) dedup on max ddate, sanity-bound nulling (expense >0.10,
turnover >20), and the universe series_filter.
"""
import io
import zipfile

import numpy as np
import pandas as pd

from steps.step9_fees_turnover import parse


def _build_fixture_zip(path):
    """One family filing (A1, filed 2024-01-15) + a second (A2, filed 2024-06-20) with
    dedup + garbage cases. Values are FRACTIONS (as the SEC stores them)."""
    sub = (
        "adsh\tfiled\n"
        "A1\t20240115\n"
        "A2\t20240620\n"
    )
    cols = "adsh\ttag\tddate\tseries\tclass\tvalue\n"
    rows = [
        # S1/C1: net + gross both present -> net wins, source=net
        "A1\tNetExpensesOverAssets\t20231231\tS1\tC1\t0.0075",
        "A1\tExpensesOverAssets\t20231231\tS1\tC1\t0.0090",
        # S1/C2: gross only -> net falls back to gross, source=gross
        "A1\tExpensesOverAssets\t20231231\tS1\tC2\t0.0120",
        # S1 turnover: series-level (class empty)
        "A1\tPortfolioTurnoverRate\t20231231\tS1\t\t0.5500",
        # S2/C3: same-adsh dup on ddate -> keep max ddate (0.0060, not 0.0050)
        "A2\tNetExpensesOverAssets\t20221231\tS2\tC3\t0.0050",
        "A2\tNetExpensesOverAssets\t20231231\tS2\tC3\t0.0060",
        # S3/C4: garbage gross (6.30 > 0.10 bound) -> nulled -> row has no value -> excluded
        "A2\tExpensesOverAssets\t20231231\tS3\tC4\t6.3000",
        # S4 turnover garbage (>20 bound) -> nulled -> excluded
        "A2\tPortfolioTurnoverRate\t20231231\tS4\t\t44577.0",
        # S5/C5: empty value -> all missing -> excluded
        "A2\tNetExpensesOverAssets\t20231231\tS5\tC5\t",
    ]
    num = cols + "\n".join(rows) + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("sub.tsv", sub)
        z.writestr("num.tsv", num)
    path.write_bytes(buf.getvalue())


def _parse(tmp_path, series_filter=None):
    zp = tmp_path / "2024q1_rr1.zip"
    _build_fixture_zip(zp)
    return parse.parse_rr_quarter(zp, {}, series_filter=series_filter)


def test_columns_and_row_filter(tmp_path):
    df = _parse(tmp_path)
    assert set(df.columns) == {
        "series_id", "class_id", "filing_date", "expense_ratio_net",
        "expense_ratio_gross", "portfolio_turnover", "fee_source",
    }
    # S3 (garbage gross), S4 (garbage turnover), S5 (empty) all excluded.
    assert set(df["series_id"]) == {"S1", "S2"}


def test_net_else_gross_and_fee_source(tmp_path):
    df = _parse(tmp_path).set_index(["series_id", "class_id"])
    c1 = df.loc[("S1", "C1")]
    assert c1["expense_ratio_net"] == 0.0075          # net wins
    assert c1["expense_ratio_gross"] == 0.0090
    assert c1["fee_source"] == "net"
    c2 = df.loc[("S1", "C2")]
    assert c2["expense_ratio_net"] == 0.0120          # fell back to gross
    assert c2["expense_ratio_gross"] == 0.0120
    assert c2["fee_source"] == "gross"


def test_turnover_is_series_level_and_filing_date(tmp_path):
    df = _parse(tmp_path)
    turn = df[df["portfolio_turnover"].notna()]
    assert len(turn) == 1
    row = turn.iloc[0]
    assert row["series_id"] == "S1"
    assert row["portfolio_turnover"] == 0.55
    assert (row["class_id"] is None) or (row["class_id"] == "") or pd.isna(row["class_id"])
    assert row["filing_date"] == "2024-01-15"
    # expense rows carry no turnover
    assert df[df["class_id"] == "C1"].iloc[0]["filing_date"] == "2024-01-15"


def test_dedup_keeps_max_ddate(tmp_path):
    df = _parse(tmp_path).set_index(["series_id", "class_id"])
    s2 = df.loc[("S2", "C3")]
    assert s2["expense_ratio_net"] == 0.0060          # 20231231 wins over 20221231
    assert s2["filing_date"] == "2024-06-20"


def test_sanity_bounds_exclude_garbage(tmp_path):
    df = _parse(tmp_path)
    # no expense above the 0.10 bound, no turnover above the 20 bound survived
    assert (df["expense_ratio_net"].dropna() <= 0.10).all()
    assert (df["expense_ratio_gross"].dropna() <= 0.10).all()
    assert (df["portfolio_turnover"].dropna() <= 20).all()


def test_series_filter_restricts_universe(tmp_path):
    df = _parse(tmp_path, series_filter={"S1"})
    assert set(df["series_id"]) == {"S1"}
