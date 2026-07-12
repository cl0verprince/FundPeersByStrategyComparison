"""step9_fees_turnover.fees — point-in-time expense/turnover per (series, quarter) from
rr_fees_raw, plus the coverage report that picks the comparison framing.

Two design rules made concrete here:
- Point-in-time (design decision 2): for (series, quarter) use the values from the most
  recent filing dated ON OR BEFORE the quarter's end; never a later filing (that would be
  the exact look-ahead this step exists to avoid). Every row keeps `source_filing_date`.
- Class->series (design amendment): MEAN across share classes. The panel's label predicts
  the mean-across-classes series return (step1's `_aggregate_class_returns_to_series`), so
  the fee must describe the same entity. This mirrors step1's RETURN aggregation, not its
  lowest-classId ticker rule; it needs no net-assets, so the plan's "class with most fee
  history" fallback (which only fires when net-assets are required) does not apply. To switch
  to a representative class, change `_collapse_classes` only.
"""
import logging

import numpy as np
import pandas as pd

from fundspeers.io import load_table, save_table

log = logging.getLogger(__name__)

_METRIC_COLS = ["expense_ratio_net", "expense_ratio_gross", "portfolio_turnover"]


def quarter_end(quarter: str) -> str:
    """'2022q1' -> '2022-03-31' (calendar quarter end, the point-in-time anchor)."""
    year, q = quarter.split("q")
    return {"1": f"{year}-03-31", "2": f"{year}-06-30",
            "3": f"{year}-09-30", "4": f"{year}-12-31"}[q]


def _collapse_classes(rr_raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse (series, class, filing) -> (series, filing): MEAN across classes of each
    metric (skipping NaN). Turnover is already series-level (one row, class NaN) so its mean
    is itself."""
    g = (rr_raw.groupby(["series_id", "filing_date"], as_index=False)[_METRIC_COLS]
         .mean())  # pandas mean skips NaN per column
    return g


def point_in_time_fees(rr_raw: pd.DataFrame, series_ids, quarters) -> pd.DataFrame:
    """One row per (series_id in `series_ids`, quarter in `quarters`) with the fee/turnover
    values from the most recent filing dated <= quarter end (NaN if none). Pure and
    unit-tested."""
    collapsed = _collapse_classes(rr_raw)
    collapsed = collapsed[collapsed["filing_date"].notna()].copy()
    collapsed["_fdate"] = pd.to_datetime(collapsed["filing_date"])
    collapsed = collapsed.sort_values("_fdate")

    series_ids = sorted(series_ids)
    target = pd.DataFrame(
        [(s, q) for q in quarters for s in series_ids], columns=["series_id", "quarter"])
    target["_qend"] = pd.to_datetime(target["quarter"].map(quarter_end))
    target = target.sort_values("_qend")

    merged = pd.merge_asof(
        target, collapsed, left_on="_qend", right_on="_fdate", by="series_id",
        direction="backward")

    merged["source_filing_date"] = merged["filing_date"]
    out = merged[["series_id", "quarter", *_METRIC_COLS, "source_filing_date"]].copy()

    # Safety assert: no source filing is dated after its quarter end.
    got = out[out["source_filing_date"].notna()]
    if len(got):
        ends = pd.to_datetime(got["quarter"].map(quarter_end))
        assert (pd.to_datetime(got["source_filing_date"]).values <= ends.values).all(), \
            "look-ahead: a source_filing_date is after its quarter end"
    return out.sort_values(["series_id", "quarter"]).reset_index(drop=True)


def build_rr_fees(cfg: dict) -> dict:
    """Build `rr_fees` (point-in-time, one row per strategy series x panel quarter) from
    `rr_fees_raw` and report coverage against the labeled `_full` panel. Returns coverage
    stats and the branch taken (>=/< the 0.80 gate)."""
    from steps.step7_unified_universe.panel import assemble_unified_panel

    rr_raw = load_table("rr_fees_raw", cfg)
    funds = load_table("funds_full", cfg)
    strategy = funds[funds["is_us_equity"] & (funds["segment"] == "strategy")]
    series_ids = sorted(strategy["series_id"].unique())
    quarters = cfg["data"]["quarters"]

    rr_fees = point_in_time_fees(rr_raw, series_ids, quarters)
    save_table(rr_fees, "rr_fees", cfg)

    # Coverage vs the labeled panel (the denominators that matter for the comparison).
    labeled, _forward, _cols = assemble_unified_panel(cfg, table_suffix="_full")
    n_labeled = len(labeled)
    n_funds = len(series_ids)

    has_expense = rr_fees[rr_fees["expense_ratio_net"].notna()]
    has_turnover = rr_fees[rr_fees["portfolio_turnover"].notna()]
    funds_with_expense = has_expense["series_id"].nunique()
    funds_with_turnover = has_turnover["series_id"].nunique()

    # Row-level coverage: fraction of the labeled (series,quarter) rows that get a value.
    key = labeled[["series_id", "quarter"]].merge(
        rr_fees, on=["series_id", "quarter"], how="left")
    rows_expense = int(key["expense_ratio_net"].notna().sum())
    rows_turnover = int(key["portfolio_turnover"].notna().sum())
    rows_both = int((key["expense_ratio_net"].notna() & key["portfolio_turnover"].notna()).sum())
    rows_either = int((key["expense_ratio_net"].notna() | key["portfolio_turnover"].notna()).sum())

    expense_row_cov = rows_expense / n_labeled if n_labeled else 0.0
    gate = cfg["fees"]["coverage_gate"]
    branch = "single-model common-covered-subset" if expense_row_cov >= gate else "dual-model with attrition"

    stats = {
        "n_strategy_funds": n_funds,
        "n_labeled_rows": n_labeled,
        "funds_with_expense": funds_with_expense,
        "funds_with_turnover": funds_with_turnover,
        "fund_expense_coverage": funds_with_expense / n_funds if n_funds else 0.0,
        "fund_turnover_coverage": funds_with_turnover / n_funds if n_funds else 0.0,
        "rows_expense": rows_expense,
        "rows_turnover": rows_turnover,
        "rows_both": rows_both,
        "rows_either": rows_either,
        "expense_row_coverage": expense_row_cov,
        "turnover_row_coverage": rows_turnover / n_labeled if n_labeled else 0.0,
        "coverage_gate": gate,
        "branch": branch,
    }
    log.info(
        "rr_fees coverage | funds: expense %d/%d (%.1f%%), turnover %d/%d (%.1f%%) | "
        "labeled rows: expense %d/%d (%.1f%%), turnover %.1f%%, both %.1f%% | gate %.2f -> %s",
        funds_with_expense, n_funds, 100 * stats["fund_expense_coverage"],
        funds_with_turnover, n_funds, 100 * stats["fund_turnover_coverage"],
        rows_expense, n_labeled, 100 * expense_row_cov,
        100 * stats["turnover_row_coverage"], 100 * rows_both / n_labeled, gate, branch)
    return stats
