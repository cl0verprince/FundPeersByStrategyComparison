"""step3_metrics — performance/risk metrics per fund, and each fund's return relative to
its quarter's cluster median (the basis for step4's underperformance label).

See steps/step3_metrics/design.md for metric definitions and the missing-cluster handling.
"""
import logging

import numpy as np
import pandas as pd

from fundspeers.io import load_table, save_table

log = logging.getLogger(__name__)


def _sorted_monthly_returns(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    df = monthly_returns.copy()
    df["r"] = df["total_return"] / 100.0
    return df.sort_values(["series_id", "quarter", "month_in_quarter"])


def compute_overall_metrics(monthly_returns: pd.DataFrame, risk_free_annual: float) -> pd.DataFrame:
    """One row per series_id: cumulative return, annualized volatility, Sharpe, max drawdown
    over the fund's full monthly-return history."""
    df = _sorted_monthly_returns(monthly_returns)
    rows = []
    for series_id, group in df.groupby("series_id"):
        r = group["r"].values
        cumulative_return = np.prod(1 + r) - 1
        annualized_vol = r.std(ddof=1) * np.sqrt(12)
        annualized_mean = r.mean() * 12
        sharpe = (annualized_mean - risk_free_annual) / annualized_vol if annualized_vol > 0 else np.nan
        wealth = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(wealth)
        drawdown = (wealth - running_max) / running_max
        rows.append({
            "series_id": series_id,
            "cumulative_return": cumulative_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": drawdown.min(),
        })
    return pd.DataFrame(rows)


def compute_quarterly_returns(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """One row per (series_id, quarter): that quarter's compounded return from its 3
    monthly returns. A missing month correctly makes the whole quarter's return NaN
    (not silently computed from the other 2) - np.prod on a pandas Series dispatches to
    Series.prod(skipna=True) and silently ignores NaN, so .values (a plain numpy array)
    is required here for correct propagation; this was caught as a real bug by comparing
    against compute_overall_metrics, which already used .values and behaved correctly."""
    df = _sorted_monthly_returns(monthly_returns)
    grouped = df.groupby(["series_id", "quarter"])["r"].apply(lambda s: np.prod(1 + s.values) - 1)
    return grouped.reset_index().rename(columns={"r": "quarterly_return"})


def compute_cluster_relative_metrics(
    quarterly_returns: pd.DataFrame, fund_clusters: pd.DataFrame
) -> pd.DataFrame:
    """Join in cluster_id, then each fund's quarterly return minus its cluster's median
    return that quarter. Fund-quarters with no resolved cluster (step2's documented
    zero-EC-holdings edge case) get NaN here, not a spurious cross-fund grouping -
    pandas groupby excludes NaN group keys by default, so transform() correctly leaves
    those rows unassigned rather than grouping them together."""
    merged = quarterly_returns.merge(fund_clusters, on=["series_id", "quarter"], how="left")
    merged["cluster_median_return"] = merged.groupby(["quarter", "cluster_id"])[
        "quarterly_return"
    ].transform("median")
    merged["return_vs_cluster_median"] = (
        merged["quarterly_return"] - merged["cluster_median_return"]
    )
    return merged


def run(cfg: dict) -> None:
    risk_free_annual = cfg["metrics"]["risk_free_annual"]

    funds = load_table("funds", cfg)
    monthly_returns = load_table("monthly_returns", cfg)
    fund_clusters = load_table("fund_clusters", cfg)

    equity_series = set(funds.loc[funds["is_us_equity"], "series_id"].unique())
    equity_returns = monthly_returns[monthly_returns["series_id"].isin(equity_series)]

    overall = compute_overall_metrics(equity_returns, risk_free_annual)
    quarterly_returns = compute_quarterly_returns(equity_returns)
    quarterly = compute_cluster_relative_metrics(quarterly_returns, fund_clusters)

    missing_cluster = quarterly["cluster_id"].isna().sum()
    if missing_cluster:
        log.warning(f"{missing_cluster} fund-quarter(s) have no resolved cluster "
                    f"(step2's zero-EC-holdings edge case) - cluster-relative metrics are NaN")

    save_table(overall, "fund_metrics_overall", cfg)
    save_table(quarterly, "fund_metrics_quarterly", cfg)
    log.info(f"saved fund_metrics_overall ({len(overall)} funds), "
             f"fund_metrics_quarterly ({len(quarterly)} fund-quarters)")
    log.info(f"overall: mean Sharpe={overall['sharpe_ratio'].mean():.3f}, "
             f"mean max_drawdown={overall['max_drawdown'].mean():.3f}")
