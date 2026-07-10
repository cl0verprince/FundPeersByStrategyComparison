"""step3_metrics — performance/risk metrics per fund, and each fund's return relative to
its quarter's cluster median (the basis for step4's underperformance label).

See steps/step3_metrics/design.md for metric definitions and the missing-cluster handling.
"""
import logging

import numpy as np
import pandas as pd

from fundspeers.category import category_tier
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


def _concentration_word(dominant_category_share: float) -> str:
    if dominant_category_share >= 0.70:
        return "Concentrated"
    if dominant_category_share >= 0.40:
        return "Leaning"
    return "Mixed"


def compute_cluster_definitions(
    fund_clusters: pd.DataFrame, funds: pd.DataFrame, fund_metrics_overall: pd.DataFrame
) -> pd.DataFrame:
    """One row per (quarter, cluster_id): a deterministic, rule-based description of that
    cluster's membership - not an LLM-generated one (step5 narrates individual funds; this
    is a hard, reproducible definition computed straight from already-verified data).

    `short_title` is allocation-only by design: clusters are formed from holdings/allocation
    similarity (step2), not performance, so the identifying name is built purely from
    `dominant_category_share` (how homogeneous the cluster is) and `dominant_category` -
    deliberately excludes avg_sharpe/avg_volatility, which are outcomes correlated with a
    cluster's composition, not what defines it. Baking a performance stat into the name would
    also collide with "underperform," which already has a specific, different meaning
    elsewhere in this pipeline (a fund's return relative to its OWN cluster's median) - a
    cluster can't be described as "underperforming" relative to itself. The longer `title`
    still reports avg_volatility/avg_sharpe separately, as descriptive stats, not identity."""
    category_by_fund_quarter = funds[["series_id", "quarter", "yahoo_category"]].drop_duplicates()
    df = fund_clusters.merge(category_by_fund_quarter, on=["series_id", "quarter"], how="left")
    df["tier"] = df["yahoo_category"].map(category_tier)
    df = df.merge(fund_metrics_overall, on="series_id", how="left")

    rows = []
    for (quarter, cluster_id), group in df.groupby(["quarter", "cluster_id"]):
        member_count = len(group)
        dominant_category = group["yahoo_category"].mode().iloc[0]
        dominant_category_share = (group["yahoo_category"] == dominant_category).mean()
        dominant_tier = group["tier"].mode().iloc[0]
        avg_volatility = group["annualized_volatility"].mean()
        avg_sharpe = group["sharpe_ratio"].mean()
        title = (
            f"{dominant_tier} tilt: {dominant_category_share:.0%} {dominant_category} "
            f"({member_count} funds), avg volatility {avg_volatility:.0%}, "
            f"avg Sharpe {avg_sharpe:.2f}"
        )
        short_title = f"{_concentration_word(dominant_category_share)} {dominant_category}"
        rows.append({
            "quarter": quarter, "cluster_id": cluster_id, "member_count": member_count,
            "dominant_category": dominant_category,
            "dominant_category_share": dominant_category_share,
            "dominant_tier": dominant_tier, "avg_volatility": avg_volatility,
            "avg_sharpe": avg_sharpe, "title": title, "short_title": short_title,
        })
    return pd.DataFrame(rows)


def run(cfg: dict, table_suffix: str = "") -> None:
    """`table_suffix` (defaults to the original behavior) reads/writes a parallel table
    namespace (e.g. "_oos") - see steps/step6_out_of_sample/design.md."""
    risk_free_annual = cfg["metrics"]["risk_free_annual"]

    funds = load_table(f"funds{table_suffix}", cfg)
    monthly_returns = load_table(f"monthly_returns{table_suffix}", cfg)
    fund_clusters = load_table(f"fund_clusters{table_suffix}", cfg)

    equity_series = set(funds.loc[funds["is_us_equity"], "series_id"].unique())
    equity_returns = monthly_returns[monthly_returns["series_id"].isin(equity_series)]

    overall = compute_overall_metrics(equity_returns, risk_free_annual)
    quarterly_returns = compute_quarterly_returns(equity_returns)
    quarterly = compute_cluster_relative_metrics(quarterly_returns, fund_clusters)

    missing_cluster = quarterly["cluster_id"].isna().sum()
    if missing_cluster:
        log.warning(f"{missing_cluster} fund-quarter(s) have no resolved cluster "
                    f"(step2's zero-EC-holdings edge case) - cluster-relative metrics are NaN")

    cluster_definitions = compute_cluster_definitions(fund_clusters, funds, overall)

    save_table(overall, f"fund_metrics_overall{table_suffix}", cfg)
    save_table(quarterly, f"fund_metrics_quarterly{table_suffix}", cfg)
    save_table(cluster_definitions, f"cluster_definitions{table_suffix}", cfg)
    log.info(f"saved fund_metrics_overall{table_suffix} ({len(overall)} funds), "
             f"fund_metrics_quarterly{table_suffix} ({len(quarterly)} fund-quarters), "
             f"cluster_definitions{table_suffix} ({len(cluster_definitions)} quarter-cluster combos)")
    log.info(f"overall: mean Sharpe={overall['sharpe_ratio'].mean():.3f}, "
             f"mean max_drawdown={overall['max_drawdown'].mean():.3f}")
