# step3_metrics — design

## Purpose (traces to Required Output)
From `monthly_returns`, compute standard performance/risk metrics for every US-equity fund,
and each fund's return relative to its own quarter's cluster (from step2) — the features step4's
prediction model will consume, and a deliverable in its own right per the Required Output.

## Scope
All 363 US-equity funds, all 12 quarters. Metrics are computed at two grains:
1. **Whole-period metrics** (one row per fund): cumulative return, annualized volatility, Sharpe
   ratio, max drawdown — computed over the fund's full 36-month return history.
2. **Per-quarter metrics** (one row per fund per quarter): that quarter's total return (compounded
   from its 3 monthly returns) and its return relative to its cluster's median return that
   quarter. This is the grain step4 needs for its pooled "features at Q -> label at Q+1" panel.

## Inputs
- `monthly_returns.parquet` (`series_id`, `quarter`, `month_in_quarter`, `total_return` - percent,
  e.g. `1.5` means +1.5% for that month).
- `fund_clusters.parquet` (`series_id`, `quarter`, `cluster_id`) from step2.
- `funds.parquet` for `is_us_equity` filtering.

## Metric definitions
- **Monthly return as a fraction:** `total_return / 100`.
- **Quarterly compounded return:** `(1+r1)*(1+r2)*(1+r3) - 1` from the quarter's 3 monthly returns.
- **Cumulative return (whole period):** compound all 36 monthly returns:
  `prod(1 + r_i) - 1` over the fund's full history.
- **Annualized volatility:** `std(monthly returns) * sqrt(12)` (sample std, ddof=1).
- **Sharpe ratio:** `(annualized mean return - risk_free_annual) / annualized volatility`, using
  `metrics.risk_free_annual` from config (already set to 0.02 in step0). Annualized mean return =
  `mean(monthly returns) * 12` (simple annualization, consistent with the volatility calc - not
  geometric, since this is a straightforward first-pass metric, not a precise industry figure).
- **Max drawdown:** build a cumulative wealth index from monthly returns
  (`cumprod(1 + r)`), then `min((wealth - running_max(wealth)) / running_max(wealth))` - the
  worst peak-to-trough decline over the 36-month history, expressed as a negative fraction.
- **Return vs. cluster median (per quarter):** `fund's quarterly return - median(quarterly
  return of all funds in its cluster that quarter)`. This is both a reportable metric AND the
  literal basis for step4's label ("underperform" = this value is negative, per the plan's
  locked decision).

## Real bug caught: `np.prod()` on a pandas Series silently skips NaN
`compute_quarterly_returns` originally computed each quarter's product via
`group["r"].apply(lambda s: np.prod(1 + s) - 1)`, passing a **pandas Series** to `np.prod`.
Confirmed on real data (2 funds have one unparseable/missing monthly return each, from
step1's `pd.to_numeric(errors="coerce")`): `np.prod()` on a Series dispatches to
`Series.prod(skipna=True)` and silently drops the NaN, producing a plausible-looking but
wrong 2-of-3-month product for that quarter - while `compute_overall_metrics` (which already
used `.values`, a plain numpy array) correctly propagated NaN for the same funds. This
inconsistency was only caught by cross-checking `.describe()` counts between the two output
tables (361 vs. 363) and tracing the discrepancy. Fixed by using `s.values` in the quarterly
calc too, with a regression test (`test_quarterly_return_with_a_missing_month_is_nan...`).
**A missing month correctly makes that whole quarter's return (and cluster-relative metrics)
NaN** - not silently computed from the other two months.

## Handling the two funds missing from a quarter's clustering (step2 finding)
A small number of fund-quarters have zero EC holdings and were excluded from step2's clustering
(logged there: e.g. `S000036624`, `S000054127`, `S000003656` in specific quarters). For those
fund-quarters, `cluster_id` is null and "return vs. cluster median" cannot be computed - left as
`NaN` for that row rather than guessing a cluster. Whole-period metrics (cumulative return,
volatility, Sharpe, max drawdown) are unaffected since they don't depend on cluster membership.

## Outputs
- **`fund_metrics_overall.parquet`**: `series_id`, `cumulative_return`, `annualized_volatility`,
  `sharpe_ratio`, `max_drawdown`.
- **`fund_metrics_quarterly.parquet`**: `series_id`, `quarter`, `quarterly_return`, `cluster_id`,
  `cluster_median_return`, `return_vs_cluster_median`.

## Determinism
Pure arithmetic over already-persisted tables - no randomness, no external calls.

## UAT (acceptance for this step) - all confirmed on real data
- Metrics computed: `fund_metrics_overall` (363 rows, one per fund) and `fund_metrics_quarterly`
  (4356 rows = 363 funds x 12 quarters). No unexpected nulls: 2 funds are NaN in
  `fund_metrics_overall` (and in their 2022q1 quarterly row) from one genuinely unparseable
  monthly return each (traced, not a bug); 7 fund-quarters are NaN in `return_vs_cluster_median`
  from step2's documented zero-EC-holdings exclusion. `9 = 7 + 2` distinct rows, no overlap.
- Spot-check: hand-recomputed cumulative return, annualized volatility, and max drawdown for a
  real fund from its raw 36 monthly returns - matched the table exactly (to floating-point
  precision).
- Sanity check on real data: no `annualized_volatility <= 0`; all `max_drawdown` in `[-1, 0]`;
  mean Sharpe 0.248, mean max drawdown -25.9% - plausible for equity funds over a window
  including 2022's bear market.
