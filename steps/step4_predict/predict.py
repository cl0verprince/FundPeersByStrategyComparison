"""step4_predict — pooled, time-safe panel predicting next-quarter peer-underperformance,
via a decision tree then a random forest.

See steps/step4_predict/design.md for the two correctness risks this is designed around:
whole-period-metric leakage, and cluster ids not being comparable across quarters.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

from fundspeers.category import category_tier
from fundspeers.io import load_table, save_model, save_table

log = logging.getLogger(__name__)


def _sorted_monthly_returns(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    df = monthly_returns.copy()
    df["r"] = df["total_return"] / 100.0
    return df.sort_values(["series_id", "quarter", "month_in_quarter"]).reset_index(drop=True)


def compute_trailing_features(
    monthly_returns: pd.DataFrame, risk_free_annual: float, window_months: int = 12
) -> pd.DataFrame:
    """One row per (series_id, quarter): trailing_return/volatility/sharpe/max_drawdown
    computed ONLY from months up to and including that quarter (a trailing window of
    `window_months`, or fewer if the fund's history is shorter) - point-in-time-safe by
    construction, unlike fund_metrics_overall (step3), which uses the WHOLE history."""
    df = _sorted_monthly_returns(monthly_returns)
    rows = []
    for series_id, group in df.groupby("series_id"):
        months = group["r"].values
        quarters = group["quarter"].values
        n = len(months)
        for i in range(2, n, 3):  # i = index of the 3rd month of each quarter
            quarter = quarters[i]
            window = months[max(0, i - window_months + 1): i + 1]
            if len(window) < 2 or np.isnan(window).any():
                trailing_return = trailing_vol = trailing_sharpe = trailing_dd = np.nan
            else:
                trailing_return = np.prod(1 + window) - 1
                trailing_vol = window.std(ddof=1) * np.sqrt(12)
                trailing_mean_annual = window.mean() * 12
                trailing_sharpe = (
                    (trailing_mean_annual - risk_free_annual) / trailing_vol
                    if trailing_vol > 0 else np.nan
                )
                wealth = np.cumprod(1 + window)
                running_max = np.maximum.accumulate(wealth)
                trailing_dd = ((wealth - running_max) / running_max).min()
            rows.append({
                "series_id": series_id, "quarter": quarter,
                "trailing_return": trailing_return, "trailing_volatility": trailing_vol,
                "trailing_sharpe": trailing_sharpe, "trailing_max_drawdown": trailing_dd,
            })
    return pd.DataFrame(rows)


def assemble_panel(
    funds: pd.DataFrame, trailing_features: pd.DataFrame, fund_metrics_quarterly: pd.DataFrame,
    quarters_ordered: list,
) -> pd.DataFrame:
    """One row per (series_id, Q) with features known at Q and the label for Q+1. Rows with
    any NaN feature or an unknown (NaN) label are dropped - not imputed."""
    fund_static = funds.drop_duplicates("series_id")[["series_id", "yahoo_category"]].copy()
    fund_static["category_tier"] = fund_static["yahoo_category"].map(category_tier)

    funds_at_q = funds[["series_id", "quarter", "net_assets"]]
    relative_return_at_q = fund_metrics_quarterly[["series_id", "quarter", "return_vs_cluster_median"]].rename(
        columns={"return_vs_cluster_median": "return_vs_cluster_median_q"}
    )

    quarter_to_next = dict(zip(quarters_ordered[:-1], quarters_ordered[1:]))
    label_source = fund_metrics_quarterly[["series_id", "quarter", "return_vs_cluster_median"]].rename(
        columns={"quarter": "next_quarter", "return_vs_cluster_median": "return_vs_cluster_median_next"}
    )

    panel = trailing_features.merge(fund_static, on="series_id", how="left")
    panel = panel.merge(funds_at_q, on=["series_id", "quarter"], how="left")
    panel = panel.merge(relative_return_at_q, on=["series_id", "quarter"], how="left")
    panel["next_quarter"] = panel["quarter"].map(quarter_to_next)
    panel = panel[panel["next_quarter"].notna()]  # last quarter has no Q+1
    panel = panel.merge(label_source, on=["series_id", "next_quarter"], how="left")
    panel["underperform_next_quarter"] = (panel["return_vs_cluster_median_next"] < 0).astype("Int64")
    panel.loc[panel["return_vs_cluster_median_next"].isna(), "underperform_next_quarter"] = pd.NA

    feature_cols = ["trailing_return", "trailing_volatility", "trailing_sharpe",
                    "trailing_max_drawdown", "return_vs_cluster_median_q", "net_assets"]
    required = feature_cols + ["underperform_next_quarter"]
    before = len(panel)
    panel = panel.dropna(subset=required)
    dropped = before - len(panel)
    if dropped:
        log.warning(f"dropped {dropped} of {before} panel rows with a missing feature or label")

    tier_dummies = pd.get_dummies(panel["category_tier"], prefix="tier")
    panel = pd.concat([panel.reset_index(drop=True), tier_dummies.reset_index(drop=True)], axis=1)
    return panel


def time_based_split(panel: pd.DataFrame, quarters_ordered: list, holdout_transitions: int):
    """Train on the earliest transitions, test on the latest - by transition, not row, so no
    test-set quarter ends earlier in time than any train-set quarter."""
    transition_quarters = quarters_ordered[:-1]  # each has a valid Q+1
    test_quarters = set(transition_quarters[-holdout_transitions:])
    train_quarters = set(transition_quarters[:-holdout_transitions])
    train = panel[panel["quarter"].isin(train_quarters)]
    test = panel[panel["quarter"].isin(test_quarters)]
    return train, test, sorted(train_quarters), sorted(test_quarters)


def run(cfg: dict) -> None:
    risk_free_annual = cfg["metrics"]["risk_free_annual"]
    holdout_transitions = cfg["model"]["test_transitions_holdout"]
    seed = cfg["seed"]

    funds = load_table("funds", cfg)
    monthly_returns = load_table("monthly_returns", cfg)
    fund_metrics_quarterly = load_table("fund_metrics_quarterly", cfg)

    equity_series = set(funds.loc[funds["is_us_equity"], "series_id"].unique())
    equity_returns = monthly_returns[monthly_returns["series_id"].isin(equity_series)]
    equity_funds = funds[funds["series_id"].isin(equity_series)]

    quarters_ordered = sorted(equity_funds["quarter"].unique())
    log.info(f"{len(equity_series)} equity funds, {len(quarters_ordered)} quarters "
             f"-> {len(quarters_ordered) - 1} chronological (Q, Q+1) transitions")

    trailing_features = compute_trailing_features(equity_returns, risk_free_annual)
    panel = assemble_panel(equity_funds, trailing_features, fund_metrics_quarterly, quarters_ordered)
    log.info(f"assembled panel: {len(panel)} usable fund-quarter rows")

    train, test, train_quarters, test_quarters = time_based_split(panel, quarters_ordered, holdout_transitions)
    log.info(f"train quarters: {train_quarters}")
    log.info(f"test quarters: {test_quarters}")
    if max(train_quarters) >= min(test_quarters):
        raise RuntimeError(
            "train/test split leaks future data into training: "
            f"train up to {max(train_quarters)}, test from {min(test_quarters)}"
        )
    log.info(f"train rows: {len(train)}, test rows: {len(test)}")

    tier_cols = sorted(c for c in panel.columns if c.startswith("tier_"))
    feature_cols = ["trailing_return", "trailing_volatility", "trailing_sharpe",
                    "trailing_max_drawdown", "return_vs_cluster_median_q", "net_assets"] + tier_cols

    x_train, y_train = train[feature_cols], train["underperform_next_quarter"].astype(int)
    x_test, y_test = test[feature_cols], test["underperform_next_quarter"].astype(int)

    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    baseline_auc = roc_auc_score(y_test, baseline.predict_proba(x_test)[:, 1]) if y_test.nunique() > 1 else 0.5

    tree = DecisionTreeClassifier(
        max_depth=cfg["model"]["tree"]["max_depth"],
        min_samples_leaf=cfg["model"]["tree"]["min_samples_leaf"],
        random_state=seed,
    ).fit(x_train, y_train)
    tree_auc = roc_auc_score(y_test, tree.predict_proba(x_test)[:, 1])
    tree_acc = accuracy_score(y_test, tree.predict(x_test))

    rf = RandomForestClassifier(
        n_estimators=cfg["model"]["rf"]["n_estimators"],
        max_depth=cfg["model"]["rf"]["max_depth"],
        min_samples_leaf=cfg["model"]["rf"]["min_samples_leaf"],
        random_state=seed,
    ).fit(x_train, y_train)
    rf_proba = rf.predict_proba(x_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)
    rf_acc = accuracy_score(y_test, rf.predict(x_test))

    log.info(f"baseline (most-frequent) AUC: {baseline_auc:.3f}")
    log.info(f"decision tree: AUC={tree_auc:.3f}, accuracy={tree_acc:.3f}")
    log.info(f"random forest: AUC={rf_auc:.3f}, accuracy={rf_acc:.3f}")
    log.info("decision tree top splits:\n" + export_text(tree, feature_names=feature_cols, max_depth=3))

    importances = pd.DataFrame({
        "feature": feature_cols, "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    log.info(f"feature importances (sum={importances['importance'].sum():.3f}):\n"
             f"{importances.to_string(index=False)}")

    predictions = pd.concat([
        train.assign(split="train", predicted_probability=rf.predict_proba(x_train)[:, 1]),
        test.assign(split="test", predicted_probability=rf_proba),
    ])[["series_id", "quarter", "predicted_probability", "underperform_next_quarter", "split"]].rename(
        columns={"underperform_next_quarter": "actual_label"}
    )

    save_table(predictions, "fund_predictions", cfg)
    save_table(importances, "model_feature_importances", cfg)
    # Bundle the model with the exact feature column list it was trained on - tier dummy
    # columns depend on which category tiers are present in a given panel, so a frozen-model
    # evaluation on a different fund set (step6) must align to this list, not just reuse its
    # own pd.get_dummies() output blindly.
    save_model({"model": rf, "feature_cols": feature_cols}, "random_forest_model", cfg)
    log.info(f"saved fund_predictions ({len(predictions)} rows), "
             f"model_feature_importances ({len(importances)} rows), "
             f"random_forest_model.joblib ({len(feature_cols)} features)")
