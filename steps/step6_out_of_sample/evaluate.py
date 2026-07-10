"""step6_out_of_sample — test whether step4's model generalizes to funds it has never seen.

Two evaluations, answering two different questions (see design.md):
1. The FROZEN step4 model applied to 1000 new, disjoint funds - does it generalize?
2. A model RETRAINED on the combined original + new funds - does more data help?
"""
import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from fundspeers.io import load_model, load_table
from steps.step4_predict.predict import assemble_panel, compute_trailing_features, time_based_split

log = logging.getLogger(__name__)


def build_panel_for(table_suffix: str, cfg: dict):
    """Assemble step4's feature/label panel for a given table namespace ("" = original
    training tables, "_oos" = the disjoint out-of-sample fund set)."""
    funds = load_table(f"funds{table_suffix}", cfg)
    monthly_returns = load_table(f"monthly_returns{table_suffix}", cfg)
    fund_metrics_quarterly = load_table(f"fund_metrics_quarterly{table_suffix}", cfg)
    risk_free_annual = cfg["metrics"]["risk_free_annual"]

    equity_series = set(funds.loc[funds["is_us_equity"], "series_id"].unique())
    equity_returns = monthly_returns[monthly_returns["series_id"].isin(equity_series)]
    equity_funds = funds[funds["series_id"].isin(equity_series)]
    quarters_ordered = sorted(equity_funds["quarter"].unique())

    trailing_features = compute_trailing_features(equity_returns, risk_free_annual)
    panel = assemble_panel(equity_funds, trailing_features, fund_metrics_quarterly, quarters_ordered)
    return panel, quarters_ordered


def align_features(panel: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Reindex a panel's feature columns to match a frozen model's exact training columns,
    in the same order. Tier dummy columns depend on which category tiers are present in a
    given panel - a tier missing from this panel becomes an all-zero column (correct: it
    means no fund here belongs to that tier), not a dropped/misaligned column."""
    return panel.reindex(columns=feature_cols, fill_value=0)


def evaluate_frozen_model_on_oos(cfg: dict) -> dict:
    """The direct answer to 'does the existing model generalize to funds it's never seen?'
    No time-based split needed - none of this panel's data was used in training, regardless
    of quarter, so the whole thing is valid held-out data."""
    bundle = load_model("random_forest_model", cfg)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    oos_panel, _ = build_panel_for("_oos", cfg)
    x_oos = align_features(oos_panel, feature_cols)
    y_oos = oos_panel["underperform_next_quarter"].astype(int)

    proba = model.predict_proba(x_oos)[:, 1]
    auc = roc_auc_score(y_oos, proba)
    log.info(f"frozen model on {len(oos_panel)} fund-quarters from 1000 new, never-trained-on "
             f"funds: AUC={auc:.3f}")
    return {"auc": auc, "n": len(oos_panel)}


def evaluate_retrained_on_combined(cfg: dict) -> dict:
    """A different question: does pooling in the new funds and retraining from scratch
    improve on the original model? Uses the same time-based split logic as step4, now over
    the combined (original + OOS) fund set."""
    original_panel, quarters_ordered = build_panel_for("", cfg)
    oos_panel, oos_quarters = build_panel_for("_oos", cfg)
    if quarters_ordered != oos_quarters:
        raise RuntimeError(
            "original and OOS quarters must match for a fair combined split: "
            f"{quarters_ordered} != {oos_quarters}"
        )

    combined = pd.concat([original_panel, oos_panel], ignore_index=True)
    tier_cols = sorted(c for c in combined.columns if c.startswith("tier_"))
    combined[tier_cols] = combined[tier_cols].fillna(0)  # a tier absent from one panel, present
    # in the other, produces NaN after concat - correctly 0, not unknown.

    holdout_transitions = cfg["model"]["test_transitions_holdout"]
    train, test, train_quarters, test_quarters = time_based_split(
        combined, quarters_ordered, holdout_transitions
    )
    if max(train_quarters) >= min(test_quarters):
        raise RuntimeError(
            "train/test split leaks future data into training: "
            f"train up to {max(train_quarters)}, test from {min(test_quarters)}"
        )

    feature_cols = ["trailing_return", "trailing_volatility", "trailing_sharpe",
                     "trailing_max_drawdown", "return_vs_cluster_median_q", "net_assets"] + tier_cols
    x_train, y_train = train[feature_cols], train["underperform_next_quarter"].astype(int)
    x_test, y_test = test[feature_cols], test["underperform_next_quarter"].astype(int)

    rf = RandomForestClassifier(
        n_estimators=cfg["model"]["rf"]["n_estimators"],
        max_depth=cfg["model"]["rf"]["max_depth"],
        min_samples_leaf=cfg["model"]["rf"]["min_samples_leaf"],
        random_state=cfg["seed"],
    ).fit(x_train, y_train)
    auc = roc_auc_score(y_test, rf.predict_proba(x_test)[:, 1])
    n_funds = pd.concat([original_panel["series_id"], oos_panel["series_id"]]).nunique()
    log.info(f"retrained on combined {n_funds} funds ({len(train)} train, {len(test)} test "
             f"fund-quarter rows): AUC={auc:.3f}")
    return {"auc": auc, "n_train": len(train), "n_test": len(test)}


def run(cfg: dict) -> None:
    frozen = evaluate_frozen_model_on_oos(cfg)
    retrained = evaluate_retrained_on_combined(cfg)
    log.info(
        "SUMMARY - original step4 test AUC=0.710 (reference, 363 funds/2024 held-out) | "
        f"frozen model on 1000 new funds AUC={frozen['auc']:.3f} | "
        f"retrained on combined ~1363 funds AUC={retrained['auc']:.3f}"
    )


if __name__ == "__main__":
    import logging as _logging

    from fundspeers.config import load_config

    _logging.basicConfig(level=_logging.INFO, format="%(message)s")
    run(load_config())
