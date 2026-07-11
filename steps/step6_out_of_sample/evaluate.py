"""step6_out_of_sample — test whether step4's model generalizes to funds it has never seen.

Two evaluations, answering two different questions (see design.md):
1. The FROZEN step4 model applied to 1000 new, disjoint funds - does it generalize?
2. A model RETRAINED on the combined original + new funds - does more data help?
"""
import logging
import shutil

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from fundspeers.io import load_model, load_table, models_dir, save_model, save_table
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


def evaluate_frozen_model_on_oos(cfg: dict, table_suffix: str = "_oos") -> dict:
    """The direct answer to 'does the existing model generalize to funds it's never seen?'
    No time-based split needed - none of this panel's data was used in training, regardless
    of quarter, so the whole thing is valid held-out data. `table_suffix` (default "_oos",
    the original batch) lets this run against any later, independently-sampled disjoint
    batch (e.g. "_oos2") - whatever is currently the OFFICIAL persisted model is loaded, so
    if that model has since been promoted/retrained on a prior OOS batch, make sure this
    batch is disjoint from that model's actual training data, not just the original one."""
    bundle = load_model("random_forest_model", cfg)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    oos_panel, _ = build_panel_for(table_suffix, cfg)
    x_oos = align_features(oos_panel, feature_cols)
    y_oos = oos_panel["underperform_next_quarter"].astype(int)

    proba = model.predict_proba(x_oos)[:, 1]
    auc = roc_auc_score(y_oos, proba)
    log.info(f"frozen model on {len(oos_panel)} fund-quarters from batch '{table_suffix}', "
             f"never-trained-on funds: AUC={auc:.3f}")
    return {"auc": auc, "n": len(oos_panel)}


def fit_retrained_on_combined(cfg: dict, table_suffixes: tuple = ("", "_oos")) -> dict:
    """Assemble the combined panel across all given table namespaces, split, and fit a
    fresh RandomForest - the shared work behind both evaluating the retrained model and
    (optionally) promoting it to be the new official model. `table_suffixes` defaults to
    the original two-batch combination (original + first OOS); pass a longer tuple (e.g.
    adding "_oos2") to fold in further disjoint batches as they accumulate."""
    panels = []
    quarters_ordered = None
    for suffix in table_suffixes:
        panel, quarters = build_panel_for(suffix, cfg)
        if quarters_ordered is None:
            quarters_ordered = quarters
        elif quarters != quarters_ordered:
            raise RuntimeError(
                "quarters must match across all combined batches for a fair split: "
                f"batch {suffix!r} has {quarters} != {quarters_ordered}"
            )
        panels.append(panel)

    combined = pd.concat(panels, ignore_index=True)
    tier_cols = sorted(c for c in combined.columns if c.startswith("tier_"))
    combined[tier_cols] = combined[tier_cols].fillna(0)  # a tier absent from one panel, present
    # in another, produces NaN after concat - correctly 0, not unknown.

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
    n_funds = pd.concat([p["series_id"] for p in panels]).nunique()
    return {
        "model": rf, "feature_cols": feature_cols, "train": train, "test": test,
        "x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test,
        "n_funds": n_funds, "table_suffixes": table_suffixes,
    }


def evaluate_retrained_on_combined(cfg: dict, table_suffixes: tuple = ("", "_oos")) -> dict:
    """A different question from the frozen-model test: does pooling in the new funds and
    retraining from scratch improve on the original model? Uses the same time-based split
    logic as step4, now over the combined fund set named by `table_suffixes`. Does NOT
    persist anything - see promote_retrained_model() for that, a separate, deliberate
    action."""
    fitted = fit_retrained_on_combined(cfg, table_suffixes)
    auc = roc_auc_score(fitted["y_test"], fitted["model"].predict_proba(fitted["x_test"])[:, 1])
    log.info(f"retrained on combined {fitted['n_funds']} funds (batches {table_suffixes}) "
             f"({len(fitted['train'])} train, {len(fitted['test'])} test fund-quarter rows): "
             f"AUC={auc:.3f}")
    return {
        "auc": auc, "n_train": len(fitted["train"]), "n_test": len(fitted["test"]),
        "n_funds": fitted["n_funds"],
    }


def promote_retrained_model(
    cfg: dict,
    table_suffixes: tuple = ("", "_oos"),
    backup_name: str = "random_forest_model_original_363funds.joblib",
) -> dict:
    """Make the combined-data retrained model (over `table_suffixes`) the new official one -
    backs up the previous model.joblib under `backup_name` (not silently overwritten - and
    refuses to proceed if that name is already taken, so a second promotion can't clobber an
    earlier backup) and regenerates fund_predictions/model_feature_importances so the
    persisted tables stay consistent with whatever model is actually saved as official. A
    deliberate, explicit action - never called automatically from run(), since promoting is
    a real decision each time, not a side effect of evaluating."""
    fitted = fit_retrained_on_combined(cfg, table_suffixes)
    rf, feature_cols = fitted["model"], fitted["feature_cols"]
    train, test = fitted["train"], fitted["test"]
    x_train, x_test = fitted["x_train"], fitted["x_test"]

    models_path = models_dir(cfg)
    old_model_path = models_path / "random_forest_model.joblib"
    if old_model_path.exists():
        backup_path = models_path / backup_name
        if backup_path.exists():
            raise RuntimeError(
                f"backup target {backup_path} already exists - pick a distinct backup_name "
                "so the currently-official model isn't lost"
            )
        shutil.copy2(old_model_path, backup_path)
        log.info(f"backed up previous model to {backup_path}")

    save_model({"model": rf, "feature_cols": feature_cols}, "random_forest_model", cfg)

    predictions = pd.concat([
        train.assign(split="train", predicted_probability=rf.predict_proba(x_train)[:, 1]),
        test.assign(split="test", predicted_probability=rf.predict_proba(x_test)[:, 1]),
    ])[["series_id", "quarter", "predicted_probability", "underperform_next_quarter", "split"]].rename(
        columns={"underperform_next_quarter": "actual_label"}
    )
    importances = pd.DataFrame({
        "feature": feature_cols, "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    save_table(predictions, "fund_predictions", cfg)
    save_table(importances, "model_feature_importances", cfg)

    auc = roc_auc_score(fitted["y_test"], rf.predict_proba(x_test)[:, 1])
    log.info(f"PROMOTED: random_forest_model.joblib now trained on {fitted['n_funds']} funds "
             f"(batches {table_suffixes}), test AUC={auc:.3f}. Previous official model backed "
             f"up to {backup_name}. fund_predictions ({len(predictions)} rows) and "
             f"model_feature_importances regenerated to match.")
    return {"auc": auc, "n_funds": fitted["n_funds"]}


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
