"""step14_webapp/extract.py - build the webapp's compact read-only extract.

Precomputes every "honest view" the public app needs from the full pipeline DB, so the
hosted app is a thin renderer with no statistics of its own. Deterministic; run on demand:

    python -m steps.step14_webapp.extract          # writes webapp/data/extract.duckdb

Wired into advance.py's dashboard stage so each quarterly refresh rebuilds it. Deploy to
the HF Space is a separate, human-gated action (steps/step14_webapp/deploy.py).
"""
import logging
import re
import unicodedata

import duckdb
import pandas as pd

log = logging.getLogger(__name__)

MIN_CLUSTER_FOR_PCTILE = 15  # percentile-of-few is noise presented as precision


def latest_quarter(src: duckdb.DuckDBPyConnection) -> str:
    return src.execute("SELECT max(quarter) FROM funds_full").fetchone()[0]


def normalize_name(s: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace - the search index key.

    Symbol characters (e.g. "™", "®") are dropped before NFKD decomposition, not after:
    NFKD expands "™" into the literal letters "TM", which would otherwise survive the
    ascii-fold and read as part of the name.
    """
    s = "".join(c for c in s if not unicodedata.category(c).startswith("S"))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def build_fund_views(src: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    asof = latest_quarter(src)

    v_fund_header = src.execute("""
        WITH span AS (
            SELECT series_id, min(quarter) AS first_quarter, max(quarter) AS last_quarter
            FROM funds_full GROUP BY series_id
        ), latest AS (
            SELECT f.series_id, f.series_name, f.ticker, f.yahoo_category, f.segment,
                   f.net_assets, f.quarter,
                   row_number() OVER (PARTITION BY f.series_id ORDER BY f.quarter DESC) AS rk
            FROM funds_full f
        ), clus AS (
            SELECT c.series_id, c.cluster_id, c.quarter,
                   row_number() OVER (PARTITION BY c.series_id ORDER BY c.quarter DESC) AS rk
            FROM fund_clusters_full c
        )
        SELECT l.series_id, l.ticker, l.series_name, l.yahoo_category, l.segment,
               l.net_assets, s.first_quarter, s.last_quarter,
               (s.last_quarter = ?) AS is_active,
               cl.cluster_id, cd.short_title AS cluster_name,
               cd.member_count AS cluster_size
        FROM latest l
        JOIN span s USING (series_id)
        LEFT JOIN clus cl ON cl.series_id = l.series_id AND cl.rk = 1
        LEFT JOIN cluster_definitions_full cd
               ON cd.cluster_id = cl.cluster_id AND cd.quarter = cl.quarter
        WHERE l.rk = 1
    """, [asof]).df()

    v_fund_search = v_fund_header[
        ["series_id", "ticker", "series_name", "cluster_id", "cluster_name",
         "net_assets", "is_active", "last_quarter"]].copy()
    v_fund_search["name_normalized"] = v_fund_search["series_name"].map(normalize_name)

    v_fund_peer_relative_ts = src.execute("""
        SELECT m.series_id, m.quarter, m.quarterly_return, m.cluster_median_return,
               m.return_vs_cluster_median, m.cluster_id,
               cd.member_count AS cluster_size,
               percent_rank() OVER (PARTITION BY m.cluster_id, m.quarter
                                    ORDER BY m.quarterly_return) AS pctile_return_in_cluster
        FROM fund_metrics_quarterly_full m
        LEFT JOIN cluster_definitions_full cd
               ON cd.cluster_id = m.cluster_id AND cd.quarter = m.quarter
        ORDER BY m.series_id, m.quarter
    """).df()

    # Percentiles within the fund's latest cluster, latest-quarter membership.
    v_fund_cluster_percentiles = src.execute("""
        WITH membership AS (
            SELECT series_id, cluster_id FROM fund_clusters_full WHERE quarter = ?
        ), fees_latest AS (
            SELECT series_id, expense_ratio_net, portfolio_turnover,
                   row_number() OVER (PARTITION BY series_id ORDER BY quarter DESC) AS rk
            FROM rr_fees
        ), joined AS (
            SELECT mb.series_id, mb.cluster_id,
                   o.annualized_volatility, o.sharpe_ratio, o.max_drawdown,
                   fl.expense_ratio_net, fl.portfolio_turnover,
                   cd.member_count AS cluster_size
            FROM membership mb
            JOIN fund_metrics_overall_full o USING (series_id)
            LEFT JOIN fees_latest fl ON fl.series_id = mb.series_id AND fl.rk = 1
            LEFT JOIN cluster_definitions_full cd
                   ON cd.cluster_id = mb.cluster_id AND cd.quarter = ?
        )
        SELECT series_id, cluster_id, cluster_size,
            percent_rank() OVER (PARTITION BY cluster_id ORDER BY annualized_volatility)
                AS pctile_volatility,
            percent_rank() OVER (PARTITION BY cluster_id ORDER BY sharpe_ratio)
                AS pctile_sharpe,
            percent_rank() OVER (PARTITION BY cluster_id ORDER BY max_drawdown)
                AS pctile_max_drawdown,
            percent_rank() OVER (PARTITION BY cluster_id ORDER BY expense_ratio_net)
                AS pctile_expense_net,
            percent_rank() OVER (PARTITION BY cluster_id ORDER BY portfolio_turnover)
                AS pctile_turnover
        FROM joined
    """, [asof, asof]).df()
    small = v_fund_cluster_percentiles["cluster_size"].fillna(0) < MIN_CLUSTER_FOR_PCTILE
    pct_cols = [c for c in v_fund_cluster_percentiles.columns if c.startswith("pctile_")]
    v_fund_cluster_percentiles.loc[small, pct_cols] = pd.NA

    v_peer_display = src.execute("""
        WITH trailing_ret AS (
            SELECT series_id, sum(quarterly_return) AS trailing_4q_return
            FROM (SELECT series_id, quarter, quarterly_return,
                         row_number() OVER (PARTITION BY series_id ORDER BY quarter DESC) AS rk
                  FROM fund_metrics_quarterly_full)
            WHERE rk <= 4 GROUP BY series_id
        ), fees_latest AS (
            SELECT series_id, expense_ratio_net,
                   row_number() OVER (PARTITION BY series_id ORDER BY quarter DESC) AS rk
            FROM rr_fees
        )
        SELECT p.series_id, p.peer_rank, p.peer_series_id, p.cosine_similarity,
               h.ticker AS peer_ticker, h.series_name AS peer_name,
               h.yahoo_category AS peer_yahoo_category,
               t.trailing_4q_return AS peer_trailing_4q_return,
               fl.expense_ratio_net AS peer_expense_net
        FROM fund_peers_full p
        JOIN (SELECT series_id, ticker, series_name, yahoo_category,
                     row_number() OVER (PARTITION BY series_id ORDER BY quarter DESC) AS rk
              FROM funds_full) h ON h.series_id = p.peer_series_id AND h.rk = 1
        LEFT JOIN trailing_ret t ON t.series_id = p.peer_series_id
        LEFT JOIN fees_latest fl ON fl.series_id = p.peer_series_id AND fl.rk = 1
        WHERE p.quarter = ?
        ORDER BY p.series_id, p.peer_rank
    """, [asof]).df()

    v_top_holdings = src.execute("""
        SELECT f.series_id, h.issuer_name, h.percentage, h.quarter, rk
        FROM (SELECT accession_number, issuer_name, percentage, quarter,
                     row_number() OVER (PARTITION BY accession_number
                                        ORDER BY percentage DESC) AS rk
              FROM holdings_full WHERE quarter = ?) h
        JOIN funds_full f ON f.accession_number = h.accession_number
        WHERE rk <= 10
    """, [asof]).df()

    return {"v_fund_header": v_fund_header, "v_fund_search": v_fund_search,
            "v_fund_peer_relative_ts": v_fund_peer_relative_ts,
            "v_fund_cluster_percentiles": v_fund_cluster_percentiles,
            "v_peer_display": v_peer_display, "v_top_holdings": v_top_holdings}


HEALTH_RULES = {
    "healthy": "Both of the last two realized quarters scored at or above the "
               "mean-reversion baseline.",
    "weak": "Above the 0.5 coin-flip in the last two realized quarters, but below the "
            "mean-reversion baseline in at least one.",
    "degraded": "At least one of the last two realized quarters scored below the 0.5 "
                "coin-flip.",
}


def compute_health_state(last_two: list) -> tuple:
    """(state, disclosed rule text) from [(auc, persistence_auc), ...] oldest-first."""
    if any(auc < 0.5 for auc, _ in last_two):
        return "degraded", HEALTH_RULES["degraded"]
    if all(auc >= base for auc, base in last_two):
        return "healthy", HEALTH_RULES["healthy"]
    return "weak", HEALTH_RULES["weak"]


def _prediction_intervals(src, cfg, forward: pd.DataFrame) -> pd.DataFrame:
    """Per-tree 10th/90th percentile of the RF's forward predictions - a real spread, not
    an invented CI. Needs cfg (model bundle + full_panel live only in the real DB); when
    cfg is None (unit tests / missing bundle) the interval is NULL and the UI says so."""
    out = forward[["series_id"]].copy()
    out["ci_low"], out["ci_high"] = pd.NA, pd.NA
    if cfg is None:
        return out
    try:
        from fundspeers.io import load_model
        import numpy as np
        bundle = load_model("full_rf_model", cfg)
        model, feature_cols = bundle["model"], bundle["feature_cols"]
        panel = src.execute("SELECT * FROM full_panel WHERE quarter = ?",
                            [forward["quarter"].iloc[0]]).df()
        panel = panel.set_index("series_id").reindex(forward["series_id"])
        x = panel.reindex(columns=feature_cols).fillna(0.0)
        per_tree = np.stack([t.predict_proba(x.values)[:, 1] for t in model.estimators_])
        out["ci_low"] = np.percentile(per_tree, 10, axis=0)
        out["ci_high"] = np.percentile(per_tree, 90, axis=0)
    except Exception as exc:  # interval is optional honesty garnish - never break the build
        log.warning("prediction intervals unavailable (%s); shipping NULL intervals", exc)
    return out


def build_model_views(src: duckdb.DuckDBPyConnection, cfg) -> dict:
    asof = latest_quarter(src)

    forward = src.execute(
        "SELECT series_id, quarter, predicted_probability FROM full_predictions "
        "WHERE split = 'forward'").df()
    target = f"the quarter after {forward['quarter'].iloc[0]}" if len(forward) else ""
    stab = src.execute("""
        SELECT series_id, flip_rate FROM (
            SELECT series_id, flip_rate,
                   row_number() OVER (PARTITION BY series_id ORDER BY quarter DESC) AS rk
            FROM full_label_stability) WHERE rk = 1""").df()
    intervals = _prediction_intervals(src, cfg, forward) if len(forward) else \
        pd.DataFrame(columns=["series_id", "ci_low", "ci_high"])
    v_fund_prediction_current = (forward.merge(intervals, on="series_id", how="left")
                                 .merge(stab, on="series_id", how="left"))
    v_fund_prediction_current["target_quarter"] = target

    v_fund_prediction_history = src.execute(
        "SELECT series_id, quarter, predicted_probability, actual_label, split "
        "FROM full_predictions WHERE split IN ('test', 'train') ORDER BY series_id, quarter").df()

    per_q = src.execute("""
        SELECT e.quarter, e.value AS auc, b.value AS persistence_auc,
               NULL AS n_scored, 'retrained' AS source
        FROM full_model_eval e
        LEFT JOIN full_model_eval b
               ON b.quarter = e.quarter AND b.metric = 'auc_persistence_baseline'
        WHERE e.metric = 'auc_pooled' AND e.quarter <> ''
        UNION ALL
        SELECT quarter, value AS auc, NULL, NULL, 'frozen' FROM oot_validation
        WHERE metric = 'auc' AND quarter <> '' AND source = 'frozen_rolled_forward'
        ORDER BY quarter
    """).df()
    v_model_health_quarters = per_q

    retrained = per_q[per_q["source"] == "retrained"].sort_values("quarter")
    last_two = [(float(r.auc), float(r.persistence_auc) if pd.notna(r.persistence_auc) else 0.5)
                for r in retrained.tail(2).itertuples()]
    state, rule_text = compute_health_state(last_two) if last_two else ("weak", "No realized quarters yet.")

    def _scalar(table, metric, source=None):
        q = f"SELECT value FROM {table} WHERE metric = ? AND quarter = ''"
        args = [metric]
        if source:
            q += " AND source = ?"
            args.append(source)
        row = src.execute(q, args).fetchone()
        return float(row[0]) if row else None

    noise_floor = src.execute(
        "SELECT avg(flip_rate) FROM full_label_stability").fetchone()[0]
    refreshed = src.execute(
        "SELECT max(refreshed_at) FROM refresh_log").fetchone()[0]
    v_model_health_current = pd.DataFrame([{
        "health_state": state, "rule_text": rule_text,
        "last_scored_quarter": retrained["quarter"].max() if len(retrained) else None,
        "auc_last": last_two[-1][0] if last_two else None,
        "auc_prev": last_two[0][0] if len(last_two) > 1 else None,
        "pooled_live_auc": _scalar("oot_validation", "auc", "published_forward"),
        "backtest_auc": _scalar("full_model_eval", "auc_pooled"),
        "base_rate": _scalar("oot_validation", "base_rate", "published_forward"),
        "label_noise_floor": float(noise_floor) if noise_floor is not None else None,
        "refreshed_at": refreshed,
    }])

    v_calibration_bins = src.execute("""
        SELECT floor(predicted_probability * 10) / 10 AS bin_low,
               floor(predicted_probability * 10) / 10 + 0.1 AS bin_high,
               count(*) AS n, avg(predicted_probability) AS predicted_mean,
               avg(actual_label) AS actual_lag_rate
        FROM full_predictions WHERE split = 'test' AND actual_label IS NOT NULL
        GROUP BY 1, 2 ORDER BY 1
    """).df()

    v_data_provenance = pd.DataFrame([{
        "last_quarter": asof, "refreshed_at": refreshed,
        "n_funds": int(src.execute(
            "SELECT count(DISTINCT series_id) FROM funds_full").fetchone()[0]),
    }])

    return {"v_fund_prediction_current": v_fund_prediction_current,
            "v_fund_prediction_history": v_fund_prediction_history,
            "v_model_health_quarters": v_model_health_quarters,
            "v_model_health_current": v_model_health_current,
            "v_calibration_bins": v_calibration_bins,
            "v_data_provenance": v_data_provenance}


def build_cluster_views(src: duckdb.DuckDBPyConnection) -> dict:
    """step15 consumers - built now so the extract schema is stable from day one."""
    asof = latest_quarter(src)
    v_cluster_summary = src.execute("""
        SELECT cd.cluster_id, cd.short_title AS cluster_name, n.narrative,
               cd.member_count, cd.dominant_category, cd.dominant_category_share,
               cd.avg_volatility, cd.avg_sharpe
        FROM cluster_definitions_full cd
        LEFT JOIN dashboard_narratives n
               ON n.cluster_id = cd.cluster_id AND n.quarter = cd.quarter
        WHERE cd.quarter = ?
    """, [asof]).df()
    v_cluster_return_dispersion = src.execute("""
        SELECT cluster_id, quarter,
               quantile_cont(quarterly_return, 0.10) AS p10,
               quantile_cont(quarterly_return, 0.25) AS p25,
               quantile_cont(quarterly_return, 0.50) AS median,
               quantile_cont(quarterly_return, 0.75) AS p75,
               quantile_cont(quarterly_return, 0.90) AS p90,
               count(*) AS n_members
        FROM fund_metrics_quarterly_full WHERE cluster_id IS NOT NULL
        GROUP BY cluster_id, quarter
    """).df()
    v_cluster_map = src.execute("""
        SELECT c.series_id, c.x, c.y, c.cluster_id, h.ticker, h.series_name
        FROM cluster_map_coords_full c
        JOIN (SELECT series_id, ticker, series_name,
                     row_number() OVER (PARTITION BY series_id ORDER BY quarter DESC) AS rk
              FROM funds_full) h ON h.series_id = c.series_id AND h.rk = 1
    """).df()
    return {"v_cluster_summary": v_cluster_summary,
            "v_cluster_return_dispersion": v_cluster_return_dispersion,
            "v_cluster_map": v_cluster_map}


def write_extract(views: dict, out_path) -> None:
    """Write a dict of {view_name: DataFrame} to a fresh duckdb file at out_path."""
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    dst = duckdb.connect(str(out_path))
    try:
        for name, df in views.items():
            dst.register("_v", df)
            dst.execute(f"CREATE TABLE {name} AS SELECT * FROM _v")
            dst.unregister("_v")
            log.info("%s: %d rows", name, len(df))
    finally:
        dst.close()
    size_mb = out_path.stat().st_size / 1e6
    log.info("extract written: %s (%.1f MB)", out_path, size_mb)
    if size_mb > 50:
        raise RuntimeError(f"extract is {size_mb:.0f} MB - over the 50 MB budget")


def run(cfg: dict, out_path=None):
    """Build all views from the real pipeline DB and write webapp/data/extract.duckdb."""
    from pathlib import Path
    from fundspeers.config import PROJECT_ROOT
    from fundspeers.io import db_path

    out_path = Path(out_path) if out_path else PROJECT_ROOT / "webapp" / "data" / "extract.duckdb"

    src = duckdb.connect(str(db_path(cfg)), read_only=True)
    try:
        views = {}
        views.update(build_fund_views(src))
        views.update(build_model_views(src, cfg))
        views.update(build_cluster_views(src))
    finally:
        src.close()

    write_extract(views, out_path)
    return out_path


if __name__ == "__main__":
    from fundspeers.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(load_config())
