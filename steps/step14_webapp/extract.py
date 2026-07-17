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
