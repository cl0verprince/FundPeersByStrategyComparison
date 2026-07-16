# step14_webapp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the credibility core of the public dynamic dashboard — extract builder, NiceGUI app shell with fuzzy search, Fund page, Model-health page, Methodology — deployed to a Hugging Face Space.

**Architecture:** A local extract builder (`steps/step14_webapp/extract.py`) precomputes ~14 honest views from the 5.6 GB pipeline DuckDB into a ~10 MB `webapp/data/extract.duckdb`. A self-contained `webapp/` NiceGUI app reads that extract read-only and deploys as a Docker HF Space. The model-health state is computed once at extract time into a singleton view; the app cannot render a probability without it.

**Tech Stack:** Python 3.12+, DuckDB, pandas, scikit-learn (extract-time only), NiceGUI ≥ 2.0, ECharts (via `ui.echart`), AG Grid (via `ui.aggrid`), huggingface_hub (deploy), pytest + nicegui.testing User fixture.

## Global Constraints (from steps/step14_webapp/design.md — binding)

- Probabilities render ONLY via `uncertain_probability` pattern: whole percents, "±N pts" interval, fixed sentence "chance this fund falls below its peers' median return next quarter", sequential-blue meter, status chip in the same card. Never red/green on predictions.
- Semantic red/green ONLY on realized deltas vs peer median (direction-aware: lower fees = good) and model-health status (always icon + label + color).
- Health rule (disclosed): Healthy = last 2 realized quarters' AUC ≥ persistence baseline that quarter; Degraded = either of last 2 < 0.5; Weak = otherwise. Computed at extract build, stored in `v_model_health_current`.
- Percentiles suppressed where cluster_size < 15 (store NULL).
- Line charts: `connectNulls: false`; solid hairline grid (never dashed); no dual axes; chart⇄table twin on /model and fund Zone C.
- Default sorts never by return. No leaderboards, rankings, star ratings, "most likely to lag" screens, custom date ranges, cluster forecasts, A-vs-B verdicts, alerts.
- Every page: freshness stamp (as-of quarter · "~60-day filing lag" · extract build date) + header one-liner "Educational analytics — not investment advice."
- Fund names from SEC filings are untrusted text: never `innerHTML`.
- Extract must stay < 50 MB (test-enforced).
- All shared-code conventions of this repo apply: config via `fundspeers.config.load_config`, tables via `fundspeers.io`, tests in `tests/`, comment style matching existing modules.

## File Structure

```
steps/step14_webapp/
  __init__.py            (empty)
  extract.py             extract builder: build_views(cfg) -> dict, run(cfg, out_path) CLI
  deploy.py              HF Space create/upload (manual, gated)
webapp/
  __init__.py            (empty)
  requirements.txt       nicegui, duckdb, pandas
  Dockerfile
  theme.py               validated palette/typography tokens (light+dark dicts)
  data.py                ExtractStore: read-only queries + in-memory search index/scorer
  components/
    __init__.py
    honesty.py           status_chip, freshness_stamp, disclaimer_line, probability_card
    charts.py            pure echarts option builders (mode-aware)
  pages/
    __init__.py
    fund.py              /fund/{ticker}
    model.py             /model
    methodology.py       /methodology
  main.py                app shell: header, omnibox, routing, 404, stale banner, ui.run
  data/.gitkeep          (extract.duckdb lands here; gitignored)
tests/
  test_step14_extract.py
  test_step14_search.py
  test_step14_charts.py
  test_step14_app.py     nicegui User-fixture smoke tests
```

---

### Task 1: Extract builder — fund-facing views

**Files:**
- Create: `steps/step14_webapp/__init__.py` (empty), `steps/step14_webapp/extract.py`
- Test: `tests/test_step14_extract.py`

**Interfaces:**
- Consumes: `fundspeers.config.load_config`, `fundspeers.io.db_path`; source tables `funds_full(series_id, series_name, quarter, net_assets, ticker, yahoo_category, segment)`, `fund_clusters_full(series_id, quarter, cluster_id)`, `cluster_definitions_full(quarter, cluster_id, member_count, dominant_category, dominant_category_share, avg_volatility, avg_sharpe, short_title)`, `fund_metrics_quarterly_full(series_id, quarter, quarterly_return, cluster_id, cluster_median_return, return_vs_cluster_median)`, `fund_metrics_overall_full(series_id, cumulative_return, annualized_volatility, sharpe_ratio, max_drawdown)`, `fund_peers_full(series_id, quarter, peer_rank, peer_series_id, cosine_similarity)`, `rr_fees(series_id, quarter, expense_ratio_net, expense_ratio_gross, portfolio_turnover)`, `holdings_full(accession_number, issuer_name, percentage, quarter)`
- Produces: `build_fund_views(src: duckdb.Connection) -> dict[str, pd.DataFrame]` returning keys `v_fund_header`, `v_fund_search`, `v_fund_peer_relative_ts`, `v_fund_cluster_percentiles`, `v_peer_display`, `v_top_holdings`. Helper `latest_quarter(src) -> str`. Helper `normalize_name(s: str) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
"""tests/test_step14_extract.py — extract builder: views exist, honest rules enforced.

All tests run against a tiny synthetic source DB built in tmp_path — never the real 5.6 GB DB.
"""
import duckdb
import pandas as pd
import pytest

from steps.step14_webapp.extract import (
    build_fund_views, latest_quarter, normalize_name)


@pytest.fixture
def src(tmp_path):
    """Synthetic source DB: 3 funds (A alive+big, B alive+small-cluster, D dead), 2 quarters."""
    con = duckdb.connect(str(tmp_path / "src.duckdb"))
    con.register("funds", pd.DataFrame([
        # series_id, series_name, quarter, net_assets, ticker, yahoo_category, segment, accession_number
        ("A", "Alpha Large Blend Fund", "2026q1", 5e9, "AAAAX", "Large Blend", "strategy", "acc-A1"),
        ("A", "Alpha Large Blend Fund", "2026q2", 6e9, "AAAAX", "Large Blend", "strategy", "acc-A2"),
        ("B", "Beta Small Value Fund", "2026q1", 1e8, "BBBBX", "Small Value", "strategy", "acc-B1"),
        ("B", "Beta Small Value Fund", "2026q2", 1e8, "BBBBX", "Small Value", "strategy", "acc-B2"),
        ("D", "Delta Dead Fund", "2026q1", 2e8, "DDDDX", "Large Blend", "strategy", "acc-D1"),
    ], columns=["series_id", "series_name", "quarter", "net_assets", "ticker",
                "yahoo_category", "segment", "accession_number"]))
    con.execute("CREATE TABLE funds_full AS SELECT * FROM funds")
    con.register("clusters", pd.DataFrame([
        ("A", "2026q1", 1), ("A", "2026q2", 1), ("B", "2026q1", 2), ("B", "2026q2", 2),
        ("D", "2026q1", 1),
    ], columns=["series_id", "quarter", "cluster_id"]))
    con.execute("CREATE TABLE fund_clusters_full AS SELECT * FROM clusters")
    con.register("cdefs", pd.DataFrame([
        ("2026q2", 1, 20, "Large Blend", 0.8, 0.15, 0.5, "Leaning Large Blend"),
        ("2026q2", 2, 5, "Small Value", 0.6, 0.20, 0.3, "Tiny Small Value"),
        ("2026q1", 1, 21, "Large Blend", 0.8, 0.15, 0.5, "Leaning Large Blend"),
        ("2026q1", 2, 5, "Small Value", 0.6, 0.20, 0.3, "Tiny Small Value"),
    ], columns=["quarter", "cluster_id", "member_count", "dominant_category",
                "dominant_category_share", "avg_volatility", "avg_sharpe", "short_title"]))
    con.execute("CREATE TABLE cluster_definitions_full AS SELECT * FROM cdefs")
    con.register("mq", pd.DataFrame([
        ("A", "2026q1", 0.04, 1, 0.03, 0.01), ("A", "2026q2", 0.02, 1, 0.03, -0.01),
        ("B", "2026q1", 0.05, 2, 0.05, 0.00), ("B", "2026q2", 0.01, 2, 0.02, -0.01),
        ("D", "2026q1", -0.02, 1, 0.03, -0.05),
    ], columns=["series_id", "quarter", "quarterly_return", "cluster_id",
                "cluster_median_return", "return_vs_cluster_median"]))
    con.execute("CREATE TABLE fund_metrics_quarterly_full AS SELECT * FROM mq")
    con.register("mo", pd.DataFrame([
        ("A", 0.30, 0.14, 0.52, -0.08), ("B", 0.10, 0.22, 0.20, -0.30),
        ("D", -0.05, 0.18, -0.10, -0.25),
    ], columns=["series_id", "cumulative_return", "annualized_volatility",
                "sharpe_ratio", "max_drawdown"]))
    con.execute("CREATE TABLE fund_metrics_overall_full AS SELECT * FROM mo")
    con.register("peers", pd.DataFrame([
        ("A", "2026q2", 1, "B", 0.91), ("A", "2026q2", 2, "D", 0.80),
        ("B", "2026q2", 1, "A", 0.91),
    ], columns=["series_id", "quarter", "peer_rank", "peer_series_id", "cosine_similarity"]))
    con.execute("CREATE TABLE fund_peers_full AS SELECT * FROM peers")
    con.register("fees", pd.DataFrame([
        ("A", "2026q2", 0.0004, 0.0005, 0.05), ("B", "2026q2", 0.0110, 0.0120, 1.20),
    ], columns=["series_id", "quarter", "expense_ratio_net", "expense_ratio_gross",
                "portfolio_turnover"]))
    con.execute("CREATE TABLE rr_fees AS SELECT * FROM fees")
    con.register("hold", pd.DataFrame([
        ("acc-A2", "MICROSOFT CORP", 6.0, "2026q2"), ("acc-A2", "APPLE INC", 5.0, "2026q2"),
        ("acc-B2", "SOME SMALL CO", 3.0, "2026q2"),
    ], columns=["accession_number", "issuer_name", "percentage", "quarter"]))
    con.execute("CREATE TABLE holdings_full AS SELECT * FROM hold")
    yield con
    con.close()


def test_latest_quarter(src):
    assert latest_quarter(src) == "2026q2"


def test_normalize_name():
    assert normalize_name("Vanguard 500 Index — Admiral™!") == "vanguard 500 index admiral"


def test_fund_views_exist_and_nonempty(src):
    views = build_fund_views(src)
    for name in ["v_fund_header", "v_fund_search", "v_fund_peer_relative_ts",
                 "v_fund_cluster_percentiles", "v_peer_display", "v_top_holdings"]:
        assert name in views and len(views[name]) > 0, name


def test_fund_header_activity_flag(src):
    header = build_fund_views(src)["v_fund_header"].set_index("series_id")
    assert bool(header.loc["A", "is_active"]) is True
    assert bool(header.loc["D", "is_active"]) is False
    assert header.loc["D", "last_quarter"] == "2026q1"
    assert header.loc["A", "cluster_name"] == "Leaning Large Blend"


def test_percentiles_suppressed_for_small_clusters(src):
    pct = build_fund_views(src)["v_fund_cluster_percentiles"].set_index("series_id")
    # cluster 1 has member_count 20 -> percentiles present; cluster 2 has 5 -> suppressed
    assert pd.notna(pct.loc["A", "pctile_volatility"])
    assert pd.isna(pct.loc["B", "pctile_volatility"])


def test_peer_display_joins_names_and_fees(src):
    peers = build_fund_views(src)["v_peer_display"]
    row = peers[(peers.series_id == "A") & (peers.peer_rank == 1)].iloc[0]
    assert row["peer_ticker"] == "BBBBX"
    assert row["peer_name"] == "Beta Small Value Fund"
    assert row["peer_expense_net"] == pytest.approx(0.0110)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_step14_extract.py -q`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'steps.step14_webapp.extract'`

- [ ] **Step 3: Implement `steps/step14_webapp/extract.py` (fund views half)**

Create empty `steps/step14_webapp/__init__.py`, then:

```python
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
    """Lowercase, strip accents/punctuation, collapse whitespace - the search index key."""
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
        WITH trailing AS (
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
        LEFT JOIN trailing t ON t.series_id = p.peer_series_id
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_step14_extract.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add steps/step14_webapp/__init__.py steps/step14_webapp/extract.py tests/test_step14_extract.py
git commit -m "step14: extract builder - fund-facing honest views (header, search, peer-relative ts, suppressed percentiles, peers, holdings)"
```

---

### Task 2: Extract builder — prediction, model-health, cluster views + CLI

**Files:**
- Modify: `steps/step14_webapp/extract.py` (append)
- Test: `tests/test_step14_extract.py` (append)

**Interfaces:**
- Consumes: source tables `full_predictions(series_id, quarter, predicted_probability, actual_label, split)`, `full_model_eval(metric, quarter, value)`, `oot_validation(metric, quarter, value, source)`, `full_label_stability(series_id, quarter, flip_rate)`, `refresh_log(quarter, refreshed_at)`, `cluster_map_coords_full(series_id, x, y, cluster_id)`, `dashboard_narratives(cluster_id, quarter, narrative)`, `cluster_validation_full(quarter, purity, adjusted_rand_index)`; model bundle `models/full_rf_model.joblib` (`{"model": RandomForestClassifier, "feature_cols": [...]}` — the bundle that produced `full_predictions`) and `full_panel` for forward-row features.
- Produces:
  - `compute_health_state(last_two: list[tuple[float, float]]) -> tuple[str, str]` — input `[(auc, persistence_auc), ...]` oldest-first, returns `(state, rule_text)` with state in `{"healthy","weak","degraded"}`.
  - `build_model_views(src, cfg) -> dict[str, pd.DataFrame]` with keys `v_fund_prediction_current` (columns: series_id, predicted_probability, ci_low, ci_high, target_quarter, flip_rate), `v_fund_prediction_history`, `v_model_health_quarters` (quarter, auc, persistence_auc, n_scored, source), `v_model_health_current` (singleton: health_state, rule_text, last_scored_quarter, auc_last, auc_prev, pooled_live_auc, backtest_auc, base_rate, label_noise_floor, refreshed_at), `v_calibration_bins`, `v_data_provenance`.
  - `build_cluster_views(src) -> dict[str, pd.DataFrame]` with keys `v_cluster_summary`, `v_cluster_return_dispersion`, `v_cluster_map` (step15 consumers; schema frozen now).
  - `run(cfg, out_path=None) -> Path` — builds ALL views, writes `webapp/data/extract.duckdb`, logs sizes; `python -m steps.step14_webapp.extract` entry point.

- [ ] **Step 1: Write the failing tests (append to tests/test_step14_extract.py)**

```python
from steps.step14_webapp.extract import compute_health_state


def test_health_state_rule_truth_table():
    # oldest-first [(auc, persistence_auc), ...] for the last two realized quarters
    assert compute_health_state([(0.60, 0.55), (0.62, 0.55)])[0] == "healthy"
    assert compute_health_state([(0.55, 0.57), (0.56, 0.57)])[0] == "weak"     # above 0.5, below baseline
    assert compute_health_state([(0.55, 0.50), (0.42, 0.57)])[0] == "degraded" # one below coin-flip
    assert compute_health_state([(0.42, 0.57), (0.41, 0.57)])[0] == "degraded"
    state, rule_text = compute_health_state([(0.42, 0.57), (0.41, 0.57)])
    assert "0.5" in rule_text  # the rule is disclosed, not proprietary


def test_model_views_from_synthetic(model_src):
    from steps.step14_webapp.extract import build_model_views
    views = build_model_views(model_src, cfg=None)  # cfg only needed for tree intervals; None skips
    cur = views["v_model_health_current"]
    assert len(cur) == 1
    assert cur.iloc[0]["health_state"] in ("healthy", "weak", "degraded")
    pred = views["v_fund_prediction_current"].set_index("series_id")
    assert "A" in pred.index and 0.0 <= pred.loc["A", "predicted_probability"] <= 1.0
    hist = views["v_fund_prediction_history"]
    assert set(hist["split"]) <= {"test", "train"}
    calib = views["v_calibration_bins"]
    assert (calib["n"] > 0).all()


@pytest.fixture
def model_src(tmp_path):
    con = duckdb.connect(str(tmp_path / "msrc.duckdb"))
    con.register("preds", pd.DataFrame([
        ("A", "2026q2", 0.41, None, "forward"),
        ("A", "2026q1", 0.70, 1.0, "test"), ("A", "2025q4", 0.30, 0.0, "test"),
        ("B", "2026q2", 0.55, None, "forward"),
        ("B", "2026q1", 0.20, 1.0, "test"),
    ], columns=["series_id", "quarter", "predicted_probability", "actual_label", "split"]))
    con.execute("CREATE TABLE full_predictions AS SELECT * FROM preds")
    con.register("ev", pd.DataFrame([
        ("auc_pooled", "", 0.578), ("auc_pooled", "2025q4", 0.457), ("auc_pooled", "2026q1", 0.427),
        ("auc_persistence_baseline", "2025q4", 0.552), ("auc_persistence_baseline", "2026q1", 0.569),
    ], columns=["metric", "quarter", "value"]))
    con.execute("CREATE TABLE full_model_eval AS SELECT * FROM ev")
    con.register("oot", pd.DataFrame([
        ("auc", "", 0.574, "published_forward"), ("base_rate", "", 0.532, "published_forward"),
        ("n_scored", "", 2057.0, "published_forward"),
        ("auc", "2026q1", 0.418, "frozen_rolled_forward"),
    ], columns=["metric", "quarter", "value", "source"]))
    con.execute("CREATE TABLE oot_validation AS SELECT * FROM oot")
    con.register("stab", pd.DataFrame([
        ("A", "2026q1", 0.05), ("B", "2026q1", 0.30),
    ], columns=["series_id", "quarter", "flip_rate"]))
    con.execute("CREATE TABLE full_label_stability AS SELECT * FROM stab")
    con.register("rlog", pd.DataFrame([("2026q2", "2026-07-16T06:45:21+00:00")],
                                      columns=["quarter", "refreshed_at"]))
    con.execute("CREATE TABLE refresh_log AS SELECT * FROM rlog")
    con.register("ff", pd.DataFrame([("A", "2026q2"), ("B", "2026q2")],
                                    columns=["series_id", "quarter"]))
    con.execute("CREATE TABLE funds_full AS SELECT * FROM ff")
    yield con
    con.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_step14_extract.py -q`
Expected: FAIL — `ImportError: cannot import name 'compute_health_state'`

- [ ] **Step 3: Implement (append to extract.py)**

```python
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


def run(cfg: dict, out_path=None):
    """Build all views from the real pipeline DB and write webapp/data/extract.duckdb."""
    from pathlib import Path
    from fundspeers.config import PROJECT_ROOT
    from fundspeers.io import db_path

    out_path = Path(out_path) if out_path else PROJECT_ROOT / "webapp" / "data" / "extract.duckdb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    src = duckdb.connect(str(db_path(cfg)), read_only=True)
    try:
        views = {}
        views.update(build_fund_views(src))
        views.update(build_model_views(src, cfg))
        views.update(build_cluster_views(src))
    finally:
        src.close()

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
    return out_path


if __name__ == "__main__":
    from fundspeers.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(load_config())
```

- [ ] **Step 4: Run all extract tests**

Run: `python -m pytest tests/test_step14_extract.py -q`
Expected: all PASS

- [ ] **Step 5: Build the real extract once and sanity-check**

Run: `python -m steps.step14_webapp.extract`
Expected: logs each view with rows > 0; final line `extract written: ...\webapp\data\extract.duckdb (~10 MB)`. Also add `webapp/data/*.duckdb` to `.gitignore` and create `webapp/data/.gitkeep`.

- [ ] **Step 6: Commit**

```bash
git add steps/step14_webapp/extract.py tests/test_step14_extract.py .gitignore webapp/data/.gitkeep
git commit -m "step14: extract builder complete - prediction intervals (per-tree spread), disclosed health rule, calibration, provenance, cluster views; <50MB enforced"
```

---

### Task 3: Webapp scaffold — theme tokens, data store, search scorer

**Files:**
- Create: `webapp/__init__.py` (empty), `webapp/requirements.txt`, `webapp/theme.py`, `webapp/data.py`, `webapp/components/__init__.py` (empty)
- Test: `tests/test_step14_search.py`

**Interfaces:**
- Produces:
  - `webapp.theme.TOKENS: dict` — `TOKENS["light"]` / `TOKENS["dark"]`, each with keys `surface, page, ink, ink2, muted, grid, s1, demph, seq (list 100→700), div_pos, div_neg, good, warning, serious, critical`. `webapp.theme.STATUS = {"healthy": ("✓", "Signal live", "good"), "weak": ("◐", "Signal weak", "warning"), "degraded": ("⚠", "Signal degraded", "critical")}`.
  - `webapp.data.ExtractStore(path)` — `.fund_header(ticker) -> dict|None`, `.fund_ts(series_id) -> pd.DataFrame`, `.fund_percentiles(series_id) -> dict|None`, `.fund_prediction(series_id) -> dict|None`, `.fund_prediction_history(series_id) -> pd.DataFrame`, `.peers(series_id) -> pd.DataFrame`, `.top_holdings(series_id) -> pd.DataFrame`, `.model_health() -> dict`, `.model_quarters() -> pd.DataFrame`, `.calibration() -> pd.DataFrame`, `.provenance() -> dict`, `.search(query, limit=8) -> list[dict]`, `.is_stale() -> bool`.
  - `webapp.data.score_query(query, index) -> list[dict]` — pure scorer; `index` = list of dicts with keys `ticker, series_name, name_normalized, net_assets, is_active, last_quarter, cluster_name, series_id`. Tier 1 exact ticker, 2 ticker prefix, 3 all-query-tokens prefix-match name tokens, 4 trigram fuzzy (Jaccard over character trigrams ≥ 0.25). Within tier: net_assets desc; inactive funds demoted one tier.

- [ ] **Step 1: Write the failing search tests**

```python
"""tests/test_step14_search.py — omnibox scorer: tiers, fuzziness, dead-fund demotion."""
from webapp.data import score_query

INDEX = [
    {"series_id": "V1", "ticker": "VFIAX", "series_name": "Vanguard 500 Index Admiral",
     "name_normalized": "vanguard 500 index admiral", "net_assets": 4e11,
     "is_active": True, "last_quarter": "2026q2", "cluster_name": "Leaning Large Blend"},
    {"series_id": "V2", "ticker": "VFINX", "series_name": "Vanguard 500 Index Investor",
     "name_normalized": "vanguard 500 index investor", "net_assets": 1e10,
     "is_active": True, "last_quarter": "2026q2", "cluster_name": "Leaning Large Blend"},
    {"series_id": "F1", "ticker": "FMAGX", "series_name": "Fidelity Magellan",
     "name_normalized": "fidelity magellan", "net_assets": 2e10,
     "is_active": True, "last_quarter": "2026q2", "cluster_name": "Leaning Large Growth"},
    {"series_id": "D1", "ticker": "VDEAD", "series_name": "Vanguard Departed Fund",
     "name_normalized": "vanguard departed fund", "net_assets": 9e11,
     "is_active": False, "last_quarter": "2024q3", "cluster_name": "Concentrated Value"},
]


def test_exact_ticker_first():
    assert score_query("vfiax", INDEX)[0]["ticker"] == "VFIAX"


def test_ticker_prefix_beats_name_match():
    got = [r["ticker"] for r in score_query("VFI", INDEX)]
    assert got[:2] == ["VFIAX", "VFINX"]  # prefix tier, AUM desc within tier


def test_name_token_prefixes():
    assert score_query("vang adm", INDEX)[0]["ticker"] == "VFIAX"


def test_fuzzy_typo():
    assert score_query("vangard 500", INDEX)[0]["ticker"] == "VFIAX"


def test_dead_fund_demoted_despite_aum():
    got = [r["ticker"] for r in score_query("vanguard", INDEX)]
    assert got.index("VDEAD") > got.index("VFIAX")  # 9e11 AUM but dead -> below living


def test_no_match_empty():
    assert score_query("zzzzqq", INDEX) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_step14_search.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'webapp'`

- [ ] **Step 3: Implement scaffold**

`webapp/requirements.txt`:
```
nicegui>=2.0
duckdb>=1.0
pandas>=2.0
```
Install locally into the project venv: `pip install nicegui` (duckdb/pandas already present).

`webapp/theme.py`:
```python
"""Design tokens - the UX spec's machine-validated palette. Charts and components read
ONLY from here; no literal colors anywhere else in webapp/."""
TOKENS = {
    "light": {
        "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#5f5e58",
        "muted": "#898781", "grid": "#e1e0d9", "s1": "#2a78d6", "demph": "#c3c2b7",
        "seq": ["#e8f0fb", "#c4d9f4", "#9dc0ec", "#6fa3e2", "#4488d9", "#2a78d6", "#1c5cab"],
        "div_pos": "#2a78d6", "div_neg": "#d03b3b",
        "good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff", "ink2": "#b5b3ac",
        "muted": "#898781", "grid": "#2c2c2a", "s1": "#3987e5", "demph": "#52514e",
        "seq": ["#12203a", "#1b3a66", "#255492", "#2f6ebd", "#3987e5", "#61a0ea", "#8ab9f0"],
        "div_pos": "#3987e5", "div_neg": "#e66767",
        "good": "#0ca30c", "warning": "#c98500", "serious": "#ec835a", "critical": "#e66767",
    },
}
STATUS = {
    "healthy": ("✓", "Signal live", "good"),
    "weak": ("◐", "Signal weak", "warning"),
    "degraded": ("⚠", "Signal degraded", "critical"),
}
DISCLAIMER = "Educational analytics — not investment advice."
PROBABILITY_SENTENCE = "chance this fund falls below its peers' median return next quarter"
```

`webapp/data.py`:
```python
"""Read-only access to the extract + the in-memory search index and scorer.

The app owns NO statistics: every number it shows was precomputed by
steps/step14_webapp/extract.py. This module only fetches and ranks."""
import os
import re
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_EXTRACT = Path(__file__).parent / "data" / "extract.duckdb"


def _trigrams(s: str) -> set:
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def score_query(query: str, index: list) -> list:
    """Tiered fuzzy scorer. Tiers: 1 exact ticker, 2 ticker prefix, 3 every query token
    prefix-matches a name token, 4 trigram Jaccard >= 0.25. Dead funds demote one tier
    (honesty: they are findable, never promoted). Within tier: AUM desc."""
    q = re.sub(r"\s+", " ", query.strip().lower())
    if not q:
        return []
    q_tokens = q.split(" ")
    q_tri = _trigrams(q)
    scored = []
    for row in index:
        ticker = row["ticker"].lower() if row["ticker"] else ""
        name = row["name_normalized"]
        if q == ticker:
            tier = 1
        elif ticker.startswith(q):
            tier = 2
        elif all(any(tok.startswith(qt) for tok in name.split(" ")) for qt in q_tokens):
            tier = 3
        else:
            tri = _trigrams(name)
            overlap = len(q_tri & tri) / max(1, len(q_tri | tri))
            if overlap < 0.25:
                continue
            tier = 4
        if not row["is_active"]:
            tier += 1
        scored.append((tier, -(row["net_assets"] or 0), row))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in scored]


class ExtractStore:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get("EXTRACT_PATH", DEFAULT_EXTRACT))
        self.con = duckdb.connect(str(self.path), read_only=True)
        self.index = self.con.execute(
            "SELECT series_id, ticker, series_name, name_normalized, net_assets, "
            "is_active, last_quarter, cluster_name FROM v_fund_search"
        ).df().to_dict("records")
        self._provenance = self.con.execute(
            "SELECT * FROM v_data_provenance").df().iloc[0].to_dict()
        self._health = self.con.execute(
            "SELECT * FROM v_model_health_current").df().iloc[0].to_dict()

    def _one(self, sql, args):
        df = self.con.execute(sql, args).df()
        return df.iloc[0].to_dict() if len(df) else None

    def fund_header(self, ticker: str):
        return self._one("SELECT * FROM v_fund_header WHERE upper(ticker) = upper(?)", [ticker])

    def fund_ts(self, series_id: str) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM v_fund_peer_relative_ts WHERE series_id = ? ORDER BY quarter",
            [series_id]).df()

    def fund_percentiles(self, series_id: str):
        return self._one("SELECT * FROM v_fund_cluster_percentiles WHERE series_id = ?",
                         [series_id])

    def fund_prediction(self, series_id: str):
        return self._one("SELECT * FROM v_fund_prediction_current WHERE series_id = ?",
                         [series_id])

    def fund_prediction_history(self, series_id: str) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM v_fund_prediction_history WHERE series_id = ? AND split = 'test' "
            "ORDER BY quarter", [series_id]).df()

    def peers(self, series_id: str) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM v_peer_display WHERE series_id = ? ORDER BY peer_rank",
            [series_id]).df()

    def top_holdings(self, series_id: str) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM v_top_holdings WHERE series_id = ? ORDER BY rk", [series_id]).df()

    def model_health(self) -> dict:
        return dict(self._health)

    def model_quarters(self) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM v_model_health_quarters ORDER BY quarter").df()

    def calibration(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM v_calibration_bins ORDER BY bin_low").df()

    def provenance(self) -> dict:
        return dict(self._provenance)

    def search(self, query: str, limit: int = 8) -> list:
        return score_query(query, self.index)[:limit]

    def is_stale(self) -> bool:
        """as-of quarter end + one quarter + 60-day filing lag + 30-day grace < today."""
        q = self._provenance["last_quarter"]
        year, qn = int(q[:4]), int(q[-1])
        end_month = qn * 3
        next_due = date(year + (1 if end_month + 6 > 12 else 0),
                        (end_month + 6 - 1) % 12 + 1, 1)
        return date.today() > next_due
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_step14_search.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add webapp/__init__.py webapp/requirements.txt webapp/theme.py webapp/data.py webapp/components/__init__.py tests/test_step14_search.py
git commit -m "step14: webapp scaffold - validated theme tokens, read-only ExtractStore, tiered fuzzy search scorer"
```

---

### Task 4: Honesty components + chart option builders

**Files:**
- Create: `webapp/components/honesty.py`, `webapp/components/charts.py`
- Test: `tests/test_step14_charts.py`

**Interfaces:**
- Consumes: `webapp.theme.TOKENS/STATUS/DISCLAIMER/PROBABILITY_SENTENCE`; NiceGUI `ui`.
- Produces:
  - `honesty.status_chip(health: dict)` — renders a `ui.link` chip to `/model`; icon+label+color from `STATUS[health["health_state"]]`. Never called without a health dict.
  - `honesty.freshness_stamp(provenance: dict, fund_last_quarter: str = None)` — the slim strip: "Data as of {q} · SEC filings publish with a ~60-day lag · extract built {date}"; appends "this fund last filed {fund_last_quarter}" when it differs.
  - `honesty.probability_card(pred: dict|None, health: dict, reason: str = None)` — the ONLY way a probability renders: value in whole %, meter div with seq-ramp fill, "±N pts" when ci present, the fixed sentence, flip-rate footnote, `status_chip` inside; `pred=None` renders the "No prediction" tile with `reason`.
  - `honesty.disclaimer_line()` — the persistent one-liner.
  - `charts.fund_vs_peers_option(mode, quarters, fund_vals, median_vals, fund_label) -> dict`
  - `charts.diverging_delta_option(mode, quarters, deltas) -> dict`
  - `charts.auc_by_quarter_option(mode, df: pd.DataFrame) -> dict` (df = v_model_health_quarters, source=='retrained')
  - `charts.dumbbell_option(mode, rows: list[tuple[str, float, float]]) -> dict`
  - `charts.histogram_option(mode, values: list, bins: int = 20) -> dict`
  - `charts.calibration_option(mode, df, base_rate) -> dict`
  All pure functions returning echarts dicts — unit-testable without a browser.

- [ ] **Step 1: Write failing chart tests**

```python
"""tests/test_step14_charts.py — the honesty rules that live in chart options."""
import pandas as pd

from webapp.components.charts import (
    auc_by_quarter_option, diverging_delta_option, fund_vs_peers_option)


def test_fund_vs_peers_never_connects_nulls_and_solid_grid():
    opt = fund_vs_peers_option("light", ["2026q1", "2026q2"], [0.04, None], [0.03, 0.02], "VFIAX")
    for s in opt["series"]:
        assert s["connectNulls"] is False
    assert opt["yAxis"]["splitLine"]["lineStyle"]["type"] == "solid"
    assert len(opt["series"]) == 2  # fund + median, emphasis form


def test_fund_line_uses_s1_median_uses_demph():
    from webapp.theme import TOKENS
    opt = fund_vs_peers_option("dark", ["2026q1"], [0.01], [0.02], "X")
    assert opt["series"][0]["lineStyle"]["color"] == TOKENS["dark"]["s1"]
    assert opt["series"][1]["lineStyle"]["color"] == TOKENS["dark"]["demph"]


def test_auc_chart_has_labeled_coinflip_line_and_flags_below_chance():
    df = pd.DataFrame({"quarter": ["2025q4", "2026q1"], "auc": [0.457, 0.427],
                       "persistence_auc": [0.552, 0.569], "source": ["retrained"] * 2})
    opt = auc_by_quarter_option("light", df)
    markline = opt["series"][0]["markLine"]["data"][0]
    assert markline["yAxis"] == 0.5 and "coin flip" in markline["label"]["formatter"].lower()
    # both points below 0.5 wear the critical status color via itemStyle per-point
    from webapp.theme import TOKENS
    pts = opt["series"][0]["data"]
    assert all(p["itemStyle"]["color"] == TOKENS["light"]["critical"] for p in pts)


def test_diverging_uses_polarity_colors():
    from webapp.theme import TOKENS
    opt = diverging_delta_option("light", ["q1", "q2"], [0.01, -0.02])
    colors = [d["itemStyle"]["color"] for d in opt["series"][0]["data"]]
    assert colors == [TOKENS["light"]["div_pos"], TOKENS["light"]["div_neg"]]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_step14_charts.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `webapp/components/charts.py`**

```python
"""Pure echarts option builders. Every option obeys the dataviz rules: emphasis form
(subject in s1 blue, context in de-emphasis gray), solid hairline grid, connectNulls
false (gaps are data), no dual axes, status colors only where value truly means bad."""
import pandas as pd

from webapp.theme import TOKENS


def _base(mode):
    t = TOKENS[mode]
    return t, {
        "backgroundColor": "transparent",
        "grid": {"left": 48, "right": 90, "top": 24, "bottom": 32},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "line"}},
    }


def fund_vs_peers_option(mode, quarters, fund_vals, median_vals, fund_label):
    t, opt = _base(mode)
    opt["xAxis"] = {"type": "category", "data": quarters,
                    "axisLine": {"lineStyle": {"color": t["grid"]}},
                    "axisLabel": {"color": t["ink2"]}}
    opt["yAxis"] = {"type": "value",
                    "splitLine": {"lineStyle": {"color": t["grid"], "type": "solid"}},
                    "axisLabel": {"color": t["ink2"], "formatter": "{value}"}}
    opt["series"] = [
        {"name": fund_label, "type": "line", "data": fund_vals, "connectNulls": False,
         "lineStyle": {"width": 2, "color": t["s1"]}, "itemStyle": {"color": t["s1"]},
         "symbolSize": 5, "endLabel": {"show": True, "color": t["ink"]}},
        {"name": "Peer median", "type": "line", "data": median_vals, "connectNulls": False,
         "lineStyle": {"width": 2, "color": t["demph"]}, "itemStyle": {"color": t["demph"]},
         "symbolSize": 4, "endLabel": {"show": True, "color": t["ink2"]}},
    ]
    return opt


def diverging_delta_option(mode, quarters, deltas):
    t, opt = _base(mode)
    opt["grid"]["top"] = 8
    opt["xAxis"] = {"type": "category", "data": quarters, "axisLabel": {"show": False},
                    "axisLine": {"lineStyle": {"color": t["grid"]}}}
    opt["yAxis"] = {"type": "value",
                    "splitLine": {"show": False},
                    "axisLabel": {"color": t["ink2"]}}
    opt["series"] = [{
        "type": "bar", "barMaxWidth": 14,
        "data": [{"value": d,
                  "itemStyle": {"color": t["div_pos"] if (d or 0) >= 0 else t["div_neg"]}}
                 for d in deltas],
        "markLine": {"silent": True, "symbol": "none",
                     "lineStyle": {"color": t["muted"], "type": "solid"},
                     "data": [{"yAxis": 0}], "label": {"show": False}},
    }]
    return opt


def auc_by_quarter_option(mode, df: pd.DataFrame):
    t, opt = _base(mode)
    df = df[df["source"] == "retrained"].sort_values("quarter") if "source" in df else df
    opt["xAxis"] = {"type": "category", "data": list(df["quarter"]),
                    "axisLine": {"lineStyle": {"color": t["grid"]}},
                    "axisLabel": {"color": t["ink2"]}}
    opt["yAxis"] = {"type": "value", "min": 0.3, "max": 0.8,
                    "splitLine": {"lineStyle": {"color": t["grid"], "type": "solid"}},
                    "axisLabel": {"color": t["ink2"]}}
    model_pts = [{"value": float(a),
                  "itemStyle": {"color": t["critical"] if a < 0.5 else t["s1"]}}
                 for a in df["auc"]]
    opt["series"] = [
        {"name": "Model", "type": "line", "data": model_pts, "connectNulls": False,
         "lineStyle": {"width": 2, "color": t["s1"]}, "symbolSize": 7,
         "endLabel": {"show": True, "color": t["ink"]},
         "markLine": {"silent": True, "symbol": "none",
                      "lineStyle": {"color": t["ink2"], "type": "solid", "width": 1},
                      "data": [{"yAxis": 0.5,
                                "label": {"formatter": "coin flip (0.5)",
                                          "color": t["ink2"]}}]}},
        {"name": "Mean-reversion rule", "type": "line",
         "data": [float(v) if pd.notna(v) else None for v in df["persistence_auc"]],
         "connectNulls": False, "lineStyle": {"width": 2, "color": t["demph"]},
         "itemStyle": {"color": t["demph"]}, "symbolSize": 4,
         "endLabel": {"show": True, "color": t["ink2"]}},
    ]
    return opt


def dumbbell_option(mode, rows):
    """rows = [(label, before, after)] -> horizontal dumbbell, one hue two shades."""
    t, opt = _base(mode)
    labels = [r[0] for r in rows]
    opt["tooltip"] = {"trigger": "item"}
    opt["xAxis"] = {"type": "value", "min": 0.4, "max": 0.8,
                    "splitLine": {"lineStyle": {"color": t["grid"], "type": "solid"}},
                    "axisLabel": {"color": t["ink2"]}}
    opt["yAxis"] = {"type": "category", "data": labels, "axisLabel": {"color": t["ink"]}}
    opt["series"] = [
        {"type": "custom", "renderItem": None,  # connector drawn by the two scatters + line
         "data": []},
        {"name": "before", "type": "scatter", "symbolSize": 10,
         "itemStyle": {"color": t["seq"][2]},
         "data": [[r[1], i] for i, r in enumerate(rows)]},
        {"name": "after", "type": "scatter", "symbolSize": 10,
         "itemStyle": {"color": t["seq"][5]},
         "data": [[r[2], i] for i, r in enumerate(rows)]},
        {"name": "gap", "type": "lines", "coordinateSystem": "cartesian2d",
         "lineStyle": {"color": t["seq"][3], "width": 2},
         "data": [{"coords": [[r[1], i], [r[2], i]]} for i, r in enumerate(rows)]},
    ]
    opt["series"] = [s for s in opt["series"] if s.get("type") != "custom"]
    return opt


def histogram_option(mode, values, bins=20):
    t, opt = _base(mode)
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int(float(v) * bins))] += 1
    opt["tooltip"] = {"trigger": "item"}
    opt["xAxis"] = {"type": "category",
                    "data": [f"{i / bins:.0%}" for i in range(bins)],
                    "axisLabel": {"color": t["ink2"], "interval": 4},
                    "axisLine": {"lineStyle": {"color": t["grid"]}}}
    opt["yAxis"] = {"type": "value",
                    "splitLine": {"lineStyle": {"color": t["grid"], "type": "solid"}},
                    "axisLabel": {"color": t["ink2"]}}
    opt["series"] = [{"type": "bar", "data": counts, "barCategoryGap": "10%",
                      "itemStyle": {"color": t["seq"][4]}}]
    return opt


def calibration_option(mode, df: pd.DataFrame, base_rate: float):
    t, opt = _base(mode)
    opt["tooltip"] = {"trigger": "item"}
    opt["xAxis"] = {"type": "value", "min": 0, "max": 1, "name": "predicted",
                    "splitLine": {"lineStyle": {"color": t["grid"], "type": "solid"}},
                    "axisLabel": {"color": t["ink2"]}}
    opt["yAxis"] = {"type": "value", "min": 0, "max": 1, "name": "actual lag rate",
                    "splitLine": {"lineStyle": {"color": t["grid"], "type": "solid"}},
                    "axisLabel": {"color": t["ink2"]}}
    opt["series"] = [
        {"name": "bins", "type": "scatter",
         "symbolSize": [max(6, min(20, int(n ** 0.5))) for n in df["n"]],
         "itemStyle": {"color": t["s1"]},
         "data": df[["predicted_mean", "actual_lag_rate"]].values.tolist()},
        {"name": "perfect", "type": "line", "data": [[0, 0], [1, 1]], "symbol": "none",
         "lineStyle": {"color": t["demph"], "type": "solid", "width": 1},
         "markLine": {"silent": True, "symbol": "none",
                      "lineStyle": {"color": t["muted"], "type": "solid"},
                      "data": [{"yAxis": base_rate,
                                "label": {"formatter": f"base rate {base_rate:.0%}",
                                          "color": t["ink2"]}}]}},
    ]
    return opt
```

- [ ] **Step 4: Implement `webapp/components/honesty.py`**

```python
"""The honesty pattern components. Placement rule enforced by construction:
probability_card is the ONLY probability renderer and it embeds status_chip."""
from nicegui import ui

from webapp.theme import DISCLAIMER, PROBABILITY_SENTENCE, STATUS, TOKENS


def status_chip(health: dict) -> None:
    icon, label, token = STATUS[health["health_state"]]
    color = TOKENS["light"][token]
    with ui.link(target="/model").classes("no-underline"):
        ui.label(f"{icon} {label}").classes(
            "px-2 py-0.5 rounded text-xs font-semibold border"
        ).style(f"color:{color}; border-color:{color}")


def freshness_stamp(provenance: dict, fund_last_quarter: str = None) -> None:
    parts = [f"Data as of {provenance['last_quarter']}",
             "SEC filings publish with a ~60-day lag",
             f"extract built {str(provenance.get('refreshed_at', ''))[:10]}"]
    if fund_last_quarter and fund_last_quarter != provenance["last_quarter"]:
        parts.append(f"this fund last filed {fund_last_quarter}")
    ui.label(" · ".join(parts)).classes("text-xs text-gray-500")


def disclaimer_line() -> None:
    ui.label(DISCLAIMER).classes("text-xs text-gray-500")


def probability_card(pred: dict, health: dict, reason: str = None) -> None:
    """The only legal rendering of a lag probability (design.md 'uncertain_probability')."""
    t = TOKENS["light"]
    with ui.card().classes("w-full max-w-sm p-4"):
        ui.label("Next-quarter outlook").classes("text-sm font-semibold")
        if pred is None:
            ui.label("No prediction this quarter").classes("text-lg")
            ui.label(reason or "This fund is not in the current forward book."
                     ).classes("text-sm text-gray-500")
            status_chip(health)
            return
        p = float(pred["predicted_probability"])
        pct = round(p * 100)
        interval = ""
        if pred.get("ci_low") is not None and not _isna(pred.get("ci_low")):
            half = (float(pred["ci_high"]) - float(pred["ci_low"])) / 2 * 100
            interval = f" ±{round(half)} pts"
        ui.label(f"{pct}%{interval}").classes("text-4xl font-semibold")
        # the meter: seq-ramp fill keyed to the value, hairline tick at 50%
        fill = t["seq"][min(6, 2 + int(p * 5))]
        ui.html(
            f'<div style="position:relative;height:10px;border-radius:5px;'
            f'background:{t["seq"][0]}">'
            f'<div style="width:{pct}%;height:10px;border-radius:5px;background:{fill}"></div>'
            f'<div style="position:absolute;left:50%;top:-2px;width:1px;height:14px;'
            f'background:{t["muted"]}"></div></div>')
        ui.label(f"{PROBABILITY_SENTENCE} ({pred['target_quarter']})"
                 ).classes("text-sm text-gray-600")
        if pred.get("flip_rate") is not None and not _isna(pred.get("flip_rate")):
            ui.label(f"The outcome itself is noisy: this fund's lag/lead label flips in "
                     f"{float(pred['flip_rate']):.0%} of peer-set draws."
                     ).classes("text-xs text-gray-500")
        status_chip(health)
        ui.label("A probability from a model whose live scorecard is public."
                 ).classes("text-xs text-gray-500")


def _isna(v) -> bool:
    return v != v  # NaN check without importing pandas here
```

- [ ] **Step 5: Run chart tests**

Run: `python -m pytest tests/test_step14_charts.py -q`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add webapp/components/charts.py webapp/components/honesty.py tests/test_step14_charts.py
git commit -m "step14: honesty components (chip, stamp, probability card) + rule-enforcing chart builders"
```

---

### Task 5: Fund page

**Files:**
- Create: `webapp/pages/__init__.py` (empty), `webapp/pages/fund.py`
- Test: `tests/test_step14_app.py` (new)

**Interfaces:**
- Consumes: `ExtractStore` methods, `honesty.*`, `charts.fund_vs_peers_option/diverging_delta_option`.
- Produces: `render_fund(store: ExtractStore, ticker: str) -> None` — full page body (no `@ui.page` decorator here; main.py routes). Also `kpi_row(header, ts, pct)` internal.

- [ ] **Step 1: Write failing smoke tests**

```python
"""tests/test_step14_app.py — NiceGUI User-fixture smoke tests against a synthetic extract.

Builds a tiny extract in tmp_path via the same builders the real one uses, sets
EXTRACT_PATH, and boots the app. The edge-state milestone test from the design: the dead
fund and the missing-prediction fund must render as designed states, not errors."""
import os
import pytest
from nicegui.testing import User

pytest_plugins = ["nicegui.testing.user_plugin"]


@pytest.fixture(scope="module", autouse=True)
def extract_env(tmp_path_factory):
    # Build a synthetic extract with the Task 1/2 builders (same fixtures as
    # test_step14_extract, extracted into tests/step14_fixtures.py by this task).
    from tests.step14_fixtures import build_synthetic_extract
    path = build_synthetic_extract(tmp_path_factory.mktemp("extract") / "extract.duckdb")
    os.environ["EXTRACT_PATH"] = str(path)
    yield
    os.environ.pop("EXTRACT_PATH", None)


@pytest.fixture(autouse=True)
def app_routes(extract_env):
    import webapp.main  # noqa: F401  (registers @ui.page routes on import)


async def test_fund_page_renders_with_honesty_elements(user: User):
    await user.open("/fund/AAAAX")
    await user.should_see("Alpha Large Blend Fund")
    await user.should_see("Signal degraded")            # chip present
    await user.should_see("Data as of 2026q2")           # freshness stamp
    await user.should_see("not investment advice")       # disclaimer
    await user.should_see("peers' median return")        # fixed sentence


async def test_dead_fund_is_archive_not_error(user: User):
    await user.open("/fund/DDDDX")
    await user.should_see("left the universe")
    await user.should_see("No forward prediction")


async def test_unknown_ticker_offers_search(user: User):
    await user.open("/fund/ZZZZ")
    await user.should_see("No fund matches")
```

Also create `tests/step14_fixtures.py`: move the two synthetic-source fixtures from `tests/test_step14_extract.py` into plain functions `make_synthetic_source(path)` (all tables from both fixtures, plus `full_panel` empty) and `build_synthetic_extract(out_path)` that calls `extract.build_fund_views/build_model_views(src, cfg=None)/build_cluster_views` and writes the extract exactly like `extract.run` does (refactor `run`'s write loop into `write_extract(views, out_path)` so both share it). Update `tests/test_step14_extract.py` to import from `tests/step14_fixtures.py`.

- [ ] **Step 2: Install test deps and run to verify failure**

Run: `pip install pytest-asyncio` then `python -m pytest tests/test_step14_app.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'webapp.main'` (main.py is Task 7; write `webapp/pages/fund.py` first and a minimal `webapp/main.py` stub is allowed to land in this task as long as Task 7 replaces it — OR mark the three tests `@pytest.mark.xfail(reason="main.py lands in task 7")` and flip in Task 7. Prefer the minimal stub: it keeps tests green per task.)

- [ ] **Step 3: Implement `webapp/pages/fund.py`**

```python
"""Fund page - zones A (outlook) / B (KPI row) / C (vs-peers chart) / D (peers grid),
plus prediction history and edge states. Content hierarchy per the UX spec."""
import pandas as pd
from nicegui import ui

from webapp.components import charts, honesty
from webapp.theme import TOKENS


def _fmt_pct(v, digits=1):
    return "—" if v is None or v != v else f"{v * 100:.{digits}f}%"


def render_fund(store, ticker: str) -> None:
    header = store.fund_header(ticker)
    if header is None:
        ui.label(f"No fund matches '{ticker}'.").classes("text-xl")
        ui.label("Try a ticker or fund name:").classes("text-sm text-gray-500")
        _search_box(store)
        return

    sid = header["series_id"]
    health = store.model_health()
    prov = store.provenance()
    ts = store.fund_ts(sid)
    pct = store.fund_percentiles(sid)

    # Header block
    with ui.row().classes("items-baseline gap-3"):
        ui.label(f"{header['ticker']} · {header['series_name']}").classes(
            "text-2xl font-semibold")
        if header.get("cluster_name"):
            ui.link(f"● {header['cluster_name']}", "/methodology#clusters").classes("text-sm")
    ui.label(f"{header['yahoo_category'] or '—'} · "
             f"${(header['net_assets'] or 0) / 1e9:.1f}B net assets").classes(
        "text-sm text-gray-600")
    honesty.freshness_stamp(prov, fund_last_quarter=header["last_quarter"])

    # Edge state: dead fund -> archive banner instead of outlook
    is_active = bool(header["is_active"])
    if not is_active:
        with ui.card().classes("w-full p-3 border").style(
                f"border-color:{TOKENS['light']['serious']}"):
            ui.label(f"⚑ This fund left the universe after {header['last_quarter']} — "
                     "final quarters shown; No forward prediction exists."
                     ).classes("text-sm font-semibold")

    with ui.row().classes("w-full gap-4 items-start"):
        # Zone A
        if is_active:
            pred = store.fund_prediction(sid)
            reason = None if pred else "insufficient coverage this quarter"
            honesty.probability_card(pred, health, reason=reason)
        # Zone B - KPI row
        with ui.row().classes("gap-3 flex-wrap"):
            _kpi_tiles(ts, pct)

    # Zone C - fund vs peer median + diverging strip + table twin
    quarters = list(ts["quarter"])
    fund_vals = [None if pd.isna(v) else float(v) for v in ts["quarterly_return"]]
    median_vals = [None if pd.isna(v) else float(v) for v in ts["cluster_median_return"]]
    deltas = [None if pd.isna(v) else float(v) for v in ts["return_vs_cluster_median"]]
    missing = [q for q, v in zip(quarters, fund_vals) if v is None]
    ui.label("Fund vs peer median, quarterly return").classes("text-lg font-semibold mt-4")
    chart_box = ui.column().classes("w-full")
    with chart_box:
        ui.echart(charts.fund_vs_peers_option(
            "light", quarters, fund_vals, median_vals, header["ticker"])
        ).classes("w-full h-64")
        ui.echart(charts.diverging_delta_option("light", quarters, deltas)
                  ).classes("w-full h-24")
        if missing:
            ui.label(f"No N-PORT filing for {', '.join(missing)}."
                     ).classes("text-xs text-gray-500")
    table_box = ui.column().classes("w-full hidden")
    with table_box:
        ui.table(rows=ts[["quarter", "quarterly_return", "cluster_median_return",
                          "return_vs_cluster_median"]].round(4).to_dict("records"))

    def _toggle():
        chart_box.classes(toggle="hidden")
        table_box.classes(toggle="hidden")
    ui.button("chart ⇄ table", on_click=_toggle).props("flat dense")

    # Zone D - peers
    peers = store.peers(sid)
    if len(peers):
        ui.label("Most-similar peers (by reported holdings)").classes(
            "text-lg font-semibold mt-4")
        ui.label("Similarity of holdings, not of future returns."
                 ).classes("text-xs text-gray-500")
        ui.aggrid({
            "columnDefs": [
                {"headerName": "Ticker", "field": "peer_ticker",
                 "cellRenderer": "agGroupCellRenderer"},
                {"headerName": "Name", "field": "peer_name", "flex": 2},
                {"headerName": "Similarity", "field": "cosine_similarity",
                 "valueFormatter": "value == null ? '—' : (value*100).toFixed(0) + '%'"},
                {"headerName": "Trailing 4Q return", "field": "peer_trailing_4q_return",
                 "valueFormatter": "value == null ? '—' : (value*100).toFixed(1) + '%'"},
                {"headerName": "Expense", "field": "peer_expense_net",
                 "valueFormatter": "value == null ? '—' : (value*100).toFixed(2) + '%'"},
            ],
            "rowData": peers.to_dict("records"),
            "defaultColDef": {"sortable": True, "resizable": True},
        }).classes("w-full").on("cellClicked",
                                lambda e: ui.navigate.to(f"/fund/{e.args['data']['peer_ticker']}"))

    # Prediction history - the per-fund misses table
    hist = store.fund_prediction_history(sid)
    if len(hist):
        ui.label("This fund's past predictions vs what happened").classes(
            "text-lg font-semibold mt-4")
        rows = [{"quarter": r.quarter,
                 "predicted": f"{r.predicted_probability * 100:.0f}%",
                 "actual": "lagged" if r.actual_label == 1 else "did not lag"}
                for r in hist.itertuples()]
        ui.table(rows=rows)

    honesty.disclaimer_line()


def _kpi_tiles(ts: pd.DataFrame, pct: dict) -> None:
    t = TOKENS["light"]
    if not len(ts):
        return
    last = ts.iloc[-1]
    tiles = [
        ("Return", last["quarterly_return"], last["cluster_median_return"], False),
    ]
    for label, val, ref, invert in tiles:
        delta = None if pd.isna(val) or pd.isna(ref) else float(val) - float(ref)
        good = delta is not None and ((delta < 0) if invert else (delta > 0))
        color = t["good"] if good else t["critical"]
        with ui.card().classes("p-3"):
            ui.label(label).classes("text-xs text-gray-500")
            ui.label(_fmt_pct(val)).classes("text-xl font-semibold")
            if delta is not None:
                ui.label(f"{'+' if delta >= 0 else ''}{delta * 100:.1f} vs peer median"
                         ).classes("text-xs").style(f"color:{color}")
    if pct:
        for label, key in [("Volatility pct'ile", "pctile_volatility"),
                           ("Sharpe pct'ile", "pctile_sharpe"),
                           ("Fees pct'ile", "pctile_expense_net")]:
            v = pct.get(key)
            with ui.card().classes("p-3"):
                ui.label(label + " (in cluster)").classes("text-xs text-gray-500")
                ui.label("n/a — cluster too small" if v is None or v != v
                         else f"{float(v) * 100:.0f}").classes("text-xl font-semibold")


def _search_box(store) -> None:
    from webapp.main import omnibox  # shared component, defined in main.py
    omnibox(store)
```

- [ ] **Step 4: Minimal `webapp/main.py` stub (replaced by Task 7)** — routes `/fund/{ticker}` to `render_fund`, defines `omnibox(store)` as a plain `ui.input` placeholder, creates a module-level `STORE = ExtractStore()`.

```python
"""App shell STUB - Task 7 replaces this with the full header/omnibox/dark-mode shell."""
from nicegui import ui

from webapp.data import ExtractStore
from webapp.pages.fund import render_fund

STORE = ExtractStore()


def omnibox(store) -> None:
    ui.input(placeholder=f"Search {len(store.index)} funds…")


@ui.page("/fund/{ticker}")
def fund_page(ticker: str):
    render_fund(STORE, ticker)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host="0.0.0.0", port=7860, show=False, title="FundsPeers")
```

- [ ] **Step 5: Run smoke tests**

Run: `python -m pytest tests/test_step14_app.py tests/test_step14_extract.py -q`
Expected: all PASS (ExtractStore reads EXTRACT_PATH; module-level STORE must construct lazily if import order bites — if it does, wrap STORE in `functools.lru_cache` getter `get_store()` and have pages call it; keep that refactor within this task.)

- [ ] **Step 6: Commit**

```bash
git add webapp/pages/__init__.py webapp/pages/fund.py webapp/main.py tests/test_step14_app.py tests/step14_fixtures.py tests/test_step14_extract.py
git commit -m "step14: fund page - outlook card, KPI tiles, vs-peers chart with table twin, peers grid, misses table, designed edge states"
```

---

### Task 6: Model-health page + Methodology page

**Files:**
- Create: `webapp/pages/model.py`, `webapp/pages/methodology.py`
- Modify: `webapp/main.py` (add routes)
- Test: `tests/test_step14_app.py` (append)

**Interfaces:**
- Consumes: `store.model_health/model_quarters/calibration/provenance`, `charts.auc_by_quarter_option/dumbbell_option/histogram_option/calibration_option`, `honesty.*`.
- Produces: `render_model(store)`, `render_methodology(store)`.

- [ ] **Step 1: Failing tests (append)**

```python
async def test_model_page_leads_with_verdict_and_shows_coinflip(user: User):
    await user.open("/model")
    await user.should_see("Can you trust the lag-probability signal right now?")
    await user.should_see("Signal degraded")
    await user.should_see("coin flip")          # AUC explainer / markline caption
    await user.should_see("The rule:")          # disclosed rule text


async def test_methodology_has_disclaimer_and_survivorship(user: User):
    await user.open("/methodology")
    await user.should_see("not investment advice")
    await user.should_see("Dead funds are included")
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_step14_app.py -q` → the two new tests FAIL (404 route).

- [ ] **Step 3: Implement `webapp/pages/model.py`**

```python
"""Model health - the page the product's credibility rests on. Verdict first, then evidence."""
from nicegui import ui

from webapp.components import charts, honesty
from webapp.theme import STATUS, TOKENS


def render_model(store) -> None:
    health = store.model_health()
    prov = store.provenance()
    t = TOKENS["light"]

    ui.label("Can you trust the lag-probability signal right now?").classes(
        "text-2xl font-semibold")
    honesty.freshness_stamp(prov)

    icon, label, token = STATUS[health["health_state"]]
    with ui.card().classes("w-full p-4 border-2").style(f"border-color:{t[token]}"):
        ui.label(f"{icon} {label.upper()}").classes("text-xl font-semibold").style(
            f"color:{t[token]}")
        ui.label(health["rule_text"]).classes("text-sm")
        ui.label(f"The rule: {health['rule_text']}").classes("text-xs text-gray-500")

    ui.label("Realized AUC by quarter vs baselines").classes("text-lg font-semibold mt-4")
    dfq = store.model_quarters()
    ui.echart(charts.auc_by_quarter_option("light", dfq)).classes("w-full h-72")
    with ui.expansion("table view"):
        ui.table(rows=dfq.round(3).to_dict("records"))

    # Plain-English AUC explainer, inline
    live = health.get("pooled_live_auc")
    if live:
        n = round(live * 100)
        ui.label(
            f"What AUC means: pick one fund that went on to lag its peers and one that "
            f"didn't. AUC {live:.3f} means the model gave the lagger the higher warning "
            f"about {n} times out of 100. A coin flip gets 50; recently this model has "
            f"been below 50.").classes("text-sm text-gray-700 max-w-2xl")

    ui.label("Backtest vs reality").classes("text-lg font-semibold mt-4")
    rows = [("backtest (hindsight)", health["backtest_auc"], health["backtest_auc"]),
            ("committed forward → reality", health["backtest_auc"],
             health["pooled_live_auc"]),
            ("last realized quarter", health["pooled_live_auc"], health["auc_last"])]
    rows = [(a, float(b), float(c)) for a, b, c in rows if b is not None and c is not None]
    ui.echart(charts.dumbbell_option("light", rows)).classes("w-full h-48")
    ui.label("The backtest is measured on the past with hindsight of the modeling "
             "choices — shown for contrast, never as the headline."
             ).classes("text-xs text-gray-500")

    ui.label("Calibration (held-out test quarters)").classes("text-lg font-semibold mt-4")
    calib = store.calibration()
    if len(calib):
        ui.echart(charts.calibration_option(
            "light", calib, float(health["base_rate"] or 0.5))).classes("w-full h-64")
        ui.label('Reading: when the model said "70%", how often did funds actually lag?'
                 ).classes("text-xs text-gray-500")

    if health.get("label_noise_floor") is not None:
        ui.label(f"Label noise floor: ~{float(health['label_noise_floor']):.0%} of "
                 "lag/lead labels flip under peer-set perturbation — a ceiling on any "
                 "model of this target.").classes("text-sm text-gray-700")

    ui.label("Open question, published: two consecutive below-chance quarters is under "
             "investigation; the signal may be retired or retrained."
             ).classes("text-sm font-semibold mt-2")
    honesty.disclaimer_line()
```

- [ ] **Step 4: Implement `webapp/pages/methodology.py`** — static prose page: data source (SEC N-PORT/RR bulk data sets, quarterly, ~60-day lag), what holdings-clustering means, the label definition (below median of top-10 peers), model limitations (relative, binary, one quarter), survivorship ("Dead funds are included in history; forward-prediction attrition is counted and disclosed, never imputed."), Yahoo categories third-party note, full disclaimer text, anchors `#clusters` and `#disclaimer`. All `ui.label`/`ui.markdown`, content copied from design.md's methodology section.

- [ ] **Step 5: Add routes to main.py stub**

```python
from webapp.pages.model import render_model
from webapp.pages.methodology import render_methodology

@ui.page("/model")
def model_page():
    render_model(STORE)

@ui.page("/methodology")
def methodology_page():
    render_methodology(STORE)
```

- [ ] **Step 6: Run tests** — `python -m pytest tests/test_step14_app.py -q` → all PASS.

- [ ] **Step 7: Commit**

```bash
git add webapp/pages/model.py webapp/pages/methodology.py webapp/main.py tests/test_step14_app.py
git commit -m "step14: model-health page (verdict-first, disclosed rule, coin-flip line, dumbbell, calibration) + methodology"
```

---

### Task 7: App shell — header, omnibox, dark mode, 404, stale banner

**Files:**
- Modify: `webapp/main.py` (replace stub with the real shell)
- Test: `tests/test_step14_app.py` (append)

**Interfaces:**
- Consumes: everything above.
- Produces: `layout(store)` context manager used by every page: header (brand link → `/`, tabs Funds/Model health, omnibox, status chip, dark toggle), freshness+disclaimer strip, stale-extract banner when `store.is_stale()`. `omnibox(store)` — real implementation. Root `/` renders hero search + model-status card (step15 replaces with full home). `/fund/{ticker}` 404 behavior stays in `render_fund`.

- [ ] **Step 1: Failing tests (append)**

```python
async def test_root_shows_hero_search_and_status_card(user: User):
    await user.open("/")
    await user.should_see("Search")
    await user.should_see("Signal degraded")


async def test_header_disclaimer_on_every_page(user: User):
    await user.open("/model")
    await user.should_see("not investment advice")
```

- [ ] **Step 2: Run to verify failure** (root route missing) then **Step 3: Implement**

```python
"""webapp/main.py - app shell: header with omnibox + status chip, layout wrapper, routes."""
from contextlib import contextmanager
from functools import lru_cache

from nicegui import ui

from webapp.components import honesty
from webapp.data import ExtractStore
from webapp.pages.fund import render_fund
from webapp.pages.methodology import render_methodology
from webapp.pages.model import render_model
from webapp.theme import DISCLAIMER


@lru_cache(maxsize=1)
def get_store() -> ExtractStore:
    return ExtractStore()


def omnibox(store) -> None:
    results = ui.column().classes("absolute z-50 bg-white shadow rounded w-96 hidden")

    def on_change(e) -> None:
        results.clear()
        hits = store.search(e.value or "", limit=8)
        results.classes(remove="hidden" if hits else None, add=None if hits else "hidden")
        with results:
            for h in hits:
                tag = "" if h["is_active"] else f" (left universe {h['last_quarter']})"
                ui.link(f"{h['ticker']}  {h['series_name']}{tag}",
                        f"/fund/{h['ticker']}").classes(
                    "px-3 py-1 text-sm no-underline" + ("" if h["is_active"] else " opacity-60"))
            if not hits and (e.value or ""):
                ui.label(f"No fund matches '{e.value}'.").classes("px-3 py-1 text-sm")

    box = ui.input(placeholder=f"Search {len(store.index)} funds by ticker or name…",
                   on_change=on_change).props("dense outlined debounce=120").classes("w-96")
    ui.keyboard(on_key=lambda e: box.run_method("focus")
                if e.key == "k" and e.modifiers.ctrl and e.action.keydown else None)


@contextmanager
def layout(store):
    dark = ui.dark_mode()
    with ui.header().classes("items-center gap-4 bg-transparent border-b px-4 py-2"):
        ui.link("◆ FundsPeers", "/").classes("text-lg font-semibold no-underline")
        ui.link("Funds", "/").classes("no-underline text-sm")
        ui.link("Model health", "/model").classes("no-underline text-sm")
        ui.link("Methodology", "/methodology").classes("no-underline text-sm")
        omnibox(store)
        honesty.status_chip(store.model_health())
        ui.button(icon="dark_mode", on_click=dark.toggle).props("flat dense round")
    with ui.row().classes("w-full px-4 py-1 gap-3 text-xs text-gray-500 border-b"):
        honesty.freshness_stamp(store.provenance())
        ui.label(DISCLAIMER)
    if store.is_stale():
        with ui.row().classes("w-full px-4 py-2 bg-yellow-100"):
            ui.label(f"This data ends {store.provenance()['last_quarter']} and a newer "
                     "quarter should exist — the extract is stale.").classes("text-sm")
    with ui.column().classes("w-full max-w-screen-xl mx-auto p-4") as content:
        yield content


@ui.page("/")
def home():
    store = get_store()
    with layout(store):
        ui.label("Which funds will lag their peers?").classes("text-3xl font-semibold")
        ui.label("Search a fund to see its holdings-based peer group, honest "
                 "peer-relative record, and the model's live scorecard."
                 ).classes("text-sm text-gray-600")
        health = store.model_health()
        with ui.card().classes("p-4 max-w-md"):
            ui.label("Can you trust the signal?").classes("font-semibold")
            honesty.status_chip(health)
            ui.label(health["rule_text"]).classes("text-sm text-gray-600")
            ui.link("See the evidence →", "/model")


@ui.page("/fund/{ticker}")
def fund_page(ticker: str):
    store = get_store()
    with layout(store):
        render_fund(store, ticker)


@ui.page("/model")
def model_page():
    store = get_store()
    with layout(store):
        render_model(store)


@ui.page("/methodology")
def methodology_page():
    store = get_store()
    with layout(store):
        render_methodology(store)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host="0.0.0.0", port=7860, show=False, title="FundsPeers",
           favicon="◆", reload=False)
```

(Adjust Task 5/6 route registrations: they move here; pages export only `render_*`.)

- [ ] **Step 4: Run the whole suite** — `python -m pytest tests/test_step14_app.py tests/test_step14_search.py tests/test_step14_charts.py tests/test_step14_extract.py -q` → all PASS.

- [ ] **Step 5: Run the app locally against the real extract and eyeball**

Run: `python -m webapp.main` then open `http://localhost:7860/fund/<a real ticker from v_fund_search>`, `/model`, `/`, a dead fund's ticker, `/fund/ZZZZ`.
Expected: all render; degraded chip everywhere; edge states designed.

- [ ] **Step 6: Commit**

```bash
git add webapp/main.py webapp/pages/fund.py webapp/pages/model.py tests/test_step14_app.py
git commit -m "step14: app shell - header omnibox with tiered search, status chip, freshness+disclaimer strip, stale banner, home"
```

---

### Task 8: Dockerfile, HF Space deploy, advance.py wiring, docs, UAT

**Files:**
- Create: `webapp/Dockerfile`, `steps/step14_webapp/deploy.py`
- Modify: `steps/step13_automation/advance.py` (`_stage_dashboard`), `tests/test_step13_advance.py` (stub the new call), `decisions.json`, `workflow.json`
- Test: manual UAT + existing suites

**Interfaces:**
- Consumes: `extract.run`, `webapp/` as a folder.
- Produces: `deploy.py: deploy(space_id: str, token=None)` using `huggingface_hub.HfApi` — `create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)` then `upload_folder(folder_path="webapp", repo_id=space_id, repo_type="space")`. CLI: `python -m steps.step14_webapp.deploy --space <user>/fundspeers`.

- [ ] **Step 1: `webapp/Dockerfile`**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/webapp
ENV PYTHONPATH=/app
EXPOSE 7860
CMD ["python", "-m", "webapp.main"]
```

- [ ] **Step 2: `steps/step14_webapp/deploy.py`**

```python
"""Deploy webapp/ (including the current extract) to a Hugging Face Space. Manual and
human-gated - never called by advance.py. Requires `hf auth login` (or HF_TOKEN)."""
import argparse
import logging

from huggingface_hub import HfApi

log = logging.getLogger(__name__)


def deploy(space_id: str) -> str:
    api = HfApi()
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path="webapp", repo_id=space_id, repo_type="space",
                      ignore_patterns=["__pycache__", "*.pyc", "data/.gitkeep"])
    url = f"https://huggingface.co/spaces/{space_id}"
    log.info("deployed: %s", url)
    return url


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="e.g. <user>/fundspeers")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    deploy(args.space)
```

Install: `pip install huggingface_hub` (dev machine only; not in webapp/requirements.txt).

- [ ] **Step 3: Wire extract rebuild into advance.py**

In `steps/step13_automation/advance.py`, `_stage_dashboard` gains one line after the static build:

```python
def _stage_dashboard(cfg: dict) -> None:
    # The refresh operates on the _full universe end-to-end; the dashboard must too
    # (the defaults would silently rebuild the stale small-universe _all dashboard).
    dashboard_build.run(cfg, narrative_mode="cached", table_suffix="_full",
                        predictions_table="full_predictions",
                        eval_table="full_model_eval",
                        stability_table="full_label_stability")
    webapp_extract.run(cfg)  # keep the hosted app's extract in sync (deploy stays manual)
```

with `from steps.step14_webapp import extract as webapp_extract` in the imports. Update `tests/test_step13_advance.py`: any test that stubs `_stage_dashboard` internals gains a `monkeypatch.setattr(advance.webapp_extract, "run", lambda cfg: None)`.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest -q`
Expected: all PASS (whole repo).

- [ ] **Step 5: Deploy and UAT (needs `hf auth login` once)**

```bash
python -m steps.step14_webapp.extract     # fresh extract from the real DB
python -m steps.step14_webapp.deploy --space <user>/fundspeers
```

UAT checklist (from design.md, verify on the live Space URL):
- [ ] `/fund/<large live ticker>` renders with real 2026q2 data; degraded chip visible
- [ ] `/model` shows verdict card, per-quarter AUC with coin-flip line, dumbbell, calibration
- [ ] Search finds a fund by misspelled name ("vangard")
- [ ] A dead fund's page renders as archive (banner, no outlook meter)
- [ ] `/fund/ZZZZ` shows the no-match + search state
- [ ] Freshness stamp + disclaimer visible on every page without scrolling

- [ ] **Step 6: Docs + gate commit**

Append a `decisions.json` entry (step14 shipped: what/where/URL, the per-tree interval decision, the extract-in-refresh wiring). Set `workflow.json` step14_webapp → `awaiting_approval`. Regenerate docs (`python scripts/render_docs.py`).

```bash
git add -A
git commit -m "step14: dynamic dashboard part 1 live - extract in refresh loop, HF Space deployed, UAT recorded"
```

**STOP at the approval gate** — report UAT results + Space URL; the human clears step14 and starts step15.

---

## Self-Review (performed)

1. **Spec coverage:** extract views (Tasks 1–2, all 14 + holdings), health rule (T2), search tiers incl. dead-fund demotion (T3), honesty components + never-naked probability (T4), fund page zones A–D + misses + edge states (T5), model page verdict/coin-flip/dumbbell/calibration/noise floor/retirement note (T6), methodology + survivorship (T6), shell/stale banner/dark/404 (T7 + render_fund), Dockerfile/deploy/advance wiring (T8). Deferred to step15 per spec: cluster pages, home hero map, 5-family palette usage, `?cmp=` overlay. Gaps accepted per spec: KPI sparklines simplified to value+delta tiles (spec's Zone B sparkline is additive, not load-bearing — noted for step15 polish); recents-in-localStorage replaced by largest-funds default (spec allows).
2. **Placeholder scan:** none — every step has code or an exact command. Task 6 Step 4 (methodology) is prose-from-spec, content enumerated.
3. **Type consistency:** `ExtractStore` method names match between Task 3 definition and Task 5–7 consumers; `compute_health_state` tuple contract consistent; `charts.*option(mode, ...)` signatures consistent; `get_store()` refactor noted where the stub's module-level STORE appears.
