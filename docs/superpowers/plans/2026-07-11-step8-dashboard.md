# step8_dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One self-contained, offline, interactive HTML dashboard (`reports/cluster_dashboard.html`) over the step7 unified universe, digestible by a financial analyst / DIY investor / advisor: cluster index, per-cluster allocation/performance/member views with forward underperformance probabilities, an honest model scorecard, cached phi-4 narratives, and a persistent liability disclaimer.

**Architecture:** Three modules in `steps/step8_dashboard/`: `data.py` (assemble one JSON payload from the step7 tables), `narratives.py` (per-cluster phi-4 paragraphs, cached in table `dashboard_narratives`), `render.py` (payload + template → single HTML file with embedded JSON, vanilla JS, hand-rolled SVG). `build.py` is the CLI. Deterministic: same tables → byte-identical HTML.

**Tech Stack:** Python (pandas, json, html), vanilla JS + SVG (no libraries, no CDNs), OpenAI client → LM Studio (existing step5 pattern), pytest.

**Authoritative spec:** `steps/step8_dashboard/design.md` (committed). Read it — and **read the `dataviz` skill before writing any chart markup** (Task 3).

**Prerequisite:** step7 executed and approved — tables `funds_all, cluster_definitions_all, fund_metrics_overall_all, fund_metrics_quarterly_all, unified_predictions, unified_model_eval, unified_label_stability, cluster_map_coords_all, holdings_all, fund_peers_all` all exist.

## Global Constraints

- Output file: `reports/cluster_dashboard.html`. Self-contained: zero external requests (no `<script src=`, `<link href=`, remote images/fonts). All data embedded as one JSON blob.
- Deterministic: two builds from the same tables are byte-identical. No timestamps, no unseeded ordering (sort every list explicitly). Narratives come from the cache table, never regenerated implicitly.
- Disclaimer text (exact, non-negotiable, footer on every view + probability column note): "Educational and informational purposes only. Not investment advice or a recommendation. Predictions are statistical estimates that may be highly inaccurate. No liability or responsibility is assumed for decisions made based on this material. Past performance does not guarantee future results."
- Allocation-segment funds: descriptive stats only, grouped by vintage — NO model probabilities.
- The dashboard reflects the LATEST quarter (2024q4) for membership/predictions, whole-period metrics for performance.
- CLI flags: `--skip-narratives` (placeholders, no LM Studio needed), `--regenerate-narratives` (explicit cache refresh; the only path that calls the LLM when a cache exists).
- Every commit: stage explicitly, scan `git diff --cached` for secret patterns first (secret-safe-commits).

---

### Task 1: Dashboard data payload

**Files:**
- Create: `steps/step8_dashboard/__init__.py` (empty)
- Create: `steps/step8_dashboard/data.py`
- Test: `tests/test_step8_data.py`

**Interfaces:**
- Produces: `data.build_payload(cfg, narratives: dict[int, str]) -> dict` — pure function of the tables + narrative mapping (cluster_id → paragraph). Payload schema (all lists sorted deterministically):

```python
{
  "universe": {"n_funds": int, "n_strategy": int, "n_allocation": int,
                "quarters": [str, ...], "latest_quarter": "2024q4"},
  "scorecard": {"auc": float, "auc_ci": [float, float], "persistence_auc": float,
                 "p_edge_le_zero": float, "per_quarter": [{"quarter": str, "auc": float,
                 "persistence_auc": float}, ...], "mean_flip_rate": float},
  "coords": [{"sid": str, "x": float, "y": float, "cluster": int}, ...],
  "clusters": [{"cluster_id": int, "short_title": str, "dominant_category": str,
                 "dominant_share": float, "member_count": int, "avg_sharpe": float,
                 "avg_volatility": float, "avg_max_drawdown": float,
                 "median_net_assets": float, "top_holdings": [{"issuer": str,
                 "weight": float}, ...],   # top 10, weight-aggregated across members
                 "narrative": str,
                 "members": [{"sid": str, "name": str, "ticker": str,
                              "net_assets": float, "sharpe": float, "volatility": float,
                              "max_drawdown": float, "cumulative_return": float,
                              "probability": float | None}, ...]}, ...],
  "allocation": [{"vintage": str,   # e.g. "Target-Date 2050", sorted
                   "members": [{"sid": str, "name": str, "ticker": str,
                                "net_assets": float, "sharpe": float,
                                "volatility": float, "max_drawdown": float,
                                "cumulative_return": float}, ...]}, ...],
  "disclaimer": str,
}
```
- Consumes: step7 tables listed in Prerequisite. `probability` = `unified_predictions` row with `split="forward"` for that series_id (None if absent). `top_holdings`: latest-quarter EC holdings of the cluster's members, sleeve-renormalized per fund then value-weight-averaged across members, top 10 by average weight.

- [ ] **Step 1: Write the failing test** — synthetic tables in a temp DB (reuse the `cfg` fixture pattern from `tests/test_step7_panel.py`: paths → tmp_path). Build minimal versions of every consumed table for 1 strategy cluster (2 funds) + 1 allocation fund; assert:

```python
"""build_payload: schema completeness, deterministic ordering, allocation separation."""
import pandas as pd
import pytest

from fundspeers.io import save_table
from steps.step8_dashboard.data import DISCLAIMER, build_payload


@pytest.fixture
def cfg(tmp_path):
    return {"paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "processed"),
                      "reports": str(tmp_path / "reports"), "models": str(tmp_path / "models")}}


@pytest.fixture
def tables(cfg):
    q = "2024q4"
    save_table(pd.DataFrame([
        {"series_id": "F1", "quarter": q, "accession_number": "a1", "series_name": "Fund One",
         "ticker": "ONE", "yahoo_category": "Large Blend", "is_us_equity": True,
         "segment": "strategy", "net_assets": 100.0},
        {"series_id": "F2", "quarter": q, "accession_number": "a2", "series_name": "Fund Two",
         "ticker": "TWO", "yahoo_category": "Large Blend", "is_us_equity": True,
         "segment": "strategy", "net_assets": 200.0},
        {"series_id": "T1", "quarter": q, "accession_number": "a3", "series_name": "Target Fund",
         "ticker": "TGT", "yahoo_category": "Target-Date 2050", "is_us_equity": True,
         "segment": "allocation", "net_assets": 50.0}]), "funds_all", cfg)
    save_table(pd.DataFrame([{
        "quarter": q, "cluster_id": 0, "member_count": 2, "dominant_category": "Large Blend",
        "dominant_category_share": 1.0, "dominant_tier": "Large", "avg_volatility": 0.15,
        "avg_sharpe": 0.5, "title": "t", "short_title": "Leaning Large Blend"}]),
        "cluster_definitions_all", cfg)
    save_table(pd.DataFrame([
        {"series_id": s, "cumulative_return": 0.2, "annualized_volatility": 0.15,
         "sharpe_ratio": 0.5, "max_drawdown": -0.1} for s in ["F1", "F2", "T1"]]),
        "fund_metrics_overall_all", cfg)
    save_table(pd.DataFrame([
        {"series_id": s, "quarter": q, "cluster_id": 0} for s in ["F1", "F2"]]),
        "fund_clusters_all", cfg)
    save_table(pd.DataFrame([
        {"series_id": "F1", "quarter": q, "predicted_probability": 0.7,
         "actual_label": None, "split": "forward"}]), "unified_predictions", cfg)
    save_table(pd.DataFrame([
        {"metric": "auc_pooled", "quarter": "", "value": 0.7},
        {"metric": "auc_persistence_baseline", "quarter": "", "value": 0.6},
        {"metric": "auc_pooled", "quarter": "2024q1", "value": 0.72},
        {"metric": "auc_persistence_baseline", "quarter": "2024q1", "value": 0.61},
        {"metric": "auc_ci_low", "quarter": "", "value": 0.65},
        {"metric": "auc_ci_high", "quarter": "", "value": 0.75},
        {"metric": "p_edge_le_zero", "quarter": "", "value": 0.02}]),
        "unified_model_eval", cfg)
    save_table(pd.DataFrame([{"series_id": "F1", "quarter": q, "flip_rate": 0.05}]),
               "unified_label_stability", cfg)
    save_table(pd.DataFrame([
        {"series_id": s, "x": 0.1, "y": 0.2, "cluster_id": 0} for s in ["F1", "F2"]]),
        "cluster_map_coords_all", cfg)
    save_table(pd.DataFrame([
        {"accession_number": a, "quarter": q, "asset_cat": "EC", "issuer_name": "ACME CORP",
         "currency_value": 100.0} for a in ["a1", "a2", "a3"]]), "holdings_all", cfg)


def test_payload_schema_and_separation(cfg, tables):
    payload = build_payload(cfg, narratives={0: "A calm cluster."})
    assert payload["universe"]["n_funds"] == 3
    assert payload["universe"]["latest_quarter"] == "2024q4"
    assert len(payload["clusters"]) == 1
    cluster = payload["clusters"][0]
    assert cluster["short_title"] == "Leaning Large Blend"
    assert cluster["narrative"] == "A calm cluster."
    members = {m["sid"]: m for m in cluster["members"]}
    assert members["F1"]["probability"] == 0.7
    assert members["F2"]["probability"] is None
    assert payload["allocation"][0]["vintage"] == "Target-Date 2050"
    assert "probability" not in payload["allocation"][0]["members"][0]
    assert payload["scorecard"]["auc"] == 0.7
    assert payload["scorecard"]["auc_ci"] == [0.65, 0.75]
    assert payload["disclaimer"] == DISCLAIMER


def test_payload_is_deterministic(cfg, tables):
    import json
    a = json.dumps(build_payload(cfg, narratives={}), sort_keys=True)
    b = json.dumps(build_payload(cfg, narratives={}), sort_keys=True)
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_step8_data.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `data.py`**

```python
"""step8_dashboard/data.py - assemble the dashboard's single JSON payload from the step7
tables. Pure read-and-shape: every number shown in the dashboard is computed upstream and
verified there; this module only selects, joins, and orders (deterministically - every
list sorted) for display.
"""
import pandas as pd

from fundspeers.io import load_table, table_exists

DISCLAIMER = (
    "Educational and informational purposes only. Not investment advice or a "
    "recommendation. Predictions are statistical estimates that may be highly inaccurate. "
    "No liability or responsibility is assumed for decisions made based on this material. "
    "Past performance does not guarantee future results.")


def _none_if_na(x):
    return None if x is None or pd.isna(x) else float(x)


def _member_row(fund_row, metrics_row, probability=None, include_probability=True):
    m = {
        "sid": fund_row["series_id"], "name": fund_row["series_name"],
        "ticker": fund_row["ticker"], "net_assets": _none_if_na(fund_row["net_assets"]),
        "sharpe": _none_if_na(metrics_row.get("sharpe_ratio")),
        "volatility": _none_if_na(metrics_row.get("annualized_volatility")),
        "max_drawdown": _none_if_na(metrics_row.get("max_drawdown")),
        "cumulative_return": _none_if_na(metrics_row.get("cumulative_return")),
    }
    if include_probability:
        m["probability"] = probability
    return m


def _top_holdings_for(members, latest_quarter, funds, holdings, top_n=10):
    """Sleeve-renormalize each member's EC holdings, then average weights across members."""
    accs = funds[(funds["series_id"].isin(members)) & (funds["quarter"] == latest_quarter)]
    acc_to_sid = dict(zip(accs["accession_number"], accs["series_id"]))
    ec = holdings[(holdings["quarter"] == latest_quarter) & (holdings["asset_cat"] == "EC")
                  & holdings["accession_number"].isin(acc_to_sid)].copy()
    if ec.empty:
        return []
    ec["sid"] = ec["accession_number"].map(acc_to_sid)
    ec["w"] = ec.groupby("sid")["currency_value"].transform(lambda v: v / v.sum())
    avg = (ec.groupby("issuer_name")["w"].sum() / len(members)).sort_values(ascending=False)
    return [{"issuer": issuer, "weight": round(float(w), 6)}
            for issuer, w in avg.head(top_n).items()]


def build_payload(cfg: dict, narratives: dict) -> dict:
    funds = load_table("funds_all", cfg)
    metrics_overall = load_table("fund_metrics_overall_all", cfg).set_index("series_id")
    clusters_tbl = load_table("fund_clusters_all", cfg)
    definitions = load_table("cluster_definitions_all", cfg)
    holdings = load_table("holdings_all", cfg)
    model_eval = load_table("unified_model_eval", cfg)
    coords = load_table("cluster_map_coords_all", cfg)
    predictions = load_table("unified_predictions", cfg)
    stability = (load_table("unified_label_stability", cfg)
                 if table_exists("unified_label_stability", cfg) else pd.DataFrame(
                     columns=["flip_rate"]))

    equity = funds[funds["is_us_equity"]]
    per_series = equity.drop_duplicates("series_id")
    quarters = sorted(equity["quarter"].unique())
    latest = quarters[-1]
    latest_funds = equity[equity["quarter"] == latest].drop_duplicates("series_id")

    forward = predictions[predictions["split"] == "forward"].set_index("series_id")

    def pooled(metric):
        row = model_eval[(model_eval["metric"] == metric) & (model_eval["quarter"] == "")]
        return float(row["value"].iloc[0]) if len(row) else None

    per_quarter = []
    q_auc = model_eval[(model_eval["metric"] == "auc_pooled") & (model_eval["quarter"] != "")]
    q_persist = model_eval[(model_eval["metric"] == "auc_persistence_baseline")
                           & (model_eval["quarter"] != "")].set_index("quarter")
    for _, row in q_auc.sort_values("quarter").iterrows():
        per_quarter.append({"quarter": row["quarter"], "auc": float(row["value"]),
                            "persistence_auc": _none_if_na(
                                q_persist["value"].get(row["quarter"]))})

    latest_clusters = clusters_tbl[clusters_tbl["quarter"] == latest]
    latest_defs = definitions[definitions["quarter"] == latest].set_index("cluster_id")

    cluster_payloads = []
    for cluster_id in sorted(latest_defs.index):
        d = latest_defs.loc[cluster_id]
        member_ids = sorted(latest_clusters.loc[
            latest_clusters["cluster_id"] == cluster_id, "series_id"])
        members = []
        for sid in member_ids:
            fund_row = latest_funds[latest_funds["series_id"] == sid]
            if fund_row.empty:
                continue
            metrics_row = (metrics_overall.loc[sid].to_dict()
                           if sid in metrics_overall.index else {})
            prob = (_none_if_na(forward.loc[sid, "predicted_probability"])
                    if sid in forward.index else None)
            members.append(_member_row(fund_row.iloc[0], metrics_row, probability=prob))
        member_metrics = pd.DataFrame(members)
        cluster_payloads.append({
            "cluster_id": int(cluster_id),
            "short_title": d["short_title"],
            "dominant_category": d["dominant_category"],
            "dominant_share": float(d["dominant_category_share"]),
            "member_count": len(members),
            "avg_sharpe": _none_if_na(member_metrics["sharpe"].mean()) if len(members) else None,
            "avg_volatility": _none_if_na(member_metrics["volatility"].mean()) if len(members) else None,
            "avg_max_drawdown": _none_if_na(member_metrics["max_drawdown"].mean()) if len(members) else None,
            "median_net_assets": _none_if_na(member_metrics["net_assets"].median()) if len(members) else None,
            "top_holdings": _top_holdings_for(set(member_ids), latest, equity, holdings),
            "narrative": narratives.get(int(cluster_id), ""),
            "members": members,
        })

    allocation_payloads = []
    alloc_funds = latest_funds[latest_funds["series_id"].isin(
        per_series.loc[per_series["segment"] == "allocation", "series_id"])]
    for vintage in sorted(alloc_funds["yahoo_category"].dropna().unique()):
        vintage_members = []
        for _, fund_row in alloc_funds[alloc_funds["yahoo_category"] == vintage].sort_values(
                "series_id").iterrows():
            metrics_row = (metrics_overall.loc[fund_row["series_id"]].to_dict()
                           if fund_row["series_id"] in metrics_overall.index else {})
            vintage_members.append(_member_row(fund_row, metrics_row,
                                               include_probability=False))
        allocation_payloads.append({"vintage": vintage, "members": vintage_members})

    return {
        "universe": {
            "n_funds": int(per_series.shape[0]),
            "n_strategy": int((per_series["segment"] == "strategy").sum()),
            "n_allocation": int((per_series["segment"] == "allocation").sum()),
            "quarters": quarters, "latest_quarter": latest,
        },
        "scorecard": {
            "auc": pooled("auc_pooled"),
            "auc_ci": [pooled("auc_ci_low"), pooled("auc_ci_high")],
            "persistence_auc": pooled("auc_persistence_baseline"),
            "p_edge_le_zero": pooled("p_edge_le_zero"),
            "per_quarter": per_quarter,
            "mean_flip_rate": _none_if_na(stability["flip_rate"].mean())
                              if len(stability) else None,
        },
        "coords": [{"sid": r["series_id"], "x": round(float(r["x"]), 4),
                    "y": round(float(r["y"]), 4), "cluster": int(r["cluster_id"])}
                   for _, r in coords.sort_values("series_id").iterrows()],
        "clusters": cluster_payloads,
        "allocation": allocation_payloads,
        "disclaimer": DISCLAIMER,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_step8_data.py tests/ -q` — Expected: all pass.

- [ ] **Step 5: Stage, scan, commit**

```bash
git add steps/step8_dashboard/__init__.py steps/step8_dashboard/data.py tests/test_step8_data.py
git diff --cached | grep -niE "BEGIN.*PRIVATE KEY|sk_live_|AKIA|ghp_|password\s*=|postgres://"   # expect no output
git commit -m "step8: dashboard data payload - deterministic shaping of step7 tables"
```

---

### Task 2: Cached phi-4 cluster narratives

**Files:**
- Create: `steps/step8_dashboard/narratives.py`
- Test: `tests/test_step8_narratives.py`

**Interfaces:**
- Produces: `narratives.get_narratives(cfg, payload_clusters: list, mode: str) -> dict[int, str]`. `mode` ∈ `"cached"` (default: return cache table contents; generate ONLY clusters missing from cache), `"skip"` (return `{}` — renderer shows placeholders), `"regenerate"` (call LLM for every cluster, overwrite cache). Cache table `dashboard_narratives` columns: `cluster_id, quarter, narrative`. Also `narratives.build_cluster_prompt(cluster: dict) -> str` (pure, testable) and `narratives.generate_one(client, cfg, cluster) -> str`.
- Consumes: step5's client pattern (`OpenAI(base_url=env["LM_STUDIO_BASE_URL"], api_key=env["LM_STUDIO_API_KEY"])` via `fundspeers.config.load_env`); cluster dicts from Task 1's payload (`short_title, dominant_category, dominant_share, member_count, avg_sharpe, avg_volatility, avg_max_drawdown, top_holdings`).

- [ ] **Step 1: Write the failing tests**

```python
"""Narrative caching: cached mode never calls the LLM for cached clusters; the prompt is
grounded (contains only the cluster's own numbers); skip mode returns empty."""
import pandas as pd
import pytest

from fundspeers.io import load_table, save_table, table_exists
from steps.step8_dashboard.narratives import build_cluster_prompt, get_narratives

CLUSTER = {
    "cluster_id": 3, "short_title": "Leaning Large Blend", "dominant_category": "Large Blend",
    "dominant_share": 0.62, "member_count": 40, "avg_sharpe": 0.51, "avg_volatility": 0.16,
    "avg_max_drawdown": -0.22,
    "top_holdings": [{"issuer": "APPLE INC", "weight": 0.06}],
}


@pytest.fixture
def cfg(tmp_path):
    return {"paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "processed"),
                      "reports": str(tmp_path / "reports"), "models": str(tmp_path / "models")},
            "llm": {"model_name": "phi-4", "temperature": 0, "top_peers_to_narrate": 5}}


def test_prompt_contains_only_cluster_facts():
    prompt = build_cluster_prompt(CLUSTER)
    assert "Leaning Large Blend" in prompt
    assert "APPLE INC" in prompt
    assert "62%" in prompt or "0.62" in prompt


def test_skip_mode_returns_empty(cfg):
    assert get_narratives(cfg, [CLUSTER], mode="skip") == {}


def test_cached_mode_uses_cache_without_llm(cfg):
    save_table(pd.DataFrame([{"cluster_id": 3, "quarter": "2024q4",
                              "narrative": "From cache."}]), "dashboard_narratives", cfg)
    # No LM Studio running in tests: if this tried the LLM it would raise.
    out = get_narratives(cfg, [CLUSTER], mode="cached")
    assert out == {3: "From cache."}


def test_cached_mode_fails_loudly_for_missing_cluster_without_llm(cfg):
    save_table(pd.DataFrame([{"cluster_id": 99, "quarter": "2024q4",
                              "narrative": "other"}]), "dashboard_narratives", cfg)
    with pytest.raises(Exception):   # cluster 3 uncached -> attempts LLM -> connection error
        get_narratives(cfg, [CLUSTER], mode="cached")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_step8_narratives.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `narratives.py`**

```python
"""step8_dashboard/narratives.py - per-cluster phi-4 paragraphs, cached in a table.

Same RAG contract as step5: the prompt contains every fact the model may use; it invents
nothing. Cached so dashboard builds are deterministic and LM-Studio-free after the first
generation (design.md: build determinism must not depend on LLM output stability).
"""
import logging

import pandas as pd

from fundspeers.io import load_table, save_table, table_exists

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a financial-data narrator writing for a financial analyst or advisor. You are "
    "given computed facts about ONE peer group (cluster) of US equity mutual funds, grouped "
    "by what they actually hold. Explain the group in one plain-English paragraph.\n\n"
    "Rules:\n"
    "- Use ONLY the facts given. Never invent a number, holding, or fund.\n"
    "- Descriptive only - no investment advice, no buy/sell/hold language.\n"
    "- One paragraph, no lists.")


def build_cluster_prompt(cluster: dict) -> str:
    holdings = ", ".join(f"{h['issuer']} ({h['weight']:.1%})"
                         for h in cluster["top_holdings"][:10]) or "unavailable"
    return (
        f"Cluster name: {cluster['short_title']}.\n"
        f"Members: {cluster['member_count']} funds; dominant category "
        f"{cluster['dominant_category']} ({cluster['dominant_share']:.0%} of members).\n"
        f"Average annualized volatility: {cluster['avg_volatility']:.1%}. "
        f"Average Sharpe ratio: {cluster['avg_sharpe']:.2f}. "
        f"Average max drawdown: {cluster['avg_max_drawdown']:.1%}.\n"
        f"Most-held stocks across members (average weight): {holdings}.")


def _get_client(cfg: dict):
    from openai import OpenAI

    from fundspeers.config import load_env
    env = load_env()
    return OpenAI(base_url=env["LM_STUDIO_BASE_URL"], api_key=env["LM_STUDIO_API_KEY"])


def generate_one(client, cfg: dict, cluster: dict) -> str:
    response = client.chat.completions.create(
        model=cfg["llm"]["model_name"], temperature=cfg["llm"]["temperature"],
        messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                  {"role": "user", "content": build_cluster_prompt(cluster)}])
    return response.choices[0].message.content


def get_narratives(cfg: dict, payload_clusters: list, mode: str = "cached",
                   quarter: str = "2024q4") -> dict:
    if mode == "skip":
        return {}
    cached = {}
    if mode != "regenerate" and table_exists("dashboard_narratives", cfg):
        tbl = load_table("dashboard_narratives", cfg)
        cached = {int(r["cluster_id"]): r["narrative"]
                  for _, r in tbl[tbl["quarter"] == quarter].iterrows()}

    missing = [c for c in payload_clusters if int(c["cluster_id"]) not in cached]
    if missing:
        log.info(f"generating {len(missing)} narrative(s) via LM Studio "
                 f"({'regenerate' if mode == 'regenerate' else 'cache misses'})")
        client = _get_client(cfg)
        for cluster in missing:
            cached[int(cluster["cluster_id"])] = generate_one(client, cfg, cluster)
        save_table(pd.DataFrame(
            [{"cluster_id": cid, "quarter": quarter, "narrative": text}
             for cid, text in sorted(cached.items())]), "dashboard_narratives", cfg)
    return cached
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_step8_narratives.py tests/ -q` — Expected: all pass.

- [ ] **Step 5: Stage, scan, commit**

```bash
git add steps/step8_dashboard/narratives.py tests/test_step8_narratives.py
git diff --cached | grep -niE "BEGIN.*PRIVATE KEY|sk_live_|AKIA|ghp_|password\s*=|postgres://"   # expect no output
git commit -m "step8: cached per-cluster phi-4 narratives with grounded prompts"
```

---

### Task 3: HTML renderer

**READ THE `dataviz` SKILL (Skill tool) BEFORE THIS TASK** — it governs chart colors, axis/legend rules, and light/dark handling for the SVG scatter and histogram strips.

**Files:**
- Create: `steps/step8_dashboard/render.py`
- Create: `steps/step8_dashboard/template.py` (the HTML/CSS/JS as one Python string constant — keeps render.py readable)
- Test: `tests/test_step8_render.py`

**Interfaces:**
- Produces: `render.render_dashboard(payload: dict) -> str` (the complete HTML document, pure function) and `render.write_dashboard(payload, cfg) -> Path` (writes `reports/cluster_dashboard.html`).
- Consumes: the Task 1 payload schema, verbatim. `template.TEMPLATE` is an HTML document with the literal token `__PAYLOAD_JSON__` replaced by `json.dumps(payload, sort_keys=True, separators=(",", ":"))`.

**Renderer behavior (all client-side, vanilla JS in the template):**
- Views: `#overview` (universe summary, SVG scatter from `coords` colored by cluster with hover tooltip = name/ticker/short_title, scorecard panel, "how to read this" copy), `#clusters` (sortable index table of `clusters`, each row → cluster section), `#allocation` (vintage groups, descriptive table only), one `#cluster-<id>` section per cluster (identity header, top-holdings bar list, member table sortable by every column including probability, narrative paragraph or `<em>narrative not generated</em>` if empty).
- Global fund search box: filters across all member tables by name/ticker substring (case-insensitive), jumps to owning cluster section.
- Table sorting: click a `<th>` to sort by that column (numeric-aware, None last); implemented once as a generic JS function applied to every `<table class="sortable">`.
- Probability column header carries `title` attr + visible footnote marker linking to the disclaimer; footer `<footer>` with the exact DISCLAIMER text on every view (fixed at page bottom).
- All numbers formatted client-side: percentages 1 decimal, Sharpe 2 decimals, net assets with thousands separators.
- Light/dark: CSS `prefers-color-scheme`, same approach as `render_docs.py`.

- [ ] **Step 1: Write the failing tests**

```python
"""Renderer: self-contained, deterministic, disclaimer everywhere, placeholders on skip."""
import re

from steps.step8_dashboard.data import DISCLAIMER
from steps.step8_dashboard.render import render_dashboard

PAYLOAD = {
    "universe": {"n_funds": 3, "n_strategy": 2, "n_allocation": 1,
                 "quarters": ["2024q4"], "latest_quarter": "2024q4"},
    "scorecard": {"auc": 0.7, "auc_ci": [0.65, 0.75], "persistence_auc": 0.6,
                  "p_edge_le_zero": 0.02, "per_quarter": [], "mean_flip_rate": 0.05},
    "coords": [{"sid": "F1", "x": 0.1, "y": 0.2, "cluster": 0}],
    "clusters": [{"cluster_id": 0, "short_title": "Leaning Large Blend",
                  "dominant_category": "Large Blend", "dominant_share": 1.0,
                  "member_count": 1, "avg_sharpe": 0.5, "avg_volatility": 0.15,
                  "avg_max_drawdown": -0.1, "median_net_assets": 100.0,
                  "top_holdings": [{"issuer": "ACME", "weight": 0.5}],
                  "narrative": "",
                  "members": [{"sid": "F1", "name": "Fund One", "ticker": "ONE",
                               "net_assets": 100.0, "sharpe": 0.5, "volatility": 0.15,
                               "max_drawdown": -0.1, "cumulative_return": 0.2,
                               "probability": 0.7}]}],
    "allocation": [{"vintage": "Target-Date 2050",
                    "members": [{"sid": "T1", "name": "Target Fund", "ticker": "TGT",
                                 "net_assets": 50.0, "sharpe": 0.5, "volatility": 0.15,
                                 "max_drawdown": -0.1, "cumulative_return": 0.2}]}],
    "disclaimer": DISCLAIMER,
}


def test_renders_and_contains_disclaimer():
    html = render_dashboard(PAYLOAD)
    assert DISCLAIMER in html
    assert "Leaning Large Blend" in html
    assert "Target-Date 2050" in html


def test_no_external_requests():
    html = render_dashboard(PAYLOAD)
    assert not re.search(r'<script[^>]+src=', html)
    assert not re.search(r'<link[^>]+href="https?://', html)
    assert "http://" not in html.replace("http://www.w3.org", "")   # SVG namespace ok
    assert "https://" not in html


def test_deterministic():
    assert render_dashboard(PAYLOAD) == render_dashboard(PAYLOAD)


def test_empty_narrative_shows_placeholder():
    html = render_dashboard(PAYLOAD)
    assert "narrative not generated" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_step8_render.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `template.py` + `render.py`**

`render.py` is small and mechanical:

```python
"""step8_dashboard/render.py - payload dict -> one self-contained HTML file."""
import json
from pathlib import Path

from fundspeers.io import reports_dir
from steps.step8_dashboard.template import TEMPLATE


def render_dashboard(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return TEMPLATE.replace("__PAYLOAD_JSON__", blob)


def write_dashboard(payload: dict, cfg: dict) -> Path:
    out = reports_dir(cfg) / "cluster_dashboard.html"
    out.write_text(render_dashboard(payload), encoding="utf-8")
    return out
```

`template.py` holds `TEMPLATE`: a complete HTML document. Structure to implement (follow the dataviz skill for palette/axis/legend specifics; server-side Python builds NOTHING per-cluster — all sections are generated client-side by JS from the embedded payload, which keeps the template static and the output deterministic):

```python
TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fund Peer-Group Dashboard</title>
<style>
  /* :root color-scheme light dark; card/table/nav styles; probability column note;
     fixed footer with the disclaimer; @media (prefers-color-scheme: dark) overrides.
     Categorical cluster colors: one 30-color palette generated in JS (HSL wheel,
     fixed order by cluster_id) - see dataviz skill for saturation/lightness bounds. */
</style>
</head>
<body>
<nav><!-- links: Overview | Clusters | Target-Date & Allocation; search input --></nav>
<main id="app"></main>
<footer id="disclaimer"></footer>
<script id="payload" type="application/json">__PAYLOAD_JSON__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("payload").textContent);
// fmtPct/fmtNum/fmtMoney helpers; clusterColor(id) -> hsl string;
// renderOverview(): universe stats, scorecard card (AUC + CI vs baselines,
//   plain-English significance sentence from p_edge_le_zero, per-quarter range,
//   mean_flip_rate sentence, "what this number is and isn't" copy),
//   SVG scatter of DATA.coords with hover tooltip;
// renderClusterIndex(): sortable table over DATA.clusters, row click -> section;
// renderClusterSection(c): identity header, top-holdings weight bars (divs, widths %),
//   narrative paragraph (or <em>narrative not generated</em>), member table with
//   probability column (header footnote marker referencing the disclaimer);
// renderAllocation(): one sub-section per DATA.allocation vintage, descriptive table;
// makeSortable(table): generic th-click numeric-aware sorting, nulls last;
// search box: substring match on name/ticker across all clusters, scroll to hit.
// Hash-based view switching (#overview default).
document.getElementById("disclaimer").textContent = DATA.disclaimer;
</script>
</body>
</html>"""
```

The implementer writes the full CSS and the JS function bodies sketched above — complete, not stubs; the tests in Step 1 plus Task 4's real-build UAT verify behavior. Keep every list render ordered by the payload's already-sorted arrays (no `Object.keys` iteration on unordered maps).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_step8_render.py tests/ -q` — Expected: all pass.

- [ ] **Step 5: Stage, scan, commit**

```bash
git add steps/step8_dashboard/render.py steps/step8_dashboard/template.py tests/test_step8_render.py
git diff --cached | grep -niE "BEGIN.*PRIVATE KEY|sk_live_|AKIA|ghp_|password\s*=|postgres://"   # expect no output
git commit -m "step8: self-contained dashboard renderer - embedded JSON, vanilla JS, disclaimer footer"
```

---

### Task 4: Build CLI, real build, UAT, docs, gate

**Files:**
- Create: `steps/step8_dashboard/build.py`
- Modify: `decisions.json`, `workflow.json` (step8 entry), `steps/step8_dashboard/design.md` (append `## UAT results`)
- Output: `reports/cluster_dashboard.html` (committed)

**Interfaces:**
- Produces: `python -m steps.step8_dashboard.build [--skip-narratives | --regenerate-narratives]`.

- [ ] **Step 1: Implement `build.py`**

```python
"""step8_dashboard/build.py - deterministic dashboard build.

    python -m steps.step8_dashboard.build                       # narratives from cache,
                                                                 # generating only misses
    python -m steps.step8_dashboard.build --skip-narratives     # no LM Studio needed
    python -m steps.step8_dashboard.build --regenerate-narratives
"""
import argparse
import logging

from fundspeers.config import load_config
from steps.step8_dashboard.data import build_payload
from steps.step8_dashboard.narratives import get_narratives
from steps.step8_dashboard.render import write_dashboard

log = logging.getLogger(__name__)


def run(cfg: dict, narrative_mode: str = "cached") -> None:
    payload = build_payload(cfg, narratives={})
    narratives = get_narratives(cfg, payload["clusters"], mode=narrative_mode,
                                quarter=payload["universe"]["latest_quarter"])
    payload = build_payload(cfg, narratives=narratives)
    out = write_dashboard(payload, cfg)
    log.info(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
             f"{len(payload['clusters'])} clusters, "
             f"{payload['universe']['n_funds']} funds)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--skip-narratives", action="store_true")
    group.add_argument("--regenerate-narratives", action="store_true")
    args = ap.parse_args()
    mode = ("skip" if args.skip_narratives
            else "regenerate" if args.regenerate_narratives else "cached")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(load_config(), narrative_mode=mode)
```

- [ ] **Step 2: First real build without LM Studio**

Run: `python -m steps.step8_dashboard.build --skip-narratives`
Expected: writes `reports/cluster_dashboard.html`, low-single-digit MB, 30 clusters.

- [ ] **Step 3: Full build with narratives** (requires LM Studio running phi-4 at localhost:1234)

Run: `python -m steps.step8_dashboard.build`
Expected: ~30 narratives generated once, cached to `dashboard_narratives`, file rewritten.

- [ ] **Step 4: Determinism check** — commit the Step 3 build first (see Step 6's git block; an intermediate `git add reports/cluster_dashboard.html && git commit -m "step8: first dashboard build"` is fine), then rebuild and confirm no diff:

```bash
python -m steps.step8_dashboard.build
git status --short reports/cluster_dashboard.html
```
Expected: no output from `git status` — the second build is byte-identical (narratives came from the cache table, everything else is a pure function of the tables).

- [ ] **Step 5: UAT against design.md** — open the file in a browser (offline). Check every design.md UAT item: all 30 cluster sections render; allocation by vintage; sorting/filter/search work; hover tooltip on the scatter; disclaimer footer always visible + probability column note; scorecard shows AUC+CI, both baselines, significance sentence, flip-rate sentence, per-quarter range; spot-check 3 narratives against their clusters' payload numbers (no invented facts); `--skip-narratives` rebuild shows placeholders and is otherwise identical. Record results.

- [ ] **Step 6: Live docs + commit + gate**

Append the step8 decisions.json entry (what was built + UAT outcome), add `{"name": "step8_dashboard", "status": "done"}` to workflow.json, append `## UAT results` to `steps/step8_dashboard/design.md`, re-render docs, run the full test suite.

```bash
python scripts/render_docs.py && python -m pytest tests/ -q
git add steps/step8_dashboard/build.py reports/cluster_dashboard.html decisions.json workflow.json reflection.html workflow.html steps/step8_dashboard/design.md
git diff --cached | grep -niE "BEGIN.*PRIVATE KEY|sk_live_|AKIA|ghp_|password\s*=|postgres://"   # expect no output
git commit -m "step8: interactive cluster dashboard - built, UAT passed, docs updated"
```

**STOP — human approval gate.** Present the dashboard + UAT results to the user.
