# step7_unified_universe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge all three ingested fund batches into one 2,243-fund universe, rebuild clustering at k=30 with TDF/allocation funds segmented out, redefine the underperformance label as kNN-peer-median, add holdings-derived features, and evaluate a retrained random forest against random + persistence baselines with a Monte Carlo uncertainty layer.

**Architecture:** New package `steps/step7_unified_universe/` with focused modules (merge → label → features → panel → model → stability) orchestrated by `build.py`. Reuses step2's `similarity.run` (extended with backward-compatible params) and step3's `metrics.run` via the existing `table_suffix="_all"` mechanism. All outputs land in new `_all`/`unified_*` tables; per-batch tables and the existing `random_forest_model.joblib` are never touched.

**Tech Stack:** Python, pandas, scikit-learn (KMeans, RandomForestClassifier, roc_auc_score), sentence-transformers (existing), DuckDB via `fundspeers.io`, numpy `default_rng` for seeded Monte Carlo, pytest.

**Authoritative spec:** `steps/step7_unified_universe/design.md` (committed). Read it before starting.

## Global Constraints

- Determinism: every stochastic operation seeded from `cfg["seed"]` (42). Same inputs → same outputs.
- Backward compatibility: `similarity.run(cfg)` / `metrics.run(cfg)` with no new args must behave byte-identically to today. New params default to preserving current behavior.
- Per-batch tables (`funds`, `funds_oos`, `funds_oos2`, …) and `models/random_forest_model.joblib` are read-only for this step.
- New config keys (exact values): `unified.n_clusters: 30`, `unified.peer_label_top_n: 10`, `unified.min_valid_peers_for_label: 5`, `unified.bootstrap_iterations: 2000`, `unified.label_stability_draws: 100`.
- Not wired into `conductor.py` — run on demand via `python -m steps.step7_unified_universe.build`.
- Segment rule (exact): `yahoo_category` starting with `"Target-Date"` or `"Allocation"` → `segment="allocation"`, else `"strategy"`. NaN/None category → `"strategy"`.
- The `_all` similarity run stores top **15** peers per fund (label uses top 10; stability perturbs 8-of-top-12).
- Every commit: stage explicitly, then scan `git diff --cached` for secret patterns before committing (secret-safe-commits).
- Test commands run from repo root: `python -m pytest tests/test_<name>.py -q`.

---

### Task 1: Close out the step6/oos2 record (pending docs + uncommitted code)

The oos2 evaluation results exist only in the session; `evaluate.py`'s generalization is uncommitted. Land them before new work starts.

**Files:**
- Modify: `steps/step6_out_of_sample/design.md` (append section at end)
- Modify: `decisions.json` (append entry)
- Commit (already modified on disk, no edits needed): `steps/step6_out_of_sample/evaluate.py`
- Commit (untracked): `reports/cluster_map_2024q4_oos2.png`

**Interfaces:** none (docs + already-tested code).

- [ ] **Step 1: Append the oos2 follow-up to `steps/step6_out_of_sample/design.md`**

Append verbatim at the end of the file:

```markdown
## Follow-up (2026-07-11): a third disjoint batch (oos2) and the composition-effect finding
A genuinely fresh 3rd batch was sampled (`table_suffix="_oos2"`: 4000-fund target, exclude_series
= union of both prior batches, 4000/4000 resolved, 1489 flagged US equity, confirmed disjoint).
Pipeline consistency held: purity=0.427/ARI=0.293 (vs. 0.409/0.249 original, 0.459/0.291 first OOS).
`evaluate_frozen_model_on_oos()` gained a `table_suffix` param (default `"_oos"`) and the
retrain/promote helpers gained a `table_suffixes` tuple param (default `("", "_oos")`) plus a
`backup_name` guard that refuses to overwrite an existing backup.

Results: the promoted 754-fund model scored **AUC=0.701 frozen on oos2** (consistent with 0.693 on
the first OOS batch - the model generalizes). But the naive 3-way retrain scored **AUC=0.680** vs
the official model's 0.725. Per-origin breakdown: 0.708 original / 0.700 oos / 0.668 oos2 test
rows, with oos2 66% of the blended test set - and the frozen model scores 0.691 on that same oos2
2024 slice. So the drop is a **composition effect** (oos2 is intrinsically harder for this method,
dominating the test mix), not training corruption. Root cause: with n_clusters fixed at 15, oos2's
clusters have median 62 members (vs 12 originally), diluting both the label and the model's top
feature. **The 3-way retrain was deliberately NOT promoted**; the diagnosis motivated the
step7_unified_universe redesign (merged universe, kNN-peer label) instead.

Operational observation, not fixed: `exclude_series` only tracks *successes* across ingestion
runs, not failures, so each new batch re-attempts (and re-fails) previously-failed candidates -
6378 attempts for 4000 resolutions on this batch.
```

- [ ] **Step 2: Append the decisions.json entry**

Append to the JSON array (before the closing `]`):

```json
{
  "date": "2026-07-11",
  "decision": "Evaluated a fresh 3rd disjoint batch (oos2, 1489 equity funds): frozen official model AUC=0.701 (generalizes), but a naive 3-way retrain scored 0.680 vs the official 0.725 - diagnosed as a composition effect and deliberately NOT promoted; motivated the step7_unified_universe redesign instead.",
  "rationale": "Per-origin test AUC (0.708 original / 0.700 oos / 0.668 oos2, with oos2 66% of test rows) plus the frozen model scoring 0.691 on the identical oos2 slice showed the blended drop reflects a harder new population, not corrupted training. Root cause traced to fixed n_clusters=15: oos2's median cluster is 62 funds vs 12 originally, so the cluster-median label and the model's top feature both degrade with population size - a task-definition problem no amount of retraining fixes, hence the redesign (merged universe, constant-size kNN-peer benchmark)."
}
```

- [ ] **Step 3: Re-render docs and run tests**

Run: `python scripts/render_docs.py && python -m pytest tests/ -q`
Expected: `wrote reflection.html and workflow.html`; all tests pass.

- [ ] **Step 4: Stage, scan, commit**

```bash
git add steps/step6_out_of_sample/evaluate.py steps/step6_out_of_sample/design.md decisions.json reflection.html reports/cluster_map_2024q4_oos2.png
git diff --cached | grep -niE "BEGIN.*PRIVATE KEY|sk_live_|AKIA|ghp_|password\s*=|postgres://"   # expect no output
git commit -m "step6: oos2 batch results - frozen AUC 0.701, 3-way retrain 0.680 diagnosed as composition effect, not promoted"
```

---

### Task 2: Config keys + unified merge module

**Files:**
- Modify: `config.json` (add `unified` section)
- Create: `steps/step7_unified_universe/__init__.py` (empty)
- Create: `steps/step7_unified_universe/merge.py`
- Test: `tests/test_step7_merge.py`

**Interfaces:**
- Produces: `merge.assign_segment(yahoo_category) -> str` ("allocation" | "strategy"); `merge.build_unified_tables(cfg) -> dict` (keys: `n_funds`, `n_strategy`, `n_allocation`) — saves `funds_all` (with new `segment` column), `holdings_all`, `monthly_returns_all`; raises `RuntimeError` on series_id overlap between batches.
- Consumes: `fundspeers.io.load_table/save_table`; batch suffixes `("", "_oos", "_oos2")`.

- [ ] **Step 1: Add the `unified` section to `config.json`** (after the `"model"` section):

```json
"unified": {
  "n_clusters": 30,
  "peer_label_top_n": 10,
  "min_valid_peers_for_label": 5,
  "bootstrap_iterations": 2000,
  "label_stability_draws": 100
}
```

- [ ] **Step 2: Write the failing tests**

```python
"""Unit tests for step7's batch merge - synthetic data, temp DB, never the real one."""
import pandas as pd
import pytest

from steps.step7_unified_universe.merge import assign_segment, build_unified_tables
from fundspeers.io import load_table, save_table


@pytest.fixture
def cfg(tmp_path):
    return {"paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "processed"),
                      "reports": str(tmp_path / "reports"), "models": str(tmp_path / "models")}}


def _make_batch(series_ids, categories):
    funds = pd.DataFrame({
        "series_id": series_ids, "quarter": "2024q4",
        "accession_number": [f"acc-{s}" for s in series_ids],
        "yahoo_category": categories, "is_us_equity": True, "net_assets": 1.0,
        "series_name": series_ids, "ticker": series_ids,
    })
    holdings = pd.DataFrame({
        "accession_number": [f"acc-{s}" for s in series_ids], "quarter": "2024q4",
        "asset_cat": "EC", "issuer_name": "ACME CORP", "currency_value": 100.0,
    })
    returns = pd.DataFrame({
        "series_id": series_ids, "quarter": "2024q4", "month_in_quarter": 1,
        "total_return": 1.0,
    })
    return funds, holdings, returns


def test_assign_segment():
    assert assign_segment("Target-Date 2045") == "allocation"
    assert assign_segment("Allocation--50% to 70% Equity") == "allocation"
    assert assign_segment("Large Blend") == "strategy"
    assert assign_segment(None) == "strategy"
    assert assign_segment(float("nan")) == "strategy"


def test_merge_concatenates_and_segments(cfg):
    for suffix, ids, cats in [("", ["A"], ["Large Blend"]),
                              ("_oos", ["B"], ["Target-Date 2050"]),
                              ("_oos2", ["C"], ["Small Value"])]:
        funds, holdings, returns = _make_batch(ids, cats)
        save_table(funds, f"funds{suffix}", cfg)
        save_table(holdings, f"holdings{suffix}", cfg)
        save_table(returns, f"monthly_returns{suffix}", cfg)
    counts = build_unified_tables(cfg)
    assert counts == {"n_funds": 3, "n_strategy": 2, "n_allocation": 1}
    funds_all = load_table("funds_all", cfg)
    assert set(funds_all["series_id"]) == {"A", "B", "C"}
    assert funds_all.set_index("series_id")["segment"].to_dict() == {
        "A": "strategy", "B": "allocation", "C": "strategy"}
    assert len(load_table("holdings_all", cfg)) == 3
    assert len(load_table("monthly_returns_all", cfg)) == 3


def test_merge_raises_on_overlapping_batches(cfg):
    for suffix in ["", "_oos", "_oos2"]:
        funds, holdings, returns = _make_batch(["DUP"], ["Large Blend"])
        save_table(funds, f"funds{suffix}", cfg)
        save_table(holdings, f"holdings{suffix}", cfg)
        save_table(returns, f"monthly_returns{suffix}", cfg)
    with pytest.raises(RuntimeError, match="overlap"):
        build_unified_tables(cfg)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_step7_merge.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'steps.step7_unified_universe'`

- [ ] **Step 4: Implement `merge.py`** (and an empty `__init__.py`)

```python
"""step7_unified_universe/merge.py - merge the three ingested batches into one universe.

Batch namespaces ("", "_oos", "_oos2") are an ingestion artifact, not a product concept
(see design.md). This module concatenates them into funds_all/holdings_all/
monthly_returns_all, asserts the batches really are disjoint, and tags each fund with its
segment: target-date/allocation funds cluster by fund FAMILY, not strategy (they are
fund-of-funds whose EC holdings are the provider's own index funds), so they are excluded
from the strategy clustering and presented separately.
"""
import logging
import re

import pandas as pd

from fundspeers.io import load_table, save_table
from steps.step2_similarity.similarity import normalize_issuer_name

log = logging.getLogger(__name__)

BATCH_SUFFIXES = ("", "_oos", "_oos2")
_ALLOCATION_PREFIXES = ("Target-Date", "Allocation")
# Fund-of-funds coverage diagnostic: issuer names that look like funds, not companies.
_FOF_ISSUER_RE = re.compile(r"\b(FUND|FD|PORTFOLIO|INDEX|ETF|TRUST|TR)\b")


def assign_segment(yahoo_category) -> str:
    if isinstance(yahoo_category, str) and yahoo_category.startswith(_ALLOCATION_PREFIXES):
        return "allocation"
    return "strategy"


def _log_fof_diagnostic(funds_all: pd.DataFrame, holdings_all: pd.DataFrame) -> None:
    """Verify (each run, not once) that the category-based segment rule caught the
    fund-of-funds: report strategy-segment equity funds whose EC holdings VALUE is
    majority fund-shaped issuer names."""
    strategy_acc = set(
        funds_all.loc[funds_all["is_us_equity"] & (funds_all["segment"] == "strategy"),
                      "accession_number"]
    )
    ec = holdings_all[
        holdings_all["accession_number"].isin(strategy_acc) & (holdings_all["asset_cat"] == "EC")
    ].copy()
    ec["is_fof_like"] = ec["issuer_name"].map(
        lambda n: bool(_FOF_ISSUER_RE.search(normalize_issuer_name(n)))
    )
    by_acc = ec.groupby("accession_number").apply(
        lambda g: g.loc[g["is_fof_like"], "currency_value"].sum() / max(g["currency_value"].sum(), 1e-9)
    )
    suspicious = (by_acc > 0.5).sum()
    log.info(f"FoF diagnostic: {suspicious} of {by_acc.shape[0]} strategy-segment filings have "
             f">50% fund-shaped EC holdings value (rule coverage check, not a filter)")


def build_unified_tables(cfg: dict) -> dict:
    funds_parts = [load_table(f"funds{s}", cfg) for s in BATCH_SUFFIXES]
    seen: set = set()
    for suffix, part in zip(BATCH_SUFFIXES, funds_parts):
        ids = set(part["series_id"].unique())
        overlap = seen & ids
        if overlap:
            raise RuntimeError(
                f"batch {suffix!r} overlaps an earlier batch on {len(overlap)} series_id(s), "
                f"e.g. {sorted(overlap)[:5]} - batches must be disjoint")
        seen |= ids

    funds_all = pd.concat(funds_parts, ignore_index=True)
    funds_all["segment"] = funds_all["yahoo_category"].map(assign_segment)
    holdings_all = pd.concat(
        [load_table(f"holdings{s}", cfg) for s in BATCH_SUFFIXES], ignore_index=True)
    monthly_all = pd.concat(
        [load_table(f"monthly_returns{s}", cfg) for s in BATCH_SUFFIXES], ignore_index=True)

    save_table(funds_all, "funds_all", cfg)
    save_table(holdings_all, "holdings_all", cfg)
    save_table(monthly_all, "monthly_returns_all", cfg)

    per_series = funds_all.drop_duplicates("series_id")
    counts = {
        "n_funds": int(per_series.shape[0]),
        "n_strategy": int((per_series["segment"] == "strategy").sum()),
        "n_allocation": int((per_series["segment"] == "allocation").sum()),
    }
    log.info(f"unified universe: {counts['n_funds']} funds "
             f"({counts['n_strategy']} strategy / {counts['n_allocation']} allocation)")
    _log_fof_diagnostic(funds_all, holdings_all)
    return counts
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_step7_merge.py -q` — Expected: 3 passed.
Run: `python -m pytest tests/ -q` — Expected: all pass (no regression).

- [ ] **Step 6: Stage, scan, commit**

```bash
git add config.json steps/step7_unified_universe/__init__.py steps/step7_unified_universe/merge.py tests/test_step7_merge.py
git diff --cached | grep -niE "BEGIN.*PRIVATE KEY|sk_live_|AKIA|ghp_|password\s*=|postgres://"   # expect no output
git commit -m "step7: unified merge module - concat 3 batches, disjointness assert, segment column, FoF diagnostic"
```

---

### Task 3: Backward-compatible extensions to `similarity.run`

**Files:**
- Modify: `steps/step2_similarity/similarity.py` (function `run`, lines ~124–215, and `_plot_cluster_map`)
- Test: `tests/test_step7_similarity_ext.py`

**Interfaces:**
- Produces: `similarity.run(cfg, table_suffix="", n_clusters=None, top_n_peers=None, require_segment=None, save_coords=False)`. `n_clusters`/`top_n_peers` default to the existing `cfg["similarity"]` values; `require_segment="strategy"` filters `funds` to that segment (column must exist); `save_coords=True` saves the latest quarter's PCA coordinates + cluster ids as table `cluster_map_coords{table_suffix}` with columns `series_id, x, y, cluster_id`.
- Consumes: `funds_all.segment` from Task 2.

- [ ] **Step 1: Write the failing test**

```python
"""The new similarity.run params must default to existing behavior and be individually testable.
Full-run behavior is covered by the real _all run in Task 9; here we unit-test the pure pieces."""
import inspect

import pandas as pd

from steps.step2_similarity.similarity import run, _filter_universe


def test_run_signature_backward_compatible():
    sig = inspect.signature(run)
    assert list(sig.parameters) == [
        "cfg", "table_suffix", "n_clusters", "top_n_peers", "require_segment", "save_coords"]
    assert sig.parameters["n_clusters"].default is None
    assert sig.parameters["require_segment"].default is None
    assert sig.parameters["save_coords"].default is False


def test_filter_universe_by_segment():
    funds = pd.DataFrame({
        "series_id": ["A", "B", "C"], "is_us_equity": [True, True, False],
        "segment": ["strategy", "allocation", "strategy"],
    })
    assert _filter_universe(funds, None) == {"A", "B"}          # today's behavior
    assert _filter_universe(funds, "strategy") == {"A"}          # segment filter
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_step7_similarity_ext.py -q`
Expected: FAIL — `ImportError: cannot import name '_filter_universe'`

- [ ] **Step 3: Implement the extension**

In `similarity.py`, add after `compute_purity`:

```python
def _filter_universe(funds: pd.DataFrame, require_segment) -> set:
    """The set of series_ids to embed/cluster: US-equity, optionally one segment only
    (step7's merged universe excludes 'allocation' fund-of-funds from strategy clustering)."""
    universe = funds[funds["is_us_equity"]]
    if require_segment is not None:
        universe = universe[universe["segment"] == require_segment]
    return set(universe["series_id"].unique())
```

Change `run`'s signature and the lines that read config/filter funds:

```python
def run(cfg: dict, table_suffix: str = "", n_clusters=None, top_n_peers=None,
        require_segment=None, save_coords: bool = False) -> None:
    """`table_suffix` (defaults to the original behavior) lets this run against a parallel,
    disjoint fund set (e.g. "_oos") - see steps/step6_out_of_sample/design.md.
    step7's merged-universe run passes n_clusters/top_n_peers overrides, require_segment
    ("strategy": exclude allocation fund-of-funds), and save_coords (persist the latest
    quarter's PCA coordinates for the step8 dashboard) - all defaults preserve the exact
    original behavior."""
    seed = cfg["seed"]
    n_clusters = n_clusters if n_clusters is not None else cfg["similarity"]["n_clusters"]
    top_n_peers = top_n_peers if top_n_peers is not None else cfg["similarity"]["top_n_peers"]
    ...
    equity_series = _filter_universe(funds, require_segment)
```

Make `_plot_cluster_map` return the coordinates it already computes (add `return` at the end):

```python
    coords_df = pd.DataFrame({
        "series_id": vectors.index, "x": coords[:, 0], "y": coords[:, 1],
        "cluster_id": cluster_labels.values,
    })
    ...  # existing plotting code unchanged
    return coords_df
```

And at the end of `run`, where `_plot_cluster_map` is called:

```python
    coords_df = _plot_cluster_map(latest_vectors, latest_labels, quarters[-1], seed, cfg,
                                  table_suffix, cluster_names)
    if save_coords:
        save_table(coords_df, f"cluster_map_coords{table_suffix}", cfg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_step7_similarity_ext.py tests/ -q` — Expected: all pass.

- [ ] **Step 5: Stage, scan, commit**

```bash
git add steps/step2_similarity/similarity.py tests/test_step7_similarity_ext.py
git diff --cached | grep -niE "BEGIN.*PRIVATE KEY|sk_live_|AKIA|ghp_|password\s*=|postgres://"   # expect no output
git commit -m "step2 similarity: backward-compatible n_clusters/top_n_peers/require_segment/save_coords params for the unified run"
```

---

### Task 4: kNN-peer label module

**Files:**
- Create: `steps/step7_unified_universe/label.py`
- Test: `tests/test_step7_label.py`

**Interfaces:**
- Produces: `label.compute_peer_labels(fund_peers, quarterly_returns, quarters_ordered, top_n, min_valid_peers) -> pd.DataFrame` with columns `series_id, quarter, peer_median_return_q, return_vs_peer_median_q, peer_median_return_next, n_valid_peers_next, underperform_next_quarter` (nullable Int64; `pd.NA` when the fund's own next-quarter return is missing, fewer than `min_valid_peers` peers have a valid next-quarter return, or `quarter` is the last quarter). Last-quarter rows are KEPT (they become the forward-prediction set).
- Consumes: `fund_peers` schema `[series_id, quarter, peer_rank, peer_series_id, cosine_similarity]` (step2); `quarterly_returns` schema `[series_id, quarter, quarterly_return]` (from `fund_metrics_quarterly`).

- [ ] **Step 1: Write the failing tests**

```python
"""kNN-peer label: below the median of your own top-N peers' next-quarter returns."""
import pandas as pd

from steps.step7_unified_universe.label import compute_peer_labels

QUARTERS = ["2024q1", "2024q2"]


def _peers(series_id, peer_ids, quarter="2024q1"):
    return pd.DataFrame({
        "series_id": series_id, "quarter": quarter,
        "peer_rank": range(1, len(peer_ids) + 1),
        "peer_series_id": peer_ids, "cosine_similarity": 0.9,
    })


def _returns(mapping, quarter):
    return pd.DataFrame({"series_id": list(mapping), "quarter": quarter,
                         "quarterly_return": list(mapping.values())})


def test_label_below_peer_median_is_1():
    peers = _peers("F", ["P1", "P2", "P3", "P4", "P5"])
    returns = pd.concat([
        _returns({"F": 0.05, "P1": 0.0, "P2": 0.0, "P3": 0.0, "P4": 0.0, "P5": 0.0}, "2024q1"),
        _returns({"F": 0.01, "P1": 0.02, "P2": 0.03, "P3": 0.04, "P4": 0.05, "P5": 0.06}, "2024q2"),
    ])
    out = compute_peer_labels(peers, returns, QUARTERS, top_n=5, min_valid_peers=5)
    row = out[(out["series_id"] == "F") & (out["quarter"] == "2024q1")].iloc[0]
    assert row["underperform_next_quarter"] == 1          # 0.01 < median(0.02..0.06)=0.04
    assert row["peer_median_return_next"] == 0.04
    assert row["return_vs_peer_median_q"] == 0.05          # 0.05 - median(0.0)=0.0
    assert row["n_valid_peers_next"] == 5


def test_too_few_valid_peers_gives_na_label_but_keeps_features():
    peers = _peers("F", ["P1", "P2", "P3", "P4", "P5"])
    returns = pd.concat([
        _returns({"F": 0.05, "P1": 0.0, "P2": 0.0, "P3": 0.0, "P4": 0.0, "P5": 0.0}, "2024q1"),
        _returns({"F": 0.01, "P1": 0.02, "P2": 0.03}, "2024q2"),   # only 2 peers valid at Q+1
    ])
    out = compute_peer_labels(peers, returns, QUARTERS, top_n=5, min_valid_peers=5)
    row = out.iloc[0]
    assert pd.isna(row["underperform_next_quarter"])
    assert row["return_vs_peer_median_q"] == 0.05          # feature still computed


def test_last_quarter_rows_kept_with_na_label():
    peers = _peers("F", ["P1", "P2", "P3", "P4", "P5"], quarter="2024q2")
    returns = _returns({"F": 0.01, "P1": 0.0, "P2": 0.0, "P3": 0.0, "P4": 0.0, "P5": 0.0}, "2024q2")
    out = compute_peer_labels(peers, returns, QUARTERS, top_n=5, min_valid_peers=5)
    assert len(out) == 1                                    # forward row present
    assert pd.isna(out.iloc[0]["underperform_next_quarter"])


def test_only_top_n_peers_used():
    peers = _peers("F", ["P1", "P2", "P3", "P4", "P5", "OUTLIER"])   # rank 6 must be ignored
    returns = pd.concat([
        _returns({"F": 0.0, "P1": 0.0, "P2": 0.0, "P3": 0.0, "P4": 0.0, "P5": 0.0, "OUTLIER": 0.0}, "2024q1"),
        _returns({"F": 0.05, "P1": 0.01, "P2": 0.02, "P3": 0.03, "P4": 0.06, "P5": 0.07,
                  "OUTLIER": 99.0}, "2024q2"),
    ])
    out = compute_peer_labels(peers, returns, QUARTERS, top_n=5, min_valid_peers=5)
    assert out.iloc[0]["peer_median_return_next"] == 0.03   # median without OUTLIER
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_step7_label.py -q`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` on `label`.

- [ ] **Step 3: Implement `label.py`**

```python
"""step7_unified_universe/label.py - the kNN-peer-relative underperformance label.

'Underperform' = the fund's next-quarter return is below the MEDIAN of its own top-N
most-similar funds' next-quarter returns (peers as-of Q, by cosine similarity on holdings
embeddings). Bespoke, constant-size peer groups by construction - immune to universe
growth, unlike the old cluster-median label (see design.md, diagnosis #1).
"""
import logging

import pandas as pd

log = logging.getLogger(__name__)

LABEL_DEFINITION = ("next-quarter return below the median of the fund's top-{top_n} "
                    "cosine-similarity peers' next-quarter returns")


def compute_peer_labels(fund_peers: pd.DataFrame, quarterly_returns: pd.DataFrame,
                        quarters_ordered: list, top_n: int, min_valid_peers: int) -> pd.DataFrame:
    """One row per (series_id, quarter) present in fund_peers. Label is pd.NA when the
    fund's own Q+1 return is missing, fewer than min_valid_peers peers have a valid Q+1
    return, or Q is the last quarter (those rows are the forward-prediction set - kept)."""
    quarter_to_next = dict(zip(quarters_ordered[:-1], quarters_ordered[1:]))
    peers = fund_peers[fund_peers["peer_rank"] <= top_n].copy()
    peers["next_quarter"] = peers["quarter"].map(quarter_to_next)

    peer_ret_q = quarterly_returns.rename(
        columns={"series_id": "peer_series_id", "quarterly_return": "peer_return_q"})
    peers = peers.merge(peer_ret_q, on=["peer_series_id", "quarter"], how="left")
    peer_ret_next = quarterly_returns.rename(
        columns={"series_id": "peer_series_id", "quarter": "next_quarter",
                 "quarterly_return": "peer_return_next"})
    peers = peers.merge(peer_ret_next, on=["peer_series_id", "next_quarter"], how="left")

    agg = peers.groupby(["series_id", "quarter"]).agg(
        peer_median_return_q=("peer_return_q", "median"),
        peer_median_return_next=("peer_return_next", "median"),
        n_valid_peers_next=("peer_return_next", "count"),
    ).reset_index()

    agg = agg.merge(quarterly_returns, on=["series_id", "quarter"], how="left")
    agg["next_quarter"] = agg["quarter"].map(quarter_to_next)
    own_next = quarterly_returns.rename(
        columns={"quarter": "next_quarter", "quarterly_return": "own_return_next"})
    agg = agg.merge(own_next, on=["series_id", "next_quarter"], how="left")

    agg["return_vs_peer_median_q"] = agg["quarterly_return"] - agg["peer_median_return_q"]

    valid = agg["own_return_next"].notna() & (agg["n_valid_peers_next"] >= min_valid_peers)
    agg["underperform_next_quarter"] = pd.Series(pd.NA, index=agg.index, dtype="Int64")
    agg.loc[valid, "underperform_next_quarter"] = (
        agg.loc[valid, "own_return_next"] < agg.loc[valid, "peer_median_return_next"]
    ).astype("Int64")

    n_dropped_label = int((~valid & agg["next_quarter"].notna()).sum())
    if n_dropped_label:
        log.info(f"{n_dropped_label} non-final fund-quarters have no label "
                 f"(own Q+1 return missing or <{min_valid_peers} valid peers)")
    return agg[["series_id", "quarter", "peer_median_return_q", "return_vs_peer_median_q",
                "peer_median_return_next", "n_valid_peers_next", "underperform_next_quarter"]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_step7_label.py tests/ -q` — Expected: all pass.

- [ ] **Step 5: Stage, scan, commit**

```bash
git add steps/step7_unified_universe/label.py tests/test_step7_label.py
git diff --cached | grep -niE "BEGIN.*PRIVATE KEY|sk_live_|AKIA|ghp_|password\s*=|postgres://"   # expect no output
git commit -m "step7: kNN-peer-median underperformance label with min-valid-peers guard and forward rows"
```

---

### Task 5: Holdings-derived feature module

**Files:**
- Create: `steps/step7_unified_universe/features.py`
- Test: `tests/test_step7_features.py`

**Interfaces:**
- Produces:
  - `features.compute_holdings_features(equity_holdings) -> pd.DataFrame [series_id, quarter, hhi, top10_weight, n_holdings]` — input needs columns `series_id, quarter, currency_value` (EC rows only, already joined to series_id).
  - `features.compute_peer_similarity_feature(fund_peers, top_n) -> pd.DataFrame [series_id, quarter, mean_peer_similarity]`.
  - `features.compute_net_asset_momentum(funds) -> pd.DataFrame [series_id, quarter, net_assets_qoq]` (NaN for each fund's first quarter).
- Consumes: `fund_peers` schema from step2; `funds_all` columns `series_id, quarter, net_assets`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd

from steps.step7_unified_universe.features import (
    compute_holdings_features, compute_net_asset_momentum, compute_peer_similarity_feature)


def test_holdings_features_concentration():
    h = pd.DataFrame({
        "series_id": ["F"] * 2 + ["G"] * 4,
        "quarter": "2024q1",
        "currency_value": [50.0, 50.0, 25.0, 25.0, 25.0, 25.0],
    })
    out = compute_holdings_features(h).set_index("series_id")
    assert out.loc["F", "hhi"] == 0.5            # two equal holdings: 2*(0.5^2)
    assert out.loc["G", "hhi"] == 0.25           # four equal holdings: 4*(0.25^2)
    assert out.loc["F", "top10_weight"] == 1.0
    assert out.loc["F", "n_holdings"] == 2


def test_holdings_features_top10_weight_caps_at_ten():
    h = pd.DataFrame({"series_id": "F", "quarter": "2024q1",
                      "currency_value": [10.0] * 20})
    out = compute_holdings_features(h)
    assert np.isclose(out.iloc[0]["top10_weight"], 0.5)   # 10 of 20 equal holdings


def test_peer_similarity_feature_uses_top_n_only():
    peers = pd.DataFrame({
        "series_id": "F", "quarter": "2024q1", "peer_rank": [1, 2, 3],
        "peer_series_id": ["A", "B", "C"], "cosine_similarity": [0.9, 0.7, 0.1],
    })
    out = compute_peer_similarity_feature(peers, top_n=2)
    assert np.isclose(out.iloc[0]["mean_peer_similarity"], 0.8)


def test_net_asset_momentum_first_quarter_is_nan():
    funds = pd.DataFrame({
        "series_id": ["F", "F", "G"], "quarter": ["2024q1", "2024q2", "2024q1"],
        "net_assets": [100.0, 110.0, 5.0],
    })
    out = compute_net_asset_momentum(funds).set_index(["series_id", "quarter"])
    assert pd.isna(out.loc[("F", "2024q1"), "net_assets_qoq"])
    assert np.isclose(out.loc[("F", "2024q2"), "net_assets_qoq"], 0.10)
    assert pd.isna(out.loc[("G", "2024q1"), "net_assets_qoq"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_step7_features.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `features.py`**

```python
"""step7_unified_universe/features.py - holdings-derived model features.

The pipeline's thesis is holdings-based similarity, yet the step4 model's features were
almost entirely trailing-return stats (see design.md, diagnosis #2). These features put
the holdings data into the model: concentration, breadth, peer-typicality, and asset flow.
"""
import pandas as pd


def compute_holdings_features(equity_holdings: pd.DataFrame) -> pd.DataFrame:
    """Per (series_id, quarter), from the fund's EC sleeve: HHI and top-10 weight share of
    sleeve-renormalized weights (consistent with step2's build_holdings_description
    renormalization), plus the raw count of EC holdings."""
    rows = []
    for (series_id, quarter), g in equity_holdings.groupby(["series_id", "quarter"]):
        total = g["currency_value"].sum()
        if total <= 0:
            continue
        w = (g["currency_value"] / total).sort_values(ascending=False)
        rows.append({
            "series_id": series_id, "quarter": quarter,
            "hhi": float((w ** 2).sum()),
            "top10_weight": float(w.head(10).sum()),
            "n_holdings": int(len(w)),
        })
    return pd.DataFrame(rows, columns=["series_id", "quarter", "hhi", "top10_weight", "n_holdings"])


def compute_peer_similarity_feature(fund_peers: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Mean cosine similarity to the fund's top-N peers - how 'typical' the fund is of its
    own niche. A low value = the fund has no close peers; its peer benchmark is weaker."""
    top = fund_peers[fund_peers["peer_rank"] <= top_n]
    return (top.groupby(["series_id", "quarter"])["cosine_similarity"].mean()
            .rename("mean_peer_similarity").reset_index())


def compute_net_asset_momentum(funds: pd.DataFrame) -> pd.DataFrame:
    """Quarter-over-quarter change in net assets (a flow/size-trend proxy). NaN for each
    fund's first observed quarter - those rows keep the panel's dropped-if-missing handling."""
    df = (funds[["series_id", "quarter", "net_assets"]]
          .drop_duplicates(["series_id", "quarter"])
          .sort_values(["series_id", "quarter"]).copy())
    df["net_assets_qoq"] = df.groupby("series_id")["net_assets"].pct_change()
    return df[["series_id", "quarter", "net_assets_qoq"]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_step7_features.py tests/ -q` — Expected: all pass.

- [ ] **Step 5: Stage, scan, commit**

```bash
git add steps/step7_unified_universe/features.py tests/test_step7_features.py
git diff --cached | grep -niE "BEGIN.*PRIVATE KEY|sk_live_|AKIA|ghp_|password\s*=|postgres://"   # expect no output
git commit -m "step7: holdings-derived features - HHI, top-10 weight, breadth, peer typicality, net-asset momentum"
```

---

### Task 6: Unified panel assembly

**Files:**
- Create: `steps/step7_unified_universe/panel.py`
- Test: `tests/test_step7_panel.py`

**Interfaces:**
- Produces: `panel.assemble_unified_panel(cfg) -> tuple[pd.DataFrame, pd.DataFrame, list]` = `(labeled, forward, feature_cols)`.
  - `labeled`: rows with all features and a non-NA label; columns include `series_id, quarter, underperform_next_quarter` + `feature_cols`.
  - `forward`: last-quarter rows with all features, label NA (the 2024q4 → 2025q1 prediction set).
  - `feature_cols`: exactly `["trailing_return", "trailing_volatility", "trailing_sharpe", "trailing_max_drawdown", "return_vs_peer_median_q", "net_assets", "net_assets_qoq", "mean_peer_similarity", "hhi", "top10_weight", "n_holdings"] + sorted tier_* dummy columns`.
- Consumes: tables `funds_all, monthly_returns_all, holdings_all, fund_peers_all`; `compute_trailing_features` (step4), `compute_peer_labels` (Task 4), the three Task 5 functions, `category_tier` (fundspeers.category); config keys `metrics.risk_free_annual`, `unified.peer_label_top_n`, `unified.min_valid_peers_for_label`.

- [ ] **Step 1: Write the failing test** — a small end-to-end synthetic panel through a temp DB.

```python
"""assemble_unified_panel on a tiny synthetic universe: 1 strategy fund + 5 peers,
2 quarters, checking feature columns, label wiring, and the forward split."""
import pandas as pd
import pytest

from fundspeers.io import save_table
from steps.step7_unified_universe.panel import assemble_unified_panel

QUARTERS = ["2024q1", "2024q2"]
IDS = ["F", "P1", "P2", "P3", "P4", "P5"]


@pytest.fixture
def cfg(tmp_path):
    return {
        "seed": 42,
        "paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "processed"),
                  "reports": str(tmp_path / "reports"), "models": str(tmp_path / "models")},
        "metrics": {"risk_free_annual": 0.02},
        "unified": {"n_clusters": 2, "peer_label_top_n": 5, "min_valid_peers_for_label": 5,
                    "bootstrap_iterations": 10, "label_stability_draws": 5},
    }


@pytest.fixture
def tables(cfg):
    funds = pd.DataFrame([
        {"series_id": s, "quarter": q, "accession_number": f"acc-{s}-{q}",
         "yahoo_category": "Large Blend", "is_us_equity": True, "segment": "strategy",
         "net_assets": 100.0 + i, "series_name": s, "ticker": s}
        for i, s in enumerate(IDS) for q in QUARTERS])
    returns = pd.DataFrame([
        {"series_id": s, "quarter": q, "month_in_quarter": m, "total_return": 1.0 + i * 0.1}
        for i, s in enumerate(IDS) for q in QUARTERS for m in (1, 2, 3)])
    holdings = pd.DataFrame([
        {"accession_number": f"acc-{s}-{q}", "quarter": q, "asset_cat": "EC",
         "issuer_name": f"CO{k}", "currency_value": 10.0}
        for s in IDS for q in QUARTERS for k in range(12)])
    peers = pd.DataFrame([
        {"series_id": s, "quarter": q, "peer_rank": r,
         "peer_series_id": p, "cosine_similarity": 0.9}
        for s in IDS for q in QUARTERS
        for r, p in enumerate([x for x in IDS if x != s], start=1)])
    save_table(funds, "funds_all", cfg)
    save_table(returns, "monthly_returns_all", cfg)
    save_table(holdings, "holdings_all", cfg)
    save_table(peers, "fund_peers_all", cfg)


def test_panel_shapes_and_features(cfg, tables):
    labeled, forward, feature_cols = assemble_unified_panel(cfg)
    assert "return_vs_peer_median_q" in feature_cols
    assert "hhi" in feature_cols and "mean_peer_similarity" in feature_cols
    assert any(c.startswith("tier_") for c in feature_cols)
    # 2024q1 rows are labelable; net_assets_qoq is NaN in q1 so labeled may be empty -
    # the forward set must still carry complete 2024q2 features:
    assert set(forward["quarter"]) == {"2024q2"}
    assert forward[feature_cols].notna().all().all()
    assert forward["underperform_next_quarter"].isna().all()


def test_labeled_rows_have_no_missing_values(cfg, tables):
    labeled, _, feature_cols = assemble_unified_panel(cfg)
    if len(labeled):
        assert labeled[feature_cols].notna().all().all()
        assert labeled["underperform_next_quarter"].notna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_step7_panel.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `panel.py`**

```python
"""step7_unified_universe/panel.py - assemble the unified feature/label panel.

One row per (strategy-segment series_id, quarter): trailing return stats (point-in-time
safe, reused from step4), holdings features, peer features, the kNN-peer label. Rows with
any missing feature are dropped, not imputed; last-quarter rows with complete features
form the FORWARD set (label unknowable yet - the dashboard's genuine forward predictions).
"""
import logging

import pandas as pd

from fundspeers.category import category_tier
from fundspeers.io import load_table
from steps.step4_predict.predict import compute_trailing_features
from steps.step7_unified_universe.features import (
    compute_holdings_features, compute_net_asset_momentum, compute_peer_similarity_feature)
from steps.step7_unified_universe.label import compute_peer_labels

log = logging.getLogger(__name__)

BASE_FEATURE_COLS = [
    "trailing_return", "trailing_volatility", "trailing_sharpe", "trailing_max_drawdown",
    "return_vs_peer_median_q", "net_assets", "net_assets_qoq", "mean_peer_similarity",
    "hhi", "top10_weight", "n_holdings",
]


def _quarterly_returns_from_monthly(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """Compound each (series_id, quarter)'s 3 monthly returns; .values so a missing month
    correctly propagates NaN (the step3 np.prod-skipna bug, not repeated here)."""
    import numpy as np
    df = monthly_returns.copy()
    df["r"] = df["total_return"] / 100.0
    df = df.sort_values(["series_id", "quarter", "month_in_quarter"])
    grouped = df.groupby(["series_id", "quarter"])["r"].apply(
        lambda s: np.prod(1 + s.values) - 1)
    return grouped.reset_index().rename(columns={"r": "quarterly_return"})


def assemble_unified_panel(cfg: dict):
    funds = load_table("funds_all", cfg)
    monthly_returns = load_table("monthly_returns_all", cfg)
    holdings = load_table("holdings_all", cfg)
    fund_peers = load_table("fund_peers_all", cfg)

    strategy = funds[funds["is_us_equity"] & (funds["segment"] == "strategy")]
    strategy_series = set(strategy["series_id"].unique())
    quarters_ordered = sorted(strategy["quarter"].unique())
    returns = monthly_returns[monthly_returns["series_id"].isin(strategy_series)]

    trailing = compute_trailing_features(returns, cfg["metrics"]["risk_free_annual"])
    quarterly_returns = _quarterly_returns_from_monthly(returns)
    labels = compute_peer_labels(
        fund_peers, quarterly_returns, quarters_ordered,
        top_n=cfg["unified"]["peer_label_top_n"],
        min_valid_peers=cfg["unified"]["min_valid_peers_for_label"])

    acc_lookup = strategy[["accession_number", "series_id"]].drop_duplicates()
    ec = holdings[(holdings["asset_cat"] == "EC")
                  & holdings["accession_number"].isin(set(acc_lookup["accession_number"]))]
    ec = ec.merge(acc_lookup, on="accession_number", how="inner")
    holdings_feats = compute_holdings_features(ec[["series_id", "quarter", "currency_value"]])

    peer_sim = compute_peer_similarity_feature(fund_peers, cfg["unified"]["peer_label_top_n"])
    momentum = compute_net_asset_momentum(strategy)

    fund_static = strategy.drop_duplicates("series_id")[["series_id", "yahoo_category"]].copy()
    fund_static["category_tier"] = fund_static["yahoo_category"].map(category_tier)
    funds_at_q = strategy[["series_id", "quarter", "net_assets"]].drop_duplicates(
        ["series_id", "quarter"])

    panel = (trailing
             .merge(labels, on=["series_id", "quarter"], how="inner")
             .merge(holdings_feats, on=["series_id", "quarter"], how="left")
             .merge(peer_sim, on=["series_id", "quarter"], how="left")
             .merge(momentum, on=["series_id", "quarter"], how="left")
             .merge(funds_at_q, on=["series_id", "quarter"], how="left")
             .merge(fund_static, on="series_id", how="left"))

    tier_dummies = pd.get_dummies(panel["category_tier"], prefix="tier")
    panel = pd.concat([panel.reset_index(drop=True), tier_dummies.reset_index(drop=True)], axis=1)
    tier_cols = sorted(c for c in panel.columns if c.startswith("tier_"))
    feature_cols = BASE_FEATURE_COLS + tier_cols

    complete = panel.dropna(subset=feature_cols)
    n_dropped = len(panel) - len(complete)
    if n_dropped:
        log.info(f"dropped {n_dropped} of {len(panel)} rows with a missing feature")

    last_quarter = quarters_ordered[-1]
    forward = complete[(complete["quarter"] == last_quarter)
                       & complete["underperform_next_quarter"].isna()]
    labeled = complete[complete["underperform_next_quarter"].notna()]
    log.info(f"unified panel: {len(labeled)} labeled rows, {len(forward)} forward rows "
             f"({last_quarter} -> next), {len(feature_cols)} features")
    return labeled, forward, feature_cols
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_step7_panel.py tests/ -q` — Expected: all pass.

- [ ] **Step 5: Stage, scan, commit**

```bash
git add steps/step7_unified_universe/panel.py tests/test_step7_panel.py
git diff --cached | grep -niE "BEGIN.*PRIVATE KEY|sk_live_|AKIA|ghp_|password\s*=|postgres://"   # expect no output
git commit -m "step7: unified panel assembly - trailing + holdings + peer features, kNN label, forward split"
```

---

### Task 7: Model, baselines, and Monte Carlo evaluation

**Files:**
- Create: `steps/step7_unified_universe/model.py`
- Test: `tests/test_step7_model.py`

**Interfaces:**
- Produces:
  - `model.fund_clustered_bootstrap(test, y_col, model_score_col, persistence_score_col, iterations, seed) -> dict` with keys `auc_ci_low, auc_ci_high, persistence_ci_low, persistence_ci_high, edge_ci_low, edge_ci_high, p_edge_le_zero` (floats).
  - `model.train_and_evaluate(cfg) -> dict` (summary numbers) — fits the RF on the time split, computes pooled + per-quarter AUC, persistence-baseline AUC (score = `-return_vs_peer_median_q`), runs the bootstrap, saves `models/unified_rf_model.joblib` (bundle: `{"model", "feature_cols", "label_definition"}`) and tables `unified_panel`, `unified_predictions` (columns `series_id, quarter, predicted_probability, actual_label, split` with split ∈ train/test/forward), `unified_feature_importances`, `unified_model_eval` (long format: `metric, quarter, value`; `quarter` empty string for pooled metrics).
- Consumes: `assemble_unified_panel` (Task 6), `time_based_split` (step4), config `model.rf`, `model.test_transitions_holdout`, `unified.bootstrap_iterations`, `seed`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd

from steps.step7_unified_universe.model import fund_clustered_bootstrap


def _fake_test(n_funds=40, rows_per_fund=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_funds):
        for q in range(rows_per_fund):
            y = int(rng.random() < 0.5)
            rows.append({"series_id": f"S{i}", "underperform_next_quarter": y,
                         # informative model score, weak persistence score:
                         "proba": y * 0.6 + rng.random() * 0.4,
                         "persist": rng.random()})
    return pd.DataFrame(rows)


def test_bootstrap_is_deterministic_under_seed():
    test = _fake_test()
    a = fund_clustered_bootstrap(test, "underperform_next_quarter", "proba", "persist",
                                 iterations=50, seed=42)
    b = fund_clustered_bootstrap(test, "underperform_next_quarter", "proba", "persist",
                                 iterations=50, seed=42)
    assert a == b


def test_bootstrap_detects_real_edge():
    test = _fake_test()
    out = fund_clustered_bootstrap(test, "underperform_next_quarter", "proba", "persist",
                                   iterations=200, seed=42)
    assert out["auc_ci_low"] > out["persistence_ci_high"] - 0.2   # model clearly better
    assert out["p_edge_le_zero"] < 0.05
    assert out["auc_ci_low"] < out["auc_ci_high"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_step7_model.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `model.py`**

```python
"""step7_unified_universe/model.py - fit the unified RF, evaluate vs baselines, and run
the Monte Carlo uncertainty layer (see design.md section 6b: MC quantifies evaluation
uncertainty; it never simulates fund returns).
"""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from fundspeers.io import save_model, save_table
from steps.step4_predict.predict import time_based_split
from steps.step7_unified_universe.label import LABEL_DEFINITION
from steps.step7_unified_universe.panel import assemble_unified_panel

log = logging.getLogger(__name__)


def fund_clustered_bootstrap(test: pd.DataFrame, y_col: str, model_score_col: str,
                             persistence_score_col: str, iterations: int, seed: int) -> dict:
    """Resample test-set FUNDS with replacement (a fund's quarters stay together - rows
    within a fund correlate, so row-level resampling would understate variance). Returns
    95% CIs for model AUC, persistence AUC, their difference, and the one-sided
    p(edge <= 0). Paired: both AUCs computed on the SAME resample each iteration."""
    rng = np.random.default_rng(seed)
    groups = {s: g for s, g in test.groupby("series_id")}
    series = np.array(sorted(groups))
    model_aucs, persist_aucs = [], []
    for _ in range(iterations):
        draw = rng.choice(series, size=len(series), replace=True)
        sample = pd.concat([groups[s] for s in draw], ignore_index=True)
        if sample[y_col].nunique() < 2:
            continue
        model_aucs.append(roc_auc_score(sample[y_col], sample[model_score_col]))
        persist_aucs.append(roc_auc_score(sample[y_col], sample[persistence_score_col]))
    model_aucs, persist_aucs = np.array(model_aucs), np.array(persist_aucs)
    edge = model_aucs - persist_aucs
    return {
        "auc_ci_low": float(np.percentile(model_aucs, 2.5)),
        "auc_ci_high": float(np.percentile(model_aucs, 97.5)),
        "persistence_ci_low": float(np.percentile(persist_aucs, 2.5)),
        "persistence_ci_high": float(np.percentile(persist_aucs, 97.5)),
        "edge_ci_low": float(np.percentile(edge, 2.5)),
        "edge_ci_high": float(np.percentile(edge, 97.5)),
        "p_edge_le_zero": float((edge <= 0).mean()),
    }


def train_and_evaluate(cfg: dict) -> dict:
    labeled, forward, feature_cols = assemble_unified_panel(cfg)
    quarters_ordered = sorted(set(labeled["quarter"]) | set(forward["quarter"]))
    train, test, train_q, test_q = time_based_split(
        labeled, quarters_ordered, cfg["model"]["test_transitions_holdout"])
    if max(train_q) >= min(test_q):
        raise RuntimeError(f"split leaks: train up to {max(train_q)}, test from {min(test_q)}")

    x_train, y_train = train[feature_cols], train["underperform_next_quarter"].astype(int)
    x_test, y_test = test[feature_cols], test["underperform_next_quarter"].astype(int)

    rf = RandomForestClassifier(
        n_estimators=cfg["model"]["rf"]["n_estimators"],
        max_depth=cfg["model"]["rf"]["max_depth"],
        min_samples_leaf=cfg["model"]["rf"]["min_samples_leaf"],
        random_state=cfg["seed"],
    ).fit(x_train, y_train)

    test = test.assign(proba=rf.predict_proba(x_test)[:, 1],
                       persist=-test["return_vs_peer_median_q"])
    pooled_auc = roc_auc_score(y_test, test["proba"])
    persistence_auc = roc_auc_score(y_test, test["persist"])

    eval_rows = [
        {"metric": "auc_pooled", "quarter": "", "value": pooled_auc},
        {"metric": "auc_persistence_baseline", "quarter": "", "value": persistence_auc},
        {"metric": "auc_random_baseline", "quarter": "", "value": 0.5},
    ]
    quarters_model_wins = 0
    for q, g in test.groupby("quarter"):
        yq = g["underperform_next_quarter"].astype(int)
        if yq.nunique() < 2:
            continue
        auc_q = roc_auc_score(yq, g["proba"])
        persist_q = roc_auc_score(yq, g["persist"])
        quarters_model_wins += int(auc_q > persist_q)
        eval_rows.append({"metric": "auc_pooled", "quarter": q, "value": auc_q})
        eval_rows.append({"metric": "auc_persistence_baseline", "quarter": q, "value": persist_q})

    boot = fund_clustered_bootstrap(
        test, "underperform_next_quarter", "proba", "persist",
        iterations=cfg["unified"]["bootstrap_iterations"], seed=cfg["seed"])
    eval_rows += [{"metric": k, "quarter": "", "value": v} for k, v in boot.items()]

    label_definition = LABEL_DEFINITION.format(top_n=cfg["unified"]["peer_label_top_n"])
    save_model({"model": rf, "feature_cols": feature_cols,
                "label_definition": label_definition}, "unified_rf_model", cfg)

    predictions = pd.concat([
        train.assign(split="train", predicted_probability=rf.predict_proba(x_train)[:, 1]),
        test.assign(split="test", predicted_probability=test["proba"]),
        forward.assign(split="forward",
                       predicted_probability=rf.predict_proba(forward[feature_cols])[:, 1]),
    ])[["series_id", "quarter", "predicted_probability", "underperform_next_quarter", "split"]]
    predictions = predictions.rename(columns={"underperform_next_quarter": "actual_label"})
    predictions["actual_label"] = predictions["actual_label"].astype("float")  # NA-safe for duckdb

    importances = pd.DataFrame({"feature": feature_cols,
                                "importance": rf.feature_importances_}
                               ).sort_values("importance", ascending=False)

    save_table(pd.concat([labeled, forward], ignore_index=True), "unified_panel", cfg)
    save_table(predictions, "unified_predictions", cfg)
    save_table(importances, "unified_feature_importances", cfg)
    save_table(pd.DataFrame(eval_rows), "unified_model_eval", cfg)

    log.info(f"unified RF: pooled test AUC={pooled_auc:.3f} "
             f"[{boot['auc_ci_low']:.3f}, {boot['auc_ci_high']:.3f}] vs persistence "
             f"{persistence_auc:.3f}; edge CI [{boot['edge_ci_low']:.3f}, "
             f"{boot['edge_ci_high']:.3f}], p(edge<=0)={boot['p_edge_le_zero']:.4f}; "
             f"model beat persistence in {quarters_model_wins}/{len(test_q)} test quarters")
    return {"auc": pooled_auc, "persistence_auc": persistence_auc,
            "quarters_model_wins": quarters_model_wins, "n_test_quarters": len(test_q), **boot}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_step7_model.py tests/ -q` — Expected: all pass.

- [ ] **Step 5: Stage, scan, commit**

```bash
git add steps/step7_unified_universe/model.py tests/test_step7_model.py
git diff --cached | grep -niE "BEGIN.*PRIVATE KEY|sk_live_|AKIA|ghp_|password\s*=|postgres://"   # expect no output
git commit -m "step7: unified RF with persistence baseline and fund-clustered bootstrap uncertainty layer"
```

---

### Task 8: Label-stability perturbation study

**Files:**
- Create: `steps/step7_unified_universe/stability.py`
- Test: `tests/test_step7_stability.py`

**Interfaces:**
- Produces: `stability.compute_label_flip_rates(fund_peers, quarterly_returns, quarters_ordered, top_n, pool_size, draw_size, draws, seed) -> pd.DataFrame [series_id, quarter, flip_rate]` — for each labelable fund-quarter with ≥ `pool_size` peers having valid next-quarter returns: base label from the top-`top_n` peer median; each of `draws` seeded draws picks `draw_size` of the top-`pool_size` peers, recomputes the label, and `flip_rate` = fraction of draws disagreeing with the base label. Defaults used by the build: `top_n=10, pool_size=12, draw_size=8, draws=cfg["unified"]["label_stability_draws"]`. Also `stability.run_stability(cfg) -> dict` (summary: `mean_flip_rate, share_flip_gt_10pct, n_evaluated`) saving table `unified_label_stability`.
- Consumes: `fund_peers_all` (top 15 stored — Task 9 passes `top_n_peers=15`), quarterly returns from `panel._quarterly_returns_from_monthly`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd

from steps.step7_unified_universe.stability import compute_label_flip_rates

QUARTERS = ["2024q1", "2024q2"]


def _setup(peer_next_returns, own_next=0.0):
    ids = [f"P{i}" for i in range(len(peer_next_returns))]
    peers = pd.DataFrame({
        "series_id": "F", "quarter": "2024q1",
        "peer_rank": range(1, len(ids) + 1), "peer_series_id": ids,
        "cosine_similarity": 0.9})
    returns = pd.concat([
        pd.DataFrame({"series_id": ["F"] + ids, "quarter": "2024q1",
                      "quarterly_return": 0.0}),
        pd.DataFrame({"series_id": ["F"] + ids, "quarter": "2024q2",
                      "quarterly_return": [own_next] + peer_next_returns}),
    ])
    return peers, returns


def test_identical_peers_never_flip():
    peers, returns = _setup([0.05] * 15, own_next=0.0)   # fund clearly below every peer
    out = compute_label_flip_rates(peers, returns, QUARTERS,
                                   top_n=10, pool_size=12, draw_size=8, draws=50, seed=42)
    assert out.iloc[0]["flip_rate"] == 0.0


def test_knife_edge_peers_flip_sometimes():
    # 6 peers above the fund, 6 below: which 8 get drawn decides the median's side.
    peers, returns = _setup([0.1] * 6 + [-0.1] * 6 + [0.1] * 3, own_next=0.0)
    out = compute_label_flip_rates(peers, returns, QUARTERS,
                                   top_n=10, pool_size=12, draw_size=8, draws=200, seed=42)
    assert 0.0 < out.iloc[0]["flip_rate"] < 1.0


def test_deterministic_under_seed():
    peers, returns = _setup([0.1] * 6 + [-0.1] * 9, own_next=0.0)
    kw = dict(top_n=10, pool_size=12, draw_size=8, draws=100, seed=7)
    a = compute_label_flip_rates(peers, returns, QUARTERS, **kw)
    b = compute_label_flip_rates(peers, returns, QUARTERS, **kw)
    pd.testing.assert_frame_equal(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_step7_stability.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `stability.py`**

```python
"""step7_unified_universe/stability.py - Monte Carlo label-stability study.

The label depends on WHICH top-10 peers the similarity step surfaced. Perturb each fund's
peer set (draw_size of the top-pool_size peers, many seeded draws), recompute the label,
and measure the flip rate - an estimate of the label's intrinsic noise floor. High flip
rates would mean AUC gains are being chased into benchmark noise (design.md section 6b).
"""
import logging

import numpy as np
import pandas as pd

from fundspeers.io import load_table, save_table
from steps.step7_unified_universe.panel import _quarterly_returns_from_monthly

log = logging.getLogger(__name__)


def compute_label_flip_rates(fund_peers: pd.DataFrame, quarterly_returns: pd.DataFrame,
                             quarters_ordered: list, top_n: int, pool_size: int,
                             draw_size: int, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quarter_to_next = dict(zip(quarters_ordered[:-1], quarters_ordered[1:]))

    pool = fund_peers[fund_peers["peer_rank"] <= pool_size].copy()
    pool["next_quarter"] = pool["quarter"].map(quarter_to_next)
    ret_next = quarterly_returns.rename(
        columns={"series_id": "peer_series_id", "quarter": "next_quarter",
                 "quarterly_return": "peer_return_next"})
    pool = pool.merge(ret_next, on=["peer_series_id", "next_quarter"], how="left")
    own_next = quarterly_returns.rename(
        columns={"quarter": "next_quarter", "quarterly_return": "own_return_next"})

    rows = []
    for (series_id, quarter), g in pool.groupby(["series_id", "quarter"]):
        g = g.sort_values("peer_rank")
        valid = g["peer_return_next"].dropna().values
        nq = quarter_to_next.get(quarter)
        if nq is None or len(valid) < pool_size:
            continue
        own = quarterly_returns[(quarterly_returns["series_id"] == series_id)
                                & (quarterly_returns["quarter"] == nq)]["quarterly_return"]
        if own.empty or pd.isna(own.iloc[0]):
            continue
        own_r = own.iloc[0]
        base_pool = g["peer_return_next"].values[:top_n]
        base_label = own_r < np.nanmedian(base_pool)
        # all draws at once: (draws, draw_size) index matrix into the top-pool_size returns
        idx = np.array([rng.choice(pool_size, size=draw_size, replace=False)
                        for _ in range(draws)])
        medians = np.median(valid[:pool_size][idx], axis=1)
        flip_rate = float(((own_r < medians) != base_label).mean())
        rows.append({"series_id": series_id, "quarter": quarter, "flip_rate": flip_rate})
    return pd.DataFrame(rows, columns=["series_id", "quarter", "flip_rate"])


def run_stability(cfg: dict) -> dict:
    funds = load_table("funds_all", cfg)
    strategy_series = set(funds.loc[funds["is_us_equity"]
                                    & (funds["segment"] == "strategy"), "series_id"])
    monthly = load_table("monthly_returns_all", cfg)
    quarterly_returns = _quarterly_returns_from_monthly(
        monthly[monthly["series_id"].isin(strategy_series)])
    fund_peers = load_table("fund_peers_all", cfg)
    quarters_ordered = sorted(funds.loc[funds["series_id"].isin(strategy_series), "quarter"].unique())

    flips = compute_label_flip_rates(
        fund_peers, quarterly_returns, quarters_ordered,
        top_n=cfg["unified"]["peer_label_top_n"], pool_size=12, draw_size=8,
        draws=cfg["unified"]["label_stability_draws"], seed=cfg["seed"])
    save_table(flips, "unified_label_stability", cfg)
    summary = {
        "mean_flip_rate": float(flips["flip_rate"].mean()),
        "share_flip_gt_10pct": float((flips["flip_rate"] > 0.10).mean()),
        "n_evaluated": int(len(flips)),
    }
    log.info(f"label stability: mean flip rate {summary['mean_flip_rate']:.3f}, "
             f"{summary['share_flip_gt_10pct']:.1%} of fund-quarters flip >10% of draws "
             f"({summary['n_evaluated']} evaluated)")
    return summary
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_step7_stability.py tests/ -q` — Expected: all pass.

- [ ] **Step 5: Stage, scan, commit**

```bash
git add steps/step7_unified_universe/stability.py tests/test_step7_stability.py
git diff --cached | grep -niE "BEGIN.*PRIVATE KEY|sk_live_|AKIA|ghp_|password\s*=|postgres://"   # expect no output
git commit -m "step7: Monte Carlo label-stability study - peer-set perturbation flip rates"
```

---

### Task 9: Orchestrator + the real full run

**Files:**
- Create: `steps/step7_unified_universe/build.py`
- Test: (no new unit test — the run itself is the test; UAT criteria from design.md checked live)

**Interfaces:**
- Produces: `build.run(cfg)` executing merge → similarity(`_all`) → metrics(`_all`) → model → stability, and a `__main__` entry. All tables from earlier tasks exist afterward.
- Consumes: everything above; `similarity.run` extension (Task 3); `metrics.run(cfg, table_suffix="_all")` (already exists).

- [ ] **Step 1: Implement `build.py`**

```python
"""step7_unified_universe/build.py - deterministic orchestrator for the unified rebuild.

Run on demand (NOT wired into conductor.py - see design.md):
    python -m steps.step7_unified_universe.build
"""
import logging

from steps.step2_similarity import similarity
from steps.step3_metrics import metrics
from steps.step7_unified_universe import merge, model, stability

log = logging.getLogger(__name__)


def run(cfg: dict) -> None:
    log.info("=== step7 unified universe: merge ===")
    merge.build_unified_tables(cfg)
    log.info("=== step7: clustering + peers (k=%s, top-15 peers, strategy segment) ===",
             cfg["unified"]["n_clusters"])
    similarity.run(cfg, table_suffix="_all", n_clusters=cfg["unified"]["n_clusters"],
                   top_n_peers=15, require_segment="strategy", save_coords=True)
    log.info("=== step7: metrics ===")
    metrics.run(cfg, table_suffix="_all")
    log.info("=== step7: model + Monte Carlo evaluation ===")
    results = model.train_and_evaluate(cfg)
    log.info("=== step7: label-stability study ===")
    stability_summary = stability.run_stability(cfg)
    log.info(f"SUMMARY: {results} | stability: {stability_summary}")


if __name__ == "__main__":
    from fundspeers.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(load_config())
```

Note: `metrics.run("_all")` computes metrics for ALL equity funds including the allocation segment (the dashboard needs their performance stats); its cluster-relative columns are NaN for unclustered allocation funds — expected, `how="left"` join.

- [ ] **Step 2: Run the full build (long: ~10–20 min of embedding 12 quarters × ~2,100 funds)**

Run: `python -m steps.step7_unified_universe.build`
Expected: completes without error; log shows ≈2,243 funds (≈2,087 strategy / ≈156 allocation), 12 quarters clustered at k=30, pooled test AUC with CI, persistence comparison, per-quarter wins, stability summary.

- [ ] **Step 3: Check design.md UAT criteria against the logs**

Verify each (from `steps/step7_unified_universe/design.md` UAT section): 2,243 unique funds & segment split; per-quarter purity/ARI logged; ≤~6 clusters under 20 members per quarter; zero TDF-dominated clusters (spot-check `cluster_definitions_all` `short_title`s contain no Target-Date); label coverage (dropped rows <2% — check the label module's log line); model beats both baselines pooled AND in a majority of test quarters; forward predictions exist (`SELECT count(*) FROM unified_predictions WHERE split='forward'` > 0). Record every number in the Task 11 docs update.

- [ ] **Step 4: Model round-trip check**

```python
python -c "
from fundspeers.config import load_config
from fundspeers.io import load_model, load_table
cfg = load_config()
b = load_model('unified_rf_model', cfg)
panel = load_table('unified_panel', cfg)
labeled = panel[panel['underperform_next_quarter'].notna()]
x = labeled[b['feature_cols']]
p = b['model'].predict_proba(x)[:, 1]
print('round-trip OK:', len(p), 'predictions;', b['label_definition'])"
```
Expected: prints count + label definition, no error.

- [ ] **Step 5: Stage, scan, commit**

```bash
git add steps/step7_unified_universe/build.py reports/cluster_map_2024q4_all.png
git diff --cached | grep -niE "BEGIN.*PRIVATE KEY|sk_live_|AKIA|ghp_|password\s*=|postgres://"   # expect no output
git commit -m "step7: orchestrator + full unified run (results recorded in design.md next)"
```

---

### Task 10: One-time approach-B sanity check (recorded, not maintained)

**Files:**
- Create (scratchpad only, NOT committed): `<scratchpad>/approach_b_check.py`
- Modify: `steps/step7_unified_universe/design.md` (record the number in section 6)

**Interfaces:** none (evidence for the design record).

- [ ] **Step 1: Run the check** — old cluster-median label on the merged universe with scaled clusters, to confirm granularity was the driver. Write a scratch script that: loads `fund_clusters_all` + quarterly returns, computes the OLD label (below cluster median next quarter, reusing `steps.step3_metrics.metrics.compute_cluster_relative_metrics` against `fund_clusters_all` at k=30), assembles the old-style feature set (trailing + `return_vs_cluster_median_q` + net_assets + tiers via step4's `assemble_panel`), fits the same seeded RF on the same time split, prints test AUC.

Run: `python <scratchpad>/approach_b_check.py`
Expected: an AUC. Whatever it is, it gets recorded.

- [ ] **Step 2: Record in design.md** — append one sentence to section 6's sanity-check paragraph: "Run on YYYY-MM-DD: scaled-cluster (k=30) old-label AUC on the merged universe = X.XXX vs the kNN-label unified model's Y.YYY — [confirms / complicates] the granularity diagnosis." Use the real numbers; report honestly either way.

- [ ] **Step 3: Stage, scan, commit**

```bash
git add steps/step7_unified_universe/design.md
git commit -m "step7: record approach-B sanity check result in design.md"
```

---

### Task 11: Live docs, UAT report, approval gate

**Files:**
- Modify: `decisions.json` (append 1–2 entries: the executed rebuild + its headline numbers; any surprise found during the run)
- Modify: `workflow.json` (append `{"name": "step7_unified_universe", "status": "done"}`)
- Modify: `steps/step7_unified_universe/design.md` (append `## UAT results` section with every measured number from Task 9 Step 3)
- Modify: `HANDOFF.md` (update to current state, or delete if the session is continuing)

**Interfaces:** none.

- [ ] **Step 1: Append the decisions.json entry** — decision: executed the unified rebuild; rationale: the real numbers (fund counts, purity/ARI at k=30, pooled AUC + CI, persistence AUC, p(edge≤0), per-quarter wins, mean flip rate). Write it from the actual run output, not from expectations.
- [ ] **Step 2: Append `## UAT results` to design.md** — one line per UAT criterion: criterion → measured value → pass/fail.
- [ ] **Step 3: Update workflow.json, re-render docs**

Run: `python scripts/render_docs.py && python -m pytest tests/ -q`
Expected: docs written; all tests pass.

- [ ] **Step 4: Stage, scan, commit**

```bash
git add decisions.json workflow.json reflection.html workflow.html steps/step7_unified_universe/design.md
git diff --cached | grep -niE "BEGIN.*PRIVATE KEY|sk_live_|AKIA|ghp_|password\s*=|postgres://"   # expect no output
git commit -m "step7: UAT results and live docs - unified universe rebuild complete"
```

- [ ] **Step 5: STOP — human approval gate.** Report the UAT results to the user and wait for approval before starting the step8 plan (`docs/superpowers/plans/2026-07-11-step8-dashboard.md`).
