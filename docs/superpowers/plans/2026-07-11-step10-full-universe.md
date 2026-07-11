# step10_full_universe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exhaust the fund universe under a relaxed ≥6-consecutive-quarters eligibility rule, extend data through 2026q1 (17 quarters), retrain with a 2025–26 test period, and score the frozen step7 model plus the dashboard's 2,086 committed forward predictions against realized returns.

**Architecture:** Backward-compatible extensions to step1's ingestion (relaxed pool, metadata reuse, uncapped exhaust mode) → `_full` table namespace → step7's generalized modules (suffix/holdout params, defaults preserve `_all` behavior) → new `steps/step10_full_universe/validate.py` (out-of-time scoring) → `build.py` orchestrator.

**Authoritative spec:** `steps/step10_full_universe/design.md` (committed). Read it first.

## Global Constraints

- Eligibility (exact): a fund qualifies for every quarter it filed, requiring ≥6 CONSECUTIVE quarters of returns somewhere in 2022q1–2026q1 (`full.min_consecutive_quarters: 6`).
- `data.quarters` extended to the 17 quarters 2022q1–2026q1. All existing tables (`_all`, per-batch) and `models/unified_rf_model.joblib` are READ-ONLY history.
- Metadata reuse (exact): series already present in ANY existing funds table (`funds`, `funds_oos`, `funds_oos2`) reuse their ticker/yahoo_category/yahoo_stock_position — ZERO repeat Yahoo lookups for known series. Only new candidates hit Yahoo.
- Exhaust mode: EVERY eligible candidate is attempted (no max_funds cap in the `_full` run).
- Backward compatibility: every modified function keeps its current behavior when new params are omitted (the step3-similarity precedent). Existing 75 tests must stay green untouched.
- Split (exact): transitions Q∈2022q1..2025q4; train Q ≤ 2024q3 (11 transitions), test Q∈2024q4..2025q4 (5 transitions) — implemented as `holdout_transitions=5` against the 17-quarter list.
- Determinism: same seed machinery throughout. Downloads idempotent (cached ZIPs skipped) — this ingestion doubles as step13's future advance-quarter mode.
- NOTHING opens the DuckDB while a pipeline process is writing (hard lesson).
- Every commit: explicit staging + secret scan (`git diff --cached | grep -niE "BEGIN.*PRIVATE KEY|sk_live_|AKIA|ghp_|password\s*=|postgres://"` → no output).
- Model bundles/tables for this run use the `full_` prefix (`full_rf_model`, `full_panel`, `full_predictions`, `full_feature_importances`, `full_model_eval`, `full_label_stability`, plus `oot_validation` from Task 5).

---

### Task 1: Ingestion extensions (eligibility, reuse, exhaust) — code only

**Files:**
- Modify: `config.json` (quarters ×17; new `"full": {"min_consecutive_quarters": 6}`)
- Modify: `steps/step1_ingest/ingest.py` (backward-compatible params)
- Test: `tests/test_step10_ingest_ext.py`

**Interfaces (contract — the implementer reads ingest.py first and integrates minimally):**
- A NEW pure function `derive_relaxed_pool(presence: pd.DataFrame, min_consecutive: int) -> set` — `presence` has columns `series_id, quarter`; returns series with ≥`min_consecutive` CONSECUTIVE quarters present (quarters compared against the sorted global quarter list; gaps break runs). Replaces the all-quarters intersection WHEN the new mode is used.
- `run(cfg, exclude_series=None, table_suffix="", relaxed_pool=False, reuse_metadata_from: list[str] | None = None, max_funds_override: int | None = None)` — new params default to today's behavior exactly. `relaxed_pool=True` swaps the pool derivation; `reuse_metadata_from=["funds","funds_oos","funds_oos2"]` builds a series→(ticker, yahoo_category, yahoo_stock_position) map from those tables and short-circuits Yahoo for hits; `max_funds_override=0` means UNCAPPED (attempt every candidate).
- A fund's `is_us_equity` flag for reused-metadata series is recomputed against `_full` holdings (the holdings-based check must run on current data; only the Yahoo-sourced fields are reused).

- [ ] **Step 1:** TDD `derive_relaxed_pool` on synthetic presence frames: (a) a fund with quarters 1–6 consecutive qualifies; (b) 5 consecutive doesn't; (c) 3+3 with a gap doesn't; (d) a fund with all 17 qualifies; (e) a fund whose 6-run ends mid-window (died) qualifies — the dead-fund case.
- [ ] **Step 2:** TDD the metadata-reuse map builder (pure function over synthetic funds frames; later table wins on conflict? NO — assert consistency, first table wins, log conflicts).
- [ ] **Step 3:** Integrate into `run()` behind the new params; existing tests untouched and green (`python -m pytest tests/ -q` → 75 + new).
- [ ] **Step 4:** Stage, scan, commit `"step10: relaxed-pool eligibility, metadata reuse, exhaust mode in ingestion (backward compatible)"`.

---

### Task 2: The `_full` ingestion run (long, network-heavy)

**Files:** none new (runs Task 1's code); Modify: `steps/step10_full_universe/design.md` (pool-size amendment)

- [ ] **Step 1:** Download the 5 new N-PORT quarterly ZIPs (2025q1–2026q1) — verify the URLs resolve; if 2026q1 is not yet published, STOP and report (the design says it should be; reality wins).
- [ ] **Step 2:** Run `run(cfg, table_suffix="_full", relaxed_pool=True, reuse_metadata_from=["funds","funds_oos","funds_oos2"], max_funds_override=0)` in the background with output to `.superpowers/sdd/step10-ingest.log`. NOTHING else touches the DB. Expect: hours-scale (thousands of new Yahoo lookups at 0.3s delay + holdings extraction × 17 quarters).
- [ ] **Step 3:** Record in design.md (`## Pool amendment`): relaxed-pool size, attempted, resolved, equity-flagged, segment split, dead-fund count (funds whose last filing < 2026q1), reuse hit-rate (must be 100% for known series — zero repeat Yahoo hits).
- [ ] **Step 4:** Stage design.md, scan, commit `"step10: _full ingestion complete - pool amendment recorded"`.

---

### Task 3: step7-module generalization (suffix/holdout/prefix params)

**Files:**
- Modify: `steps/step7_unified_universe/panel.py`, `model.py`, `stability.py` (backward-compatible params)
- Test: `tests/test_step10_generalization.py`

**Interfaces:**
- `assemble_unified_panel(cfg, table_suffix="_all")` — replaces the hardcoded `funds_all` etc. with `f"funds{table_suffix}"` (default preserves today).
- `train_and_evaluate(cfg, table_suffix="_all", holdout_transitions=None, output_prefix="unified")` — holdout defaults to `cfg["model"]["test_transitions_holdout"]`; all `save_table`/`save_model` names use `output_prefix` (`unified_panel` → `f"{output_prefix}_panel"`, bundle `f"{output_prefix}_rf_model"`). Defaults reproduce today's names exactly.
- `run_stability(cfg, table_suffix="_all", output_table=None)` — output defaults to `unified_label_stability`.
- ALSO fix here (the step7 final review's deferred batch — same files, right moment): empty-`forward` guard before `predict_proba`; per-quarter win-count denominator counts only comparable quarters; delete stability's dead `own_next` local; hoist panel.py's `import numpy` to module top; add the dedup assert on `quarterly_returns` input in `label.py`.
- [ ] **Step 1:** TDD: signature/default tests (the step3-similarity pattern) + one temp-DB test that `assemble_unified_panel(cfg, "_x")` reads `funds_x` tables (reuse the test_step7_panel fixture with renamed tables).
- [ ] **Step 2:** Implement; full suite green (existing 75+ untouched).
- [ ] **Step 3:** Stage, scan, commit `"step10: generalize step7 modules to table_suffix/output_prefix + deferred hardening batch"`.

---

### Task 4: k re-check + clustering + metrics on `_full`

- [ ] **Step 1:** Scratch k-sweep (scratchpad script, latest quarter, strategy segment of `_full`, k∈{30,40,50}, the step7 §3 metrics). Record the choice + table in design.md (`## k re-check`). If the winner ≠ 30, set `unified.n_clusters` override for the `_full` run via a new config key `full.n_clusters` (else omit).
- [ ] **Step 2:** Run `similarity.run(cfg, table_suffix="_full", n_clusters=<chosen>, top_n_peers=15, require_segment="strategy", save_coords=True)` then `metrics.run(cfg, table_suffix="_full")` — background, log file, DB exclusive. Expect 1.5–2h (17 quarters × ~2× universe).
- [ ] **Step 3:** Stage design.md + `reports/cluster_map_2026q1_full.png`, scan, commit `"step10: _full clustered - k re-check recorded"`.

---

### Task 5: Out-of-time validation module

**Files:**
- Create: `steps/step10_full_universe/__init__.py`, `steps/step10_full_universe/validate.py`
- Test: `tests/test_step10_validate.py`

**Interfaces:**
- `validate.score_published_forward_predictions(cfg) -> dict` — loads `unified_predictions` (split="forward": the 2,086 committed 2024q4→2025q1 predictions), `fund_peers_all` (peers as-of 2024q4), and `_full` quarterly returns; computes each fund's REALIZED 2025q1 label (own realized return < median of its 2024q4 top-10 peers' realized returns, min 5 valid — reuse `compute_peer_labels` machinery where possible); returns `{auc, n_scored, n_missing, base_rate}`. `n_missing` = forward funds without a realized 2025q1 return (died/merged/late) — reported, never imputed.
- `validate.score_frozen_model_rolled_forward(cfg) -> dict` — loads the FROZEN `unified_rf_model` bundle, assembles the `_full` panel (Task 3's generalized function), filters to labeled transitions Q ≥ 2025q1, aligns features to the bundle's `feature_cols` (reindex, fill 0 for missing tier dummies — the step6 `align_features` pattern), returns pooled + per-quarter AUC + row counts.
- Pure-logic parts TDD'd on synthetic frames (realized-label correctness incl. the min-valid-peers and dead-fund cases; feature alignment with a missing tier column).
- [ ] **Step 1:** TDD → implement → suite green.
- [ ] **Step 2:** Stage, scan, commit `"step10: out-of-time validation - published predictions vs reality + frozen model rolled forward"`.

---

### Task 6: Orchestrated model run + validation + UAT + docs + gate

**Files:**
- Create: `steps/step10_full_universe/build.py` (validate → train_and_evaluate(`_full`, holdout=5, prefix="full") → stability(`_full`) — clustering already done in Task 4; `__main__`; not in conductor)
- Modify: design.md (`## UAT results`), `decisions.json`, `workflow.json`, HANDOFF.md (not committed)

- [ ] **Step 1:** `build.py`; run it (background, log, DB exclusive). Validation runs FIRST (frozen-model numbers are committed before the retrained model's numbers exist — no peeking incentive).
- [ ] **Step 2:** UAT per design.md — every number recorded: out-of-time scores (BOTH variants + attrition), retrained pooled/per-quarter AUC + CI vs all baselines (random, persistence, REVERSED persistence), fund-disjoint check, stability, dead-fund counts, k re-check outcome.
- [ ] **Step 3:** decisions.json entry (honest headline: did the committed predictions survive contact with reality?), workflow.json step10 done, render docs, full suite green.
- [ ] **Step 4:** Stage, scan, commit `"step10: full universe run - UAT results recorded"`. **STOP — human approval gate.** step9 (fees on `_full`) follows.
