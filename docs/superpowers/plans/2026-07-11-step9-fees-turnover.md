# step9_fees_turnover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Point-in-time expense ratios + portfolio turnover from the SEC Risk/Return Summary bulk datasets, an expense-rank baseline, a fund-disjoint split check, and a re-evaluation of the unified model — answering whether fees/turnover move the edge past the mean-reversion tie.

**Architecture:** New package `steps/step9_fees_turnover/`: `acquire.py` (download/cache RR ZIPs) → `parse.py` (raw class/filing rows) → `fees.py` (point-in-time as-of join + class→series aggregation → `rr_fees`) → `evaluate.py` (with-fees panel variant, expense-rank baseline, chronological + fund-disjoint evals reusing step7's `fund_clustered_bootstrap`) → `build.py` orchestrator. step7's tables/model are read-only.

**Tech Stack:** requests (existing SEC download pattern from step1), pandas (`merge_asof` for point-in-time), scikit-learn, DuckDB via `fundspeers.io`, pytest.

**Authoritative spec:** `steps/step9_fees_turnover/design.md` (committed). Read it first.

## Global Constraints

- **This is a discovery-first plan.** The RR datasets' exact schema (file names inside the ZIPs, tag names for net/gross expense ratio and turnover, series/class identifier columns, date fields) is UNKNOWN until Task 1 inspects a real quarter. Task 1 produces a schema amendment in design.md; Tasks 2–3 implement against THAT amendment, not against guesses. If the real schema contradicts an interface below, the design amendment governs and the deviation is flagged in the task report.
- Point-in-time rule (exact): for (series_id, Q), use the most recent filing dated ≤ Q's end; never a later filing; every `rr_fees` row keeps `source_filing_date`; assert no future-dated sources.
- Net expense ratio is the model feature; gross stored alongside, never fed to the model.
- Class→series: reuse step1's existing class policy so fees describe the same share class as returns.
- Coverage gate 0.80 (config `fees.coverage_gate`) decides the comparison framing (common-covered-subset comparison vs dual-model with attrition stated).
- Fund-disjoint split: seeded 80/20 fund split (config `fees.fund_disjoint_test_share`: 0.20, `cfg["seed"]`), train = 2022–2023 quarters of the 80%, test = 2024 quarters of the 20%.
- Determinism: everything seeded; downloads cached/idempotent under `data/raw/`; SEC User-Agent from env (step1's pattern).
- step7 outputs are read-only. New tables: `rr_fees_raw`, `rr_fees`, `fees_model_eval`. Model bundle saved only on explicit decision at the gate.
- Config addition (exact): `"fees": {"rr_years": [2021, 2022, 2023, 2024], "coverage_gate": 0.80, "fund_disjoint_test_share": 0.20}`.
- Never open the DuckDB from a second process while a pipeline is writing.
- Every commit: stage explicitly, secret-scan `git diff --cached` first.

---

### Task 1: Acquire + schema discovery

**Files:**
- Create: `steps/step9_fees_turnover/__init__.py`, `steps/step9_fees_turnover/acquire.py`
- Modify: `config.json` (the `fees` section), `steps/step9_fees_turnover/design.md` (schema amendment)
- Test: `tests/test_step9_acquire.py`

**Interfaces:**
- Produces: `acquire.rr_zip_path(year, quarter, cfg) -> Path` (local cache path); `acquire.download_rr_quarter(year, quarter, cfg) -> Path` (skip if cached, SEC User-Agent header, same politeness as step1's downloader — read `steps/step1_ingest/ingest.py`'s download helper and mirror it); `acquire.download_all(cfg) -> list[Path]` for all quarters of `fees.rr_years`.
- The RR dataset landing page: https://www.sec.gov/dera/data/mutual-fund-prospectus-risk-return-summary-data-sets — Task 1 determines the actual ZIP URL pattern from it (expected shape similar to the N-PORT one; verify, don't assume).

- [ ] **Step 1:** Add the `fees` config section (exact values from Global Constraints).
- [ ] **Step 2:** TDD the cache-path/skip logic with a fake tiny ZIP in tmp_path (no network in unit tests): test that `download_rr_quarter` returns the cached path without re-downloading when the file exists (mirror how step1's tests handle this — read them first).
- [ ] **Step 3:** Implement `acquire.py` mirroring step1's download conventions (User-Agent from env, streaming to `data/raw/`, log lines).
- [ ] **Step 4:** REAL download of ONE quarter (e.g. 2024q4) and inspect: `python -c` listing the ZIP's members, then read the first rows of each member file. Record in `design.md` under a new `## Schema amendment (verified YYYY-MM-DD)` heading: file names, delimiter, the exact column names for series id / class id / filing date / period, and the exact tag or column names for net expense ratio, gross expense ratio, portfolio turnover. THIS AMENDMENT IS THE CONTRACT FOR TASKS 2–3.
- [ ] **Step 5:** Real download of all remaining quarters (16 ZIPs total; log sizes).
- [ ] **Step 6:** Full suite green; stage `config.json`, `steps/step9_fees_turnover/__init__.py`, `acquire.py`, `design.md`, `tests/test_step9_acquire.py`; secret-scan; commit `"step9: RR dataset acquisition + verified schema amendment"`.

---

### Task 2: Parse → `rr_fees_raw`

**Files:**
- Create: `steps/step9_fees_turnover/parse.py`
- Test: `tests/test_step9_parse.py`

**Interfaces:**
- Produces: `parse.parse_rr_quarter(zip_path, cfg) -> pd.DataFrame` and `parse.run_parse(cfg)` saving table `rr_fees_raw` with EXACTLY these output columns (mapping from whatever the schema amendment says the source columns are): `series_id, class_id, filing_date, expense_ratio_net, expense_ratio_gross, portfolio_turnover` — one row per (filing, class) where at least one of the three values is present. `filing_date` as ISO string. Numeric values as fractions (0.0075 = 75bps), converting from percent if the source tags are percent-scaled — the schema amendment must state which, from inspected real values.
- Consumes: the Task 1 schema amendment in design.md (the implementer reads it and codes to it).

- [ ] **Step 1:** Write failing tests against a synthetic fixture ZIP built in-test to match the AMENDED schema (small: 2 filings, 3 classes, one missing turnover). Assert: column mapping, fraction scaling, the at-least-one-value row filter, and that a class with all three missing is excluded.
- [ ] **Step 2:** Implement `parse.py` to the amendment. Run tests green.
- [ ] **Step 3:** REAL parse of all cached quarters → `rr_fees_raw`; log row count, distinct series count, and value sanity (median net expense ratio should land in the 0.2%–1.5% ballpark for equity funds — if it doesn't, the scaling assumption is wrong: STOP and fix before committing).
- [ ] **Step 4:** Full suite green; stage; secret-scan; commit `"step9: parse RR filings to rr_fees_raw"`.

---

### Task 3: Point-in-time join + class→series → `rr_fees` + coverage report

**Files:**
- Create: `steps/step9_fees_turnover/fees.py`
- Test: `tests/test_step9_fees.py`

**Interfaces:**
- Produces: `fees.build_rr_fees(cfg) -> dict` (coverage stats) saving `rr_fees`: one row per (series_id, quarter) over the panel's 12 quarters × strategy funds, columns `series_id, quarter, expense_ratio_net, expense_ratio_gross, portfolio_turnover, source_filing_date` (NaN/None where no filing ≤ quarter end exists). Core logic `fees.point_in_time_fees(rr_raw, series_ids, quarters) -> pd.DataFrame` is pure and unit-testable: for each series and quarter, the LAST filing with `filing_date <= quarter_end(quarter)`; quarter_end("2022q1") = "2022-03-31" etc.
- Class→series: select the class per series consistent with step1's policy (read step1's design/code for the exact rule and mirror it; if class-level net assets aren't available in RR data, fall back to: the class with the most non-null fee history, tie-broken by class_id — a DOCUMENTED deviation to flag).
- Consumes: `rr_fees_raw` (Task 2), `funds_all` (strategy series ids + quarters).

- [ ] **Step 1:** Failing tests for `point_in_time_fees` on synthetic frames: (a) picks the latest filing ≤ quarter end, (b) never picks a later filing (a fund whose only filing is 2023-05-01 has NaN for all 2022 quarters and 2023q1), (c) `source_filing_date <= quarter_end` asserted for every non-null row, (d) carries values forward across quarters until superseded.
- [ ] **Step 2:** Implement (`pd.merge_asof` on sorted frames or an explicit per-series loop — correctness over cleverness), tests green.
- [ ] **Step 3:** REAL run → `rr_fees`; print the coverage report (per design §4): % of 2,087 strategy funds with any fees, % of the 20,838 labeled panel rows covered; state which coverage branch applies. Record the numbers in the task report AND design.md's UAT-results-to-be.
- [ ] **Step 4:** Full suite green; stage; secret-scan; commit `"step9: point-in-time rr_fees with coverage report"`.

---

### Task 4: Evaluation — with-fees model, expense-rank baseline, fund-disjoint splits

**Files:**
- Create: `steps/step9_fees_turnover/evaluate.py`
- Test: `tests/test_step9_evaluate.py`

**Interfaces:**
- Produces:
  - `evaluate.fund_disjoint_split(panel, quarters_ordered, holdout_transitions, test_share, seed) -> (train, test)`: seeded `rng.choice` of `floor(test_share * n_funds)` series_ids as the test-fund set; train = train-quarter rows of NON-test funds; test = test-quarter rows of test funds. Pure, unit-tested.
  - `evaluate.run_evaluation(cfg) -> dict` — loads `unified_panel` + `rr_fees`, builds the with-fees panel variant (join fees, drop rows missing them — variant only), restricts BOTH variants to the common covered subset for the apples-to-apples comparison, then evaluates: {unified, unified+fees} × {chronological split, fund-disjoint split}, plus baselines on the chronological test rows: random 0.5, persistence (`-return_vs_peer_median_q`) with its reversed reading, expense-rank (`expense_ratio_net` as the score). Pooled AUC + `fund_clustered_bootstrap` CI for the model-vs-expense-rank edge; per-quarter AUCs. Saves `fees_model_eval` (long format `metric, quarter, value, variant`).
- Consumes: `unified_panel` (step7; includes labeled rows with feature columns + `underperform_next_quarter`), `steps.step7_unified_universe.model.fund_clustered_bootstrap` (reuse, do not copy), `time_based_split` (step4), config `model.rf`, `fees.*`, `seed`.

- [ ] **Step 1:** Failing tests: (a) `fund_disjoint_split` — no series_id overlap between train and test, test rows only from test quarters, deterministic under seed, approximate test-share honored; (b) expense-rank baseline scoring — on a tiny synthetic frame where high-fee funds are exactly the underperformers, AUC = 1.0, and inverted labels give 0.0.
- [ ] **Step 2:** Implement `evaluate.py`; tests green.
- [ ] **Step 3:** Full suite green; stage; secret-scan; commit `"step9: with-fees evaluation, expense-rank baseline, fund-disjoint splits"`.

---

### Task 5: Orchestrator, real run, UAT, docs, gate

**Files:**
- Create: `steps/step9_fees_turnover/build.py` (acquire→parse→fees→evaluate; `__main__`; NOT in conductor)
- Modify: `steps/step9_fees_turnover/design.md` (`## UAT results`), `decisions.json`, `workflow.json` (step9 entry), HANDOFF.md (not committed)

- [ ] **Step 1:** `build.py` (same shape as step7's build.py — plain sequential orchestrator with phase log lines).
- [ ] **Step 2:** REAL full run. Nothing else touches the DuckDB while it runs. Output to a log file for monitoring.
- [ ] **Step 3:** UAT per design.md — record every number: coverage %s and branch, expense-rank baseline AUC + CI, unified vs unified+fees on common subset (pooled + per-quarter + CI), fund-disjoint AUCs for both, feature importances (where do fees rank?), no-future-filings assertion result.
- [ ] **Step 4:** decisions.json entry with the honest headline (does anything beat reversed persistence ~0.715 now?); `## UAT results` in design.md; workflow.json step9 done; render docs; full suite green.
- [ ] **Step 5:** Stage docs + code; secret-scan; commit `"step9: fees & turnover evaluated - UAT results recorded"`. **STOP — human approval gate.** The saved-model decision and the README rewrite (with these final numbers) both happen after the gate.
