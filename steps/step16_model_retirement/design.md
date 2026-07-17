# step16_model_retirement — design

## Purpose
Retire the lag-probability model **publicly and falsifiably**, as the honest ending of the
project's arc: backtest AUC 0.717 → committed-forward reality 0.574 → two consecutive
realized quarters below coin-flip (0.457 on 2025q4→2026q1, 0.427 on 2026q1→2026q2) while
the persistence baseline flipped positive. Diagnosis, stated publicly: the model had
mostly learned mean-reversion; the market regime flipped to momentum in late 2025,
inverting the signal.

Approved decisions (2026-07-17): falsifiable retirement (keep scoring the frozen model
every quarter and publishing the result); fund pages replace the outlook card with a
retirement notice (prediction history stays); the refresh scores-frozen-only (no
retraining, no new forward predictions, no with-fees re-eval); the statement reaches the
webapp (/model + chip), the README, and the static dashboard.

## One source of truth
`config.json` gains:
```json
"model": { ...existing keys...,
  "retirement": {
    "as_of": "2026q2",
    "statement": "Retired after two consecutive realized quarters below the 0.5 coin-flip (0.457, 0.427) while the mean-reversion baseline inverted. The model had largely learned mean-reversion; when the market regime flipped to momentum in late 2025 the signal inverted. The frozen model continues to be scored against reality each quarter below - if that record shows sustained recovery, retirement will be revisited."
  }
}
```
Absent key = not retired (all existing behavior). Every consumer reads retirement ONLY
from config (pipeline) or from the extract's health singleton (webapp).

## Changes by component

### extract.py (steps/step14_webapp)
- `build_model_views(src, cfg)`: when `cfg["model"]["retirement"]` exists —
  `v_model_health_current.health_state = "retired"`, plus new columns `retired_as_of`,
  `retirement_statement`; `rule_text` = the statement. (Non-retired: new columns NULL.)
- `v_fund_prediction_current` is built EMPTY when retired (no live probabilities in the
  extract at all). `v_fund_prediction_history` unchanged — the misses table is the record.
- New view `v_model_retirement_record` (quarter, auc, n_rows): frozen-model out-of-time
  scores for label quarters AFTER `as_of`, from `oot_validation`
  `source='frozen_rolled_forward'` per-quarter rows with quarter > as_of. Empty initially.
  Built unconditionally (empty when not retired) so the schema is stable.
- `compute_health_state` itself stays pure/rule-based (unchanged); the retirement override
  happens in `build_model_views` where cfg is visible.

### theme.py / honesty.py (webapp)
- `STATUS` gains `"retired": ("✕", "Signal retired", "muted")` — neutral ink, a fact not
  an alarm; chip otherwise identical (icon + label + color, links /model).
- New `honesty.retirement_card(health, mode="light")`: the fund-page tile — "Model
  retired ({retired_as_of})", one-sentence why, link "The full record →" to /model,
  status chip inside. No probability, no meter, ever.
- `probability_card` unchanged (it simply is not called when retired).

### webapp/pages/fund.py
- When `health["health_state"] == "retired"`: Zone A renders `retirement_card` instead of
  `probability_card` for ALL funds (live or dead; dead keeps its archive banner too).
- Prediction-history table and everything else unchanged.

### webapp/pages/model.py
- Verdict card in the retired state: "✕ RETIRED as of {as_of}" (muted/ink treatment, not
  critical red) + the statement.
- New section directly under the verdict: **"Since retirement"** — table/strip from
  `v_model_retirement_record`; empty state text: "First post-retirement score expected
  ~Nov 2026 (when the next N-PORT data set publishes). The frozen model's predictions are
  scored against reality every quarter; this record is what would justify — or refute —
  this retirement."
- Forward-book section: replaced when retired by one line ("No live forward book — the
  model is retired; no new predictions are generated."). All historical evidence sections
  (AUC by quarter, dumbbell, calibration, baselines, noise floor) remain — they ARE the
  case for retirement. The published open-question sentence is replaced by the resolution
  ("Resolved 2026-07-17: retired.").

### advance.py (steps/step13_automation) + step10 build
- New `full_build.run_retired(cfg)`: pipeline repair (`ensure_funds_full_segment`) +
  `score_frozen_model_rolled_forward` + `_write_oot_validation`-equivalent append of the
  new quarter's frozen score (published-forward score is fixed history; recompute-idempotent
  as today) + `run_stability` (the label-noise record is model-independent and continues).
  NO `train_and_evaluate`, NO `fund_disjoint_auc`.
- `advance._stage_evaluate(cfg)`: if retirement set → `full_build.run_retired(cfg)` and
  SKIP `fees_evaluate.run_evaluation` entirely; else current behavior. Stage log line says
  which path ran.
- Net effect each future quarter: the falsifiable record grows by one row; nothing else
  about the refresh changes (ingest/cluster/metrics/fees-data/dashboards/extract all run).

### Static dashboard (steps/step8_dashboard)
- Build reads `cfg["model"]["retirement"]`; when set, the scorecard section renders a
  retirement banner (as_of + statement, linking nothing — it's self-contained HTML) above
  the existing reality-first numbers. Narratives/clusters unchanged.

### README
- The honest-arc section gets its ending: what was tried, what reality said (the numbers
  above), the regime diagnosis, the retirement, and the standing falsifiable record.

## What retirement does NOT change
Clusters, peers, metrics, fees data, narratives, search, the static dashboard's cluster
content, step15's scope (explorer/home — still pending), the quarterly probe routine.

## Tests / UAT (gate)
- Unit: build_model_views with a retired cfg → health_state "retired", statement carried,
  empty v_fund_prediction_current, v_model_retirement_record schema present;
  non-retired cfg → all current behavior (regression).
- Unit: advance retired path calls run_retired and skips fees evaluation (stub test in
  test_step13_advance.py style); run_retired composes the right calls (stubbed).
- App smoke (retired synthetic extract): fund page shows "Model retired" + NO probability
  + history intact; /model shows "RETIRED as of", the since-retirement empty state, and no
  forward book; chip "Signal retired" on /, /fund/*, /model.
- App smoke (non-retired synthetic extract): existing 7 tests still pass unchanged.
- Live UAT: rebuild real extract with retirement set; drive /fund/VTSMX + /model locally;
  static dashboard rebuilt with banner; README section present; full suite green.

## Out of scope
Regime-gated or retrained revival models; automated un-retirement (the statement names
the reopening condition in prose only); removing historical tables/models (nothing is
deleted — retirement is additive honesty, not erasure).
