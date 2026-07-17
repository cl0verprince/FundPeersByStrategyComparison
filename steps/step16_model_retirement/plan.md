# step16_model_retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire the lag-probability model publicly and falsifiably: config-driven retired state across chip/fund pages//model/static dashboard/README; refresh switches to score-frozen-only; a growing "since retirement" public record.

**Architecture:** One source of truth — `config.json` `model.retirement` — read by the extract builder (which stamps the health singleton, empties the live forward book, and builds the since-retirement view), by advance.py (score-frozen-only path), and by the static-dashboard payload. The webapp renders purely from the extract, so retired rendering follows automatically from the health state plus one new tile component.

**Tech Stack:** unchanged (Python, DuckDB, NiceGUI, pytest + nicegui.testing User fixture).

## Global Constraints (from steps/step16_model_retirement/design.md — binding)

- Retirement read ONLY from `cfg["model"]["retirement"]` (pipeline/extract/static dash) or from `v_model_health_current` (webapp). Absent key = fully non-retired current behavior (regression-tested).
- When retired: NO live probabilities anywhere — `v_fund_prediction_current` built EMPTY; fund Zone A renders the retirement tile (never a probability, never a grayed meter); /model has no forward book.
- Retired chip: `STATUS["retired"] = ("✕", "Signal retired", "muted")` — neutral fact treatment, never critical red; icon + label + color as always; links /model.
- The falsifiable record: `v_model_retirement_record` (quarter, auc, n_rows) = frozen_rolled_forward per-quarter oot_validation rows with quarter > as_of; built unconditionally (empty when not retired); /model "Since retirement" section renders it with the designed empty-state text.
- Prediction HISTORY stays everywhere (the misses table is the permanent record). Nothing is deleted — retirement is additive honesty, not erasure.
- Refresh retired path: `full_build.run_retired(cfg)` = segment repair + frozen scoring (recreate oot_validation idempotently via the existing `_write_oot_validation(published, frozen, cfg)`) + `run_stability`; NO `train_and_evaluate`, NO `fund_disjoint_auc`, NO `fees_evaluate.run_evaluation`.
- The retirement statement text is EXACTLY the one in design.md's config block (copy verbatim into config.json).
- Colors from TOKENS only; all existing step14 honesty constraints remain binding.
- `compute_health_state` stays pure/rule-based; the override lives in `build_model_views`.

## File Structure

```
config.json                          modify: model.retirement block
steps/step14_webapp/extract.py       modify: build_model_views retirement handling + v_model_retirement_record
webapp/theme.py                      modify: STATUS "retired"
webapp/components/honesty.py         modify: add retirement_card()
webapp/pages/fund.py                 modify: Zone A retired branch
webapp/pages/model.py                modify: retired verdict, Since-retirement section, forward-book replacement
steps/step10_full_universe/build.py  modify: add run_retired()
steps/step13_automation/advance.py   modify: _stage_evaluate retired branch
steps/step8_dashboard/data.py        modify: payload carries retirement
steps/step8_dashboard/template.py    modify: oot panel retirement banner
README.md                            modify: the arc's ending
tests/step14_fixtures.py             modify: build_synthetic_extract(retired=False) param
tests/test_step14_extract.py         append retirement-view tests
tests/test_step14_app.py             append retired app-smoke tests (second fixture module scope — see Task 3)
tests/test_step13_advance.py         append retired-path stage test
tests/test_step10_build.py           append run_retired composition test
```

---

### Task 1: Extract + config — retirement state, empty forward book, retirement record view

**Files:**
- Modify: `config.json`, `steps/step14_webapp/extract.py`
- Test: `tests/test_step14_extract.py` (append), `tests/step14_fixtures.py` (modify)

**Interfaces:**
- Consumes: existing `build_model_views(src, cfg)` (cfg currently used only for prediction intervals; None in unit tests), `latest_quarter`, existing views.
- Produces (binding for Tasks 2–3):
  - `v_model_health_current` gains columns `retired_as_of` (str|NULL), `retirement_statement` (str|NULL); when `cfg` has `model.retirement`: `health_state="retired"`, `rule_text=statement`.
  - `v_fund_prediction_current`: EMPTY (same columns) when retired.
  - `v_model_retirement_record` (quarter, auc, n_rows): frozen per-quarter rows with quarter > as_of; empty when not retired. NOTE: oot_validation's frozen per-quarter rows have no n_rows per quarter — build with `n_rows` NULL; the record's auc + quarter are the substance.
  - `tests/step14_fixtures.py: build_synthetic_extract(out_path, retired=False)` — when True, passes a minimal retired cfg `{"model": {"retirement": {"as_of": "2026q1", "statement": "Retired for the synthetic record."}}}` into `build_model_views` (interval lookup fails harmlessly → NULL intervals, existing behavior).

- [ ] **Step 1: Write the failing tests (append to tests/test_step14_extract.py)**

```python
RETIRED_CFG = {"model": {"retirement": {
    "as_of": "2026q1", "statement": "Retired for the synthetic record."}}}


def test_retired_health_state_and_statement(model_src):
    from steps.step14_webapp.extract import build_model_views
    views = build_model_views(model_src, cfg=RETIRED_CFG)
    cur = views["v_model_health_current"].iloc[0]
    assert cur["health_state"] == "retired"
    assert cur["retired_as_of"] == "2026q1"
    assert cur["rule_text"] == "Retired for the synthetic record."
    assert cur["retirement_statement"] == "Retired for the synthetic record."


def test_retired_empties_live_forward_book(model_src):
    from steps.step14_webapp.extract import build_model_views
    views = build_model_views(model_src, cfg=RETIRED_CFG)
    pred = views["v_fund_prediction_current"]
    assert len(pred) == 0
    assert "predicted_probability" in pred.columns  # schema stable


def test_retirement_record_only_after_as_of(model_src):
    from steps.step14_webapp.extract import build_model_views
    views = build_model_views(model_src, cfg=RETIRED_CFG)
    rec = views["v_model_retirement_record"]
    # model_src fixture has one frozen per-quarter row at 2026q1 (== as_of, excluded)
    assert list(rec.columns) == ["quarter", "auc", "n_rows"]
    assert len(rec) == 0


def test_not_retired_is_regression_free(model_src):
    from steps.step14_webapp.extract import build_model_views
    views = build_model_views(model_src, cfg=None)
    cur = views["v_model_health_current"].iloc[0]
    assert cur["health_state"] in ("healthy", "weak", "degraded")
    assert pd.isna(cur["retired_as_of"])
    assert len(views["v_fund_prediction_current"]) > 0
    assert len(views["v_model_retirement_record"]) == 0
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_step14_extract.py -q` → new tests FAIL (missing columns/view).

- [ ] **Step 3: Implement in `steps/step14_webapp/extract.py`**

In `build_model_views(src, cfg)`:

```python
    retirement = (cfg or {}).get("model", {}).get("retirement") if isinstance(cfg, dict) else None
```
(place near the top; `cfg` may be a real config dict or None). Then:
- After the existing forward/interval assembly: `if retirement: v_fund_prediction_current = v_fund_prediction_current.iloc[0:0]` (schema-stable empty).
- After the existing `v_model_health_current` DataFrame is built, add the two new columns:
```python
    v_model_health_current["retired_as_of"] = retirement["as_of"] if retirement else pd.NA
    v_model_health_current["retirement_statement"] = retirement["statement"] if retirement else pd.NA
    if retirement:
        v_model_health_current.loc[0, "health_state"] = "retired"
        v_model_health_current.loc[0, "rule_text"] = retirement["statement"]
```
- New view (always built):
```python
    frozen_q = src.execute(
        "SELECT quarter, value AS auc FROM oot_validation "
        "WHERE metric = 'auc' AND quarter <> '' AND source = 'frozen_rolled_forward' "
        "ORDER BY quarter").df()
    if retirement:
        rec = frozen_q[frozen_q["quarter"] > retirement["as_of"]].copy()
    else:
        rec = frozen_q.iloc[0:0].copy()
    rec["n_rows"] = pd.NA
    v_model_retirement_record = rec[["quarter", "auc", "n_rows"]]
```
Add it to the returned dict. Update `tests/step14_fixtures.py`'s `build_synthetic_extract` with the `retired=False` param per the Interfaces block.

Also `config.json`: add the `retirement` block (verbatim statement from design.md) inside the existing `"model"` object. NOTE: adding it now flips the REAL pipeline/extract to retired — that is the point of this step; later tasks make every surface render it.

- [ ] **Step 4: Run** — `python -m pytest tests/test_step14_extract.py tests/test_step14_app.py -q` → extract tests PASS; app tests must still pass (they build with retired=False default).

- [ ] **Step 5: Commit** — `step16: retirement state in config + extract (retired health, empty forward book, since-retirement record view)`

---

### Task 2: Chip + retirement tile (theme.py, honesty.py)

**Files:**
- Modify: `webapp/theme.py`, `webapp/components/honesty.py`
- Test: `tests/test_step14_charts.py` (append — pure checks)

**Interfaces:**
- Produces: `STATUS["retired"] = ("✕", "Signal retired", "muted")`; `honesty.retirement_card(health: dict, mode: str = "light")` — card with "Model retired ({retired_as_of})" heading, the statement (ink2), link "The full record →" to `/model`, `status_chip(health, mode)` inside. No probability, no meter.

- [ ] **Step 1: Failing tests (append to tests/test_step14_charts.py)**

```python
def test_status_has_retired_state():
    from webapp.theme import STATUS, TOKENS
    icon, label, token = STATUS["retired"]
    assert (icon, label) == ("✕", "Signal retired")
    assert token in TOKENS["light"] and token not in ("critical", "warning", "serious", "good")
```

- [ ] **Step 2: RED** — `python -m pytest tests/test_step14_charts.py -q` → KeyError.

- [ ] **Step 3: Implement**

theme.py: add the entry to `STATUS`. honesty.py:

```python
def retirement_card(health: dict, mode: str = "light") -> None:
    """The fund-page Zone A tile when the model is retired: a fact, not an alarm.
    Never renders a probability or meter (design: no live probabilities anywhere)."""
    t = TOKENS[mode]
    with ui.card().classes("w-full max-w-sm p-4"):
        ui.label(f"Model retired ({health.get('retired_as_of', '')})").classes(
            "text-sm font-semibold")
        ui.label(str(health.get("retirement_statement") or "")).classes(
            "text-sm").style(f"color:{t['ink2']}")
        ui.link("The full record →", "/model").classes("text-sm")
        status_chip(health, mode)
```

- [ ] **Step 4: GREEN** — charts tests pass; full suite green.
- [ ] **Step 5: Commit** — `step16: retired status state + retirement_card tile`

---

### Task 3: Fund + /model retired rendering, retired app-smoke tests

**Files:**
- Modify: `webapp/pages/fund.py`, `webapp/pages/model.py`
- Test: new `tests/test_step16_app_retired.py`

**Interfaces:**
- Consumes: `retirement_card`, `v_model_retirement_record` via a new `ExtractStore.retirement_record() -> pd.DataFrame` (add to `webapp/data.py`: `SELECT * FROM v_model_retirement_record ORDER BY quarter`).
- Fund page: in `render_fund`, where Zone A currently does `if is_active: probability_card(...)` — the retired branch takes precedence for ALL funds: `if health["health_state"] == "retired": honesty.retirement_card(health)` (dead funds keep their archive banner above it; the banner's "No forward prediction exists" line remains true).
- /model page: when retired — verdict card renders "✕ RETIRED as of {retired_as_of}" using muted/ink treatment (NOT `t[token]` red path; reuse the STATUS token which is "muted") + the statement; directly below it the **Since retirement** section: if `retirement_record` empty → the exact empty-state text from design.md; else a `ui.table` of the rows. The forward-book section is replaced by `ui.label("No live forward book — the model is retired; no new predictions are generated.")`. The open-question sentence is replaced by "Resolved 2026-07-17: retired." Historical evidence sections all remain.

- [ ] **Step 1: Failing tests — new file `tests/test_step16_app_retired.py`** (module-scoped retired extract; mirrors test_step14_app.py's pattern)

```python
"""Retired-state app smoke tests against a RETIRED synthetic extract."""
import os
import pytest
from nicegui.testing import User

pytest_plugins = ["nicegui.testing.user_plugin"]


@pytest.fixture(scope="module", autouse=True)
def retired_extract_env(tmp_path_factory):
    from tests.step14_fixtures import build_synthetic_extract
    path = build_synthetic_extract(
        tmp_path_factory.mktemp("retired") / "extract.duckdb", retired=True)
    os.environ["EXTRACT_PATH"] = str(path)
    # get_store() is lru_cached per process - clear it so this module's env var wins.
    import webapp.main
    webapp.main.get_store.cache_clear()
    yield
    os.environ.pop("EXTRACT_PATH", None)
    webapp.main.get_store.cache_clear()


async def test_fund_page_shows_retirement_not_probability(user: User):
    await user.open("/fund/AAAAX")
    await user.should_see("Model retired")
    await user.should_see("Signal retired")
    await user.should_see("Retired for the synthetic record.")


async def test_model_page_retired_verdict_and_empty_record(user: User):
    await user.open("/model")
    await user.should_see("RETIRED as of")
    await user.should_see("First post-retirement score expected")
    await user.should_see("No live forward book")
```

(If `cache_clear` ordering vs test_step14_app.py's module fixture proves flaky under one pytest run, the sanctioned fallback is to run this file in its own pytest invocation via a marker — document which was needed.)

- [ ] **Step 2: RED** — new file fails (no retirement rendering yet).
- [ ] **Step 3: Implement** fund.py + model.py + `ExtractStore.retirement_record()` per Interfaces.
- [ ] **Step 4: GREEN** — `python -m pytest tests/test_step16_app_retired.py tests/test_step14_app.py -q` then full suite.
- [ ] **Step 5: Commit** — `step16: retired rendering - fund tile, /model retired verdict + since-retirement record, no forward book`

---

### Task 4: Refresh retired path (run_retired + advance branch)

**Files:**
- Modify: `steps/step10_full_universe/build.py`, `steps/step13_automation/advance.py`
- Test: `tests/test_step10_build.py` (append), `tests/test_step13_advance.py` (append)

**Interfaces:**
- `full_build.run_retired(cfg) -> None`:
```python
def run_retired(cfg: dict) -> None:
    """Retired-model refresh (design step16): keep the falsifiable record growing -
    segment repair, frozen out-of-time scoring, label-stability - and nothing that
    trains or emits new predictions."""
    log.info("=== step10 (RETIRED): pipeline repair (funds_full.segment) ===")
    ensure_funds_full_segment(cfg)
    log.info("=== step10 (RETIRED): out-of-time scoring of the frozen model ===")
    published = score_published_forward_predictions(cfg)
    frozen = score_frozen_model_rolled_forward(cfg)
    _write_oot_validation(published, frozen, cfg)
    log.info("=== step10 (RETIRED): label-stability study ===")
    run_stability(cfg, table_suffix="_full", output_table="full_label_stability")
    log.info("frozen record updated; no retraining, no new forward predictions (model retired)")
```
- `advance._stage_evaluate(cfg)` becomes:
```python
def _stage_evaluate(cfg: dict) -> dict:
    if cfg.get("model", {}).get("retirement"):
        log.info("model retired (%s) - scoring frozen only, no retrain/fees eval",
                 cfg["model"]["retirement"]["as_of"])
        full_build.run_retired(cfg)
        return {"retired": True}
    full_build.run(cfg)
    return fees_evaluate.run_evaluation(cfg)
```

- [ ] **Step 1: Failing tests** — test_step10_build.py: stub `ensure_funds_full_segment`, `score_published_forward_predictions`, `score_frozen_model_rolled_forward`, `_write_oot_validation`, `run_stability` via monkeypatch recording calls; assert run_retired calls all five and never touches `train_and_evaluate`/`fund_disjoint_auc` (monkeypatch those to raise). test_step13_advance.py: with a retired cfg, `_stage_evaluate` calls `full_build.run_retired` (stub) and NOT `fees_evaluate.run_evaluation` (stub to raise); non-retired cfg keeps current path (existing tests already cover — verify none breaks).
- [ ] **Step 2: RED** → **Step 3: Implement** → **Step 4: GREEN + full suite** → **Step 5: Commit** — `step16: refresh retired path - score frozen only, stability kept, no retrain/fees eval`

---

### Task 5: Static dashboard banner + README + real rebuilds + UAT prep

**Files:**
- Modify: `steps/step8_dashboard/data.py` (payload gains `scorecard.retirement` from cfg), `steps/step8_dashboard/template.py` (oot panel: when `s.retirement` set, render a banner div first: "✕ MODEL RETIRED as of {as_of}" + statement, styled with existing `.oot` vars, muted not critical), `README.md` (the honest-arc ending section)
- Test: `tests/test_step8_data.py` (append: payload carries retirement when cfg has it; absent otherwise)

- [ ] **Step 1: Failing payload test** → **Step 2: RED** → **Step 3: Implement data.py/template.py** → **Step 4: README section** (content: the arc 0.717 → 0.574 → 0.457/0.427 inverted; regime diagnosis; retirement statement; the standing falsifiable record and where to watch it) → **Step 5: GREEN + full suite**
- [ ] **Step 6: Real rebuilds** — `python -m steps.step14_webapp.extract` (retired extract; verify v_model_health_current shows retired) and `python -m steps.step8_dashboard.build --table-suffix _full --prefix full` equivalent (use the exact invocation advance.py's `_stage_dashboard` uses) to rebuild `reports/cluster_dashboard.html` with the banner. Report both outputs.
- [ ] **Step 7: Commit** — `step16: static dashboard retirement banner + README ending; real extract and dashboard rebuilt retired`

(Controller then drives live UAT + docs/gate per the design's UAT list.)

## Self-Review (performed)
1. **Spec coverage:** config block (T1), extract state/empty book/record view (T1), chip+tile (T2), fund+/model rendering incl. empty-state text and forward-book replacement (T3), refresh score-frozen-only (T4), static banner + README (T5), UAT live drive = controller's wrap-up. Nothing in design.md unassigned.
2. **Placeholder scan:** none; all new logic has code, page edits have precise behavioral anchors to named functions.
3. **Type consistency:** `retirement_card(health, mode)` matches Task 3 usage; `retirement_record()` naming consistent; `run_retired` imports already exist in build.py's namespace (`run_stability` imported at module top there — verify; if not, add the import).
