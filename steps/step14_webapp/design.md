# step14_webapp — design (part 1 of 2; step15 completes the app)

## Purpose
A public, hosted, dynamic dashboard for financial analysts, retail investors, and
data-curious enthusiasts, on top of the pipeline's tables — the interactive complement to
the static `reports/cluster_dashboard.html` (which stays; the refresh keeps rebuilding it).

Designed by two expert passes (2026-07-16): a financial-data content spec and a UI/UX
design governed by the dataviz skill (palettes machine-validated). This document is the
synthesis and the contract.

## The one design axiom
**The differentiator is not the model — it's the honesty.** The model's live health is a
global condition: a three-state chip (Healthy / Weak / Degraded) computed from one
precomputed singleton view, rendered in the header and inside every card that shows a
prediction. No code path can show a probability without its health state. Today the app
ships in the **Degraded** state (last two realized quarters AUC 0.428, 0.418 — below
coin-flip) and is designed to look right in that state first.

## Scope split
- **step14 (this step):** extract builder + NiceGUI app shell (header, omnibox search,
  status chip, freshness stamp, disclaimer, dark/light) + **Fund page** + **Model health
  page** + **Methodology page** + Dockerfile + deployed Hugging Face Space.
- **step15 (next step):** Cluster explorer (`/clusters`), cluster detail (`/cluster/{id}`),
  Home (`/`). Until then `/` renders the search hero + model-status card only, and cluster
  links on the fund page point to an "arriving in the next release" stub with the cluster's
  narrative text inline.

## Architecture
```
[local 5.6GB DuckDB] --steps/step14_webapp/extract.py--> [webapp/data/extract.duckdb ~10MB]
                                                              |
                                   [webapp/ NiceGUI app, read-only] --Docker--> HF Space
```
- **`webapp/`** (top-level, the deployable unit): `main.py` (app + routing), `pages/`
  (fund.py, model.py, methodology.py; step15 adds clusters.py, home.py), `components/`
  (status_chip.py, probability_tile.py, freshness.py, search.py, charts.py = mode-aware
  echarts option builders), `data.py` (read-only DuckDB access + in-memory search index),
  `theme.py` (validated palette tokens), `Dockerfile`, `requirements.txt`,
  `data/extract.duckdb` (built artifact, gitignored in the main repo; pushed to the Space).
- **`steps/step14_webapp/extract.py`**: builds the extract from the `_full` tables as
  ~14 precomputed views (below), single deterministic entry point
  (`python -m steps.step14_webapp.extract`). Wired into `advance.py` as part of the
  dashboard stage so every quarterly refresh rebuilds it. Deploy to the Space is a separate
  explicit command (`--deploy`), never automatic — same human-gate philosophy as `--push`.
- **Hosting:** Hugging Face Space, Docker SDK, port 7860, public. No secrets (all data is
  public SEC-derived). Quarterly update = rebuild extract + `hf upload` the Space.

## Extract data contract (precomputed views; grain → key columns)
From the financial expert's spec — names are binding for implementation:
- `v_fund_header` (series_id): identity, ticker, yahoo_category, segment, cluster_id/name,
  net_assets, first/last quarter, is_active
- `v_fund_search` (series_id): ticker, name, name_normalized, cluster info, aum, is_active
- `v_fund_peer_relative_ts` (series_id × quarter): quarterly_return, cluster_median_return,
  return_vs_cluster_median, cluster_size, pctile_return_in_cluster
- `v_fund_cluster_percentiles` (series_id @ latest): pctile of vol/sharpe/maxdd/fees/turnover
  within cluster; **percentiles suppressed where cluster_size < 15** (noise as precision)
- `v_fund_prediction_current` (series_id): predicted_probability, target_quarter, flip_rate,
  health_state_at_publish
- `v_fund_prediction_history` (series_id × quarter): predicted vs actual — the per-fund
  "our misses" table
- `v_peer_display` (series_id × rank @ latest): top-15 peers with similarity, category,
  trailing return, fees
- `v_cluster_summary`, `v_cluster_return_dispersion`, `v_cluster_map` (step15 consumers;
  built now so the extract format is stable)
- `v_model_health_quarters` (quarter × source): auc, persistence_auc, n_scored, from
  full_model_eval + oot_validation, source-labeled
- `v_model_health_current` (singleton): health_state, rule_text, last_two_aucs,
  pooled_live_auc, backtest_auc, base_rate, label_noise_floor, refreshed_at
- `v_calibration_bins` (prob_bin): predicted_mean vs actual_lag_rate
- `v_data_provenance` (singleton): last_quarter, refreshed_at, next_expected_publication
- `v_top_holdings` (series_id/cluster × rank @ latest): top-10 holdings

Measured: full extract ≈ 10 MB parquet/duckdb (core tables 8.8 MB + holdings 1.3 MB).

## Health-state rule (disclosed on /model, computed at extract build)
- **Healthy**: last 2 realized quarters' AUC ≥ best baseline that quarter
- **Weak**: above 0.5 but below the persistence baseline in either
- **Degraded**: either of the last 2 realized quarters < 0.5

## Pages (step14)
### /fund/{ticker}
Zones, top to bottom: **A** next-quarter outlook — probability as stat tile + sequential-
blue meter (never red/green), whole percents, "±N pts" bootstrap CI, fixed sentence
("chance this fund falls below its peers' median return next quarter"), label-noise
footnote, status chip fused in-card. **B** KPI row — return/vol/Sharpe/maxDD/fees as stat
tiles, each value + delta vs peer median (the one place semantic red/green is earned;
direction-aware: lower fees = good) + 18-quarter sparkline. **C** fund vs peer median,
18 quarters — emphasis-form line (fund slot-1 blue, median gray, end-labels), diverging
fund−median bar strip beneath on the same x-axis, metric toggle above, `connectNulls:false`
(gaps are data), chart⇄table twin. **D** top-15 peers aggrid — similarity in-cell bar,
default sort by similarity (never by return), real `<a>` links. Prediction history table
(the misses) below. Header: identity + cluster link + freshness stamp.

### /model
Hero verdict card (status color + icon + words, rule shown). Realized AUC per quarter vs
baselines: line chart, model emphasized, labeled solid 0.5 "coin flip" hairline, below-0.5
model points flagged in critical status color, pending quarter as open circle.
Backtest-vs-reality dumbbell (0.717 → 0.614 → 0.574 arc). Plain-English AUC explainer
inline ("pick one lagger and one non-lagger; the model ranked the lagger higher 57/100
times"). Calibration panel from v_calibration_bins with base_rate marked. Baselines table
(random/persistence/reversed/expense-rank/fund-disjoint/label-noise floor). Forward book:
histogram of 2,970 live probabilities + promised scoring date. Published retirement
criterion: "two consecutive below-chance quarters is under investigation; the signal may
be retired." Chart⇄table twins throughout.

### /methodology
Data source (SEC N-PORT/RR), what clustering means, label definition, model limitations,
survivorship statement (dead funds included; forward attrition counted), Yahoo third-party
note, full disclaimer text.

### Search (global omnibox)
Server-side scorer over an in-memory index (3,295 rows loaded at startup). Tiers: exact
ticker → ticker prefix → name-token prefix → trigram fuzzy; within tier by AUM desc. Dead
funds included, demoted, tagged "left universe YYYY qQ". Ctrl+K / `/` focus; debounce
120 ms; ≤8 rows; empty/no-match states designed (browse-clusters escape hatch). Never
innerHTML (SEC-sourced names).

## Honesty pattern components (built once, reused)
`status_chip()` (three states, icon + label + color, links /model), `uncertain_probability()`
(the only legal probability rendering), `freshness_stamp()` (as-of quarter · ~60-day filing
lag · extract build date; every page, same position), disclaimer (persistent header
one-liner "Educational analytics — not investment advice" + contextual sentence inside the
prediction card + full text on /methodology; no modal).

## Visual system (binding tokens in theme.py)
Validated palettes: 5-slot style-family categorical for step15's map (light PASS ΔE 13.3;
dark floor-band legal with label+tooltip relief); sequential blue ramp for
probabilities/similarity; blue↔red diverging for fund−median; status tokens (good/warning/
serious/critical) only for realized deltas and model health — never on predictions.
System sans; tabular-nums only in grids/tickers/axes. Hairline-border cards, no shadows;
4 px grid; max width 1440 px. `ui.dark_mode` toggle, per-mode chart tokens (dark is a
selected palette, not inversion). Responsive: desktop-first, stacked <1024 px, phone-legible.

## Edge states (designed, not error'd)
Missing quarters → line gaps + quiet caption. Dead fund → archive banner, outlook zone
replaced by explanation ("no forward prediction exists"), history browsable. Prediction
unavailable → reason shown, chip still renders. Stale extract (as_of + 1q + 60d + grace
< today) → global warning banner. Unknown ticker → 404 with pre-filled search. 40-cluster
color: 5 families + position + emphasis (validator proved 8 scatter hues illegal in dark).

## Exclusions (binding — from the financial expert)
No leaderboards/rankings; no star ratings; no "most likely to lag" sortable screens; no
absolute return predictions; backtest never a headline; no cluster-level forecasts; no
portfolio upload; no A-vs-B verdicts; no custom date ranges; no alerts on lag probability.
Default sorts never by return.

## Testing
- **pytest (offline):** extract builder — every view exists, row counts > 0, health-state
  rule truth-table, percentile suppression < 15, extract size < 50 MB; search scorer —
  tier ordering, fuzzy typos ("vangard"), dead-fund demotion; probability rendering — CI
  formatting, fixed sentence; chart option builders — connectNulls false, no dashed grid,
  tokens per mode.
- **App smoke:** NiceGUI test client — /fund/{live}, /fund/{dead}, /fund/{gappy},
  /fund/ZZZZ, /model render without error and contain their required honesty elements
  (chip, stamp, disclaimer) — the edge-state milestone test.
- **UAT (gate):** deployed Space URL serves /fund/VFIAX-equivalent + /model with real
  2026q2 data; degraded chip visible on both; search finds a fund by misspelled name;
  dead-fund page renders as archive; extract rebuild via advance.py path produces an
  identical-schema extract.

## Out of scope (step14)
Cluster explorer/detail/home (step15). Auth, user accounts, alerts (excluded permanently).
Custom domain. Always-on paid Space tier (free-tier cold start accepted initially).
