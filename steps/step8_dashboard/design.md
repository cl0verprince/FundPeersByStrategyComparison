# step8_dashboard — design

## Purpose
The project's human-facing final output: an **interactive dashboard** over the step7
unified universe, digestible by a financial data analyst / economist / DIY investor /
financial advisor — not a data scientist. One artifact that answers: what peer groups
exist, what does each one hold and how has it performed, which funds are in it, and what
does the model expect next quarter — with the model's actual measured skill and a clear
liability disclaimer alongside every prediction.

## Deliverable and constraints
- **One self-contained HTML file**: `reports/cluster_dashboard.html`, generated
  deterministically by `python -m steps.step8_dashboard.build` from the step7 tables.
  Same offline pattern as `reflection.html`: double-clickable, no server, no external
  CDNs/fonts/libraries — all data embedded as JSON, interactivity in vanilla JS, charts
  hand-rolled SVG rendered client-side from the embedded data.
- **Deterministic**: same input tables → byte-identical dashboard (narratives included,
  temperature 0). No timestamps of generation-time randomness in the output.
- Rebuild is cheap and idempotent; the file is committed like the cluster-map PNGs.

## Structure — three levels

### 1. Overview (landing view)
- Universe summary: fund count, quarters covered (2022q1–2024q4), data source (SEC N-PORT
  bulk data + Yahoo categories), segment split (strategy vs allocation/target-date).
- **Interactive cluster map**: PCA scatter of the latest quarter's strategy-segment
  embeddings (coordinates precomputed in the build), colored by cluster, hover tooltip =
  fund name, ticker, cluster short_title.
- "How to read this" — plain-English method summary: grouped by *what funds actually
  hold* (regulatory filings), not by marketing labels; ~200 words, no jargon.
- **Model scorecard** — the honest frame for every probability shown deeper in: pooled
  test AUC vs the random and persistence baselines, per-quarter AUC range plotted, and a
  short "what this number is and isn't" paragraph (a probability of lagging peers, not a
  return forecast; measured on past data; can be wrong).

### 2. Cluster index
- Sortable/filterable table of the 30 strategy clusters: short_title, member count,
  dominant category + share, avg Sharpe / volatility / max drawdown, median net assets.
  Click-through to the cluster section. Global fund search (name/ticker) jumps to the
  owning cluster.
- **Allocation segment shown separately, organized by vintage** (Target-Date 2030/2040/
  2050/...), presented as provider glide-path suites with the family named — the grouping
  a human expects there, per the step7 finding that holdings-clustering groups TDFs by
  family, not strategy. No model predictions for this segment (it is outside the strategy
  clustering and the model's training population); descriptive stats only.

### 3. Per-cluster section (x30)
- **Identity**: short_title, dominant category/tier + share, member count.
- **Allocation profile**: top representative holdings across members (weight-aggregated),
  concentration stats (median HHI, top-10 weight).
- **Performance panel**: distributions (histogram strips) of member Sharpe / volatility /
  max drawdown / cumulative return, each vs the all-universe distribution for context.
- **Member table**: fund name, ticker, net assets, trailing metrics, and the model's
  **underperformance probability for the coming quarter** (the genuine forward prediction:
  2024q4 features → 2025q1) — sortable, filterable, with the disclaimer note attached to
  the probability column header.
- **Plain-English narrative**: one phi-4-generated paragraph per cluster (reusing step5's
  LM Studio machinery, temperature 0), grounded ONLY in that cluster's computed numbers
  (top holdings, dominant category, performance stats, size) passed in the prompt —
  RAG-style, same anti-hallucination pattern as step5. Narratives are generated once and
  **cached to a `dashboard_narratives` table**; rebuilds reuse the cache (so build
  determinism never depends on LLM output stability), regenerating only via an explicit
  `--regenerate-narratives` flag. Build flag `--skip-narratives` emits a visible
  "narrative not generated" placeholder so the dashboard remains fully buildable without
  LM Studio running (UAT covers both modes).

## Disclaimer (required, non-negotiable)
A persistent footer on every view plus a note beside each probability column:
educational and informational purposes only; not investment advice or a recommendation;
predictions are statistical estimates that may be highly inaccurate; no liability or
responsibility is assumed for decisions made based on this material; past performance
does not guarantee future results.

## Visual design
Follows the dataviz skill's system (read before implementing charts): consistent palette,
light/dark friendly, every mark labeled, no chart junk. Tables are the primary interface;
charts support, not decorate.

## Data budget
~2,243 funds × (metrics + predictions + coordinates) + 30 cluster aggregates + top-holdings
lists ≈ low single-digit MB of embedded JSON — fine for a single offline HTML file.

## UAT (acceptance for this step)
- File opens offline (no network requests) in a browser; all 30 cluster sections render;
  allocation segment renders grouped by vintage.
- Sorting, filtering, and fund search work; cluster-map hover shows fund name/ticker.
- Every probability display has the disclaimer note; the footer disclaimer is always
  visible; the model scorecard shows AUC vs both baselines and the per-quarter range.
- Narratives: each cluster's paragraph mentions only facts present in its grounding data
  (spot-check 3 clusters); `--skip-narratives` build renders placeholders and is otherwise
  identical.
- Two consecutive builds from the same tables produce byte-identical output.
