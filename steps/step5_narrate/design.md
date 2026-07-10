# step5_narrate — design

## Purpose (traces to Required Output)
Given a fund and a quarter, retrieve everything already computed about it (category, peers,
cluster, performance/risk metrics, prediction) and have phi-4 (via LM Studio's local API) explain
it in plain English, grounded in that retrieved data - not the model's own guess, not training.

## Scope and role of the LLM (locked decisions from earlier in this project)
- phi-4, served by LM Studio at `LM_STUDIO_BASE_URL` (confirmed live: `http://127.0.0.1:1234/v1`,
  model id `phi-4`), via the OpenAI-compatible chat completions API (verified with a real call).
- **RAG-style narration only.** All the facts in the explanation come from retrieval (steps
  1-4's tables); the LLM's job is to turn structured numbers into readable prose and never to
  invent a number, a peer, or a prediction itself. No fine-tuning, no training - already decided
  and out of scope.
- This is the **one deliberately non-deterministic external-service boundary** in the pipeline
  (documented per the deterministic-conductors principle) - LLM text output can vary between
  runs even at fixed input; the retrieved facts it's grounded in are fully deterministic.

## Retrieval: what gets assembled for one (series_id, quarter)
A single `build_context(series_id, quarter, cfg) -> dict` pulls, via `fundspeers.io.load_table`:
- **Identity:** `series_name`, `yahoo_category`, `ticker` (from `funds`).
- **Peers:** top-N from `fund_peers` (already has `peer_series_id`, `cosine_similarity`),
  resolved to peer names + categories via a second lookup into `funds`.
- **Cluster:** `cluster_id` from `fund_clusters`, and that quarter's `purity`/`adjusted_rand_index`
  from `cluster_validation` (how trustworthy the cluster assignment is, quarter-appropriate
  caveat).
- **Metrics:** whole-period `cumulative_return`/`annualized_volatility`/`sharpe_ratio`/
  `max_drawdown` from `fund_metrics_overall`, and that quarter's `quarterly_return`/
  `return_vs_cluster_median` from `fund_metrics_quarterly`.
- **Prediction:** `predicted_probability` and `actual_label` from `fund_predictions` for that
  (series_id, quarter), if present (not every fund-quarter is in the panel - step4 dropped 18
  rows and only covers transitions with a valid Q+1; missing prediction is reported as such,
  not hidden or fabricated).

## Prompting
- **System prompt** fixes the LLM's role and ground rules explicitly: explain only the provided
  facts, do not invent numbers, do not add investment advice/recommendations (this is a
  descriptive tool, not a financial advisor), flag missing data instead of guessing.
- **User content** is the retrieved context serialized as clearly-labeled plain text (not raw
  JSON dumped at the model - easier for a small local model to ground on cleanly).
- `temperature=0` for reproducible-as-possible output (verified working in the connectivity
  test); this doesn't make it fully deterministic (LLM inference isn't guaranteed bit-stable
  even at temp=0 across backends) - documented as the known non-deterministic boundary, not
  papered over.

## Output
- `narrate_fund(series_id, quarter, cfg) -> str` - the callable most tests and ad-hoc use will
  call directly (no persisted table for this step; narrations are generated on demand, not
  precomputed for the whole panel - there's no requirement to narrate all 4,356 fund-quarters,
  and doing so would be slow/costly for a local model with no clear use for the bulk output).
- `run(cfg)` (the step entry point the conductor calls) narrates one illustrative example fund
  (deterministically chosen: highest-`predicted_probability` fund in the test split, i.e. the
  fund the model is most confident will underperform next quarter - a natural, interesting case
  to showcase) and logs the result, so the conductor's full run always exercises this step
  without needing user input.

## Determinism
Retrieval is fully deterministic. The LLM call is the one documented exception (see above).
`run()`'s choice of *which* fund to narrate is deterministic (max by `predicted_probability`,
ties broken by `series_id` sort) - only the narration text itself may vary between runs.

## UAT (acceptance for this step) - all confirmed against the live LM Studio server
- LM Studio confirmed live before building (`GET /v1/models` listed `phi-4`; a real chat
  completion round-tripped correctly).
- `run(cfg)` picked the fund with the highest predicted underperformance probability in the
  test split (Invesco S&P Smallcap 600 Pure Value ETF, 2024q3, 87.8%) and produced a coherent,
  well-structured paragraph via a live call to phi-4.
- Spot-checked every claim in that narration against the raw retrieved context: ticker, category,
  net assets, all four whole-history metrics, this quarter's return and cluster-relative return,
  cluster id, purity/ARI, two named peers, predicted probability, and the actual outcome all
  matched exactly - no invented numbers, no invented peers.
- Both documented edge cases tested directly and handled gracefully (reported as "unavailable",
  not fabricated): a fund/quarter outside step4's panel (2024q4, the last quarter, has no Q+1)
  and a fund/quarter with no resolved cluster (`S000003656`, 2024q4 - the same zero-EC-holdings
  case step2 and step3 both documented).
