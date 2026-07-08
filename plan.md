# Goal
From SEC N-PORT holdings, group US equity funds into strategy peer groups by what they hold,
then train a model that predicts — beating a naive baseline — which funds will underperform
their peer group's next-quarter total return, with phi-4 (LM Studio) generating plain-English
explanations of each grouping and prediction.

# Required Output
- **Deliverable:** a reproducible pipeline = Python modules (`ingest`, `similarity`, `metrics`,
  `model`, `llm`) + thin, explained notebooks per method + a trained random-forest model +
  phi-4-generated narrations, all runnable from one deterministic conductor entry point.
- **Format / interface:** run the conductor script (or per-step notebooks); outputs are local
  data tables (parquet/DuckDB), peer-group assignments, per-fund metrics, an evaluated model,
  and text explanations.
- **Acceptance criteria (each objectively checkable):**
    - Ingestion builds clean `funds`, `holdings`, and `monthly_returns` tables from the chosen
      N-PORT quarters; schema validates and row counts > 0; a US-equity flag is populated.
    - For any given fund, the system returns its top-N cosine-similarity peers and a cluster id.
    - Holdings-based clusters are scored against Yahoo fund categories (purity / adjusted Rand
      index) — a reported number showing how well the clusters recover the known categories.
    - Per-fund metrics computed for all funds: cumulative return, volatility, Sharpe, max drawdown,
      plus each fund's return relative to its cluster median.
    - A random forest predicts next-quarter peer-underperformance with **AUC above a 0.5 baseline**
      on a time-based held-out test set; feature importances are reported.
    - phi-4 (LM Studio API) emits a coherent plain-English explanation, grounded in the computed
      data, for a chosen fund's cluster and prediction.
- **Out of scope:** live/real-time data, paid vendors, LLM fine-tuning, non-equity or non-US funds,
  portfolio construction / trading, per-filing XML scraping.

# Key decisions & verified facts
- **Data source:** bulk **Form N-PORT Data Sets** — quarterly ZIPs of tab-delimited files at
  `https://www.sec.gov/files/dera/data/form-n-port-data-sets/{YYYY}q{N}_nport.zip`
  (coverage 2019Q4–2026Q1). No XML scraping. Requires a descriptive `User-Agent` header; ≤10 req/s.
- **Tables used:** `SUBMISSION`, `REGISTRANT`, `FUND_REPORTED_INFO` (`NET_ASSETS`),
  `MONTHLY_TOTAL_RETURN` (per `CLASS_ID`, 3 monthly returns/filing), `FUND_REPORTED_HOLDING`
  (`ASSET_CAT`, `PERCENTAGE`, `INVESTMENT_COUNTRY`, issuer fields). Join fund-level on
  `ACCESSION_NUMBER`, holdings on `HOLDING_ID`.
- **Data span:** US funds, ~2–3 years of quarterly snapshots (default 2022Q1–2024Q4; adjustable) —
  enough for a pooled, time-split train/test across multiple prediction periods.
- **Universe:** US equity funds, identified two ways that cross-check: **Yahoo fund category**
  (pre-filter) and holdings inference (value-weighted `ASSET_CAT='EC'` + `INVESTMENT_COUNTRY='US'`).
  No fund-level equity label exists in N-PORT.
- **Yahoo Finance (heuristic + validator):** join N-PORT funds → tickers via the SEC mutual-fund
  ticker map (`https://www.sec.gov/files/company_tickers_mf.json`), then pull each fund's **category**
  via `yfinance`. Used for the equity pre-filter and for validating the holdings-based clusters.
  `yfinance` is unofficial/best-effort — categories are a heuristic, not authoritative.
- **Cadence:** holdings snapshots are effectively **quarterly** (only 3rd-month reports public
  historically), but each carries **all 3 monthly returns** → "features at quarter Q → label at Q+1,"
  pooled across quarters.
- **Peers:** cosine-similarity clusters over holdings-based strategy vectors.
- **Underperform (label):** a fund's next-quarter total return is below its cluster's median.
- **Multi-class returns:** `MONTHLY_TOTAL_RETURN` is per share-class; aggregate to fund/series level
  (net-asset-weighted or representative largest class) — resolved in step1 `design.md`.
- **LLM:** phi-4 (`Q4_K_S` GGUF) served by **LM Studio** at its OpenAI-compatible local API
  (`http://localhost:1234/v1`), used for **RAG-style narration only** (no training/fine-tuning).

# Steps
1. **step0_setup** — Scaffold once: `.gitignore` + `.env.example`, dependency setup
   (pandas/pyarrow or duckdb, scikit-learn, requests, yfinance, matplotlib, openai client),
   module + notebook layout, the deterministic conductor entry-point skeleton, and browser-readable
   docs (`render_docs.py` + seeded `decisions.json`/`workflow.json`) → step0_setup/design.md
2. **step1_ingest** — Download the chosen N-PORT quarterly ZIPs, load the key tables, build clean
   `funds`, `holdings`, `monthly_returns` tables; enrich `funds` with ticker (SEC
   `company_tickers_mf.json`) + Yahoo category; set US-equity flag; persist → step1_ingest/design.md
3. **step2_similarity** — Build value-weighted strategy feature vectors, compute cosine similarity,
   cluster into peer groups; nearest-peers lookup + visualization; score clusters against Yahoo
   categories (purity / adjusted Rand index) → step2_similarity/design.md
4. **step3_metrics** — From monthly returns compute cumulative return, volatility, Sharpe, max
   drawdown per fund, plus each fund's return vs its cluster median → step3_metrics/design.md
5. **step4_predict** — Assemble the supervised panel (features at Q → underperform at Q+1) pooled
   across quarters; time-based train/test split; decision tree → random forest; evaluate vs baseline;
   feature importances → step4_predict/design.md
6. **step5_narrate** — Given a fund, retrieve its cluster/peers/metrics/prediction and have phi-4
   (LM Studio API) emit a grounded plain-English explanation → step5_narrate/design.md

# Process
Built with the step-wise-methodology: contract-first; one numbered step at a time; each step has its
own `design.md` tracing to the Required Output, is built and tested, updates the live docs
(`decisions.json`, `workflow.json`) in the same change, runs a UAT against its acceptance criteria,
is committed with a staged-diff secret scan, and then **stops at a human approval gate** before the
next step starts. `plan.md` `# Steps` + the generated `workflow.html` are the roadmap; there is no
separate roadmap file.

# Verification
- **Per step:** run the step's code/notebook and check its UAT acceptance criteria; report at the gate.
- **End-to-end (SIT):** run the deterministic conductor start→finish and confirm every Required
  Output criterion. The phi-4 narration criterion requires LM Studio running phi-4 locally.
