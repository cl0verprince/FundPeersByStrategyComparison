# step1_ingest — design

## Purpose (traces to Required Output)
Produce clean `funds`, `holdings`, and `monthly_returns` parquet tables for a bounded, US-equity
fund universe spanning multiple quarters — the foundation every later step (similarity, metrics,
prediction, narration) reads from.

## Verified facts (checked against live SEC/Yahoo responses, not assumed)
- N-PORT ZIP internal files (tab-delimited) confirmed by downloading `2024q4_nport.zip`:
  `SUBMISSION.tsv`, `REGISTRANT.tsv`, `FUND_REPORTED_INFO.tsv`, `MONTHLY_TOTAL_RETURN.tsv`,
  `FUND_REPORTED_HOLDING.tsv` (+ many derivative/debt tables we don't need).
- **`FUND_REPORTED_HOLDING.tsv` is ~850MB uncompressed per quarter** (all filers, all asset
  classes) — far too large to load in full for every quarter. Must filter to our fund universe's
  `ACCESSION_NUMBER`s before/while loading it.
- Confirmed columns match the plan: `ASSET_CAT` (e.g. `EC`=equity common, `LON`, `DBT`, `ABS-MBS`,
  ...), `INVESTMENT_COUNTRY`, `PERCENTAGE`, `CURRENCY_VALUE` present in `FUND_REPORTED_HOLDING.tsv`.
- `MONTHLY_TOTAL_RETURN.tsv` confirmed per-`CLASS_ID`, 3 monthly returns per filing.
- SEC's `company_tickers_mf.json` confirmed format: flat rows of `[cik, seriesId, classId, symbol]`
  — one row per share class. `CLASS_ID` values (e.g. `C000024954`) are globally unique SEC-assigned
  IDs, so join directly on `classId == CLASS_ID` (no need to also match CIK).
- **Correction to the plan's assumed Yahoo path:** `yfinance`'s `Ticker(symbol).info` does **not**
  carry a fund category for mutual funds in the installed version (1.5.1). The category instead
  lives at `Ticker(symbol).funds_data.fund_overview["categoryName"]` (e.g. `"Mid-Cap Growth"`).
  `funds_data.asset_classes` additionally gives a direct `stockPosition` fraction (0–1) — a
  **cheaper equity pre-filter than parsing holdings at all**, usable before touching N-PORT holdings.
- Category/asset-class calls are **one per fund series** (not per quarter) — Yahoo reflects the
  fund's current category, applied uniformly across all quarters for that series. This is a
  simplifying assumption (a fund's category rarely changes) — documented, not hidden.

## Ingestion order (staged, to avoid loading the 850MB/quarter holdings table wholesale)
1. **Fund-level tables** (small: 1–4MB/quarter) — download+load `SUBMISSION`, `REGISTRANT`,
   `FUND_REPORTED_INFO` for every configured quarter; join on `ACCESSION_NUMBER` into a raw
   fund-quarter table (`SERIES_ID`, `SERIES_NAME`, `CIK`, `REPORT_DATE`, `NET_ASSETS`, ...).
2. **Returns table** (small) — load `MONTHLY_TOTAL_RETURN` for every quarter (per `CLASS_ID`).
3. **Panel filter** — keep only `SERIES_ID`s present in **every** configured quarter (guarantees a
   complete panel for the pooled multi-period prediction task in step4).
4. **Universe sampling — iterative fill, not a fixed pre-sample.** A 20-fund pilot run showed
   ~45% of candidate series have no ticker mapping at all (before Yahoo is even attempted), so a
   fixed sample of `max_funds` candidates would under-shoot the target after attrition. Instead:
   deterministically shuffle the full complete-panel candidate pool (seeded via `config.seed`),
   then resolve ticker + Yahoo data one candidate at a time until `max_funds` are **successfully
   resolved** or the pool is exhausted (logged if so). Default **`max_funds: 1000`** (confirmed
   with user — larger universe for richer clustering/prediction).
5. **Ticker mapping** — download `company_tickers_mf.json` once; for each sampled series pick one
   canonical class (lowest `CLASS_ID` that also appears in `MONTHLY_TOTAL_RETURN`) and its ticker.
6. **Yahoo enrichment** — for each sampled series' ticker, call `funds_data.fund_overview` +
   `funds_data.asset_classes` once; record `yahoo_category` and `yahoo_stock_position`.
   - `yahoo_stock_position` measures **asset class only** (stock vs. bond/cash) — it does **not**
     capture geography. A fund with `yahoo_stock_position=0.99` can still be "Foreign Large Value"
     or "Diversified Emerging Mkts". Confirmed empirically: an early is-us-equity flag defined as
     `yahoo_stock_position >= 0.80` alone incorrectly flagged international/emerging-market equity
     funds as US equity. **Geography must come from holdings**, per the original plan decision.
   - Small delay between calls (`time.sleep`) — polite to the unofficial API, not a hard rate limiter.
   - A ticker that fails to resolve (delisted/no data) drops that series from the universe (logged,
     not an error) — bounded universe already has margin from `max_funds`.
7. **Holdings** — now that the fund universe (and its `ACCESSION_NUMBER`s per surviving quarter) is
   fixed and small, stream-parse each quarter's `FUND_REPORTED_HOLDING.tsv` in chunks
   (`pd.read_csv(..., chunksize=...)`), keeping only rows whose `ACCESSION_NUMBER` is in the
   universe. Build the holdings table from these rows (used for strategy vectors in step2).
   - Per `accession_number`, compute `holdings_us_equity_share` = value-weighted share of holdings
     where `ASSET_CAT=='EC'` **and** `INVESTMENT_COUNTRY=='US'`, out of total portfolio value.
     Average across a series' 12 quarters to get one per-series figure.
   - **Final `is_us_equity` flag** = `yahoo_stock_position >= data.equity_stock_position_min` **AND**
     `holdings_us_equity_share >= data.us_holdings_share_min` — the two sources cross-check each
     other exactly as the original plan intended (Yahoo for the fast equity/bond split, holdings
     for geography). `holdings_us_equity_share` is also persisted as its own column for step2's
     cluster-vs-category validation to reference.
8. **Multi-class return aggregation** — `MONTHLY_TOTAL_RETURN` is per `CLASS_ID`; aggregate to
   series level via **simple mean across classes present that quarter** (documented simplification;
   share-class-level net-asset weights aren't available in the bulk data set).
9. **Persist** three tables via `fundspeers.io.save_table`: `funds`, `holdings`, `monthly_returns`.

## Schema (persisted tables)
- **`funds`**: `series_id`, `series_name`, `cik`, `ticker`, `yahoo_category`,
  `yahoo_stock_position`, `holdings_us_equity_share`, `is_us_equity` (bool), plus per-quarter
  `accession_number`/`net_assets` rows (one row per series per quarter — a panel, not one row
  per fund).
- **`holdings`**: `accession_number`, `holding_id`, `issuer_name`, `asset_cat`, `issuer_type`,
  `investment_country`, `percentage`, `currency_value`.
- **`monthly_returns`**: `series_id`, `report_date` (quarter), `month_in_quarter` (1/2/3),
  `total_return` (mean across classes).

## Config additions (`config.json`)
- `data.max_funds: 1000`
- `data.equity_stock_position_min: 0.80`
- `data.us_holdings_share_min: 0.70`
- `data.yahoo_request_delay_seconds: 0.3`
- `data.yahoo_max_retries: 3`

## Determinism
Universe sampling uses `random.Random(cfg["seed"])` over the sorted, complete-panel series list —
same seed + same input data → same sample every run. Downloaded ZIPs are cached under
`data/raw/` (git-ignored) so re-runs don't re-download.

## Known limitation: fund-of-funds wrappers fool the geography check
`INVESTMENT_COUNTRY` in N-PORT is the country where a holding's **issuer** is organized, not
where its economic exposure lies. A fund that gains international exposure by holding *another*
US-domiciled fund (e.g. a currency-hedged EAFE ETF holding shares of a plain US-registered
`iShares MSCI EAFE ETF`) shows up as `INVESTMENT_COUNTRY='US'` for that holding, inflating
`holdings_us_equity_share` even though the fund is 100% international. Verified on real data:
3 of 1000 sampled funds (all with recognizably international Yahoo categories) were misflagged
this way. This is an inherent property of the data model (not a bug in this ingestion), affects
~0.3% of the sample, and is not worth solving here (would require recursively resolving nested
fund-of-fund holdings) - documented as a known, accepted limitation.

## UAT (acceptance for this step)
- Running the step downloads the 12 configured quarterly ZIPs (cached after first run) and
  completes without error.
- `funds`, `holdings`, `monthly_returns` parquet files exist; row counts > 0.
- `funds.is_us_equity` is populated (not all-null) and reflects both the `yahoo_stock_position`
  and `holdings_us_equity_share` thresholds.
- Every `series_id` in `funds` has rows in `monthly_returns` for all 12 quarters (panel completeness).
- Spot-check: US domestic equity funds (e.g. "Large Blend", "Small Value") are flagged
  `is_us_equity=True`; bond funds and the large majority of international/global equity funds
  (e.g. "Foreign Large Value", "Diversified Emerging Mkts") are flagged `False` - confirmed 87/90
  on the real 1000-fund run, with the 3 exceptions traced to the documented fund-of-funds
  limitation above, not an ingestion defect.

## Resolved: universe size
Confirmed with user: **`max_funds: 1000`**. Downloading ~12 quarters x ~400MB ZIPs (~5GB total,
cached under `data/raw/` after first run) plus ~1000 sequential `yfinance` calls (~5-10 min with
a polite delay). Retries with backoff on transient failures; a ticker that never resolves is
dropped from the universe and logged, not treated as a fatal error.
