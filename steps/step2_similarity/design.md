# step2_similarity — design

## Follow-up (2026-07-11): cluster map legend using `short_title`
The PCA cluster-map PNG originally used a plain numeric `cluster_id` colorbar - not
informative on its own (step2's cluster ids are arbitrary per-quarter/per-run labels, per the
KMeans note below). `_plot_cluster_map` now plots one scatter series per cluster, labeled with
its `short_title` (e.g. "Leaning Large Blend", "Concentrated Real Estate") in a proper legend.
`short_title` only needs `dominant_category`/`dominant_category_share` - not the
metrics-dependent fields step3's `cluster_definitions` also carries - so
`fundspeers.category.compute_dominant_category_info` (shared with step3) is called directly
in step2's `run()`, before step3 ever runs. Concretely useful in practice: comparing the
original and out-of-sample cluster maps (step6), the legend makes it visible that the same
*type* of fund (Target-Date allocation funds) forms the separated left island in both plots,
even though the arbitrary `cluster_id` numbers differ between the two independent runs -
exactly the kind of structural consistency check that's otherwise invisible from color alone.

## Purpose (traces to Required Output)
For the US-equity fund universe built in step1, build strategy vectors from actual portfolio
overlap, compute cosine similarity, cluster into peer groups per quarter, and validate the
clusters against Yahoo's fund categories.

## Scope
Only the **363 funds flagged `is_us_equity=True`** in `funds.parquet` — the 637 non-equity
funds are out of scope for peer grouping (they exist in the data only as part of step1's
broader universe/attrition margin). Only **`asset_cat=='EC'`** (common equity) holdings rows
are used to build strategy vectors — a fund's residual bond/cash sleeve doesn't reflect its
equity strategy and would dilute the overlap signal.

## Why category-only features would fail (real finding, not hypothetical)
The master plan named "allocation across asset category/issuer type/country" as the feature
space. Checked against real data: since this step's universe is already equity-filtered, nearly
every fund's category breakdown looks like ~95%+ `EC`/`CORP`/`US` — there's almost no variance
left to cluster on. **Confirmed and rejected** in favor of issuer-overlap vectors (below),
which is also how real portfolio-overlap tools (e.g. Morningstar X-Ray) measure fund similarity.

## Feature vectors: local text-embeddings of a holdings description (final approach)

**Issuer-overlap vectors were built and tested first, then rejected on measured evidence** (see
below) in favor of embeddings. `normalize_issuer_name` (uppercase, strip trailing punctuation,
collapse whitespace) is the one piece of that work retained - it fixed a real data defect
(`"NVIDIA Corp"` / `"NVIDIA CORP"` / `"Microsoft Corp."` all being the same issuer reported
differently by different filers) and is reused to build clean holdings-description text below.

**Why issuer-overlap failed:** a value-weighted top-K allocation vector over held issuers is
literally what tools like Morningstar X-Ray use for portfolio overlap, so it was the natural
first approach. Tested at K=500 and K=3000 (breadth-selected, i.e. by distinct funds holding an
issuer rather than by dollar value, since value-based selection was *also* tested and found to
starve small-cap funds of any representation). Result on real data: **adjusted Rand Index vs.
Yahoo categories stayed at 0.01-0.07 (chance level) regardless of K or `n_clusters` (tested
5 through 60)** - ruled out as a tuning problem. Root cause: popular mega-cap stocks (Microsoft,
Apple, NVIDIA) are held broadly *across* growth/value/blend categories, so raw stock overlap
mostly captures "which companies are popular" rather than the value/growth tilt that actually
defines Morningstar's style-box taxonomy.

**The embedding approach:**
1. For each fund-quarter, take its top 25 `EC` holdings by weight (renormalized within the
   fund's own equity sleeve) and build one text string, e.g.
   `"Fund holdings: NVIDIA CORP 8.2%, MICROSOFT CORP 7.5%, ..."` using the normalized issuer
   names.
2. Embed this text with a small local sentence-embedding model, **`all-MiniLM-L6-v2`**
   (`sentence-transformers`, 384-dim, ~80MB, CPU-only, no GPU required). Downloaded once from
   the Hugging Face Hub and cached locally - the **one documented external-network dependency**
   of this step (first run only; fully offline afterward). Deterministic at inference time (no
   sampling).
3. L2-normalize the embedding vectors (standard practice for sentence embeddings - makes KMeans
   behave like spherical k-means and keeps cosine similarity well-defined for peer lookup).

**Empirically confirmed improvement:** on the same 362-fund 2024q4 snapshot, embeddings raised
adjusted Rand Index from ~0.02 (issuer-overlap) to **0.25 at `n_clusters=15`** (the configured
value) and up to 0.30 at `n_clusters=39` (matching the real category count) - roughly a 10x
improvement, and purity roughly doubled at the same cluster count (0.27 -> 0.41). A general-
purpose text embedding model appears to encode broader associations about company identity and
investment style that plain stock-overlap cannot. Also much simpler (no top-K tuning) and fast
(~60s for the full 12-quarter run, including one-time model download).

**Honest characterization of what the ARI improvement represents** (confirmed on the full
12-quarter run: overall purity=0.409, ARI=0.249): breaking validation down by broad market-cap
tier (Large/Mid/Small/Sector-Other, pooled across quarters) shows **within-tier ARI is only
0.01-0.04** - close to chance - while the strong overall ARI is driven mostly by the clusters
correctly separating **broad segments** (large-cap vs. small-cap vs. sector-concentrated funds)
rather than distinguishing **growth vs. value vs. blend within a given cap-size tier**. This
makes sense: which companies a fund holds strongly signals its market-cap/sector tier (small-cap
funds hold different companies than large-cap funds), but a stock's growth/value tilt is a
valuation-ratio characteristic, not something holdings-identity alone captures well - even via
embeddings. For this project's purposes this is still a legitimate, useful notion of "peer" (a
Large Blend fund's peer group is dominated by other large-cap funds, which is a more relevant
comparison set than random funds), even though it doesn't fully recreate Morningstar's finer
style-box taxonomy.

**New dependencies:** `sentence-transformers` (pulls in `torch`, `transformers`) - confirmed
Python 3.14-compatible wheels exist (checked before installing, since this project already hit
a Python-3.14-wheel wall with `pandas` in step0).

## Per-quarter clustering (not one static clustering)
Because step4's prediction panel is "features at quarter Q -> underperform label at Q+1" pooled
across quarters, peer groups must be computed **separately for each of the 12 quarters** - a
fund's peer group as of Q1 may differ from Q4 as its holdings evolve. Cluster ids are **not**
meaningful across quarters (KMeans labels are arbitrary per run) - only within a given quarter.
- **Clustering:** KMeans, `similarity.n_clusters` (config, 15), one fit per quarter on that
  quarter's 363 vectors.
- **Peers:** cosine similarity matrix per quarter; `similarity.top_n_peers` (config, 10)
  nearest peers per fund (excluding itself).
- **Determinism:** KMeans seeded from `cfg["seed"]`.

## Cluster validation against Yahoo categories
Per quarter, score the KMeans cluster labels against the (time-invariant) `yahoo_category`
ground truth:
- **Purity:** for each cluster, the fraction of funds belonging to its majority category,
  weighted by cluster size.
- **Adjusted Rand Index (ARI):** `sklearn.metrics.adjusted_rand_score` - standard clustering-
  vs-ground-truth agreement measure, corrected for chance.
Both are computed per quarter and averaged into one reported figure for the Required Output's
acceptance criterion ("a reported number showing how well clusters recover known categories").

## Outputs (persisted + one visualization)
- **`fund_clusters.parquet`**: `series_id`, `quarter`, `cluster_id`.
- **`fund_peers.parquet`**: `series_id`, `quarter`, `peer_rank` (1..10), `peer_series_id`,
  `cosine_similarity`.
- **`cluster_validation.parquet`**: `quarter`, `purity`, `adjusted_rand_index`.
- **`reports/cluster_map_<latest_quarter>.png`**: 2D PCA projection of the latest quarter's
  363 fund embedding vectors, colored by cluster id, as the required cluster visualization.
  New `paths.reports` config entry.
- A `get_peers(series_id, quarter)` - style lookup function in `similarity.py` for ad-hoc
  querying (backs the "for any fund, return its peers" acceptance criterion).

## Config additions (`config.json`)
- `similarity.embedding_model: "all-MiniLM-L6-v2"`
- `similarity.top_holdings_for_description: 25`
- `paths.reports: "reports"`
(`similarity.n_clusters: 15` and `similarity.top_n_peers: 10` already exist from step0;
`similarity.issuer_top_k` from the rejected issuer-overlap approach is removed.)

## Determinism
KMeans seeded from `cfg["seed"]`; embedding inference is deterministic (no sampling) once the
model is downloaded and cached; PCA for the plot uses `random_state=cfg["seed"]`.

## UAT (acceptance for this step)
- For any `(series_id, quarter)` in the equity universe, `get_peers()` returns its top-10
  cosine-similarity peers and its cluster id.
- `cluster_validation.parquet` has one row per quarter with `purity` and `adjusted_rand_index`
  populated (non-null, valid ranges) - reported as the required validation score. Given the
  embedding approach measured ARI ~0.25 on a real quarter (vs. ~0.02 for the rejected
  issuer-overlap approach), the full 12-quarter run should land in a similar range, not near zero.
- `reports/cluster_map_<quarter>.png` exists and shows visibly separated clusters (not one
  giant blob), spot-checked visually.
- Spot-check: two funds with very similar Yahoo categories (e.g. two "Large Blend" funds) show
  higher cosine similarity to each other than to a fund in an unrelated category (e.g. a
  "Small Value" fund), on real data.
