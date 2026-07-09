"""step2_similarity — build local text-embedding strategy vectors from holdings, cluster
into peer groups per quarter, and validate clusters against Yahoo fund categories.

See steps/step2_similarity/design.md: issuer-overlap vectors were tried first and rejected
(near chance-level agreement with Yahoo categories) in favor of embedding a holdings
description with a small local sentence-embedding model, which measured ~10x better.
"""
import logging
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from fundspeers.io import load_table, reports_dir, save_table

log = logging.getLogger(__name__)

_TRAILING_PUNCT_RE = re.compile(r"[.,]+$")
_WHITESPACE_RE = re.compile(r"\s+")

_model_cache = {}


def normalize_issuer_name(name: str) -> str:
    """Conservative normalization to merge casing/punctuation variants of the same
    issuer (e.g. "NVIDIA Corp" / "NVIDIA CORP" / "Microsoft Corp.") without merging
    genuinely distinct companies - no suffix-stripping, just case/punctuation/whitespace."""
    if not isinstance(name, str):
        return ""
    name = name.strip().upper()
    name = _TRAILING_PUNCT_RE.sub("", name)
    name = _WHITESPACE_RE.sub(" ", name)
    return name


def build_holdings_description(fund_holdings: pd.DataFrame, top_n: int) -> str:
    """One text string per fund-quarter: its top-N EC holdings by weight (renormalized
    within the fund's own equity sleeve), as "ISSUER weight%, ..." - the input to the
    embedding model."""
    total_value = fund_holdings["currency_value"].sum()
    weighted = fund_holdings.assign(weight=fund_holdings["currency_value"] / total_value)
    top = weighted.nlargest(top_n, "weight")
    parts = [f"{row.issuer_norm} {row.weight * 100:.1f}%" for row in top.itertuples()]
    return "Fund holdings: " + ", ".join(parts)


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_quarter_funds(quarter_holdings: pd.DataFrame, model_name: str, top_n: int) -> pd.DataFrame:
    """One row per series_id: its L2-normalized holdings-description embedding."""
    model = _get_embedding_model(model_name)
    series_ids, texts = [], []
    for series_id, group in quarter_holdings.groupby("series_id"):
        series_ids.append(series_id)
        texts.append(build_holdings_description(group, top_n))
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = normalize(embeddings)
    return pd.DataFrame(embeddings, index=pd.Index(series_ids, name="series_id"))


def compute_purity(cluster_labels: pd.Series, true_labels: pd.Series) -> float:
    """Weighted-by-size fraction of each cluster belonging to its majority true label."""
    df = pd.DataFrame({"cluster": cluster_labels.values, "truth": true_labels.values})
    total = len(df)
    if total == 0:
        return float("nan")
    majority_counts = df.groupby("cluster")["truth"].agg(lambda s: s.value_counts().max())
    return majority_counts.sum() / total


def get_peers(series_id: str, quarter: str, cfg: dict) -> pd.DataFrame:
    """Return the top-N cosine-similarity peers (with cluster id) for a fund in a quarter."""
    peers = load_table("fund_peers", cfg)
    clusters = load_table("fund_clusters", cfg)
    fund_peers = peers[(peers["series_id"] == series_id) & (peers["quarter"] == quarter)]
    fund_peers = fund_peers.sort_values("peer_rank")
    cluster_row = clusters[(clusters["series_id"] == series_id) & (clusters["quarter"] == quarter)]
    cluster_id = cluster_row["cluster_id"].iloc[0] if not cluster_row.empty else None
    fund_peers = fund_peers.assign(cluster_id=cluster_id)
    return fund_peers[["series_id", "quarter", "cluster_id", "peer_rank", "peer_series_id",
                        "cosine_similarity"]]


def _plot_cluster_map(vectors: pd.DataFrame, cluster_labels: pd.Series, quarter: str,
                       seed: int, cfg: dict) -> None:
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(vectors.values)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=cluster_labels.values, cmap="tab20", s=25)
    ax.set_title(f"Fund strategy clusters ({quarter}) - PCA projection of holdings embeddings")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="cluster_id")
    out_path = reports_dir(cfg) / f"cluster_map_{quarter}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote {out_path}")


def run(cfg: dict) -> None:
    seed = cfg["seed"]
    n_clusters = cfg["similarity"]["n_clusters"]
    top_n_peers = cfg["similarity"]["top_n_peers"]
    embedding_model = cfg["similarity"]["embedding_model"]
    top_holdings = cfg["similarity"]["top_holdings_for_description"]

    funds = load_table("funds", cfg)
    holdings = load_table("holdings", cfg)

    equity_series = set(funds.loc[funds["is_us_equity"], "series_id"].unique())
    equity_accessions = set(
        funds.loc[funds["series_id"].isin(equity_series), "accession_number"]
    )
    equity_holdings = holdings[
        holdings["accession_number"].isin(equity_accessions) & (holdings["asset_cat"] == "EC")
    ].copy()
    equity_holdings["issuer_norm"] = equity_holdings["issuer_name"].map(normalize_issuer_name)

    # accession_number -> series_id (holdings already carries its own "quarter" column from
    # step1, consistent with funds' quarter for the same accession_number by construction -
    # only series_id needs to come across the join).
    acc_lookup = funds[funds["series_id"].isin(equity_series)][
        ["accession_number", "series_id"]
    ].drop_duplicates()
    equity_holdings = equity_holdings.merge(acc_lookup, on="accession_number", how="inner")

    log.info(f"{len(equity_series)} equity funds; embedding holdings descriptions with "
             f"'{embedding_model}' (top {top_holdings} holdings per fund-quarter)")

    category_by_series = funds.drop_duplicates("series_id").set_index("series_id")["yahoo_category"]

    cluster_rows, peer_rows, validation_rows = [], [], []
    quarters = sorted(equity_holdings["quarter"].unique())
    latest_vectors = latest_labels = None

    for quarter in quarters:
        quarter_holdings = equity_holdings[equity_holdings["quarter"] == quarter]

        missing = equity_series - set(quarter_holdings["series_id"].unique())
        if missing:
            log.warning(f"{quarter}: {len(missing)} equity fund(s) have zero EC holdings "
                        f"this quarter, excluded from clustering: {sorted(missing)}")

        vectors = embed_quarter_funds(quarter_holdings, embedding_model, top_holdings)

        sim_matrix = cosine_similarity(vectors.values)
        series_ids = vectors.index.tolist()
        for i, series_id in enumerate(series_ids):
            sims = pd.Series(sim_matrix[i], index=series_ids).drop(index=series_id)
            top_peers = sims.sort_values(ascending=False).head(top_n_peers)
            for rank, (peer_id, score) in enumerate(top_peers.items(), start=1):
                peer_rows.append({
                    "series_id": series_id, "quarter": quarter, "peer_rank": rank,
                    "peer_series_id": peer_id, "cosine_similarity": score,
                })

        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = pd.Series(kmeans.fit_predict(vectors.values), index=series_ids, name="cluster_id")
        for series_id, cluster_id in labels.items():
            cluster_rows.append({"series_id": series_id, "quarter": quarter, "cluster_id": cluster_id})

        truth = category_by_series.reindex(series_ids).fillna("Unknown")
        purity = compute_purity(labels, truth)
        ari = adjusted_rand_score(truth.values, labels.values)
        validation_rows.append({"quarter": quarter, "purity": purity, "adjusted_rand_index": ari})
        log.info(f"{quarter}: {len(series_ids)} funds clustered into {n_clusters} groups "
                 f"(purity={purity:.3f}, ARI={ari:.3f})")

        latest_vectors, latest_labels = vectors, labels

    save_table(pd.DataFrame(cluster_rows), "fund_clusters", cfg)
    save_table(pd.DataFrame(peer_rows), "fund_peers", cfg)
    save_table(pd.DataFrame(validation_rows), "cluster_validation", cfg)

    _plot_cluster_map(latest_vectors, latest_labels, quarters[-1], seed, cfg)

    mean_purity = pd.DataFrame(validation_rows)["purity"].mean()
    mean_ari = pd.DataFrame(validation_rows)["adjusted_rand_index"].mean()
    log.info(f"averaged across {len(quarters)} quarters: purity={mean_purity:.3f}, "
             f"ARI={mean_ari:.3f}")

    _log_validation_by_category_tier(cluster_rows, category_by_series)


_CATEGORY_TIERS = {
    "Large": ("Large Blend", "Large Value", "Large Growth"),
    "Mid": ("Mid-Cap Blend", "Mid-Cap Value", "Mid-Cap Growth"),
    "Small": ("Small Blend", "Small Value", "Small Growth"),
}


def _category_tier(category: str) -> str:
    for tier, names in _CATEGORY_TIERS.items():
        if category in names:
            return tier
    return "Sector/Other"


def _log_validation_by_category_tier(cluster_rows: list, category_by_series: pd.Series) -> None:
    """Purity/ARI pooled across all quarters, broken down by broad market-cap tier - so a
    strong overall average can't hide segments the method serves less well."""
    df = pd.DataFrame(cluster_rows)
    df["category"] = df["series_id"].map(category_by_series).fillna("Unknown")
    df["tier"] = df["category"].map(_category_tier)
    for tier, group in df.groupby("tier"):
        purity = compute_purity(group["cluster_id"], group["category"])
        ari = adjusted_rand_score(group["category"].values, group["cluster_id"].values)
        log.info(f"  tier={tier} (n={len(group)} fund-quarters): purity={purity:.3f}, ARI={ari:.3f}")
