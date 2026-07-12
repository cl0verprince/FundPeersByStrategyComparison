"""Shared fund-category bucketing. Used by step2 (cluster validation), step3 (cluster
definitions), and step4 (a stable, cross-quarter-comparable feature) - previously duplicated
in step2 and step4; centralized here once a third caller needed the same logic."""
import pandas as pd

CATEGORY_TIERS = {
    "Large": ("Large Blend", "Large Value", "Large Growth"),
    "Mid": ("Mid-Cap Blend", "Mid-Cap Value", "Mid-Cap Growth"),
    "Small": ("Small Blend", "Small Value", "Small Growth"),
}


def category_tier(category: str) -> str:
    """Bucket a Yahoo/Morningstar-style category into Large/Mid/Small/Sector-Other."""
    for tier, names in CATEGORY_TIERS.items():
        if category in names:
            return tier
    return "Sector/Other"


def concentration_word(dominant_category_share: float) -> str:
    """How homogeneous a cluster is, in one word - the first half of a short_title."""
    if dominant_category_share >= 0.70:
        return "Concentrated"
    if dominant_category_share >= 0.40:
        return "Leaning"
    return "Mixed"


def compute_dominant_category_info(cluster_assignments: pd.DataFrame,
                                    category_by_series: pd.Series) -> pd.DataFrame:
    """For one quarter's cluster assignments (columns: series_id, cluster_id), the dominant
    category, its share, and a short allocation-only bucket name per cluster_id - e.g.
    "Concentrated Real Estate". Deliberately excludes any performance stat (avg_sharpe,
    avg_volatility): clusters are formed from holdings/allocation similarity, not
    performance, so the identifying name is built only from cluster composition. Shared by
    step2 (the cluster-map legend, computed before step3 exists) and step3 (the full
    cluster_definitions table, which adds performance stats as separate descriptive fields,
    not part of the name)."""
    df = cluster_assignments.copy()
    # fillna("Unknown") mirrors step2's purity-truth convention. Without it, a cluster whose
    # members ALL lack a yahoo_category crashes the empty mode() below - first hit for real
    # on the _full universe (2022q3, cluster 37: every member category-unresolved).
    df["yahoo_category"] = df["series_id"].map(category_by_series).fillna("Unknown")
    rows = []
    for cluster_id, group in df.groupby("cluster_id"):
        dominant_category = group["yahoo_category"].mode().iloc[0]
        dominant_category_share = (group["yahoo_category"] == dominant_category).mean()
        rows.append({
            "cluster_id": cluster_id,
            "dominant_category": dominant_category,
            "dominant_category_share": dominant_category_share,
            "short_title": f"{concentration_word(dominant_category_share)} {dominant_category}",
        })
    return pd.DataFrame(rows)
