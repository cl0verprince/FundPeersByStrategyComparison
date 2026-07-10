"""Shared fund-category bucketing. Used by step2 (cluster validation), step3 (cluster
definitions), and step4 (a stable, cross-quarter-comparable feature) - previously duplicated
in step2 and step4; centralized here once a third caller needed the same logic."""

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
