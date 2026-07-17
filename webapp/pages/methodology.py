"""Methodology - static prose explaining what the numbers mean, where they come from, and
what they don't claim. Content sourced from design.md's /methodology + Exclusions sections
and README.md's data-source / limitations language. Anchored (`#clusters`, `#disclaimer`)
so other pages (e.g. the cluster-name link on /fund/{ticker}) can deep-link into it."""
from nicegui import ui

from webapp.components import honesty
from webapp.theme import DISCLAIMER


def render_methodology(store) -> None:
    prov = store.provenance()

    ui.label("Methodology").classes("text-2xl font-semibold")
    honesty.freshness_stamp(prov)

    ui.label("Data source").classes("text-lg font-semibold mt-4")
    ui.markdown(
        "Holdings and fund identity come from the SEC's bulk **Form N-PORT Data Sets**; "
        "fees and turnover come from the SEC DERA **Risk/Return (RR) Summary** data sets. "
        "Both are as-filed, point-in-time regulatory filings — no paid vendor data, no "
        "screen-scraping. Funds report quarterly, and filings publish with a lag of "
        "roughly **60 days** after quarter-end, so this extract always trails the "
        "calendar by at least that long."
    ).classes("max-w-2xl text-sm")

    with ui.column().props("id=clusters").classes("w-full gap-0"):
        ui.label("What 'clustering' means").classes("text-lg font-semibold mt-4")
        ui.markdown(
            "Each fund's top holdings are turned into a short text description, embedded "
            "locally, and grouped with KMeans into peer clusters. A cluster is a "
            "**holdings-similarity** peer group — funds whose reported portfolios look "
            "alike — not a group formed from returns, risk, or performance. Cluster names "
            "are allocation labels only (e.g. \"Leaning Large Blend\"), taken from each "
            "cluster's dominant third-party category, never from how its funds "
            "subsequently performed."
        ).classes("max-w-2xl text-sm")

    ui.label("What the label means").classes("text-lg font-semibold mt-4")
    ui.markdown(
        "\"Underperform\" (the model's prediction target) means a fund's next-quarter "
        "return falls **below the median next-quarter return of its own top-10 "
        "most-similar peers** — a constant-size, nearest-neighbor benchmark, not a "
        "cluster-wide median that would silently change meaning as a cluster grows or "
        "shrinks."
    ).classes("max-w-2xl text-sm")

    ui.label("What the model can and can't tell you").classes("text-lg font-semibold mt-4")
    ui.markdown(
        "The prediction is **relative** — versus a fund's own peer group, never versus "
        "the market or an absolute return — **binary** — lag or not, never a return "
        "magnitude — and scoped to **one quarter ahead**. There are no rankings, no star "
        "ratings, no multi-quarter forecasts, and no cluster-level predictions: a single "
        "probability for a single fund for a single quarter, nothing more."
    ).classes("max-w-2xl text-sm")

    ui.label("Survivorship").classes("text-lg font-semibold mt-4")
    ui.markdown(
        "Dead funds are included in the historical record: a fund that leaves the "
        "universe keeps its final, often worst, quarters in every chart and average shown "
        "here, which structurally reduces survivorship bias rather than papering over it. "
        "Funds that die going forward are counted as forward-prediction attrition and "
        "disclosed on the scorecard — never dropped from history or imputed."
    ).classes("max-w-2xl text-sm")

    ui.label("Third-party categories").classes("text-lg font-semibold mt-4")
    ui.markdown(
        "Yahoo Finance categories are shown for orientation and used to sanity-check "
        "clusters against an independent label. `yfinance` is an unofficial, third-party "
        "scraper, not a regulatory filing — treat these categories as a noisy validator, "
        "never as ground truth."
    ).classes("max-w-2xl text-sm")

    with ui.column().props("id=disclaimer").classes("w-full gap-0"):
        ui.label("Disclaimer").classes("text-lg font-semibold mt-4")
        honesty.disclaimer_line()
        ui.markdown(
            f"**{DISCLAIMER}** Nothing on this site is a recommendation to buy, sell, or "
            "hold any security. Predicted and realized relative performance shown here "
            "are not a guarantee of future results. The probabilities on this site are "
            "model outputs that carry real, disclosed uncertainty (see /model) — always "
            "treat them as one noisy input among many, never as investment advice."
        ).classes("max-w-2xl text-sm")
