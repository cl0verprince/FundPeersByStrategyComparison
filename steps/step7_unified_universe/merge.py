"""step7_unified_universe/merge.py - merge the three ingested batches into one universe.

Batch namespaces ("", "_oos", "_oos2") are an ingestion artifact, not a product concept
(see design.md). This module concatenates them into funds_all/holdings_all/
monthly_returns_all, asserts the batches really are disjoint, and tags each fund with its
segment: target-date/allocation funds cluster by fund FAMILY, not strategy (they are
fund-of-funds whose EC holdings are the provider's own index funds), so they are excluded
from the strategy clustering and presented separately.
"""
import logging
import re

import pandas as pd

from fundspeers.io import load_table, save_table
from steps.step2_similarity.similarity import normalize_issuer_name

log = logging.getLogger(__name__)

BATCH_SUFFIXES = ("", "_oos", "_oos2")
_ALLOCATION_PREFIXES = ("Target-Date", "Allocation")
# Fund-of-funds coverage diagnostic: issuer names that look like funds, not companies.
_FOF_ISSUER_RE = re.compile(r"\b(FUND|FD|PORTFOLIO|INDEX|ETF|TRUST|TR)\b")


def assign_segment(yahoo_category) -> str:
    if isinstance(yahoo_category, str) and yahoo_category.startswith(_ALLOCATION_PREFIXES):
        return "allocation"
    return "strategy"


def _log_fof_diagnostic(funds_all: pd.DataFrame, holdings_all: pd.DataFrame) -> None:
    """Verify (each run, not once) that the category-based segment rule caught the
    fund-of-funds: report strategy-segment equity funds whose EC holdings VALUE is
    majority fund-shaped issuer names."""
    strategy_acc = set(
        funds_all.loc[funds_all["is_us_equity"] & (funds_all["segment"] == "strategy"),
                      "accession_number"]
    )
    ec = holdings_all[
        holdings_all["accession_number"].isin(strategy_acc) & (holdings_all["asset_cat"] == "EC")
    ].copy()
    ec["is_fof_like"] = ec["issuer_name"].map(
        lambda n: bool(_FOF_ISSUER_RE.search(normalize_issuer_name(n)))
    )
    by_acc = ec.groupby("accession_number").apply(
        lambda g: g.loc[g["is_fof_like"], "currency_value"].sum() / max(g["currency_value"].sum(), 1e-9)
    )
    suspicious = (by_acc > 0.5).sum()
    log.info(f"FoF diagnostic: {suspicious} of {by_acc.shape[0]} strategy-segment filings have "
             f">50% fund-shaped EC holdings value (rule coverage check, not a filter)")


def build_unified_tables(cfg: dict) -> dict:
    funds_parts = [load_table(f"funds{s}", cfg) for s in BATCH_SUFFIXES]
    seen: set = set()
    for suffix, part in zip(BATCH_SUFFIXES, funds_parts):
        ids = set(part["series_id"].unique())
        overlap = seen & ids
        if overlap:
            raise RuntimeError(
                f"batch {suffix!r} overlaps an earlier batch on {len(overlap)} series_id(s), "
                f"e.g. {sorted(overlap)[:5]} - batches must be disjoint")
        seen |= ids

    funds_all = pd.concat(funds_parts, ignore_index=True)
    funds_all["segment"] = funds_all["yahoo_category"].map(assign_segment)
    holdings_all = pd.concat(
        [load_table(f"holdings{s}", cfg) for s in BATCH_SUFFIXES], ignore_index=True)
    monthly_all = pd.concat(
        [load_table(f"monthly_returns{s}", cfg) for s in BATCH_SUFFIXES], ignore_index=True)

    save_table(funds_all, "funds_all", cfg)
    save_table(holdings_all, "holdings_all", cfg)
    save_table(monthly_all, "monthly_returns_all", cfg)

    per_series = funds_all.drop_duplicates("series_id")
    counts = {
        "n_funds": int(per_series.shape[0]),
        "n_strategy": int((per_series["segment"] == "strategy").sum()),
        "n_allocation": int((per_series["segment"] == "allocation").sum()),
    }
    log.info(f"unified universe: {counts['n_funds']} funds "
             f"({counts['n_strategy']} strategy / {counts['n_allocation']} allocation)")
    _log_fof_diagnostic(funds_all, holdings_all)
    return counts
