"""step9_fees_turnover.parse — flatten the RR num.tsv/sub.tsv rows into `rr_fees_raw`,
one row per (filing, share class) carrying point-in-time expense/turnover facts.

Codes to design.md "Schema amendment (verified 2026-07-12)":
- num.tsv holds `series` (S000######), `class` (C000######, empty for series-level facts),
  `tag`, `ddate`, `value`; join `adsh -> sub.filed` for the look-ahead-safe filing date.
- Tags: NetExpensesOverAssets (net), ExpensesOverAssets (gross), PortfolioTurnoverRate.
  All are FRACTIONS already (0.0075 = 0.75%), so no percent rescaling.
- Net-else-gross fallback (mandatory for coverage): expense_ratio_net = net if present else
  gross; a `fee_source` column records which per row. gross kept alongside for the record.
- Sanity bounds (design amendment): expenses in [0, 0.10], turnover in [0, 20]. Out-of-bound
  VALUES are nulled (not whole rows dropped); a row surviving with >=1 value is kept. This
  matters because gross feeds expense_ratio_net for gross-only funds, and gross has garbage
  (max 6.30) the way turnover does (max 44577).
- Dedup: within (adsh, series, class, tag) keep max `ddate` (multiple fiscal-year periods
  per filing), tie-break max `value`, so the pipeline is reproducible.

rr_fees_raw is built for our strategy universe only (documented scope choice — keeps the
large num.tsv parse memory-sane); the point-in-time join in fees.py consumes it.
"""
import logging
import zipfile

import numpy as np
import pandas as pd

from fundspeers.io import load_table, raw_dir, save_table
from steps.step9_fees_turnover.acquire import rr_zip_path

log = logging.getLogger(__name__)

NET_TAG = "NetExpensesOverAssets"
GROSS_TAG = "ExpensesOverAssets"
TURNOVER_TAG = "PortfolioTurnoverRate"
TAGS = (NET_TAG, GROSS_TAG, TURNOVER_TAG)

# Sanity bounds — design amendment. Deviates from the plan's example (0, 0.05]: verified
# net max is 0.0598 (clean, legit US-equity fund), so a 0.05 cap would drop real funds;
# 0.10 is the amendment's own recommendation and still drops the gross garbage (>1.0).
EXPENSE_MAX = 0.10
EXPENSE_MIN = 0.0
TURNOVER_MAX = 20.0
TURNOVER_MIN = 0.0

_OUT_COLS = ["series_id", "class_id", "filing_date", "expense_ratio_net",
            "expense_ratio_gross", "portfolio_turnover", "fee_source"]


def _iso_filed(yyyymmdd: str) -> str:
    s = str(yyyymmdd).strip()
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def _read_num_filtered(z: zipfile.ZipFile, series_filter) -> pd.DataFrame:
    """Stream num.tsv in chunks, keeping only our three tags (and, if given, our universe
    series) — the memory-sane parse the acquisition report calls for."""
    frames = []
    with z.open("num.tsv") as f:
        for chunk in pd.read_csv(
            f, sep="\t", dtype=str, chunksize=500_000,
            usecols=["adsh", "tag", "ddate", "series", "class", "value"],
            na_filter=False, encoding="utf-8", encoding_errors="replace",
        ):
            keep = chunk["tag"].isin(TAGS)
            if series_filter is not None:
                keep &= chunk["series"].isin(series_filter)
            if keep.any():
                frames.append(chunk[keep])
    if not frames:
        return pd.DataFrame(columns=["adsh", "tag", "ddate", "series", "class", "value"])
    return pd.concat(frames, ignore_index=True)


def parse_rr_quarter(zip_path, cfg: dict, series_filter=None) -> pd.DataFrame:
    """Parse one RR quarterly ZIP into rr_fees_raw rows (one per filing x share class,
    plus series-level turnover rows). `series_filter` (a set of series ids) restricts to our
    universe; None keeps everything (used by the unit tests)."""
    with zipfile.ZipFile(zip_path) as z:
        num = _read_num_filtered(z, series_filter)
        sub = pd.read_csv(z.open("sub.tsv"), sep="\t", dtype=str,
                          usecols=["adsh", "filed"], na_filter=False,
                          encoding="utf-8", encoding_errors="replace")

    if num.empty:
        return pd.DataFrame(columns=_OUT_COLS)

    num["value"] = pd.to_numeric(num["value"], errors="coerce")
    num = num[num["value"].notna()]
    num = num[num["ddate"].str.len() == 8]

    # Dedup within (adsh, series, class, tag): keep max ddate, tie-break max value.
    num = (num.sort_values(["ddate", "value"])
              .drop_duplicates(["adsh", "series", "class", "tag"], keep="last"))

    # Pivot tags to columns, per (adsh, series, class).
    wide = num.pivot_table(index=["adsh", "series", "class"], columns="tag",
                           values="value", aggfunc="last").reset_index()
    for t in TAGS:
        if t not in wide.columns:
            wide[t] = np.nan

    filed = dict(zip(sub["adsh"], sub["filed"]))
    wide["filing_date"] = wide["adsh"].map(filed).map(
        lambda v: _iso_filed(v) if isinstance(v, str) and len(v) == 8 else None)

    # Net-else-gross fallback + provenance.
    net = wide[NET_TAG]
    gross = wide[GROSS_TAG]
    expense_net = net.where(net.notna(), gross)
    fee_source = np.where(net.notna(), "net", np.where(gross.notna(), "gross", None))

    out = pd.DataFrame({
        "series_id": wide["series"],
        "class_id": wide["class"].replace("", None),
        "filing_date": wide["filing_date"],
        "expense_ratio_net": expense_net.values,
        "expense_ratio_gross": gross.values,
        "portfolio_turnover": wide[TURNOVER_TAG].values,
        "fee_source": fee_source,
    })

    # Sanity-bound: null out-of-range VALUES (keep the row if anything else survives).
    n_exp_bad = int(((out["expense_ratio_net"].notna())
                     & ~out["expense_ratio_net"].between(EXPENSE_MIN, EXPENSE_MAX)).sum())
    n_grs_bad = int(((out["expense_ratio_gross"].notna())
                     & ~out["expense_ratio_gross"].between(EXPENSE_MIN, EXPENSE_MAX)).sum())
    n_trn_bad = int(((out["portfolio_turnover"].notna())
                     & ~out["portfolio_turnover"].between(TURNOVER_MIN, TURNOVER_MAX)).sum())

    bad_net = out["expense_ratio_net"].notna() & ~out["expense_ratio_net"].between(EXPENSE_MIN, EXPENSE_MAX)
    out.loc[bad_net, ["expense_ratio_net", "fee_source"]] = None
    out.loc[out["expense_ratio_gross"].notna()
            & ~out["expense_ratio_gross"].between(EXPENSE_MIN, EXPENSE_MAX), "expense_ratio_gross"] = np.nan
    out.loc[out["portfolio_turnover"].notna()
            & ~out["portfolio_turnover"].between(TURNOVER_MIN, TURNOVER_MAX), "portfolio_turnover"] = np.nan

    # Keep rows with at least one surviving value.
    has_value = (out["expense_ratio_net"].notna() | out["expense_ratio_gross"].notna()
                 | out["portfolio_turnover"].notna())
    out = out[has_value].reset_index(drop=True)

    if n_exp_bad or n_grs_bad or n_trn_bad:
        log.info(f"  sanity-bound nulled: {n_exp_bad} net, {n_grs_bad} gross, "
                 f"{n_trn_bad} turnover values out of range")
    return out[_OUT_COLS]


def _strategy_series(cfg: dict) -> set:
    funds = load_table("funds_full", cfg)
    strat = funds[funds["is_us_equity"] & (funds["segment"] == "strategy")]
    return set(strat["series_id"].unique())


def run_parse(cfg: dict) -> pd.DataFrame:
    """Parse every cached RR ZIP for cfg['fees']['rr_years'] into `rr_fees_raw`, restricted
    to our strategy universe. Returns the combined frame (also saved)."""
    series_filter = _strategy_series(cfg)
    log.info(f"parsing RR quarters for {len(series_filter)} strategy series")
    rdir = raw_dir(cfg)
    frames = []
    for year in cfg["fees"]["rr_years"]:
        for q in (1, 2, 3, 4):
            zp = rr_zip_path(year, q, cfg)
            if not zp.exists():
                continue
            df = parse_rr_quarter(zp, cfg, series_filter=series_filter)
            log.info(f"{zp.name}: {len(df)} rows "
                     f"({df['expense_ratio_net'].notna().sum()} net, "
                     f"{df['portfolio_turnover'].notna().sum()} turnover)")
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_OUT_COLS)
    save_table(combined, "rr_fees_raw", cfg)
    n_series = combined["series_id"].nunique()
    med_net = combined["expense_ratio_net"].median()
    log.info(f"rr_fees_raw: {len(combined)} rows, {n_series} distinct series; "
             f"median net expense {med_net:.4f} (equity-fund sanity: 0.2%-1.5%)")
    return combined
