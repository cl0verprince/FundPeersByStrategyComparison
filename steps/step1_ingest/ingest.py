"""step1_ingest — build funds/holdings/monthly_returns tables from SEC N-PORT + Yahoo Finance.

See steps/step1_ingest/design.md for the staged approach and why each stage exists.
"""
import logging
import random
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from fundspeers.config import load_env
from fundspeers.io import raw_dir, save_table

log = logging.getLogger(__name__)

NPORT_URL = "https://www.sec.gov/files/dera/data/form-n-port-data-sets/{quarter}_nport.zip"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers_mf.json"

FUND_LEVEL_TABLES = ["SUBMISSION.tsv", "REGISTRANT.tsv", "FUND_REPORTED_INFO.tsv"]
RETURN_TABLE = "MONTHLY_TOTAL_RETURN.tsv"
HOLDING_TABLE = "FUND_REPORTED_HOLDING.tsv"


def _headers() -> dict:
    ua = load_env()["SEC_USER_AGENT"]
    if not ua:
        raise RuntimeError("SEC_USER_AGENT must be set in .env (see .env.example)")
    return {"User-Agent": ua}


def _download_quarter_zip(quarter: str, cfg: dict) -> Path:
    dest = raw_dir(cfg) / f"{quarter}_nport.zip"
    if dest.exists():
        return dest
    url = NPORT_URL.format(quarter=quarter)
    log.info(f"downloading {url}")
    resp = requests.get(url, headers=_headers(), timeout=300)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def _read_tsv_from_zip(zip_path: Path, member: str, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        with z.open(member) as f:
            return pd.read_csv(f, sep="\t", dtype=str, low_memory=False, **kwargs)


def _dedupe_one_filing_per_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep exactly one filing per SERIES_ID. A small number of series file more than
    once in the same quarter (restated/refiled reports not marked as an amendment) -
    keep the lowest ACCESSION_NUMBER deterministically, so each series contributes
    exactly one row per quarter downstream (funds panel, returns, holdings)."""
    return frame.sort_values("ACCESSION_NUMBER").drop_duplicates("SERIES_ID", keep="first")


def _load_fund_level_frame(zip_path: Path, quarter: str) -> pd.DataFrame:
    submission = _read_tsv_from_zip(
        zip_path, "SUBMISSION.tsv",
        usecols=["ACCESSION_NUMBER", "REPORT_DATE", "SUB_TYPE"],
    )
    registrant = _read_tsv_from_zip(
        zip_path, "REGISTRANT.tsv",
        usecols=["ACCESSION_NUMBER", "CIK", "REGISTRANT_NAME"],
    )
    fund_info = _read_tsv_from_zip(
        zip_path, "FUND_REPORTED_INFO.tsv",
        usecols=["ACCESSION_NUMBER", "SERIES_NAME", "SERIES_ID", "NET_ASSETS"],
    )
    frame = submission.merge(registrant, on="ACCESSION_NUMBER", how="inner")
    frame = frame.merge(fund_info, on="ACCESSION_NUMBER", how="inner")
    frame = frame[frame["SUB_TYPE"] == "NPORT-P"].drop(columns=["SUB_TYPE"])
    frame["quarter"] = quarter
    frame["NET_ASSETS"] = pd.to_numeric(frame["NET_ASSETS"], errors="coerce")
    return _dedupe_one_filing_per_series(frame)


def _aggregate_class_returns_to_series(returns: pd.DataFrame, quarter: str) -> pd.DataFrame:
    """Collapse per-CLASS_ID monthly returns to one row per series per month, via the
    mean across classes present that quarter (see design.md - share-class net-asset
    weights aren't available in the bulk data set, so mean is the documented simplification)."""
    for col in ["MONTHLY_TOTAL_RETURN1", "MONTHLY_TOTAL_RETURN2", "MONTHLY_TOTAL_RETURN3"]:
        returns[col] = pd.to_numeric(returns[col], errors="coerce")

    per_series = returns.groupby("SERIES_ID")[
        ["MONTHLY_TOTAL_RETURN1", "MONTHLY_TOTAL_RETURN2", "MONTHLY_TOTAL_RETURN3"]
    ].mean().reset_index()

    rows = []
    for _, row in per_series.iterrows():
        for month_in_quarter, col in enumerate(
            ["MONTHLY_TOTAL_RETURN1", "MONTHLY_TOTAL_RETURN2", "MONTHLY_TOTAL_RETURN3"], start=1
        ):
            rows.append({
                "series_id": row["SERIES_ID"],
                "quarter": quarter,
                "month_in_quarter": month_in_quarter,
                "total_return": row[col],
            })
    return pd.DataFrame(rows)


def _load_returns_frame(zip_path: Path, fund_level: pd.DataFrame, quarter: str) -> pd.DataFrame:
    returns = _read_tsv_from_zip(
        zip_path, RETURN_TABLE,
        usecols=["ACCESSION_NUMBER", "CLASS_ID",
                 "MONTHLY_TOTAL_RETURN1", "MONTHLY_TOTAL_RETURN2", "MONTHLY_TOTAL_RETURN3"],
    )
    returns = returns.merge(
        fund_level[["ACCESSION_NUMBER", "SERIES_ID"]], on="ACCESSION_NUMBER", how="inner"
    )
    return _aggregate_class_returns_to_series(returns, quarter)


def _load_ticker_map(cfg: dict) -> pd.DataFrame:
    cache = raw_dir(cfg) / "company_tickers_mf.json"
    if not cache.exists():
        resp = requests.get(TICKER_MAP_URL, headers=_headers(), timeout=60)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
    import json
    data = json.loads(cache.read_text(encoding="utf-8"))
    return pd.DataFrame(data["data"], columns=data["fields"])


_PERMANENT_MISS_MARKERS = ("no fund data found", "404")


def _fetch_yahoo_fund_data(ticker: str, cfg: dict) -> dict | None:
    delay = cfg["data"]["yahoo_request_delay_seconds"]
    max_retries = cfg["data"]["yahoo_max_retries"]
    for attempt in range(max_retries):
        try:
            fd = yf.Ticker(ticker).funds_data
            overview = fd.fund_overview or {}
            asset_classes = fd.asset_classes or {}
            time.sleep(delay)
            return {
                "yahoo_category": overview.get("categoryName"),
                "yahoo_stock_position": asset_classes.get("stockPosition"),
            }
        except Exception as exc:  # noqa: BLE001 - yfinance raises assorted exceptions
            log.warning(f"yahoo lookup failed for {ticker} (attempt {attempt + 1}): {exc}")
            # "No Fund data found" / 404 are permanent misses (delisted, wrong ticker, not a
            # fund) - retrying never helps and just burns ~2s/candidate across ~40% attrition.
            # Only back off and retry for other (plausibly transient) errors.
            if any(marker in str(exc).lower() for marker in _PERMANENT_MISS_MARKERS):
                return None
            time.sleep(delay * (attempt + 1))
    return None


def _load_holdings_for_universe(
    zip_path: Path, accession_numbers: set, quarter: str
) -> pd.DataFrame:
    chunks = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open(HOLDING_TABLE) as f:
            reader = pd.read_csv(
                f, sep="\t", dtype=str, low_memory=False, chunksize=200_000,
                usecols=["ACCESSION_NUMBER", "HOLDING_ID", "ISSUER_NAME", "ASSET_CAT",
                         "ISSUER_TYPE", "INVESTMENT_COUNTRY", "PERCENTAGE", "CURRENCY_VALUE"],
            )
            for chunk in reader:
                matched = chunk[chunk["ACCESSION_NUMBER"].isin(accession_numbers)]
                if not matched.empty:
                    chunks.append(matched)
    if not chunks:
        return pd.DataFrame(columns=[
            "accession_number", "holding_id", "issuer_name", "asset_cat",
            "issuer_type", "investment_country", "percentage", "currency_value", "quarter",
        ])
    holdings = pd.concat(chunks, ignore_index=True)
    holdings = holdings.rename(columns={
        "ACCESSION_NUMBER": "accession_number", "HOLDING_ID": "holding_id",
        "ISSUER_NAME": "issuer_name", "ASSET_CAT": "asset_cat", "ISSUER_TYPE": "issuer_type",
        "INVESTMENT_COUNTRY": "investment_country", "PERCENTAGE": "percentage",
        "CURRENCY_VALUE": "currency_value",
    })
    holdings["percentage"] = pd.to_numeric(holdings["percentage"], errors="coerce")
    holdings["currency_value"] = pd.to_numeric(holdings["currency_value"], errors="coerce")
    holdings["quarter"] = quarter
    return holdings


def _compute_holdings_us_equity_share(holdings: pd.DataFrame) -> pd.DataFrame:
    """Per accession_number, the value-weighted share of holdings that are US-domiciled
    common equity (ASSET_CAT=='EC' and INVESTMENT_COUNTRY=='US'), out of total portfolio
    value. Vectorized (groupby+sum, no per-group Python calls) - a naive groupby().apply()
    version was measured to be the pipeline's dominant cost on ~4M holdings rows."""
    is_us_equity_holding = (holdings["asset_cat"] == "EC") & (holdings["investment_country"] == "US")
    us_value = is_us_equity_holding * holdings["currency_value"].fillna(0.0)
    per_accession = holdings.assign(us_value=us_value).groupby("accession_number").agg(
        us_value=("us_value", "sum"), total_value=("currency_value", "sum")
    )
    return (
        (per_accession["us_value"] / per_accession["total_value"]).fillna(0.0)
        .rename("holdings_us_equity_share")
        .reset_index()
    )


def _combine_us_equity_flag(
    funds_static: pd.DataFrame,
    holdings: pd.DataFrame,
    accession_to_series: pd.DataFrame,
    us_holdings_share_min: float,
) -> pd.DataFrame:
    """Combine Yahoo's asset-class signal (`yahoo_is_equity`) with a holdings-based
    geography check to produce the final `is_us_equity` flag. yahoo_stock_position alone
    cannot distinguish a US equity fund from an international/emerging-market one - both
    can show stockPosition close to 1.0 (verified empirically). Geography must come from
    N-PORT holdings. See design.md for the known fund-of-funds edge case this doesn't catch."""
    holdings_us_share_by_accession = _compute_holdings_us_equity_share(holdings)
    holdings_us_share_by_series = (
        holdings_us_share_by_accession.merge(accession_to_series, on="accession_number", how="inner")
        .groupby("series_id")["holdings_us_equity_share"].mean()
        .reset_index()
    )
    funds_static = funds_static.merge(holdings_us_share_by_series, on="series_id", how="left")
    funds_static["is_us_equity"] = (
        funds_static["yahoo_is_equity"]
        & (funds_static["holdings_us_equity_share"] >= us_holdings_share_min)
    )
    return funds_static


def run(cfg: dict) -> None:
    quarters = cfg["data"]["quarters"]
    max_funds = cfg["data"]["max_funds"]
    stock_position_min = cfg["data"]["equity_stock_position_min"]

    # Stage 1-2: fund-level + returns tables, all quarters (small tables).
    fund_level_by_quarter = {}
    returns_frames = []
    for quarter in quarters:
        zip_path = _download_quarter_zip(quarter, cfg)
        fund_level = _load_fund_level_frame(zip_path, quarter)
        fund_level_by_quarter[quarter] = fund_level
        returns_frames.append(_load_returns_frame(zip_path, fund_level, quarter))
        log.info(f"{quarter}: {len(fund_level)} fund filings loaded")

    monthly_returns = pd.concat(returns_frames, ignore_index=True)

    # Stage 3: panel filter - series present in every configured quarter.
    series_per_quarter = [
        set(frame["SERIES_ID"].dropna()) for frame in fund_level_by_quarter.values()
    ]
    complete_panel_series = sorted(set.intersection(*series_per_quarter))
    log.info(f"{len(complete_panel_series)} series present in all {len(quarters)} quarters")

    # Stage 4-6: shuffle the full candidate pool deterministically, then resolve
    # ticker + Yahoo data one candidate at a time until `max_funds` are resolved
    # (or the pool is exhausted). A fixed-size pre-filter sample would under-shoot
    # max_funds once ticker-mapping and Yahoo-lookup attrition are accounted for.
    t_yahoo_start = time.time()
    rng = random.Random(cfg["seed"])
    candidate_order = complete_panel_series.copy()
    rng.shuffle(candidate_order)

    ticker_map = _load_ticker_map(cfg)
    ticker_map = ticker_map.sort_values("classId").drop_duplicates("seriesId", keep="first")
    series_to_ticker = dict(zip(ticker_map["seriesId"], ticker_map["symbol"]))

    funds_rows = []
    attempted = 0
    for series_id in candidate_order:
        if len(funds_rows) >= max_funds:
            break
        attempted += 1
        ticker = series_to_ticker.get(series_id)
        if not ticker:
            continue
        yahoo = _fetch_yahoo_fund_data(ticker, cfg)
        if yahoo is None or yahoo["yahoo_stock_position"] is None:
            log.info(f"{series_id} ({ticker}): yahoo lookup failed, dropping from universe")
            continue
        funds_rows.append({
            "series_id": series_id,
            "ticker": ticker,
            "yahoo_category": yahoo["yahoo_category"],
            "yahoo_stock_position": yahoo["yahoo_stock_position"],
            "yahoo_is_equity": yahoo["yahoo_stock_position"] >= stock_position_min,
        })
        if len(funds_rows) % 50 == 0:
            log.info(f"resolved {len(funds_rows)}/{max_funds} funds "
                     f"({attempted} candidates attempted so far)")

    funds_static = pd.DataFrame(funds_rows)
    log.info(f"{len(funds_static)} series resolved via Yahoo out of {attempted} attempted "
             f"({len(candidate_order)} candidates available) in {time.time() - t_yahoo_start:.1f}s; "
             f"{funds_static['yahoo_is_equity'].sum()} flagged equity (asset-class only, "
             f"geography resolved from holdings below)")
    if len(funds_static) < max_funds:
        log.warning(
            f"only resolved {len(funds_static)} funds, short of the {max_funds} target "
            f"- the candidate pool ({len(candidate_order)}) was exhausted"
        )

    final_series = set(funds_static["series_id"])

    # Attach per-quarter panel rows (series_name, cik, net_assets, accession_number).
    panel_rows = []
    for quarter, frame in fund_level_by_quarter.items():
        subset = frame[frame["SERIES_ID"].isin(final_series)]
        for _, row in subset.iterrows():
            panel_rows.append({
                "series_id": row["SERIES_ID"],
                "series_name": row["SERIES_NAME"],
                "cik": row["CIK"],
                "quarter": quarter,
                "accession_number": row["ACCESSION_NUMBER"],
                "net_assets": row["NET_ASSETS"],
            })
    funds_panel = pd.DataFrame(panel_rows)

    monthly_returns = monthly_returns[monthly_returns["series_id"].isin(final_series)]

    # Stage 7: holdings, filtered to our universe's accession numbers, per quarter.
    holdings_frames = []
    for quarter, frame in fund_level_by_quarter.items():
        t_quarter = time.time()
        accession_numbers = set(
            frame[frame["SERIES_ID"].isin(final_series)]["ACCESSION_NUMBER"]
        )
        if not accession_numbers:
            continue
        zip_path = raw_dir(cfg) / f"{quarter}_nport.zip"
        quarter_holdings = _load_holdings_for_universe(zip_path, accession_numbers, quarter)
        holdings_frames.append(quarter_holdings)
        log.info(f"{quarter}: {len(quarter_holdings)} holding rows loaded "
                 f"({time.time() - t_quarter:.1f}s)")
    holdings = pd.concat(holdings_frames, ignore_index=True) if holdings_frames else pd.DataFrame()

    # Stage 8: combine Yahoo's asset-class signal with a holdings-based geography check.
    us_holdings_share_min = cfg["data"]["us_holdings_share_min"]
    accession_to_series = funds_panel[["accession_number", "series_id"]]
    funds_static = _combine_us_equity_flag(
        funds_static, holdings, accession_to_series, us_holdings_share_min
    )
    funds = funds_panel.merge(funds_static, on="series_id", how="inner")
    log.info(f"{funds_static['is_us_equity'].sum()} of {len(funds_static)} series "
             f"flagged US equity (yahoo equity AND holdings US-share >= {us_holdings_share_min})")

    # Stage 9: persist.
    save_table(funds, "funds", cfg)
    save_table(holdings, "holdings", cfg)
    save_table(monthly_returns, "monthly_returns", cfg)
    log.info(
        f"saved funds ({len(funds)} rows), holdings ({len(holdings)} rows), "
        f"monthly_returns ({len(monthly_returns)} rows)"
    )
