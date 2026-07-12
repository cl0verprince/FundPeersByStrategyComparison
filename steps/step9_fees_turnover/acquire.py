"""step9_fees_turnover.acquire — download + cache the SEC DERA Mutual Fund Prospectus
Risk/Return Summary Data Sets (the quarterly "rr1" ZIPs).

Mirrors step1_ingest's download conventions: the SEC User-Agent comes from the environment
(.env), files stream to data/raw/, and an already-cached ZIP is never re-downloaded. The
schema of the ZIP members (sub/num/tag/... .tsv) and the exact fee/turnover tags are recorded
in design.md under "Schema amendment (verified 2026-07-12)" — Tasks 2-3 code to that.
"""
import logging
import re
import zipfile
from pathlib import Path

import requests

from fundspeers.config import load_env
from fundspeers.io import raw_dir

log = logging.getLogger(__name__)

# Verified 2026-07-12 from the landing page + a real 200 download. Note the path segment
# "mutual-fund-prospectus-risk/return-summary-data-sets" (SEC really does split "risk" and
# "return" with a slash) and the "_rr1" suffix.
RR_LANDING_URL = (
    "https://www.sec.gov/dera/data/mutual-fund-prospectus-risk-return-summary-data-sets"
)
RR_URL = (
    "https://www.sec.gov/files/dera/data/mutual-fund-prospectus-risk/"
    "return-summary-data-sets/{year}q{quarter}_rr1.zip"
)


def _headers() -> dict:
    ua = load_env()["SEC_USER_AGENT"]
    if not ua:
        raise RuntimeError("SEC_USER_AGENT must be set in .env (see .env.example)")
    return {"User-Agent": ua}


def rr_zip_path(year: int, quarter: int, cfg: dict) -> Path:
    """Local cache path for one quarter's RR ZIP, e.g. data/raw/2024q4_rr1.zip."""
    return raw_dir(cfg) / f"{year}q{quarter}_rr1.zip"


def rr_url(year: int, quarter: int) -> str:
    return RR_URL.format(year=year, quarter=quarter)


def download_rr_quarter(year: int, quarter: int, cfg: dict) -> Path:
    """Download one quarter's RR ZIP to the cache and return its path. Idempotent: an
    existing cached file is returned untouched with no network call (same skip contract as
    step1's _download_quarter_zip)."""
    dest = rr_zip_path(year, quarter, cfg)
    if dest.exists():
        return dest
    url = rr_url(year, quarter)
    log.info(f"downloading {url}")
    resp = requests.get(url, headers=_headers(), timeout=600)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    log.info(f"saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def available_quarters(cfg: dict) -> set[tuple[int, int]]:
    """The (year, quarter) pairs the SEC has actually published, parsed from the landing
    page's ZIP hrefs. Used to skip-with-log configured quarters that don't exist yet (e.g.
    unpublished 2026q2-q4) instead of hard-failing on a 404."""
    resp = requests.get(RR_LANDING_URL, headers=_headers(), timeout=60)
    resp.raise_for_status()
    pairs = set()
    for m in re.finditer(r"(\d{4})q([1-4])_rr1\.zip", resp.text):
        pairs.add((int(m.group(1)), int(m.group(2))))
    return pairs


def download_all(cfg: dict) -> list[Path]:
    """Download every quarter of every year in cfg['fees']['rr_years']. Quarters that the
    SEC has not published yet are skipped with a log line (never a hard failure), so a
    partially-published trailing year is fine. Returns the cached paths, in year/quarter
    order, for the quarters that exist."""
    years = cfg["fees"]["rr_years"]
    published = available_quarters(cfg)
    paths: list[Path] = []
    for year in years:
        for quarter in (1, 2, 3, 4):
            if (year, quarter) not in published:
                log.info(f"{year}q{quarter}: not published on the SEC landing page yet, skipping")
                continue
            paths.append(download_rr_quarter(year, quarter, cfg))
    log.info(f"RR acquisition: {len(paths)} quarterly ZIPs available/cached")
    return paths


def zip_members(zip_path: Path) -> list[str]:
    """Convenience for schema inspection: the member file names inside an RR ZIP."""
    with zipfile.ZipFile(zip_path) as z:
        return z.namelist()
