"""step13_automation.advance — deterministic conductor for one quarterly refresh.

Run on demand (the scheduled cloud routine invokes this same entry point):

    python -m steps.step13_automation.advance              # probe + full refresh, no push
    python -m steps.step13_automation.advance --dry-run     # print the plan, write nothing
    python -m steps.step13_automation.advance --push        # refresh AND push the commit

RUNBOOK (the design's 8-stage flow — this docstring IS the checklist):

  1. PROBE     Is a NEW N-PORT quarter published beyond data.quarters' last entry?
               HEAD the next quarter's N-PORT ZIP (ingest's URL pattern + SEC UA).
               Nothing new -> log and exit 0; the routine treats that as success.
  2. EXTEND    Append the new quarter to data.quarters (and fees.rr_years if the year is
               new). config.json is rewritten in place (targeted insert, valid JSON);
               cfg is reloaded from disk so every downstream stage sees the new quarter.
  3. INGEST    Re-ingest _full with the step10 parameters: relaxed_pool=True,
               reuse_metadata_from INCLUDING funds_full itself (skips Yahoo for the
               ~137k already-resolved series), max_funds_override=0 (uncapped). The ZIP
               cache makes this incremental; Yahoo is hit only for genuinely new candidates.
  4. CLUSTER   similarity.run(_full, n_clusters=cfg["full"]["n_clusters"], top_n_peers=15,
               require_segment="strategy", save_coords=True)  --  DORMANT-CONFIG TRAP:
               n_clusters MUST come from full.n_clusters (40), NOT similarity.n_clusters
               (15, the default this call would otherwise pick up). Then metrics.run(_full).
  5. FEES      step9 acquire (new RR quarter) -> parse -> point-in-time rr_fees join.
  6. EVALUATE  step10 build.run (validation-first out-of-time scoring of the previous
               official model on the newly-realized quarter, then retrain) + step9's
               with-fees evaluation.
  7. DASHBOARD step8 build.run (narratives from cache; new clusters get placeholders when
               LM Studio is down — never a failure).
  8. DOCS      Append the quarter to a rolling refresh_log table, then commit everything
               with a standard message. Never push unless --push (push=True) is set.

NO scheduling code lives here (the cloud routine is created separately via /schedule).
Keep the DuckDB exclusive to this process while a real refresh runs.
"""
import argparse
import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from fundspeers.config import PROJECT_ROOT, load_config, load_env
from fundspeers.io import save_table
from steps.step1_ingest import ingest
from steps.step1_ingest.ingest import NPORT_URL
from steps.step2_similarity import similarity
from steps.step3_metrics import metrics
from steps.step8_dashboard import build as dashboard_build
from steps.step9_fees_turnover import acquire as fees_acquire
from steps.step9_fees_turnover import evaluate as fees_evaluate
from steps.step9_fees_turnover import fees as fees_fees
from steps.step9_fees_turnover import parse as fees_parse
from steps.step10_full_universe import build as full_build

log = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_ROOT / "config.json"

# Yahoo-metadata reuse sources, in priority order (first wins on conflict). funds_full — the
# just-completed full universe — comes first so the advance path skips Yahoo for every series
# already resolved; ingest guards missing tables with table_exists, so older fallbacks are
# harmless if absent.
REUSE_METADATA_FROM = ["funds_full", "funds_all", "funds"]

STAGE_PLAN = [
    "extend config (data.quarters, fees.rr_years)",
    "ingest _full (relaxed pool, reuse incl. funds_full, uncapped)",
    "cluster + metrics (_full, n_clusters=full.n_clusters, top_n_peers=15)",
    "fees refresh (acquire -> parse -> rr_fees join)",
    "re-evaluate (step10 out-of-time + retrain; step9 with-fees eval)",
    "dashboard rebuild (step8)",
    "docs: refresh_log + commit (push gated by --push)",
]


def next_quarter_after(quarter: str) -> str:
    """"YYYYqQ" -> the immediately following calendar quarter, rolling the year over at q4:
    "2026q1" -> "2026q2"; "2026q4" -> "2027q1". Pure/deterministic (no I/O)."""
    year_str, q_str = quarter.lower().split("q")
    year, q = int(year_str), int(q_str)
    if q < 1 or q > 4:
        raise ValueError(f"quarter out of range: {quarter!r}")
    if q == 4:
        return f"{year + 1}q1"
    return f"{year}q{q + 1}"


def probe_next_quarter(cfg: dict) -> str | None:
    """The next quarter after cfg['data']['quarters'][-1], IF the SEC has published its
    N-PORT bulk ZIP; otherwise None. A HEAD request on the N-PORT URL (same pattern + SEC
    User-Agent as step1 ingest): 200 -> published (return the quarter), 404 -> not yet
    (return None). Any other status raises — a 403/429 rate-limit must NOT be misread as
    "nothing new" and silently swallowed by the caller."""
    next_quarter = next_quarter_after(cfg["data"]["quarters"][-1])
    url = NPORT_URL.format(quarter=next_quarter)
    ua = load_env()["SEC_USER_AGENT"]
    if not ua:
        raise RuntimeError("SEC_USER_AGENT must be set in .env (see .env.example)")
    log.info("probing %s for a published N-PORT data set: %s", next_quarter, url)
    resp = requests.head(url, headers={"User-Agent": ua}, timeout=60, allow_redirects=True)
    if resp.status_code == 200:
        log.info("%s is published", next_quarter)
        return next_quarter
    if resp.status_code == 404:
        log.info("%s not published yet (404)", next_quarter)
        return None
    resp.raise_for_status()
    raise RuntimeError(f"unexpected status {resp.status_code} probing {url}")


def _insert_into_json_array(text: str, key: str, quoted_item: str) -> str:
    """Append `quoted_item` (already a JSON token, e.g. '"2026q2"' or '2027') to the JSON
    array under top-level-ish `key`, in place, preserving the file's formatting. Inserts
    right after the array's last existing element so the surrounding layout is untouched."""
    m = re.search(rf'("{key}"\s*:\s*\[)(.*?)(\])', text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"could not locate array for key {key!r} in config")
    prefix, body, suffix = m.group(1), m.group(2), m.group(3)
    # Find the last non-whitespace char of the body (the final element or its comma).
    stripped = body.rstrip()
    trailing_ws = body[len(stripped):]
    new_body = f"{stripped}, {quoted_item}{trailing_ws}"
    return text[:m.start()] + prefix + new_body + suffix + text[m.end():]


def extend_config(config_path: Path, new_quarter: str) -> None:
    """Rewrite config.json in place: append `new_quarter` to data.quarters, and its year to
    fees.rr_years when that year is not already present. Targeted string insert (formatting
    preserved); the result is parsed to guarantee it stays valid JSON."""
    config_path = Path(config_path)
    text = config_path.read_text(encoding="utf-8")
    current = json.loads(text)

    if new_quarter in current["data"]["quarters"]:
        raise ValueError(f"{new_quarter} already present in data.quarters")
    text = _insert_into_json_array(text, "quarters", f'"{new_quarter}"')

    new_year = int(new_quarter.split("q")[0])
    if new_year not in current.get("fees", {}).get("rr_years", []):
        text = _insert_into_json_array(text, "rr_years", str(new_year))

    parsed = json.loads(text)  # fail loudly if the insert broke JSON
    assert new_quarter in parsed["data"]["quarters"]
    config_path.write_text(text, encoding="utf-8")
    log.info("config extended: data.quarters += %s (rr_years covers %d)", new_quarter, new_year)


# --- stage functions (thin wrappers; each a single module attribute so tests can stub) ---


def _stage_ingest(cfg: dict) -> None:
    ingest.run(cfg, table_suffix="_full", relaxed_pool=True,
               reuse_metadata_from=REUSE_METADATA_FROM, max_funds_override=0)


def _stage_cluster_and_metrics(cfg: dict) -> None:
    similarity.run(cfg, table_suffix="_full", n_clusters=cfg["full"]["n_clusters"],
                   top_n_peers=15, require_segment="strategy", save_coords=True)
    metrics.run(cfg, table_suffix="_full")


def _stage_fees(cfg: dict) -> dict:
    fees_acquire.download_all(cfg)
    fees_parse.run_parse(cfg)
    return fees_fees.build_rr_fees(cfg)


def _stage_evaluate(cfg: dict) -> dict:
    full_build.run(cfg)
    return fees_evaluate.run_evaluation(cfg)


def _stage_dashboard(cfg: dict) -> None:
    # The refresh operates on the _full universe end-to-end; the dashboard must too
    # (the defaults would silently rebuild the stale small-universe _all dashboard).
    dashboard_build.run(cfg, narrative_mode="cached", table_suffix="_full",
                        predictions_table="full_predictions",
                        eval_table="full_model_eval",
                        stability_table="full_label_stability")


def _stage_docs_and_commit(cfg: dict, new_quarter: str, summary: dict, push: bool) -> None:
    """Append one row to the rolling refresh_log table, then commit everything. Push is
    gated: never pushes unless push=True (the human reviews the refresh commit first)."""
    row = {"quarter": new_quarter,
           "refreshed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for k, v in summary.items():
        if isinstance(v, (int, float, str, bool)):
            row[f"summary_{k}"] = v
    save_table(pd.DataFrame([row]), "refresh_log", cfg)
    _commit(new_quarter, push)


def _commit(new_quarter: str, push: bool) -> None:
    msg = f"step13: automated quarterly refresh - {new_quarter}"
    subprocess.run(["git", "add", "-A"], cwd=PROJECT_ROOT, check=True)
    subprocess.run(["git", "commit", "-m", msg], cwd=PROJECT_ROOT, check=True)
    if push:
        subprocess.run(["git", "push"], cwd=PROJECT_ROOT, check=True)
        log.info("pushed refresh commit for %s", new_quarter)
    else:
        log.info("committed refresh for %s (not pushed; pass --push to publish)", new_quarter)


def run(cfg: dict, dry_run: bool = False, push: bool = False,
        config_path: Path = None) -> dict:
    """Execute one quarterly refresh (the design's 8-stage flow). Returns a status dict.

    dry_run=True prints the plan (which quarter, which stages) and returns WITHOUT any write
    — including no config mutation. "No new quarter" returns {"status": "up_to_date"}."""
    config_path = Path(config_path) if config_path else CONFIG_PATH

    new_quarter = probe_next_quarter(cfg)
    if new_quarter is None:
        log.info("no new N-PORT quarter beyond %s — nothing to do",
                 cfg["data"]["quarters"][-1])
        return {"status": "up_to_date", "quarter": None}

    if dry_run:
        log.info("DRY RUN — would refresh for %s; no writes performed.", new_quarter)
        for i, stage in enumerate(STAGE_PLAN, start=1):
            log.info("  plan step %d: %s", i, stage)
        return {"status": "dry_run", "quarter": new_quarter, "stages": STAGE_PLAN}

    # Stage 2: extend config on disk, then reload so every stage sees the new quarter.
    log.info("=== step13 [1/8]: new quarter %s — extending config ===", new_quarter)
    extend_config(config_path, new_quarter)
    cfg = load_config(config_path)

    log.info("=== step13 [2/8]: re-ingest _full ===")
    _stage_ingest(cfg)

    log.info("=== step13 [3/8]: re-cluster + metrics (_full) ===")
    _stage_cluster_and_metrics(cfg)

    log.info("=== step13 [4/8]: fees refresh ===")
    coverage = _stage_fees(cfg)

    log.info("=== step13 [5/8]: re-evaluate (out-of-time + retrain; with-fees eval) ===")
    evaluation = _stage_evaluate(cfg)

    log.info("=== step13 [6/8]: dashboard rebuild ===")
    _stage_dashboard(cfg)

    log.info("=== step13 [7/8]: docs + commit ===")
    summary = {"fees_coverage": coverage.get("coverage") if isinstance(coverage, dict) else None}
    _stage_docs_and_commit(cfg, new_quarter, summary, push)

    log.info("=== step13 [8/8]: refresh complete for %s ===", new_quarter)
    return {"status": "refreshed", "quarter": new_quarter,
            "coverage": coverage, "evaluation": evaluation, "pushed": push}


def main() -> None:
    ap = argparse.ArgumentParser(description="Advance the pipeline by one quarter.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (quarter + stages) and write nothing")
    ap.add_argument("--push", action="store_true",
                    help="push the refresh commit (default: commit only, no push)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run(load_config(), dry_run=args.dry_run, push=args.push)
    log.info("result: %s", result)


if __name__ == "__main__":
    main()
