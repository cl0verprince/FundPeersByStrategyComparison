"""step9_fees_turnover/build.py — plain sequential orchestrator: acquire -> parse -> fees
-> evaluate. NOT wired into conductor.py (same rationale as steps 6-10). Run on demand:

    python -m steps.step9_fees_turnover.build

acquire skips already-cached ZIPs; parse/fees/evaluate are deterministic. Keep the DuckDB
exclusive while this runs (it writes rr_fees_raw, rr_fees, fees_model_eval,
fees_feature_importances).
"""
import logging

from steps.step9_fees_turnover.acquire import download_all
from steps.step9_fees_turnover.evaluate import run_evaluation
from steps.step9_fees_turnover.fees import build_rr_fees
from steps.step9_fees_turnover.parse import run_parse

log = logging.getLogger(__name__)


def run(cfg: dict) -> dict:
    log.info("=== step9: acquire RR ZIPs (skip-cached) ===")
    download_all(cfg)

    log.info("=== step9: parse RR filings -> rr_fees_raw ===")
    run_parse(cfg)

    log.info("=== step9: point-in-time rr_fees + coverage ===")
    coverage = build_rr_fees(cfg)

    log.info("=== step9: evaluation (with-fees model, baselines, fund-disjoint) ===")
    evaluation = run_evaluation(cfg)

    return {"coverage": coverage, "evaluation": evaluation}


if __name__ == "__main__":
    from fundspeers.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(load_config())
