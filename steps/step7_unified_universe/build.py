"""step7_unified_universe/build.py - deterministic orchestrator for the unified rebuild.

Run on demand (NOT wired into conductor.py - see design.md):
    python -m steps.step7_unified_universe.build
"""
import logging

from steps.step2_similarity import similarity
from steps.step3_metrics import metrics
from steps.step7_unified_universe import merge, model, stability

log = logging.getLogger(__name__)


def run(cfg: dict) -> None:
    log.info("=== step7 unified universe: merge ===")
    merge.build_unified_tables(cfg)
    log.info("=== step7: clustering + peers (k=%s, top-15 peers, strategy segment) ===",
             cfg["unified"]["n_clusters"])
    similarity.run(cfg, table_suffix="_all", n_clusters=cfg["unified"]["n_clusters"],
                   top_n_peers=15, require_segment="strategy", save_coords=True)
    log.info("=== step7: metrics ===")
    metrics.run(cfg, table_suffix="_all")
    log.info("=== step7: model + Monte Carlo evaluation ===")
    results = model.train_and_evaluate(cfg)
    log.info("=== step7: label-stability study ===")
    stability_summary = stability.run_stability(cfg)
    log.info(f"SUMMARY: {results} | stability: {stability_summary}")


if __name__ == "__main__":
    from fundspeers.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(load_config())
