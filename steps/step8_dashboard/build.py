"""step8_dashboard/build.py - deterministic dashboard build.

    python -m steps.step8_dashboard.build                       # narratives from cache,
                                                                 # generating only misses
    python -m steps.step8_dashboard.build --skip-narratives     # no LM Studio needed
    python -m steps.step8_dashboard.build --regenerate-narratives
"""
import argparse
import logging

from fundspeers.config import load_config
from steps.step8_dashboard.data import build_payload
from steps.step8_dashboard.narratives import get_narratives
from steps.step8_dashboard.render import write_dashboard

log = logging.getLogger(__name__)


def run(cfg: dict, narrative_mode: str = "cached") -> None:
    payload = build_payload(cfg, narratives={})
    narratives = get_narratives(cfg, payload["clusters"], mode=narrative_mode,
                                quarter=payload["universe"]["latest_quarter"])
    payload = build_payload(cfg, narratives=narratives)
    out = write_dashboard(payload, cfg)
    log.info(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
             f"{len(payload['clusters'])} clusters, "
             f"{payload['universe']['n_funds']} funds)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--skip-narratives", action="store_true")
    group.add_argument("--regenerate-narratives", action="store_true")
    args = ap.parse_args()
    mode = ("skip" if args.skip_narratives
            else "regenerate" if args.regenerate_narratives else "cached")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(load_config(), narrative_mode=mode)
