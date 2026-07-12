"""step8_dashboard/build.py - deterministic dashboard build.

    python -m steps.step8_dashboard.build                       # narratives from cache,
                                                                 # generating only misses
    python -m steps.step8_dashboard.build --skip-narratives     # no LM Studio needed
    python -m steps.step8_dashboard.build --regenerate-narratives
    python -m steps.step8_dashboard.build --table-suffix _full --prefix full  # full universe
"""
import argparse
import logging

from fundspeers.config import load_config
from steps.step8_dashboard.data import build_payload
from steps.step8_dashboard.narratives import get_narratives
from steps.step8_dashboard.render import write_dashboard

log = logging.getLogger(__name__)


def run(cfg: dict, narrative_mode: str = "cached", table_suffix: str = "_all",
        predictions_table: str = "unified_predictions",
        eval_table: str = "unified_model_eval",
        stability_table: str = "unified_label_stability") -> None:
    kw = dict(table_suffix=table_suffix, predictions_table=predictions_table,
              eval_table=eval_table, stability_table=stability_table)
    payload = build_payload(cfg, narratives={}, **kw)
    narratives = get_narratives(cfg, payload["clusters"], mode=narrative_mode,
                                quarter=payload["universe"]["latest_quarter"])
    payload = build_payload(cfg, narratives=narratives, **kw)
    out = write_dashboard(payload, cfg)
    log.info(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
             f"{len(payload['clusters'])} clusters, "
             f"{payload['universe']['n_funds']} funds)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--skip-narratives", action="store_true")
    group.add_argument("--regenerate-narratives", action="store_true")
    ap.add_argument("--table-suffix", default="_all",
                    help="universe-table suffix, e.g. _all (default) or _full")
    ap.add_argument("--prefix", default="unified",
                    help="prefix for the predictions/eval/stability tables: "
                         "'unified' (default) -> unified_predictions/_model_eval/_label_stability; "
                         "'full' -> full_predictions/_model_eval/_label_stability")
    ap.add_argument("--predictions-table", default=None,
                    help="explicit predictions table name (overrides --prefix)")
    ap.add_argument("--eval-table", default=None,
                    help="explicit model-eval table name (overrides --prefix)")
    ap.add_argument("--stability-table", default=None,
                    help="explicit label-stability table name (overrides --prefix)")
    args = ap.parse_args()
    mode = ("skip" if args.skip_narratives
            else "regenerate" if args.regenerate_narratives else "cached")
    predictions_table = args.predictions_table or f"{args.prefix}_predictions"
    eval_table = args.eval_table or f"{args.prefix}_model_eval"
    stability_table = args.stability_table or f"{args.prefix}_label_stability"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(load_config(), narrative_mode=mode, table_suffix=args.table_suffix,
        predictions_table=predictions_table, eval_table=eval_table,
        stability_table=stability_table)
