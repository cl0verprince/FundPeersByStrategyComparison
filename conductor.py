"""General conductor — runs the whole FundsPeersStrategy pipeline deterministically.

Same config.json -> same output, every run.
Run:  python conductor.py
"""
import logging
import subprocess
import sys

try:
    from tqdm import tqdm  # live, in-place progress bar
except ImportError:  # graceful fallback: still runs, just no bar
    def tqdm(it, **_):
        return it

from fundspeers.config import load_config
from fundspeers.seeding import seed_everything

logging.basicConfig(level=logging.INFO, format="%(message)s")


def build_pipeline(cfg: dict):
    """Return the ordered steps as (label, callable(cfg))."""
    from steps.step1_ingest.ingest import run as step1_run
    from steps.step2_similarity.similarity import run as step2_run
    from steps.step3_metrics.metrics import run as step3_run
    from steps.step4_predict.predict import run as step4_run
    from steps.step5_narrate.narrate import run as step5_run

    return [
        ("step1_ingest", step1_run),
        ("step2_similarity", step2_run),
        ("step3_metrics", step3_run),
        ("step4_predict", step4_run),
        ("step5_narrate", step5_run),
    ]


def regenerate_docs() -> None:
    """Rebuild reflection.html / workflow.html from decisions.json / workflow.json.

    Wired into the conductor so the browser-readable docs never go stale.
    """
    subprocess.run(
        [sys.executable, "scripts/render_docs.py"],
        check=True,
    )


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])

    steps = build_pipeline(cfg)
    for label, run in tqdm(steps, desc="pipeline", unit="step"):
        run(cfg)

    regenerate_docs()


if __name__ == "__main__":
    main()
