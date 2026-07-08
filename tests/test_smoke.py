"""Smoke test: the conductor runs end-to-end (no-op pipeline) and docs render."""
from pathlib import Path

from fundspeers.config import PROJECT_ROOT
import conductor


def test_conductor_runs_end_to_end():
    conductor.main()


def test_docs_are_rendered():
    assert (PROJECT_ROOT / "reflection.html").exists()
    assert (PROJECT_ROOT / "workflow.html").exists()
