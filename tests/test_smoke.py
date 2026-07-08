"""Smoke test: the conductor mechanism (seeding, progress bar, doc regeneration) runs
end-to-end. Uses stub steps, not the real pipeline - step1_ingest hits the network and
takes minutes, which isn't appropriate for a fast test suite. Real step logic is covered
by its own focused unit tests (e.g. tests/test_ingest.py)."""
from pathlib import Path

from fundspeers.config import PROJECT_ROOT
import conductor

STUB_PIPELINE = [(f"stub_step{i}", lambda cfg: None) for i in range(3)]


def test_conductor_runs_end_to_end():
    conductor.main(pipeline=STUB_PIPELINE)


def test_docs_are_rendered():
    assert (PROJECT_ROOT / "reflection.html").exists()
    assert (PROJECT_ROOT / "workflow.html").exists()
