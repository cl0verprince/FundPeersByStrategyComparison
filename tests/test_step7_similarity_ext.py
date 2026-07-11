"""The new similarity.run params must default to existing behavior and be individually testable.
Full-run behavior is covered by the real _all run in Task 9; here we unit-test the pure pieces."""
import inspect

import pandas as pd

from steps.step2_similarity.similarity import run, _filter_universe


def test_run_signature_backward_compatible():
    sig = inspect.signature(run)
    assert list(sig.parameters) == [
        "cfg", "table_suffix", "n_clusters", "top_n_peers", "require_segment", "save_coords"]
    assert sig.parameters["n_clusters"].default is None
    assert sig.parameters["require_segment"].default is None
    assert sig.parameters["save_coords"].default is False


def test_filter_universe_by_segment():
    funds = pd.DataFrame({
        "series_id": ["A", "B", "C"], "is_us_equity": [True, True, False],
        "segment": ["strategy", "allocation", "strategy"],
    })
    assert _filter_universe(funds, None) == {"A", "B"}          # today's behavior
    assert _filter_universe(funds, "strategy") == {"A"}          # segment filter
