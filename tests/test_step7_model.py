import numpy as np
import pandas as pd

from steps.step7_unified_universe.model import fund_clustered_bootstrap


def _fake_test(n_funds=40, rows_per_fund=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_funds):
        for q in range(rows_per_fund):
            y = int(rng.random() < 0.5)
            rows.append({"series_id": f"S{i}", "underperform_next_quarter": y,
                         # informative model score, weak persistence score:
                         "proba": y * 0.6 + rng.random() * 0.4,
                         "persist": rng.random()})
    return pd.DataFrame(rows)


def test_bootstrap_is_deterministic_under_seed():
    test = _fake_test()
    a = fund_clustered_bootstrap(test, "underperform_next_quarter", "proba", "persist",
                                 iterations=50, seed=42)
    b = fund_clustered_bootstrap(test, "underperform_next_quarter", "proba", "persist",
                                 iterations=50, seed=42)
    assert a == b


def test_bootstrap_detects_real_edge():
    test = _fake_test()
    out = fund_clustered_bootstrap(test, "underperform_next_quarter", "proba", "persist",
                                   iterations=200, seed=42)
    assert out["auc_ci_low"] > out["persistence_ci_high"] - 0.2   # model clearly better
    assert out["p_edge_le_zero"] < 0.05
    assert out["auc_ci_low"] < out["auc_ci_high"]
