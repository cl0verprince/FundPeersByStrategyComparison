"""Unit tests for step9_fees_turnover.evaluate — the two pure pieces: the fund-disjoint
split (doubly-disjoint: unseen funds in an unseen period) and the expense-rank baseline
scorer. The full run_evaluation is exercised by build.py's real run, not unit tests.
"""
import numpy as np
import pandas as pd

from steps.step9_fees_turnover import evaluate


def _panel():
    quarters = ["2022q1", "2022q2", "2022q3", "2022q4", "2023q1", "2023q2"]
    rows = []
    for i in range(10):
        for q in quarters:
            rows.append({"series_id": f"S{i}", "quarter": q,
                         "underperform_next_quarter": float(i % 2), "feat": float(i)})
    return pd.DataFrame(rows), quarters


def test_fund_disjoint_split_disjoint_and_period_restricted():
    panel, quarters = _panel()
    # holdout_transitions=2 -> transitions q1..q5, test_q={q4,q5}, train_q={q1,q2,q3}
    train, test = evaluate.fund_disjoint_split(
        panel, quarters, holdout_transitions=2, test_share=0.2, seed=42)
    assert set(train["series_id"]).isdisjoint(set(test["series_id"]))
    # quarters[:-1] = 5 transitions; holdout 2 -> test_q = {2022q4, 2023q1}
    assert set(test["quarter"]) <= {"2022q4", "2023q1"}
    assert set(train["quarter"]) <= {"2022q1", "2022q2", "2022q3"}
    # ~20% of 10 funds = 2 test funds
    assert test["series_id"].nunique() == 2


def test_fund_disjoint_split_is_deterministic():
    panel, quarters = _panel()
    a = evaluate.fund_disjoint_split(panel, quarters, 2, 0.2, seed=42)
    b = evaluate.fund_disjoint_split(panel, quarters, 2, 0.2, seed=42)
    assert set(a[1]["series_id"]) == set(b[1]["series_id"])
    # a different seed picks a (generally) different test-fund set
    c = evaluate.fund_disjoint_split(panel, quarters, 2, 0.2, seed=7)
    assert isinstance(c[1], pd.DataFrame)


def test_expense_rank_auc_perfect_and_inverted():
    df = pd.DataFrame({
        "underperform_next_quarter": [1.0, 1.0, 0.0, 0.0],
        "expense_ratio_net": [0.02, 0.015, 0.005, 0.003],
    })
    assert evaluate.expense_rank_auc(df) == 1.0
    inv = df.assign(underperform_next_quarter=[0.0, 0.0, 1.0, 1.0])
    assert evaluate.expense_rank_auc(inv) == 0.0
