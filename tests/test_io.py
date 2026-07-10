"""Unit tests for fundspeers.io's DuckDB-backed table read/write helpers.

Uses a temp directory config (via tmp_path) so tests never touch the real
data/processed/fundspeers.duckdb.
"""
import pandas as pd
import pytest

from fundspeers.io import load_model, load_table, save_model, save_table, table_exists


@pytest.fixture
def cfg(tmp_path):
    return {"paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "processed"),
                       "reports": str(tmp_path / "reports"), "models": str(tmp_path / "models")}}


def test_save_and_load_table_round_trips(cfg):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    save_table(df, "widgets", cfg)
    result = load_table("widgets", cfg)
    pd.testing.assert_frame_equal(result, df)


def test_table_exists_reflects_saved_tables(cfg):
    assert table_exists("widgets", cfg) is False
    save_table(pd.DataFrame({"a": [1]}), "widgets", cfg)
    assert table_exists("widgets", cfg) is True


def test_save_table_overwrites_on_rerun(cfg):
    save_table(pd.DataFrame({"a": [1, 2]}), "widgets", cfg)
    save_table(pd.DataFrame({"a": [9]}), "widgets", cfg)
    result = load_table("widgets", cfg)
    assert result["a"].tolist() == [9]


def test_invalid_table_name_is_rejected(cfg):
    with pytest.raises(ValueError):
        save_table(pd.DataFrame({"a": [1]}), "widgets; DROP TABLE funds", cfg)


def test_save_and_load_model_round_trips(cfg):
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(random_state=0).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    save_model(model, "test_model", cfg)
    reloaded = load_model("test_model", cfg)
    assert reloaded.predict([[0], [3]]).tolist() == model.predict([[0], [3]]).tolist()
