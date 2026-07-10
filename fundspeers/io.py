"""Path resolution and SQL table read/write helpers, driven by config.

Tables live in one DuckDB file (data/processed/fundspeers.duckdb) - an embedded, serverless
SQL database (no server process, no credentials). Open it directly with any DuckDB client
(the `duckdb` CLI, DBeaver, etc.) to run ad-hoc SQL against the pipeline's output.
"""
import re
from pathlib import Path

import duckdb
import joblib
import pandas as pd

from fundspeers.config import PROJECT_ROOT

_VALID_TABLE_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def raw_dir(cfg: dict) -> Path:
    p = PROJECT_ROOT / cfg["paths"]["raw"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir(cfg: dict) -> Path:
    p = PROJECT_ROOT / cfg["paths"]["processed"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports_dir(cfg: dict) -> Path:
    p = PROJECT_ROOT / cfg["paths"]["reports"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path(cfg: dict) -> Path:
    return processed_dir(cfg) / "fundspeers.duckdb"


def models_dir(cfg: dict) -> Path:
    p = PROJECT_ROOT / cfg["paths"]["models"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_model(model, name: str, cfg: dict) -> Path:
    path = models_dir(cfg) / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str, cfg: dict):
    return joblib.load(models_dir(cfg) / f"{name}.joblib")


def _validate_table_name(name: str) -> None:
    # Table names are always internal string literals (e.g. "funds", "holdings"), never
    # user input - this is defensive, not a real injection risk, but cheap to check.
    if not _VALID_TABLE_NAME.match(name):
        raise ValueError(f"invalid table name: {name!r}")


def save_table(df: pd.DataFrame, name: str, cfg: dict) -> str:
    _validate_table_name(name)
    with duckdb.connect(str(db_path(cfg))) as con:
        con.register("_df_to_save", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df_to_save")
    return name


def load_table(name: str, cfg: dict) -> pd.DataFrame:
    _validate_table_name(name)
    with duckdb.connect(str(db_path(cfg))) as con:
        return con.execute(f"SELECT * FROM {name}").df()


def table_exists(name: str, cfg: dict) -> bool:
    _validate_table_name(name)
    with duckdb.connect(str(db_path(cfg))) as con:
        row = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
        ).fetchone()
        return row[0] > 0
