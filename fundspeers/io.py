"""Path resolution and parquet table read/write helpers, driven by config."""
from pathlib import Path

import pandas as pd

from fundspeers.config import PROJECT_ROOT


def raw_dir(cfg: dict) -> Path:
    p = PROJECT_ROOT / cfg["paths"]["raw"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir(cfg: dict) -> Path:
    p = PROJECT_ROOT / cfg["paths"]["processed"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_table(df: pd.DataFrame, name: str, cfg: dict) -> Path:
    path = processed_dir(cfg) / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_table(name: str, cfg: dict) -> pd.DataFrame:
    path = processed_dir(cfg) / f"{name}.parquet"
    return pd.read_parquet(path)


def table_exists(name: str, cfg: dict) -> bool:
    return (processed_dir(cfg) / f"{name}.parquet").exists()
