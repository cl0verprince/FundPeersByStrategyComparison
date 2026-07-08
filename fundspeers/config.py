"""Load project configuration and environment variables.

config.json holds all pipeline knobs (seed, paths, model params) — never
hardcode these in step code. .env holds secrets (API keys, user agents).
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = None) -> dict:
    path = Path(path) if path else PROJECT_ROOT / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_env() -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    return {
        "SEC_USER_AGENT": os.environ.get("SEC_USER_AGENT", ""),
        "LM_STUDIO_BASE_URL": os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        "LM_STUDIO_API_KEY": os.environ.get("LM_STUDIO_API_KEY", "lm-studio"),
    }
