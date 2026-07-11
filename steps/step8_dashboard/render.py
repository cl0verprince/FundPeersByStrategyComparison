"""step8_dashboard/render.py - payload dict -> one self-contained HTML file."""
import json
from pathlib import Path

from fundspeers.io import reports_dir
from steps.step8_dashboard.template import TEMPLATE


def render_dashboard(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return TEMPLATE.replace("__PAYLOAD_JSON__", blob)


def write_dashboard(payload: dict, cfg: dict) -> Path:
    out = reports_dir(cfg) / "cluster_dashboard.html"
    out.write_text(render_dashboard(payload), encoding="utf-8")
    return out
