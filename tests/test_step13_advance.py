"""Unit tests for step13's advance-quarter conductor. No network, no DuckDB, no real
pipeline stages — the stages are integration-tested by their own steps and are stubbed
here. Covers: quarter arithmetic (incl. year rollover), the HEAD probe (404 -> None,
200 -> the quarter), config extension, and the dry-run / up-to-date early exits which must
write NOTHING."""
import json

import pytest

from steps.step13_automation import advance


# --- next_quarter_after (pure arithmetic) ----------------------------------


def test_next_quarter_within_year():
    assert advance.next_quarter_after("2026q1") == "2026q2"
    assert advance.next_quarter_after("2026q2") == "2026q3"
    assert advance.next_quarter_after("2026q3") == "2026q4"


def test_next_quarter_year_rollover():
    assert advance.next_quarter_after("2026q4") == "2027q1"
    assert advance.next_quarter_after("2024q4") == "2025q1"


def test_next_quarter_rejects_bad_quarter():
    with pytest.raises(ValueError):
        advance.next_quarter_after("2026q5")


# --- probe_next_quarter (HEAD mocked) --------------------------------------


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code

    def raise_for_status(self):
        pass


def _cfg(last_quarter="2026q1"):
    return {"data": {"quarters": ["2025q4", last_quarter]}}


def test_probe_returns_quarter_when_published(monkeypatch):
    monkeypatch.setattr(advance, "load_env", lambda: {"SEC_USER_AGENT": "test ua"})
    monkeypatch.setattr(advance.requests, "head", lambda *a, **k: _Resp(200))
    assert advance.probe_next_quarter(_cfg("2026q1")) == "2026q2"


def test_probe_returns_none_when_not_published(monkeypatch):
    monkeypatch.setattr(advance, "load_env", lambda: {"SEC_USER_AGENT": "test ua"})
    monkeypatch.setattr(advance.requests, "head", lambda *a, **k: _Resp(404))
    assert advance.probe_next_quarter(_cfg("2026q1")) is None


def test_probe_raises_on_unexpected_status(monkeypatch):
    # A 429/403 must NOT be silently read as "nothing new".
    monkeypatch.setattr(advance, "load_env", lambda: {"SEC_USER_AGENT": "test ua"})
    monkeypatch.setattr(advance.requests, "head", lambda *a, **k: _Resp(429))
    with pytest.raises(Exception):
        advance.probe_next_quarter(_cfg("2026q1"))


# --- extend_config ---------------------------------------------------------


_SAMPLE_CONFIG = """{
  "seed": 42,
  "data": {
    "quarters": [
      "2025q4",
      "2026q1"
    ],
    "max_funds": 1000
  },
  "fees": {
    "rr_years": [2024, 2025, 2026]
  }
}
"""


def _write_config(tmp_path, text=_SAMPLE_CONFIG):
    p = tmp_path / "config.json"
    p.write_text(text, encoding="utf-8")
    return p


def test_extend_config_appends_quarter_valid_json(tmp_path):
    p = _write_config(tmp_path)
    advance.extend_config(p, "2026q2")
    parsed = json.loads(p.read_text(encoding="utf-8"))
    assert parsed["data"]["quarters"] == ["2025q4", "2026q1", "2026q2"]
    # Same year already covered -> rr_years unchanged.
    assert parsed["fees"]["rr_years"] == [2024, 2025, 2026]


def test_extend_config_adds_new_rr_year_on_rollover(tmp_path):
    text = _SAMPLE_CONFIG.replace('"2026q1"', '"2026q4"')
    p = _write_config(tmp_path, text)
    advance.extend_config(p, "2027q1")
    parsed = json.loads(p.read_text(encoding="utf-8"))
    assert parsed["data"]["quarters"][-1] == "2027q1"
    assert parsed["fees"]["rr_years"] == [2024, 2025, 2026, 2027]


def test_extend_config_rejects_duplicate_quarter(tmp_path):
    p = _write_config(tmp_path)
    before = p.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        advance.extend_config(p, "2026q1")
    assert p.read_text(encoding="utf-8") == before  # no partial write


# --- run(): dry-run and up-to-date write nothing ---------------------------


def _no_stages_allowed(monkeypatch):
    """Trip-wire: any real stage call fails the test."""
    def boom(*a, **k):
        raise AssertionError("a pipeline stage was called")
    for name in ("_stage_ingest", "_stage_cluster_and_metrics", "_stage_fees",
                 "_stage_evaluate", "_stage_dashboard", "_stage_docs_and_commit",
                 "extend_config"):
        monkeypatch.setattr(advance, name, boom)


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    p = _write_config(tmp_path)
    before = p.read_text(encoding="utf-8")
    _no_stages_allowed(monkeypatch)
    monkeypatch.setattr(advance, "probe_next_quarter", lambda cfg: "2026q2")

    result = advance.run({"data": {"quarters": ["2026q1"]}}, dry_run=True, config_path=p)

    assert result["status"] == "dry_run"
    assert result["quarter"] == "2026q2"
    assert p.read_text(encoding="utf-8") == before  # config untouched


def test_up_to_date_writes_nothing(tmp_path, monkeypatch):
    p = _write_config(tmp_path)
    before = p.read_text(encoding="utf-8")
    _no_stages_allowed(monkeypatch)
    monkeypatch.setattr(advance, "probe_next_quarter", lambda cfg: None)

    result = advance.run({"data": {"quarters": ["2026q1"]}}, config_path=p)

    assert result["status"] == "up_to_date"
    assert result["quarter"] is None
    assert p.read_text(encoding="utf-8") == before


# --- run(): the dormant-config-key trap (full.n_clusters, NOT similarity's 15) ---


def test_cluster_stage_uses_full_n_clusters_not_similarity_default(monkeypatch):
    """The design's headline trap: similarity.run must receive n_clusters from
    full.n_clusters (40), never the dormant similarity.n_clusters default (15)."""
    calls = {}
    monkeypatch.setattr(advance.full_build, "ensure_funds_full_segment", lambda cfg: None)
    monkeypatch.setattr(advance.similarity, "run",
                        lambda cfg, **kw: calls.update(kw))
    monkeypatch.setattr(advance.metrics, "run", lambda cfg, **kw: None)

    cfg = {"full": {"n_clusters": 40}, "similarity": {"n_clusters": 15}}
    advance._stage_cluster_and_metrics(cfg)

    assert calls["n_clusters"] == 40
    assert calls["top_n_peers"] == 15
    assert calls["require_segment"] == "strategy"
    assert calls["save_coords"] is True
    assert calls["table_suffix"] == "_full"


def test_cluster_stage_repairs_segment_before_clustering(monkeypatch):
    """The first live run's crash: a fresh stage-2 ingest rewrites funds_full WITHOUT the
    `segment` column (ingest never writes it), and stage 3's
    similarity.run(require_segment="strategy") KeyErrors. step10's idempotent
    ensure_funds_full_segment repair must therefore run before clustering — not first
    at stage 5 inside full_build.run."""
    order = []
    monkeypatch.setattr(advance.full_build, "ensure_funds_full_segment",
                        lambda cfg: order.append("segment_repair"))
    monkeypatch.setattr(advance.similarity, "run",
                        lambda cfg, **kw: order.append("similarity"))
    monkeypatch.setattr(advance.metrics, "run",
                        lambda cfg, **kw: order.append("metrics"))

    advance._stage_cluster_and_metrics({"full": {"n_clusters": 40}})

    assert order == ["segment_repair", "similarity", "metrics"]


# --- _stage_evaluate: retired path (design step16) vs. current non-retired path ---


def _boom(*_a, **_k):
    raise AssertionError("must not be called on the retired path")


def test_stage_evaluate_retired_calls_run_retired_not_fees(monkeypatch):
    calls = []
    monkeypatch.setattr(advance.full_build, "run_retired",
                        lambda cfg: calls.append("run_retired"))
    monkeypatch.setattr(advance.full_build, "run", _boom)
    monkeypatch.setattr(advance.fees_evaluate, "run_evaluation", _boom)

    cfg = {"model": {"retirement": {"as_of": "2026q1",
                                     "statement": "Retired for the test record."}}}
    result = advance._stage_evaluate(cfg)

    assert calls == ["run_retired"]
    assert result == {"retired": True}


def test_stage_evaluate_non_retired_calls_full_run_and_fees(monkeypatch):
    calls = []
    monkeypatch.setattr(advance.full_build, "run", lambda cfg: calls.append("run"))
    monkeypatch.setattr(advance.full_build, "run_retired", _boom)
    monkeypatch.setattr(advance.fees_evaluate, "run_evaluation",
                        lambda cfg: calls.append("run_evaluation") or {"coverage": 0.9})

    cfg = {"model": {"rf": {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 5}}}
    result = advance._stage_evaluate(cfg)

    assert calls == ["run", "run_evaluation"]
    assert result == {"coverage": 0.9}
