"""Unit tests for step9_fees_turnover.acquire — cache/skip logic and the availability
filter, with no network. A tiny fake ZIP stands in for a real RR data set in tmp_path;
requests.get is monkeypatched to prove the cached path short-circuits the network."""
import io
import zipfile

import pytest

from steps.step9_fees_turnover import acquire


def _cfg(tmp_path):
    # Only the paths the acquire helpers touch; raw_dir(cfg) resolves relative to the
    # project root, so give an absolute tmp path.
    return {"paths": {"raw": str(tmp_path / "raw")}, "fees": {"rr_years": [2024]}}


def _fake_rr_zip(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("sub.tsv", "adsh\tfiled\n0001-24-1\t20241206\n")
        z.writestr("num.tsv", "adsh\ttag\tseries\tclass\tvalue\n0001-24-1\tNetExpensesOverAssets\tS1\tC1\t0.0075\n")
    path.write_bytes(buf.getvalue())


def test_rr_zip_path_uses_quarter_naming(tmp_path):
    cfg = _cfg(tmp_path)
    p = acquire.rr_zip_path(2024, 4, cfg)
    assert p.name == "2024q4_rr1.zip"


def test_rr_url_pattern():
    url = acquire.rr_url(2024, 4)
    assert url.endswith("/return-summary-data-sets/2024q4_rr1.zip")
    assert "mutual-fund-prospectus-risk/return-summary-data-sets" in url


def test_download_rr_quarter_returns_cached_without_network(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    dest = acquire.rr_zip_path(2024, 4, cfg)
    _fake_rr_zip(dest)
    original_bytes = dest.read_bytes()

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("network must not be hit when the ZIP is already cached")

    monkeypatch.setattr(acquire.requests, "get", _boom)
    monkeypatch.setenv("SEC_USER_AGENT", "test agent test@example.com")

    result = acquire.download_rr_quarter(2024, 4, cfg)
    assert result == dest
    assert result.read_bytes() == original_bytes  # untouched


def test_download_rr_quarter_downloads_when_missing(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    payload = b"fake-zip-bytes"

    class _Resp:
        content = payload

        def raise_for_status(self):
            pass

    captured = {}

    def _fake_get(url, headers=None, timeout=None):  # noqa: ANN001
        captured["url"] = url
        captured["headers"] = headers
        return _Resp()

    monkeypatch.setattr(acquire.requests, "get", _fake_get)
    monkeypatch.setenv("SEC_USER_AGENT", "test agent test@example.com")

    result = acquire.download_rr_quarter(2024, 3, cfg)
    assert result.read_bytes() == payload
    assert captured["url"].endswith("2024q3_rr1.zip")
    assert "User-Agent" in captured["headers"]


def test_download_rr_quarter_requires_user_agent(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(acquire.requests, "get", lambda *a, **k: None)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    # load_env reads .env too; force the empty value to win by pointing at a dir with none.
    monkeypatch.setenv("SEC_USER_AGENT", "")
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        acquire.download_rr_quarter(2024, 2, cfg)


def test_download_all_skips_unpublished_quarters(tmp_path, monkeypatch):
    cfg = {"paths": {"raw": str(tmp_path / "raw")}, "fees": {"rr_years": [2026]}}
    # SEC has published only 2026q1; q2-q4 must be skipped, not fetched/failed.
    monkeypatch.setattr(acquire, "available_quarters", lambda cfg: {(2026, 1)})

    downloaded = {}

    def _fake_download(year, quarter, cfg):  # noqa: ANN001
        p = acquire.rr_zip_path(year, quarter, cfg)
        downloaded[(year, quarter)] = p
        return p

    monkeypatch.setattr(acquire, "download_rr_quarter", _fake_download)
    paths = acquire.download_all(cfg)
    assert list(downloaded) == [(2026, 1)]
    assert len(paths) == 1
