"""Read-only access to the extract + the in-memory search index and scorer.

The app owns NO statistics: every number it shows was precomputed by
steps/step14_webapp/extract.py. This module only fetches and ranks."""
import os
import re
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_EXTRACT = Path(__file__).parent / "data" / "extract.duckdb"


def _trigrams(s: str) -> set:
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def score_query(query: str, index: list) -> list:
    """Tiered fuzzy scorer. Tiers: 1 exact ticker, 2 ticker prefix, 3 every query token
    prefix-matches a name token, 4 trigram Jaccard >= 0.25. Dead funds demote one tier
    (honesty: they are findable, never promoted). Within tier: AUM desc."""
    q = re.sub(r"\s+", " ", query.strip().lower())
    if not q:
        return []
    q_tokens = q.split(" ")
    q_tri = _trigrams(q)
    scored = []
    for row in index:
        ticker = row["ticker"].lower() if row["ticker"] else ""
        name = row["name_normalized"]
        if q == ticker:
            tier = 1
        elif ticker.startswith(q):
            tier = 2
        elif all(any(tok.startswith(qt) for tok in name.split(" ")) for qt in q_tokens):
            tier = 3
        else:
            tri = _trigrams(name)
            overlap = len(q_tri & tri) / max(1, len(q_tri | tri))
            if overlap < 0.25:
                continue
            tier = 4
        if not row["is_active"]:
            tier += 1
        scored.append((tier, -(row["net_assets"] or 0), row))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in scored]


class ExtractStore:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get("EXTRACT_PATH", DEFAULT_EXTRACT))
        self.con = duckdb.connect(str(self.path), read_only=True)
        self.index = self.con.execute(
            "SELECT series_id, ticker, series_name, name_normalized, net_assets, "
            "is_active, last_quarter, cluster_name FROM v_fund_search"
        ).df().to_dict("records")
        self._provenance = self.con.execute(
            "SELECT * FROM v_data_provenance").df().iloc[0].to_dict()
        self._health = self.con.execute(
            "SELECT * FROM v_model_health_current").df().iloc[0].to_dict()

    def _one(self, sql, args):
        df = self.con.execute(sql, args).df()
        return df.iloc[0].to_dict() if len(df) else None

    def fund_header(self, ticker: str):
        return self._one("SELECT * FROM v_fund_header WHERE upper(ticker) = upper(?)", [ticker])

    def fund_ts(self, series_id: str) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM v_fund_peer_relative_ts WHERE series_id = ? ORDER BY quarter",
            [series_id]).df()

    def fund_percentiles(self, series_id: str):
        return self._one("SELECT * FROM v_fund_cluster_percentiles WHERE series_id = ?",
                         [series_id])

    def fund_prediction(self, series_id: str):
        return self._one("SELECT * FROM v_fund_prediction_current WHERE series_id = ?",
                         [series_id])

    def fund_prediction_history(self, series_id: str) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM v_fund_prediction_history WHERE series_id = ? AND split = 'test' "
            "ORDER BY quarter", [series_id]).df()

    def peers(self, series_id: str) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM v_peer_display WHERE series_id = ? ORDER BY peer_rank",
            [series_id]).df()

    def top_holdings(self, series_id: str) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM v_top_holdings WHERE series_id = ? ORDER BY rk", [series_id]).df()

    def model_health(self) -> dict:
        return dict(self._health)

    def model_quarters(self) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM v_model_health_quarters ORDER BY quarter").df()

    def calibration(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM v_calibration_bins ORDER BY bin_low").df()

    def provenance(self) -> dict:
        return dict(self._provenance)

    def search(self, query: str, limit: int = 8) -> list:
        return score_query(query, self.index)[:limit]

    def is_stale(self) -> bool:
        """as-of quarter end + one quarter + 60-day filing lag + 30-day grace < today."""
        q = self._provenance["last_quarter"]
        year, qn = int(q[:4]), int(q[-1])
        end_month = qn * 3
        next_due = date(year + (1 if end_month + 6 > 12 else 0),
                        (end_month + 6 - 1) % 12 + 1, 1)
        return date.today() > next_due
