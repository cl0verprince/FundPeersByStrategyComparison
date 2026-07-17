"""Fund page - zones A (outlook) / B (KPI row) / C (vs-peers chart) / D (peers grid),
plus prediction history and edge states. Content hierarchy per the UX spec."""
import pandas as pd
from nicegui import ui

from webapp.components import charts, honesty
from webapp.theme import TOKENS


def _fmt_pct(v, digits=1):
    return "—" if v is None or v != v else f"{v * 100:.{digits}f}%"


def render_fund(store, ticker: str) -> None:
    header = store.fund_header(ticker)
    if header is None:
        ui.label(f"No fund matches '{ticker}'.").classes("text-xl")
        ui.label("Try a ticker or fund name:").classes("text-sm text-gray-500")
        _search_box(store)
        return

    sid = header["series_id"]
    health = store.model_health()
    prov = store.provenance()
    ts = store.fund_ts(sid)
    pct = store.fund_percentiles(sid)

    # Header block
    with ui.row().classes("items-baseline gap-3"):
        ui.label(f"{header['ticker']} · {header['series_name']}").classes(
            "text-2xl font-semibold")
        if header.get("cluster_name"):
            ui.link(f"● {header['cluster_name']}", "/methodology#clusters").classes("text-sm")
    na = header["net_assets"]
    assets = "—" if na is None or na != na else f"${na / 1e9:.1f}B"
    ui.label(f"{header['yahoo_category'] or '—'} · "
             f"{assets} net assets").classes("text-sm text-gray-600")
    honesty.freshness_stamp(prov, fund_last_quarter=header["last_quarter"])

    # Edge state: dead fund -> archive banner instead of outlook
    is_active = bool(header["is_active"])
    if not is_active:
        with ui.card().classes("w-full p-3 border").style(
                f"border-color:{TOKENS['light']['serious']}"):
            ui.label(f"⚑ This fund left the universe after {header['last_quarter']} — "
                     "final quarters shown; No forward prediction exists."
                     ).classes("text-sm font-semibold")
            # No outlook card on a dead fund, so the model-health context rides here.
            honesty.status_chip(health)

    with ui.row().classes("w-full gap-4 items-start"):
        # Zone A
        if is_active:
            pred = store.fund_prediction(sid)
            reason = None if pred else "insufficient coverage this quarter"
            honesty.probability_card(pred, health, reason=reason)
        # Zone B - KPI row
        with ui.row().classes("gap-3 flex-wrap"):
            _kpi_tiles(ts, pct)

    # Zone C - fund vs peer median + diverging strip + table twin
    quarters = list(ts["quarter"])
    fund_vals = [None if pd.isna(v) else float(v) for v in ts["quarterly_return"]]
    median_vals = [None if pd.isna(v) else float(v) for v in ts["cluster_median_return"]]
    deltas = [None if pd.isna(v) else float(v) for v in ts["return_vs_cluster_median"]]
    missing = [q for q, v in zip(quarters, fund_vals) if v is None]
    ui.label("Fund vs peer median, quarterly return").classes("text-lg font-semibold mt-4")
    chart_box = ui.column().classes("w-full")
    with chart_box:
        ui.echart(charts.fund_vs_peers_option(
            "light", quarters, fund_vals, median_vals, header["ticker"])
        ).classes("w-full h-64")
        ui.echart(charts.diverging_delta_option("light", quarters, deltas)
                  ).classes("w-full h-24")
        if missing:
            ui.label(f"No N-PORT filing for {', '.join(missing)}."
                     ).classes("text-xs text-gray-500")
    table_box = ui.column().classes("w-full hidden")
    with table_box:
        ui.table(rows=ts[["quarter", "quarterly_return", "cluster_median_return",
                          "return_vs_cluster_median"]].round(4).to_dict("records"))

    def _toggle():
        chart_box.classes(toggle="hidden")
        table_box.classes(toggle="hidden")
    ui.button("chart ⇄ table", on_click=_toggle).props("flat dense")

    # Zone D - peers
    peers = store.peers(sid)
    if len(peers):
        ui.label("Most-similar peers (by reported holdings)").classes(
            "text-lg font-semibold mt-4")
        ui.label("Similarity of holdings, not of future returns."
                 ).classes("text-xs text-gray-500")
        ui.aggrid({
            "columnDefs": [
                # Real <a> links (design rule for Zone D): new-tab / right-click must
                # work. Tickers are alphanumeric SEC tickers, so no escaping risk —
                # never do this string-built renderer for free-text fields like names.
                {"headerName": "Ticker", "field": "peer_ticker",
                 ":cellRenderer": "params => '<a href=\"/fund/' + params.value + '\" "
                                  "class=\"no-underline\">' + params.value + '</a>'"},
                {"headerName": "Name", "field": "peer_name", "flex": 2},
                {"headerName": "Similarity", "field": "cosine_similarity",
                 "valueFormatter": "value == null ? '—' : (value*100).toFixed(0) + '%'"},
                {"headerName": "Trailing 4Q return", "field": "peer_trailing_4q_return",
                 "valueFormatter": "value == null ? '—' : (value*100).toFixed(1) + '%'"},
                {"headerName": "Expense", "field": "peer_expense_net",
                 "valueFormatter": "value == null ? '—' : (value*100).toFixed(2) + '%'"},
            ],
            "rowData": peers.to_dict("records"),
            "defaultColDef": {"sortable": True, "resizable": True},
        }).classes("w-full")

    # Prediction history - the per-fund misses table
    hist = store.fund_prediction_history(sid)
    if len(hist):
        ui.label("This fund's past predictions vs what happened").classes(
            "text-lg font-semibold mt-4")
        rows = [{"quarter": r.quarter,
                 "predicted": f"{r.predicted_probability * 100:.0f}%",
                 "actual": "lagged" if r.actual_label == 1 else "did not lag"}
                for r in hist.itertuples()]
        ui.table(rows=rows)

    honesty.disclaimer_line()


def _kpi_tiles(ts: pd.DataFrame, pct: dict) -> None:
    t = TOKENS["light"]
    if not len(ts):
        return
    last = ts.iloc[-1]
    tiles = [
        ("Return", last["quarterly_return"], last["cluster_median_return"], False),
    ]
    for label, val, ref, invert in tiles:
        delta = None if pd.isna(val) or pd.isna(ref) else float(val) - float(ref)
        good = delta is not None and ((delta < 0) if invert else (delta > 0))
        color = t["good"] if good else t["critical"]
        with ui.card().classes("p-3"):
            ui.label(label).classes("text-xs text-gray-500")
            ui.label(_fmt_pct(val)).classes("text-xl font-semibold")
            if delta is not None:
                ui.label(f"{'+' if delta >= 0 else ''}{delta * 100:.1f} vs peer median"
                         ).classes("text-xs").style(f"color:{color}")
    if pct:
        for label, key in [("Volatility pct'ile", "pctile_volatility"),
                           ("Sharpe pct'ile", "pctile_sharpe"),
                           ("Fees pct'ile", "pctile_expense_net")]:
            v = pct.get(key)
            with ui.card().classes("p-3"):
                ui.label(label + " (in cluster)").classes("text-xs text-gray-500")
                ui.label("n/a — cluster too small" if v is None or v != v
                         else f"{float(v) * 100:.0f}/100").classes("text-xl font-semibold")


def _search_box(store) -> None:
    from webapp.main import omnibox  # shared component, defined in main.py
    omnibox(store)
