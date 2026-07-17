"""webapp/main.py - app shell: header with omnibox + status chip, layout wrapper, routes.

Dark-mode note: `ui.dark_mode()` is wired to a header toggle button, but the honesty
components (`status_chip`, `freshness_stamp`, `disclaimer_line`, `probability_card`) all
default to `mode="light"` and pages call them without threading a mode through. Full
dark-mode token threading (passing mode="dark" down into every component call site) is
explicitly OUT of scope for this task per design carry-over notes; it's deferred to a
later polish step. The toggle only flips NiceGUI's own dark-mode styling for now.
"""
from contextlib import contextmanager
from functools import lru_cache

from nicegui import ui

from webapp.components import honesty
from webapp.data import ExtractStore
from webapp.pages.fund import render_fund
from webapp.pages.methodology import render_methodology
from webapp.pages.model import render_model
from webapp.theme import DISCLAIMER, TOKENS


@lru_cache(maxsize=1)
def get_store() -> ExtractStore:
    return ExtractStore()


def omnibox(store) -> None:
    t = TOKENS["light"]  # per-mode threading deferred; see module docstring
    with ui.element("div").classes("relative"):
        # Declared before the input so the on_change closure can reference it;
        # absolute + top-full anchors it below the input regardless of DOM order.
        results = ui.column().classes(
            "absolute top-full left-0 mt-1 z-50 shadow rounded w-96 hidden"
        ).style(f"background:{t['surface']}")

        def on_change(e) -> None:
            results.clear()
            hits = store.search(e.value or "", limit=8)
            show = bool(hits) or bool(e.value or "")
            results.classes(remove="hidden" if show else None, add=None if show else "hidden")
            with results:
                for h in hits:
                    tag = "" if h["is_active"] else f" (left universe {h['last_quarter']})"
                    ui.link(f"{h['ticker']}  {h['series_name']}{tag}",
                            f"/fund/{h['ticker']}").classes(
                        "px-3 py-1 text-sm no-underline" + ("" if h["is_active"] else " opacity-60"))
                if not hits and (e.value or ""):
                    ui.label(f"No fund matches '{e.value}'.").classes("px-3 py-1 text-sm")

        box = ui.input(placeholder=f"Search {len(store.index)} funds by ticker or name…",
                       on_change=on_change).props("dense outlined debounce=120").classes("w-96")
    def on_key(e) -> None:
        # Ctrl+K or plain "/" focuses the search box (design shortcut). No cheap way to
        # detect "an input already has focus" from NiceGUI's KeyEventArguments, so "/"
        # will also fire while typing in another field — accepted per review guidance.
        if not e.action.keydown:
            return
        if e.key == "k" and e.modifiers.ctrl:
            box.run_method("focus")
        elif e.key == "/" and not (e.modifiers.ctrl or e.modifiers.alt or e.modifiers.meta):
            box.run_method("focus")

    ui.keyboard(on_key=on_key)


@contextmanager
def layout(store):
    dark = ui.dark_mode()
    with ui.header().classes("items-center gap-4 bg-transparent border-b px-4 py-2"):
        ui.link("◆ FundsPeers", "/").classes("text-lg font-semibold no-underline")
        ui.link("Funds", "/").classes("no-underline text-sm")
        ui.link("Model health", "/model").classes("no-underline text-sm")
        ui.link("Methodology", "/methodology").classes("no-underline text-sm")
        omnibox(store)
        honesty.status_chip(store.model_health())
        ui.button(icon="dark_mode", on_click=dark.toggle).props("flat dense round")
    with ui.row().classes("w-full px-4 py-1 gap-3 text-xs text-gray-500 border-b"):
        honesty.freshness_stamp(store.provenance())
        ui.label(DISCLAIMER)
    if store.is_stale():
        t = TOKENS["light"]  # per-mode threading deferred; see module docstring
        with ui.row().classes("w-full px-4 py-2 items-center gap-2").style(
                f"background:{t['surface']}; border-left:4px solid {t['warning']}"):
            ui.label("⚠").style(f"color:{t['warning']}")
            ui.label(f"This data ends {store.provenance()['last_quarter']} and a newer "
                     "quarter should exist — the extract is stale."
                     ).classes("text-sm").style(f"color:{t['warning']}")
    with ui.column().classes("w-full max-w-screen-xl mx-auto p-4") as content:
        yield content


@ui.page("/")
def home():
    store = get_store()
    with layout(store):
        ui.label("Which funds will lag their peers?").classes("text-3xl font-semibold")
        ui.label("Search a fund to see its holdings-based peer group, honest "
                 "peer-relative record, and the model's live scorecard."
                 ).classes("text-sm text-gray-600")
        health = store.model_health()
        with ui.card().classes("p-4 max-w-md"):
            ui.label("Can you trust the signal?").classes("font-semibold")
            honesty.status_chip(health)
            ui.label(health["rule_text"]).classes("text-sm text-gray-600")
            ui.link("See the evidence →", "/model")


@ui.page("/fund/{ticker}")
def fund_page(ticker: str):
    store = get_store()
    with layout(store):
        render_fund(store, ticker)


@ui.page("/model")
def model_page():
    store = get_store()
    with layout(store):
        render_model(store)


@ui.page("/methodology")
def methodology_page():
    store = get_store()
    with layout(store):
        render_methodology(store)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host="0.0.0.0", port=7860, show=False, title="FundsPeers",
           favicon="◆", reload=False)
