"""App shell STUB - Task 7 replaces this with the full header/omnibox/dark-mode shell."""
from nicegui import ui

from webapp.data import ExtractStore
from webapp.pages.fund import render_fund
from webapp.pages.methodology import render_methodology
from webapp.pages.model import render_model

STORE = ExtractStore()


def omnibox(store) -> None:
    ui.input(placeholder=f"Search {len(store.index)} funds…")


@ui.page("/fund/{ticker}")
def fund_page(ticker: str):
    render_fund(STORE, ticker)


@ui.page("/model")
def model_page():
    render_model(STORE)


@ui.page("/methodology")
def methodology_page():
    render_methodology(STORE)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host="0.0.0.0", port=7860, show=False, title="FundsPeers")
