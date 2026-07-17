"""App shell STUB - Task 7 replaces this with the full header/omnibox/dark-mode shell."""
from nicegui import ui

from webapp.data import ExtractStore
from webapp.pages.fund import render_fund

STORE = ExtractStore()


def omnibox(store) -> None:
    ui.input(placeholder=f"Search {len(store.index)} funds…")


@ui.page("/fund/{ticker}")
def fund_page(ticker: str):
    render_fund(STORE, ticker)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host="0.0.0.0", port=7860, show=False, title="FundsPeers")
