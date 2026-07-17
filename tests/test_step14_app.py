"""tests/test_step14_app.py — NiceGUI User-fixture smoke tests against a synthetic extract.

Builds a tiny extract in tmp_path via the same builders the real one uses, sets
EXTRACT_PATH, and boots the app. The edge-state milestone test from the design: the dead
fund and the missing-prediction fund must render as designed states, not errors."""
import os
import pytest
from nicegui.testing import User

pytest_plugins = ["nicegui.testing.user_plugin"]


@pytest.fixture(scope="module", autouse=True)
def extract_env(tmp_path_factory):
    # Build a synthetic extract with the Task 1/2 builders (same fixtures as
    # test_step14_extract, extracted into tests/step14_fixtures.py by this task).
    from tests.step14_fixtures import build_synthetic_extract
    path = build_synthetic_extract(tmp_path_factory.mktemp("extract") / "extract.duckdb")
    os.environ["EXTRACT_PATH"] = str(path)
    yield
    os.environ.pop("EXTRACT_PATH", None)


@pytest.fixture(autouse=True)
def app_routes(extract_env):
    import webapp.main  # noqa: F401  (registers @ui.page routes on import)


async def test_fund_page_renders_with_honesty_elements(user: User):
    await user.open("/fund/AAAAX")
    await user.should_see("Alpha Large Blend Fund")
    await user.should_see("Signal degraded")            # chip present
    await user.should_see("Data as of 2026q2")           # freshness stamp
    await user.should_see("not investment advice")       # disclaimer
    await user.should_see("peers' median return")        # fixed sentence


async def test_dead_fund_is_archive_not_error(user: User):
    await user.open("/fund/DDDDX")
    await user.should_see("left the universe")
    await user.should_see("No forward prediction")
    # no outlook card on a dead fund, but the model-health chip must still be present
    await user.should_see("Signal degraded")


async def test_unknown_ticker_offers_search(user: User):
    await user.open("/fund/ZZZZ")
    await user.should_see("No fund matches")


async def test_model_page_leads_with_verdict_and_shows_coinflip(user: User):
    await user.open("/model")
    await user.should_see("Can you trust the lag-probability signal right now?")
    await user.should_see("Signal degraded")
    await user.should_see("coin flip")          # AUC explainer / markline caption
    await user.should_see("The rule:")          # disclosed rule text


async def test_methodology_has_disclaimer_and_survivorship(user: User):
    await user.open("/methodology")
    await user.should_see("not investment advice")
    await user.should_see("Dead funds are included")
