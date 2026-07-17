"""Retired-state app smoke tests against a RETIRED synthetic extract."""
import os
import pytest
from nicegui.testing import User

pytest_plugins = ["nicegui.testing.user_plugin"]


@pytest.fixture(scope="module", autouse=True)
def retired_extract_env(tmp_path_factory):
    from tests.step14_fixtures import build_synthetic_extract
    path = build_synthetic_extract(
        tmp_path_factory.mktemp("retired") / "extract.duckdb", retired=True)
    os.environ["EXTRACT_PATH"] = str(path)
    # get_store() is lru_cached per process - clear it so this module's env var wins.
    import webapp.main
    webapp.main.get_store.cache_clear()
    yield
    os.environ.pop("EXTRACT_PATH", None)
    webapp.main.get_store.cache_clear()


async def test_fund_page_shows_retirement_not_probability(user: User):
    await user.open("/fund/AAAAX")
    await user.should_see("Model retired")
    await user.should_see("Signal retired")
    await user.should_see("Retired for the synthetic record.")


async def test_model_page_retired_verdict_and_empty_record(user: User):
    await user.open("/model")
    await user.should_see("RETIRED as of")
    await user.should_see("First post-retirement score expected")
    await user.should_see("No live forward book")
