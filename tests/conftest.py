from pathlib import Path

import pytest

from backtest import load_daily

FIXTURE_CACHE = Path(__file__).parent / "fixtures" / "prices"
FIXTURE_TICKERS = ("SPY", "QQQ", "JPM")


@pytest.fixture(scope="session")
def load():

    def _load(ticker: str):
        return load_daily(ticker, str(FIXTURE_CACHE), allow_download=False)

    return _load


@pytest.fixture(scope="session")
def spy(load):
    return load("SPY")
