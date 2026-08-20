import pytest

from backtest import load_daily
from tests.conftest import FIXTURE_CACHE, FIXTURE_TICKERS


def test_every_fixture_ticker_loads_offline(load):
    for ticker in FIXTURE_TICKERS:
        df = load(ticker)
        assert len(df) == 2000
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert df.index.is_monotonic_increasing
        assert (df["Close"] > 0).all()
        assert df.notna().all().all()


def test_a_cache_miss_offline_fails_instead_of_downloading():
    with pytest.raises(ValueError, match="downloads are disabled"):
        load_daily("NOSUCHTICKER", str(FIXTURE_CACHE), allow_download=False)


def test_tickers_are_normalised_before_the_cache_lookup(load):
    assert load_daily("  spy  ", str(FIXTURE_CACHE), allow_download=False).equals(load("SPY"))
