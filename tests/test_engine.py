import pandas as pd
import pytest

from backtest.engine import run
from backtest.strategy import sma_crossover


def make_prices(values):
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_no_lookahead():
    # Price doubles on day 3. A position entered on day 2 catches it;
    # the engine must not credit day 2's signal with day 2's return.
    prices = make_prices([100, 100, 200, 200])
    positions = pd.Series([0.0, 1.0, 0.0, 0.0], index=prices.index)
    result = run(prices, positions, cost_bps=0.0)
    assert result.equity.iloc[-1] == pytest.approx(2.0)

    # Entering on the day of the jump itself earns nothing.
    late = pd.Series([0.0, 0.0, 1.0, 0.0], index=prices.index)
    result = run(prices, late, cost_bps=0.0)
    assert result.equity.iloc[-1] == pytest.approx(1.0)


def test_costs_charged_on_position_changes():
    prices = make_prices([100] * 4)  # flat prices: only costs remain
    positions = pd.Series([1.0, 1.0, 0.0, 0.0], index=prices.index)
    result = run(prices, positions, cost_bps=10.0)
    # Two changes (0->1 entry, 1->0 exit) at 10 bps each.
    assert result.equity.iloc[-1] == pytest.approx((1 - 0.001) ** 2)
    assert result.stats["trades"] == 2


def test_max_drawdown():
    prices = make_prices([100, 120, 60, 90])
    positions = pd.Series(1.0, index=prices.index)
    result = run(prices, positions, cost_bps=0.0)
    assert result.stats["max_drawdown"] == pytest.approx(-0.5)


def test_sma_crossover_positions_are_zero_or_one():
    prices = make_prices(list(range(100, 130)))
    df = pd.DataFrame({"Close": prices})
    positions = sma_crossover(df, fast=3, slow=10)
    assert set(positions.unique()) <= {0.0, 1.0}
    # Steadily rising prices: fast SMA ends above slow SMA.
    assert positions.iloc[-1] == 1.0
