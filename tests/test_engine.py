import pandas as pd
import pytest

from backtest.engine import TRADING_DAYS, _stats, run
from backtest.strategy import sma_crossover, vol_target


def make_prices(values):
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_rejects_inputs_that_would_fail_silently():
    prices = make_prices([100, 101, 102])
    positions = pd.Series(1.0, index=prices.index)

    with pytest.raises(ValueError):
        run(make_prices([100, 0, 100]), positions)
    with pytest.raises(ValueError):
        run(prices, positions, cost_bps=-5.0)
    with pytest.raises(ValueError):
        run(prices, positions, rf_annual=-1.5)
    with pytest.raises(ValueError):
        run(make_prices([]), pd.Series([], dtype=float))
    with pytest.raises(ValueError):
        run(make_prices([100]), pd.Series([1.0], index=make_prices([100]).index))
    with pytest.raises(ValueError):
        sma_crossover(pd.DataFrame({"Close": prices}), fast=200, slow=50)
    with pytest.raises(ValueError):
        vol_target(pd.DataFrame({"Close": prices}), target_vol=-0.1)
    with pytest.raises(ValueError):
        sma_crossover(pd.DataFrame({"Close": prices}), fast=0, slow=3)
    with pytest.raises(ValueError):
        vol_target(pd.DataFrame({"Close": prices}), lookback=0)
    with pytest.raises(ValueError):
        vol_target(pd.DataFrame({"Close": prices}), max_leverage=-2.0)


def test_rejects_costs_that_drive_equity_non_positive():
    prices = make_prices([100, 100, 40])
    positions = pd.Series([1.0, 1.0, 0.0], index=prices.index)
    with pytest.raises(ValueError, match="wipe out"):
        run(prices, positions, cost_bps=5000)

    result = run(prices, positions, cost_bps=5)
    assert (result.equity > 0).all()


def test_vol_target_is_flat_when_volatility_is_unusable():
    flat = pd.DataFrame({"Close": make_prices([100.0] * 8)})
    positions = vol_target(flat, target_vol=0.1, lookback=3, max_leverage=1.0)
    assert (positions == 0.0).all()


def test_vol_target_zero_is_a_legitimate_always_cash_strategy():
    prices = make_prices([100, 101, 99, 102, 98, 103, 97])
    df = pd.DataFrame({"Close": prices})
    positions = vol_target(df, target_vol=0.0, lookback=2)
    assert (positions == 0.0).all()


def test_sharpe_at_zero_dispersion_takes_the_sign_of_the_mean():
    idx = pd.bdate_range("2024-01-01", periods=4)
    equity = pd.Series([1.02, 1.02**2, 1.02**3, 1.02**4], index=idx)
    positions = pd.Series(1.0, index=idx)

    daily = pd.Series([0.02, 0.02, 0.02, 0.02], index=idx)
    stats = _stats(equity, daily, positions, rf_daily=0.0)
    assert stats["sharpe"] == float("inf")
    assert stats["sortino"] == float("inf")
    assert stats["sharpe_se"] == float("inf")
    assert stats["sharpe_ci_low"] == stats["sharpe_ci_high"] == float("inf")

    daily = pd.Series([-0.02, -0.02, -0.02, -0.02], index=idx)
    stats = _stats(equity, daily, positions, rf_daily=0.0)
    assert stats["sharpe"] == float("-inf")
    assert stats["sortino"] == pytest.approx(-0.02 / 0.02 * TRADING_DAYS**0.5)

    daily = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)
    stats = _stats(equity, daily, positions, rf_daily=0.0)
    assert stats["sharpe"] == 0.0
    assert stats["sortino"] == 0.0


def test_negative_risk_free_rate_is_allowed():
    prices = make_prices([100] * 5)
    cash = pd.Series(0.0, index=prices.index)
    result = run(prices, cash, cost_bps=0.0, rf_annual=-0.02)

    rho = 0.98 ** (1 / TRADING_DAYS) - 1
    assert rho < 0
    assert result.equity.iloc[-1] == pytest.approx((1 + rho) ** 5)
    assert result.equity.iloc[-1] < 1.0


def test_no_lookahead():
    prices = make_prices([100, 100, 200, 200])
    positions = pd.Series([0.0, 1.0, 0.0, 0.0], index=prices.index)
    result = run(prices, positions, cost_bps=0.0)
    assert result.equity.iloc[-1] == pytest.approx(2.0)

    late = pd.Series([0.0, 0.0, 1.0, 0.0], index=prices.index)
    result = run(prices, late, cost_bps=0.0)
    assert result.equity.iloc[-1] == pytest.approx(1.0)


def test_costs_charged_on_position_changes():
    prices = make_prices([100] * 4) 
    positions = pd.Series([1.0, 1.0, 0.0, 0.0], index=prices.index)
    result = run(prices, positions, cost_bps=10.0)
    assert result.equity.iloc[-1] == pytest.approx((1 - 0.001) ** 2)
    assert result.stats["trades"] == 2


def test_max_drawdown():
    prices = make_prices([100, 120, 60, 90])
    positions = pd.Series(1.0, index=prices.index)
    result = run(prices, positions, cost_bps=0.0)
    assert result.stats["max_drawdown"] == pytest.approx(-0.5)


def test_max_drawdown_includes_starting_capital_as_a_peak():
    prices = make_prices([100] * 4)
    positions = pd.Series(1.0, index=prices.index)
    result = run(prices, positions, cost_bps=500.0)
    assert result.equity.iloc[0] == pytest.approx(0.95)
    assert result.stats["max_drawdown"] == pytest.approx(-0.05)


def test_cash_earns_the_risk_free_rate():
    prices = make_prices([100] * 5)
    positions = pd.Series(0.0, index=prices.index)
    result = run(prices, positions, cost_bps=0.0, rf_annual=0.05)

    rho = 1.05 ** (1 / TRADING_DAYS) - 1
    assert result.equity.iloc[-1] == pytest.approx((1 + rho) ** 5)
    assert result.stats["sharpe"] == 0.0


def test_sortino_exceeds_sharpe_when_downside_is_small():
    prices = make_prices([100, 110, 109, 120, 119, 130])
    positions = pd.Series(1.0, index=prices.index)
    stats = run(prices, positions, cost_bps=0.0).stats
    assert stats["sortino"] > stats["sharpe"]


def test_vol_target_scales_inversely_with_volatility():
    calm = [100 + (1 if i % 2 else -1) for i in range(40)]
    wild = [100 + (2 if i % 2 else -2) for i in range(40)]
    df = pd.DataFrame({"Close": make_prices(calm + wild)})

    positions = vol_target(df, target_vol=0.10, lookback=20, max_leverage=1.0)
    assert (positions >= 0).all() and (positions <= 1.0).all()

    calm_pos = positions.iloc[35]
    wild_pos = positions.iloc[-1]
    assert wild_pos == pytest.approx(calm_pos / 2, rel=0.2)


def test_sma_crossover_positions_are_zero_or_one():
    prices = make_prices(list(range(100, 130)))
    df = pd.DataFrame({"Close": prices})
    positions = sma_crossover(df, fast=3, slow=10)
    assert set(positions.unique()) <= {0.0, 1.0}
    assert positions.iloc[-1] == 1.0
