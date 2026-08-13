"""Vectorized daily backtest engine."""

from dataclasses import dataclass

import pandas as pd

TRADING_DAYS = 252


@dataclass
class Result:
    equity: pd.Series          # cumulative equity curve, starts at 1.0
    daily_returns: pd.Series   # net daily strategy returns
    stats: dict


def run(prices: pd.Series, positions: pd.Series, cost_bps: float = 5.0) -> Result:
    """Backtest daily positions against a close price series.

    A position set on day t (using information through t's close) earns
    day t+1's close-to-close return. Each unit of position change pays
    `cost_bps` basis points of that day's traded notional.
    """
    positions = positions.reindex(prices.index).fillna(0.0)
    asset_returns = prices.pct_change().fillna(0.0)
    held = positions.shift(1).fillna(0.0)
    costs = positions.diff().abs().fillna(positions.abs()) * cost_bps / 10_000
    daily = held * asset_returns - costs
    equity = (1.0 + daily).cumprod()
    return Result(equity=equity, daily_returns=daily, stats=_stats(equity, daily, positions))


def _stats(equity: pd.Series, daily: pd.Series, positions: pd.Series) -> dict:
    n_days = len(daily)
    total_return = equity.iloc[-1] - 1.0
    years = n_days / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1.0 if years > 0 else 0.0
    vol = daily.std() * TRADING_DAYS**0.5
    sharpe = daily.mean() / daily.std() * TRADING_DAYS**0.5 if daily.std() > 0 else 0.0
    max_drawdown = (equity / equity.cummax() - 1.0).min()
    trades = int((positions.diff().fillna(positions.abs()) != 0).sum())
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "trades": trades,
        "days": n_days,
    }
