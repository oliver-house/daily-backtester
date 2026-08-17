from dataclasses import dataclass

import pandas as pd

TRADING_DAYS = 252


@dataclass
class Result:
    equity: pd.Series
    daily_returns: pd.Series
    stats: dict


def run(
    prices: pd.Series,
    positions: pd.Series,
    cost_bps: float = 5.0,
    rf_annual: float = 0.0,
) -> Result:
    if len(prices) < 2:
        raise ValueError(
            f"need at least 2 days of data to compute statistics, got {len(prices)} "
            "(check the ticker and any --start date)"
        )
    if (prices <= 0).any():
        raise ValueError("prices must be strictly positive (found a zero or negative price)")
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be non-negative, got {cost_bps}")
    if rf_annual <= -1:
        raise ValueError(f"rf_annual must exceed -1, got {rf_annual}")

    positions = positions.reindex(prices.index).fillna(0.0)
    asset_returns = prices.pct_change().fillna(0.0)
    held = positions.shift(1).fillna(0.0)
    costs = positions.diff().abs().fillna(positions.abs()) * cost_bps / 10_000
    rf_daily = (1.0 + rf_annual) ** (1 / TRADING_DAYS) - 1.0
    daily = held * asset_returns + (1.0 - held) * rf_daily - costs

    growth = 1.0 + daily
    if (growth <= 0).any():
        first = growth.index[growth <= 0][0]
        raise ValueError(
            f"costs wipe out all equity on {first}: net daily return "
            f"{daily.loc[first]:.4f} <= -1 (cost_bps={cost_bps})"
        )

    equity = growth.cumprod()
    stats = _stats(equity, daily, positions, rf_daily)
    return Result(equity=equity, daily_returns=daily, stats=stats)


def _risk_adjusted_ratio(mean_excess: float, dispersion: float) -> float:
    if dispersion > 0:
        return mean_excess / dispersion * TRADING_DAYS**0.5
    if mean_excess > 0:
        return float("inf")
    if mean_excess < 0:
        return float("-inf")
    return 0.0


def _stats(
    equity: pd.Series, daily: pd.Series, positions: pd.Series, rf_daily: float = 0.0
) -> dict:
    n_days = len(daily)
    total_return = equity.iloc[-1] - 1.0
    years = n_days / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1.0 if years > 0 else 0.0
    vol = daily.std() * TRADING_DAYS**0.5

    excess = daily - rf_daily
    sharpe = _risk_adjusted_ratio(excess.mean(), excess.std())

    downside_dev = ((excess.clip(upper=0.0) ** 2).mean()) ** 0.5
    sortino = _risk_adjusted_ratio(excess.mean(), downside_dev)

    if sharpe in (float("inf"), float("-inf")):
        sharpe_se = float("inf")
        sharpe_ci_low = sharpe_ci_high = sharpe
    else:
        sharpe_se = ((TRADING_DAYS + 0.5 * sharpe**2) / n_days) ** 0.5
        sharpe_ci_low = sharpe - 1.96 * sharpe_se
        sharpe_ci_high = sharpe + 1.96 * sharpe_se

    running_peak = equity.cummax().clip(lower=1.0)
    max_drawdown = (equity / running_peak - 1.0).min()
    trades = int((positions.diff().fillna(positions.abs()) != 0).sum())
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "sharpe_se": sharpe_se,
        "sharpe_ci_low": sharpe_ci_low,
        "sharpe_ci_high": sharpe_ci_high,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "trades": trades,
        "days": n_days,
    }
