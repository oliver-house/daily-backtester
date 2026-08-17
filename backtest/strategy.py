import pandas as pd

from .engine import TRADING_DAYS


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def sma_crossover(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    if fast < 1:
        raise ValueError(f"fast window must be at least 1, got {fast}")
    if fast >= slow:
        raise ValueError(f"fast window must be shorter than slow ({fast} >= {slow})")

    fast_sma = df["Close"].rolling(fast).mean()
    slow_sma = df["Close"].rolling(slow).mean()
    return (fast_sma > slow_sma).astype(float)


def vol_target(
    df: pd.DataFrame,
    target_vol: float = 0.10,
    lookback: int = 20,
    max_leverage: float = 1.0,
) -> pd.Series:
    if target_vol < 0:
        raise ValueError(f"target_vol must be non-negative, got {target_vol}")
    if lookback < 1:
        raise ValueError(f"lookback must be at least 1, got {lookback}")
    if not 0 <= max_leverage <= 1:
        raise ValueError(f"max_leverage must lie in [0, 1], got {max_leverage}")

    returns = df["Close"].pct_change()
    realised = returns.rolling(lookback).std() * TRADING_DAYS**0.5
    usable = realised.where(realised > 0)
    return (target_vol / usable).clip(upper=max_leverage).fillna(0.0)
