"""Strategies: plain functions mapping OHLCV data to a daily position Series.

A position is the fraction of equity held in the asset that day, in [-1, 1].
The engine applies each day's position to the *next* day's return.
"""

import pandas as pd


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def sma_crossover(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    fast_sma = df["Close"].rolling(fast).mean()
    slow_sma = df["Close"].rolling(slow).mean()
    return (fast_sma > slow_sma).astype(float)
