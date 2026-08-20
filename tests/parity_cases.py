import numpy as np
import pandas as pd

from backtest import load_daily
from tests.conftest import FIXTURE_CACHE


def _walk(n, seed, start=100.0, drift=0.0002, vol=0.012):
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, vol, n)
    return list(np.round(start * np.exp(np.cumsum(steps)), 6))


def cases() -> list[dict]:
    spy = load_daily("SPY", str(FIXTURE_CACHE), allow_download=False)
    real = [round(float(v), 6) for v in spy["Close"].iloc[-400:]]

    flat = [100.0] * 40
    regime_shift = [100 + (1 if i % 2 else -1) for i in range(30)] + [
        100 + (4 if i % 2 else -4) for i in range(30)
    ]

    return [
        {
            "name": "sma warm-up on a short series",
            "prices": _walk(60, seed=1),
            "strategy": "sma", "params": {"fast": 5, "slow": 20},
            "cost_bps": 5.0, "rf": 0.0,
        },
        {
            "name": "sma where the sample barely exceeds the slow window",
            "prices": _walk(205, seed=2),
            "strategy": "sma", "params": {"fast": 50, "slow": 200},
            "cost_bps": 5.0, "rf": 0.04,
        },
        {
            "name": "sma on real prices at a realistic risk-free rate",
            "prices": real,
            "strategy": "sma", "params": {"fast": 20, "slow": 100},
            "cost_bps": 5.0, "rf": 0.04,
        },
        {
            "name": "sma on a perfectly flat series, every window tied",
            "prices": flat,
            "strategy": "sma", "params": {"fast": 3, "slow": 10},
            "cost_bps": 5.0, "rf": 0.0,
        },
        {
            "name": "vol target across a volatility regime change",
            "prices": regime_shift,
            "strategy": "vol", "params": {"targetVol": 0.10, "lookback": 20},
            "cost_bps": 5.0, "rf": 0.0,
        },
        {
            "name": "vol target with a zero-variance window",
            "prices": flat,
            "strategy": "vol", "params": {"targetVol": 0.10, "lookback": 5},
            "cost_bps": 5.0, "rf": 0.0,
        },
        {
            "name": "vol target on real prices, short lookback",
            "prices": real,
            "strategy": "vol", "params": {"targetVol": 0.15, "lookback": 10},
            "cost_bps": 2.5, "rf": 0.04,
        },
        {
            "name": "vol target at zero, a legitimate always-cash strategy",
            "prices": _walk(120, seed=3),
            "strategy": "vol", "params": {"targetVol": 0.0, "lookback": 20},
            "cost_bps": 5.0, "rf": 0.03,
        },
        {
            "name": "buy and hold with no costs and no interest",
            "prices": _walk(300, seed=4),
            "strategy": "hold", "params": {},
            "cost_bps": 0.0, "rf": 0.0,
        },
        {
            "name": "buy and hold under a negative risk-free rate",
            "prices": _walk(300, seed=5),
            "strategy": "hold", "params": {},
            "cost_bps": 5.0, "rf": -0.02,
        },
        {
            "name": "zero dispersion, where Sharpe is signed infinity",
            "prices": [100.0 * 1.001**i for i in range(50)],
            "strategy": "hold", "params": {},
            "cost_bps": 0.0, "rf": 0.0,
        },
        {
            "name": "costs large enough to wipe out equity",
            "prices": [100.0, 100.0, 40.0, 60.0],
            "strategy": "sma", "params": {"fast": 1, "slow": 2},
            "cost_bps": 9000.0, "rf": 0.0,
        },
        {
            "name": "a cost high enough to bite but not to ruin",
            "prices": _walk(200, seed=6),
            "strategy": "sma", "params": {"fast": 2, "slow": 5},
            "cost_bps": 250.0, "rf": 0.0,
        },
    ]


def python_positions(case: dict, prices: pd.Series) -> pd.Series:
    from backtest import buy_and_hold, sma_crossover, vol_target

    df = pd.DataFrame({"Close": prices})
    params = case["params"]
    if case["strategy"] == "sma":
        return sma_crossover(df, params["fast"], params["slow"])
    if case["strategy"] == "vol":
        return vol_target(df, params["targetVol"], params["lookback"])
    return buy_and_hold(df)
