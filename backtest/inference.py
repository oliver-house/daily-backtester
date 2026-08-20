import math

import pandas as pd


def _default_lags(n: int) -> int:
    if n < 2:
        return 0
    return int(4 * (n / 100) ** (2 / 9))


def newey_west_se(x, lags: int | None = None) -> float:
    series = pd.Series(x).dropna()
    n = len(series)
    if n < 2:
        return 0.0

    if lags is None:
        lags = _default_lags(n)
    lags = max(0, min(lags, n - 1))

    dev = (series - series.mean()).to_numpy()
    iid_variance = float((dev * dev).sum() / n)
    variance = iid_variance
    for j in range(1, lags + 1):
        cov = float((dev[j:] * dev[:-j]).sum() / n)
        variance += 2.0 * (1.0 - j / (lags + 1)) * cov

    if variance <= 0:
        variance = iid_variance
    if variance <= 0:
        return 0.0
    return (variance / n) ** 0.5


def paired_test(strategy_daily, benchmark_daily) -> dict:
    diff = (pd.Series(strategy_daily) - pd.Series(benchmark_daily)).dropna()
    n = len(diff)
    if n == 0:
        return {"mean_diff": 0.0, "se": 0.0, "t_stat": 0.0, "p_value": 1.0,
                "days": 0, "lags": 0}

    mean_diff = float(diff.mean())
    se = newey_west_se(diff)
    if se > 0:
        t_stat = mean_diff / se
        p_value = math.erfc(abs(t_stat) / 2**0.5)
    else:
        t_stat = 0.0
        p_value = 1.0

    return {
        "mean_diff": mean_diff,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "days": n,
        "lags": _default_lags(n),
    }
