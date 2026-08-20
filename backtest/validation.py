import pandas as pd

from .engine import TRADING_DAYS, annualised_ratio, run
from .inference import paired_test
from .strategy import buy_and_hold, warmup_days


def feasible_folds(n: int, embargo: int, preferred: int) -> int:
    for folds in range(preferred, 0, -1):
        if n // (folds + 1) >= 2 * max(embargo, 1):
            return folds
    return 0


def _windows(n: int, n_folds: int) -> list[tuple[int, int, int]]:
    if n_folds < 1:
        raise ValueError(f"n_folds must be at least 1, got {n_folds}")
    block = n // (n_folds + 1)
    if block < 2:
        raise ValueError(
            f"{n} rows split into {n_folds + 1} blocks leaves {block} rows per block; "
            "use fewer folds or a longer sample"
        )

    windows = []
    for fold in range(n_folds):
        test_start = block * (fold + 1)
        test_end = n if fold == n_folds - 1 else block * (fold + 2)
        windows.append((0, test_start, test_end))
    return windows


def walk_forward(
    df: pd.DataFrame,
    fn,
    grid: list[dict],
    n_folds: int = 8,
    cost_bps: float = 5.0,
    rf: float = 0.0,
    embargo: int | None = None,
) -> dict:
    if not grid:
        raise ValueError("grid must contain at least one parameter set")

    n = len(df)
    windows = _windows(n, n_folds)
    if embargo is None:
        embargo = max(warmup_days(fn, params) for params in grid)
    if embargo < 0:
        raise ValueError(f"embargo must be non-negative, got {embargo}")

    rf_daily = (1.0 + rf) ** (1 / TRADING_DAYS) - 1.0

    def daily_returns(positions):
        return run(df["Close"], positions, cost_bps=cost_bps, rf_annual=rf).daily_returns

    grid_daily = [(params, daily_returns(fn(df, **params))) for params in grid]
    hold_daily = daily_returns(buy_and_hold(df))

    def sharpe(daily, start: int, end: int) -> float:
        excess = daily.iloc[start:end] - rf_daily
        return annualised_ratio(excess.mean(), excess.std())

    folds, selected_returns, benchmark_returns = [], [], []
    for index, (train_start, test_start, test_end) in enumerate(windows):
        scored = [
            (params, sharpe(daily, train_start, test_start), daily)
            for params, daily in grid_daily
        ]
        best_params, best_is, best_daily = max(scored, key=lambda row: row[1])

        scored_start = min(test_start + embargo, test_end)
        if scored_start >= test_end:
            raise ValueError(
                f"fold {index + 1} has {test_end - test_start} test days but the embargo "
                f"removes {embargo}; use fewer folds or a shorter warm-up"
            )

        selected_returns.append(best_daily.iloc[scored_start:test_end])
        benchmark_returns.append(hold_daily.iloc[scored_start:test_end])
        folds.append(
            {
                "fold": index + 1,
                "train": (df.index[train_start].date(), df.index[test_start - 1].date()),
                "test": (df.index[scored_start].date(), df.index[test_end - 1].date()),
                "train_days": test_start - train_start,
                "test_days": test_end - scored_start,
                "embargoed_days": scored_start - test_start,
                "params": best_params,
                "is_sharpe": best_is,
                "oos_sharpe": sharpe(best_daily, scored_start, test_end),
                "hold_oos_sharpe": sharpe(hold_daily, scored_start, test_end),
            }
        )

    strategy_series = pd.concat(selected_returns)
    benchmark_series = pd.concat(benchmark_returns)
    wins = sum(1 for f in folds if f["oos_sharpe"] > f["hold_oos_sharpe"])
    decays = sorted(f["is_sharpe"] - f["oos_sharpe"] for f in folds)
    middle = len(decays) // 2

    return {
        "folds": folds,
        "n_folds": n_folds,
        "embargo": embargo,
        "folds_won": wins,
        "median_decay": (
            decays[middle] if len(decays) % 2 else (decays[middle - 1] + decays[middle]) / 2
        ),
        "pooled": paired_test(strategy_series, benchmark_series),
        "pooled_sharpe": annualised_ratio(
            (strategy_series - rf_daily).mean(), (strategy_series - rf_daily).std()
        ),
        "pooled_hold_sharpe": annualised_ratio(
            (benchmark_series - rf_daily).mean(), (benchmark_series - rf_daily).std()
        ),
        "pooled_days": len(strategy_series),
    }
