import argparse
import math
from statistics import median

from backtest import buy_and_hold, load_daily, run, sma_crossover
from backtest.engine import TRADING_DAYS, _risk_adjusted_ratio

FAST_WINDOWS = (10, 20, 30, 50, 75, 100)
SLOW_WINDOWS = (50, 100, 150, 200, 250)


def main() -> None:
    parser = argparse.ArgumentParser(description="In-sample / out-of-sample SMA sweep")
    parser.add_argument("ticker", help="US equity/ETF ticker, e.g. SPY")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="cost per trade, bps")
    parser.add_argument("--rf", type=float, default=0.0, help="annual risk-free rate")
    parser.add_argument("--start", help="start date, e.g. 2010-01-01")
    args = parser.parse_args()

    df = load_daily(args.ticker)
    if args.start:
        df = df.loc[args.start:]

    split = len(df) // 2
    train, test = df.iloc[:split], df.iloc[split:]
    rf_daily = (1.0 + args.rf) ** (1 / TRADING_DAYS) - 1.0

    def sharpe(positions, window):
        full = run(df["Close"], positions, cost_bps=args.cost_bps, rf_annual=args.rf)
        excess = full.daily_returns.loc[window.index] - rf_daily
        return _risk_adjusted_ratio(excess.mean(), excess.std())

    grid = [(f, s) for f in FAST_WINDOWS for s in SLOW_WINDOWS if f < s]
    scored = []
    for fast, slow in grid:
        positions = sma_crossover(df, fast, slow)
        scored.append((fast, slow, sharpe(positions, train), sharpe(positions, test)))

    scored.sort(key=lambda row: row[2], reverse=True)
    best_fast, best_slow, best_is, best_oos = scored[0]
    median_oos = median(row[3] for row in scored)

    hold = buy_and_hold(df)
    hold_oos = sharpe(hold, test)

    se = (TRADING_DAYS / len(train)) ** 0.5
    noise_max = se * math.sqrt(2 * math.log(len(grid)))

    print(f"\n{args.ticker.upper()} — SMA parameter sweep")
    print(f"  Train: {train.index[0].date()} to {train.index[-1].date()}  ({len(train)} days)")
    print(f"  Test:  {test.index[0].date()} to {test.index[-1].date()}  ({len(test)} days)")
    print(f"  Grid:  {len(grid)} (fast, slow) pairs\n")

    print("  Top 5 by in-sample Sharpe")
    print(f"    {'fast':>5}{'slow':>6}{'IS Sharpe':>12}{'OOS Sharpe':>13}")
    for fast, slow, is_sharpe, oos_sharpe in scored[:5]:
        print(f"    {fast:>5}{slow:>6}{is_sharpe:>12.2f}{oos_sharpe:>13.2f}")

    print(f"\n  Best in-sample pair            ({best_fast}, {best_slow})")
    print(f"    its in-sample Sharpe         {best_is:>7.2f}")
    print(f"    its out-of-sample Sharpe     {best_oos:>7.2f}")
    print(f"  Median OOS Sharpe, all pairs   {median_oos:>7.2f}")
    print(f"  Buy & hold OOS Sharpe          {hold_oos:>7.2f}")
    print(f"  Best-of-{len(grid)} Sharpe under pure noise {noise_max:>7.2f}")


if __name__ == "__main__":
    main()
