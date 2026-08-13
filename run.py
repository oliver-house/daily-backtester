"""Run a daily backtest from the command line.

Example:
    python run.py SPY --strategy sma --fast 50 --slow 200
"""

import argparse
from pathlib import Path

from backtest import load_daily, run, buy_and_hold, sma_crossover

FIGURES_DIR = Path("figures")
EQUITY_PLOT_FILENAME = "equity.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal daily backtester")
    parser.add_argument("ticker", help="US equity/ETF ticker, e.g. SPY")
    parser.add_argument("--strategy", choices=["sma", "hold"], default="sma")
    parser.add_argument("--fast", type=int, default=50, help="fast SMA window")
    parser.add_argument("--slow", type=int, default=200, help="slow SMA window")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="cost per trade, bps")
    parser.add_argument("--start", help="start date, e.g. 2010-01-01")
    parser.add_argument("--plot", action="store_true", help="show equity curve (needs matplotlib)")
    args = parser.parse_args()

    df = load_daily(args.ticker)
    if args.start:
        df = df.loc[args.start:]

    if args.strategy == "sma":
        positions = sma_crossover(df, args.fast, args.slow)
        label = f"SMA {args.fast}/{args.slow}"
    else:
        positions = buy_and_hold(df)
        label = "Buy & hold"

    result = run(df["Close"], positions, cost_bps=args.cost_bps)

    print(f"\n{args.ticker.upper()} — {label}  "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    fmt = {
        "total_return": ("Total return", "{:+.1%}"),
        "cagr": ("CAGR", "{:+.2%}"),
        "annual_vol": ("Annual vol", "{:.2%}"),
        "sharpe": ("Sharpe", "{:.2f}"),
        "max_drawdown": ("Max drawdown", "{:.1%}"),
        "trades": ("Trades", "{}"),
        "days": ("Days", "{}"),
    }
    for key, (name, spec) in fmt.items():
        print(f"  {name:<14}{spec.format(result.stats[key])}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")  # save to file, no GUI window / no blocking
        import matplotlib.pyplot as plt

        result.equity.plot(title=f"{args.ticker.upper()} — {label}", ylabel="Equity")
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / EQUITY_PLOT_FILENAME
        plt.savefig(out_path)
        print(f"\nEquity curve saved to {out_path}")


if __name__ == "__main__":
    main()
