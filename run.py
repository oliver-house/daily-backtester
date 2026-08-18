import argparse
from pathlib import Path

from backtest import (
    buy_and_hold,
    load_daily,
    paired_test,
    run,
    sma_crossover,
    vol_target,
)
from backtest.engine import TRADING_DAYS

FIGURES_DIR = Path("figures")
EQUITY_PLOT_FILENAME = "equity.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal daily backtester")
    parser.add_argument("ticker", help="US equity/ETF ticker, e.g. SPY")
    parser.add_argument("--strategy", choices=["sma", "hold", "vol"], default="sma")
    parser.add_argument("--fast", type=int, default=50, help="fast SMA window")
    parser.add_argument("--slow", type=int, default=200, help="slow SMA window")
    parser.add_argument("--target-vol", type=float, default=0.10,
                        help="annual volatility target for --strategy vol")
    parser.add_argument("--lookback", type=int, default=20,
                        help="realised-vol lookback for --strategy vol")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="cost per trade, bps")
    parser.add_argument("--rf", type=float, default=0.0,
                        help="annual risk-free rate earned on cash, e.g. 0.04")
    parser.add_argument("--start", help="start date, e.g. 2010-01-01")
    parser.add_argument("--no-plot", action="store_true", help="skip saving the equity curve")
    args = parser.parse_args()

    df = load_daily(args.ticker)
    if args.start:
        df = df.loc[args.start:]

    if args.strategy == "sma":
        positions = sma_crossover(df, args.fast, args.slow)
        label = f"SMA {args.fast}/{args.slow}"
    elif args.strategy == "vol":
        positions = vol_target(df, args.target_vol, args.lookback)
        label = f"Vol target {args.target_vol:.0%}"
    else:
        positions = buy_and_hold(df)
        label = "Buy & hold"

    result = run(df["Close"], positions, cost_bps=args.cost_bps, rf_annual=args.rf)

    benchmark = None
    if args.strategy != "hold":
        benchmark = run(df["Close"], buy_and_hold(df),
                        cost_bps=args.cost_bps, rf_annual=args.rf)

    print(f"\n{args.ticker.upper()} — {label}  "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    rows = [
        ("total_return", "Total return", "{:+.1%}", None),
        ("cagr", "CAGR", "{:+.2%}", "{:+.2%}"),
        ("annual_vol", "Annual vol", "{:.2%}", None),
        ("sharpe", "Sharpe", "{:.2f}", "{:+.2f}"),
        ("sortino", "Sortino", "{:.2f}", None),
        ("max_drawdown", "Max drawdown", "{:.1%}", None),
        ("trades", "Trades", "{}", None),
        ("days", "Days", "{}", None),
    ]
    if benchmark:
        print(f"  {'':<14}{'Strategy':>12}{'Buy & hold':>13}{'Diff':>10}")
    for key, name, spec, diff_spec in rows:
        line = f"  {name:<14}{spec.format(result.stats[key]):>12}"
        if benchmark:
            line += f"{spec.format(benchmark.stats[key]):>13}"
            line += (f"{diff_spec.format(result.stats[key] - benchmark.stats[key]):>10}"
                     if diff_spec else " " * 10)
        print(line)

    stats = result.stats
    print(f"\n  Sharpe 95% CI: {stats['sharpe_ci_low']:+.2f} to {stats['sharpe_ci_high']:+.2f}"
          f"  (se {stats['sharpe_se']:.2f})")

    if benchmark:
        test = paired_test(result.daily_returns, benchmark.daily_returns)
        print(f"  vs buy & hold: {test['mean_diff'] * TRADING_DAYS:+.2%}/yr, "
              f"t = {test['t_stat']:+.2f}, p = {test['p_value']:.3f} "
              f"(Newey-West, {test['lags']} lags)")

    if not args.no_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n(matplotlib not installed, skipping plot: pip install matplotlib)")
        else:
            result.equity.plot(title=f"{args.ticker.upper()} — {label}", ylabel="Equity")
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            out_path = FIGURES_DIR / EQUITY_PLOT_FILENAME
            plt.savefig(out_path)
            print(f"\nEquity curve saved to {out_path}")


if __name__ == "__main__":
    main()
