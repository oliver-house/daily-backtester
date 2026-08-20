import argparse
import math
from statistics import median

import pandas as pd

from backtest import (
    add_data_args,
    buy_and_hold,
    load_daily,
    loader,
    paired_test,
    run,
    sma_crossover,
    vol_target,
)
from backtest.bootstrap import reality_check
from backtest.engine import TRADING_DAYS, annualised_ratio
from backtest.strategy import warmup_days
from backtest.validation import feasible_folds, walk_forward

FAST_WINDOWS = (10, 20, 30, 50, 75, 100)
SLOW_WINDOWS = (50, 100, 150, 200, 250)
TARGET_VOLS = (0.05, 0.10, 0.15, 0.20, 0.30)
LOOKBACKS = (10, 20, 40, 60, 120)

UNIVERSE = ("SPY", "QQQ", "DIA", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN",
            "JPM", "XOM", "JNJ", "KO",
            "BAC", "CAT", "WMT", "DIS", "GLD", "TLT")

STRATEGY_GRIDS = {
    "sma": (
        sma_crossover,
        [{"fast": f, "slow": s} for f in FAST_WINDOWS for s in SLOW_WINDOWS if f < s],
    ),
    "vol": (
        vol_target,
        [{"target_vol": v, "lookback": n} for v in TARGET_VOLS for n in LOOKBACKS],
    ),
}


def label(params: dict) -> str:
    return "/".join(str(v) for v in params.values())


def sweep_ticker(ticker, strategy="sma", cost_bps=2.0, rf=0.04, start=None,
                 load=load_daily, n_boot=1000, seed=0, n_folds=8) -> dict:
    df = load(ticker)
    if start:
        df = df.loc[start:]

    split = len(df) // 2
    train, test = df.iloc[:split], df.iloc[split:]
    rf_daily = (1.0 + rf) ** (1 / TRADING_DAYS) - 1.0
    fn, grid = STRATEGY_GRIDS[strategy]

    def daily_returns(positions):
        return run(df["Close"], positions, cost_bps=cost_bps, rf_annual=rf).daily_returns

    def sharpe(daily, window):
        excess = daily.loc[window.index] - rf_daily
        return annualised_ratio(excess.mean(), excess.std())

    scored = []
    for params in grid:
        daily = daily_returns(fn(df, **params))
        scored.append((params, sharpe(daily, train), sharpe(daily, test), daily))
    scored.sort(key=lambda row: row[1], reverse=True)

    best_params, best_is, best_oos, best_daily = scored[0]
    hold_daily = daily_returns(buy_and_hold(df))

    train_differences = pd.DataFrame(
        {label(params): (daily - hold_daily).loc[train.index] for params, _, _, daily in scored}
    )
    rc = reality_check(train_differences, n_boot=n_boot, rng=seed)

    embargo = max(warmup_days(fn, params) for params in grid)
    folds = feasible_folds(len(df), embargo, n_folds)
    wf = walk_forward(df, fn, grid, n_folds=folds, cost_bps=cost_bps, rf=rf) if folds else None

    se = (TRADING_DAYS / len(train)) ** 0.5
    return {
        "ticker": ticker,
        "strategy": strategy,
        "train": (train.index[0].date(), train.index[-1].date(), len(train)),
        "test": (test.index[0].date(), test.index[-1].date(), len(test)),
        "grid_size": len(grid),
        "top": [(p, i, o) for p, i, o, _ in scored[:5]],
        "best_params": best_params,
        "best_is": best_is,
        "best_oos": best_oos,
        "median_oos": median(row[2] for row in scored),
        "hold_oos": sharpe(hold_daily, test),
        "noise_max": se * math.sqrt(2 * math.log(len(grid))),
        "reality_check": rc,
        "walk_forward": wf,
        "significance": paired_test(
            best_daily.loc[test.index], hold_daily.loc[test.index]
        ),
    }


def print_report(r: dict) -> None:
    print(f"\n{r['ticker']} — {r['strategy'].upper()} parameter sweep")
    print(f"  Train: {r['train'][0]} to {r['train'][1]}  ({r['train'][2]} days)")
    print(f"  Test:  {r['test'][0]} to {r['test'][1]}  ({r['test'][2]} days)")
    print(f"  Grid:  {r['grid_size']} parameter sets\n")

    print("  Top 5 by in-sample Sharpe")
    print(f"    {'params':>12}{'IS Sharpe':>12}{'OOS Sharpe':>13}")
    for params, is_sharpe, oos_sharpe in r["top"]:
        print(f"    {label(params):>12}{is_sharpe:>12.2f}{oos_sharpe:>13.2f}")

    sig = r["significance"]
    print(f"\n  Best in-sample set             {label(r['best_params'])}")
    print(f"    its in-sample Sharpe         {r['best_is']:>7.2f}")
    print(f"    its out-of-sample Sharpe     {r['best_oos']:>7.2f}")
    print(f"  Median OOS Sharpe, all sets    {r['median_oos']:>7.2f}")
    print(f"  Buy & hold OOS Sharpe          {r['hold_oos']:>7.2f}")

    rc = r["reality_check"]
    print()
    print(f"  Correcting for having tried {r['grid_size']} parameter sets, in-sample:")
    print(f"    Best-of-{r['grid_size']} Sharpe under pure noise {r['noise_max']:>7.2f}"
          f"   (assumes {r['grid_size']} independent trials)")
    print(f"    Reality Check p-value          {rc['p_value']:>7.3f}"
          f"   ({rc['n_boot']} stationary-bootstrap draws, "
          f"mean block {rc['mean_block']:.0f}d)")

    wf = r["walk_forward"]
    print()
    if wf is None:
        print("  Walk-forward: skipped, not enough history for even one embargoed fold")
    else:
        pooled = wf["pooled"]
        print(f"  Walk-forward, {wf['n_folds']} anchored folds"
              f"  ({wf['embargo']}d embargo, {wf['pooled_days']} scored days)")
        print(f"    Folds the selected set won     {wf['folds_won']:>3}/{wf['n_folds']}")
        print(f"    Median IS to OOS decay         {wf['median_decay']:>7.2f}")
        print(f"    Pooled Sharpe                  {wf['pooled_sharpe']:>7.2f}"
              f"   vs {wf['pooled_hold_sharpe']:.2f} buy & hold")
        print(f"    Pooled difference              {pooled['mean_diff'] * TRADING_DAYS:>+7.2%}/yr"
              f"   t = {pooled['t_stat']:+.2f}, p = {pooled['p_value']:.3f}")

    print()
    print(f"  Best set vs buy & hold (single-split OOS):  "
          f"{sig['mean_diff'] * TRADING_DAYS:+.2%}/yr, "
          f"t = {sig['t_stat']:+.2f}, p = {sig['p_value']:.3f}")


def print_summary(records: list) -> None:
    n = len(records)
    hold_wins = sum(1 for r in records if r["hold_oos"] > r["best_oos"])
    cleared = sum(1 for r in records if r["best_is"] > r["noise_max"])
    significant = sum(1 for r in records if r["significance"]["p_value"] < 0.05)
    survived_rc = sum(1 for r in records if r["reality_check"]["p_value"] < 0.05)
    with_folds = [r["walk_forward"] for r in records if r["walk_forward"] is not None]
    wf_won = sum(wf["folds_won"] for wf in with_folds)
    wf_total = sum(wf["n_folds"] for wf in with_folds)
    wf_significant = sum(
        1 for wf in with_folds
        if wf["pooled"]["p_value"] < 0.05 and wf["pooled_sharpe"] > wf["pooled_hold_sharpe"]
    )
    gaps = [r["hold_oos"] - r["best_oos"] for r in records]
    decay = [r["best_is"] - r["best_oos"] for r in records]

    print(f"\n{'=' * 62}")
    print(f"SUMMARY — {n} tickers, {records[0]['strategy'].upper()} strategy")
    print(f"{'=' * 62}")
    print(f"  Buy & hold beat the selected set OOS    {hold_wins}/{n}")
    print(f"  Median OOS Sharpe gap (hold - best)     {median(gaps):+.2f}")
    print(f"  Median in-sample to OOS decay           {median(decay):+.2f}")
    print(f"  In-sample best cleared noise ceiling    {cleared}/{n}")
    print(f"  In-sample best survived Reality Check   {survived_rc}/{n}")
    print(f"  Gap significant at p < 0.05             {significant}/{n}")
    if with_folds:
        print(f"  Walk-forward folds the strategy won     {wf_won}/{wf_total}")
        print(f"  Walk-forward pooled win, p < 0.05       {wf_significant}/{len(with_folds)}")
    else:
        print("  Walk-forward: skipped for every ticker (not enough history)")
    print("\n  Universe is survivorship-biased: these tickers are liquid today,")
    print("  so delisted and bankrupt names are absent by construction.")


def main() -> None:
    parser = argparse.ArgumentParser(description="In-sample / out-of-sample sweep")
    parser.add_argument("tickers", nargs="*", help="one or more tickers, e.g. SPY QQQ")
    parser.add_argument("--universe", action="store_true",
                        help=f"sweep the built-in {len(UNIVERSE)}-ticker universe")
    parser.add_argument("--strategy", choices=sorted(STRATEGY_GRIDS), default="sma")
    parser.add_argument("--cost-bps", type=float, default=2.0, help="cost per trade, bps")
    parser.add_argument("--rf", type=float, default=0.04,
                        help="annual risk-free rate earned on cash, e.g. 0.04. Idle cash still earns interest at this rate, so Sharpe/Sortino are computed on returns in excess of it rather than crediting idle cash with zero")
    parser.add_argument("--start", help="start date, e.g. 2010-01-01")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="stationary-bootstrap draws for the Reality Check")
    parser.add_argument("--seed", type=int, default=0,
                        help="bootstrap seed, so a reported p-value is reproducible")
    parser.add_argument("--folds", type=int, default=8,
                        help="walk-forward folds, each reselecting from the grid")
    add_data_args(parser)
    args = parser.parse_args()

    tickers = list(UNIVERSE) if args.universe else args.tickers
    if not tickers:
        parser.error("give at least one ticker, or --universe")

    records = []
    for ticker in tickers:
        try:
            record = sweep_ticker(ticker, args.strategy, args.cost_bps,
                                  args.rf, args.start, load=loader(args),
                                  n_boot=args.n_boot, seed=args.seed,
                                  n_folds=args.folds)
        except (ValueError, RuntimeError) as exc:
            print(f"\n{ticker}: skipped ({exc})")
            continue
        records.append(record)
        print_report(record)

    if len(records) > 1:
        print_summary(records)


if __name__ == "__main__":
    main()
