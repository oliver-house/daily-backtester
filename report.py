import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import add_data_args, buy_and_hold, load_daily, loader, run
from backtest.bootstrap import reality_check
from backtest.engine import TRADING_DAYS, annualised_ratio
from backtest.strategy import warmup_days
from backtest.validation import feasible_folds, walk_forward
from sweep import STRATEGY_GRIDS, UNIVERSE, label

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE = TEMPLATE_DIR / "dashboard.html"
ENGINE = TEMPLATE_DIR / "engine.js"
OUTPUT = Path("docs") / "index.html"

DEFAULTS = {"cost_bps": 2.0, "rf": 0.04, "fast": 50, "slow": 200,
            "target_vol": 0.10, "lookback": 20}

EVIDENCE_COST_BPS = 2.0
EVIDENCE_RF = 0.04
EVIDENCE_FOLDS = 8
EVIDENCE_BOOTSTRAP_DRAWS = 1000
EVIDENCE_SEED = 0
NULL_HISTOGRAM_BINS = 40


def _histogram(values, statistic, bins=NULL_HISTOGRAM_BINS) -> dict:
    low = float(min(values.min(), statistic))
    high = float(max(values.max(), statistic))
    if high <= low:
        high = low + 1e-9
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    return {"edges": [round(float(e), 6) for e in edges],
            "counts": [int(c) for c in counts]}


def build_evidence(df, strategy: str) -> dict:
    fn, grid = STRATEGY_GRIDS[strategy]
    rf_daily = (1.0 + EVIDENCE_RF) ** (1 / TRADING_DAYS) - 1.0
    split = len(df) // 2

    def daily_returns(positions):
        return run(df["Close"], positions,
                   cost_bps=EVIDENCE_COST_BPS, rf_annual=EVIDENCE_RF).daily_returns

    def sharpe(daily, start, end):
        excess = daily.iloc[start:end] - rf_daily
        return annualised_ratio(excess.mean(), excess.std())

    hold = daily_returns(buy_and_hold(df))
    points, differences = [], {}
    for params in grid:
        daily = daily_returns(fn(df, **params))
        difference = daily - hold
        differences[label(params)] = difference.iloc[:split]
        points.append({
            "label": label(params),
            "is": round(sharpe(daily, 0, split), 4),
            "oos": round(sharpe(daily, split, len(df)), 4),
        })

    best = max(range(len(points)), key=lambda i: points[i]["is"])
    rc = reality_check(pd.DataFrame(differences), n_boot=EVIDENCE_BOOTSTRAP_DRAWS,
                       rng=EVIDENCE_SEED)
    embargo = max(warmup_days(fn, params) for params in grid)
    folds = feasible_folds(len(df), embargo, EVIDENCE_FOLDS)
    wf = walk_forward(df, fn, grid, n_folds=folds, cost_bps=EVIDENCE_COST_BPS,
                      rf=EVIDENCE_RF) if folds else None

    return {
        "grid": points,
        "best": best,
        "hold_is": round(sharpe(hold, 0, split), 4),
        "hold_oos": round(sharpe(hold, split, len(df)), 4),
        "split_date": df.index[split].strftime("%Y-%m-%d"),
        "reality_check": {
            "p_value": round(rc["p_value"], 4),
            "statistic": round(rc["statistic"], 6),
            "n_boot": rc["n_boot"],
            "mean_block": round(rc["mean_block"], 1),
            "null": _histogram(rc["null"], rc["statistic"]),
        },
        "walk_forward": None if wf is None else {
            "embargo": wf["embargo"],
            "folds_won": wf["folds_won"],
            "pooled_p": round(wf["pooled"]["p_value"], 4),
            "pooled_diff": round(wf["pooled"]["mean_diff"] * TRADING_DAYS, 6),
            "folds": [{
                "fold": f["fold"],
                "from": str(f["test"][0]),
                "to": str(f["test"][1]),
                "params": label(f["params"]),
                "oos": round(f["oos_sharpe"], 4),
                "hold": round(f["hold_oos_sharpe"], 4),
            } for f in wf["folds"]],
        },
    }


def build(tickers, load=load_daily) -> dict:
    data = {
        "generated": date.today().isoformat(),
        "defaults": DEFAULTS,
        "tickers": [],
        "series": {},
        "evidence": {},
        "evidence_settings": {
            "cost_bps": EVIDENCE_COST_BPS,
            "rf": EVIDENCE_RF,
            "folds": EVIDENCE_FOLDS,
            "n_boot": EVIDENCE_BOOTSTRAP_DRAWS,
        },
    }

    for ticker in tickers:
        try:
            df = load(ticker)
        except (ValueError, RuntimeError) as exc:
            print(f"  {ticker}: skipped ({exc})")
            continue

        data["tickers"].append(ticker)
        data["series"][ticker] = {
            "dates": [d.strftime("%Y-%m-%d") for d in df.index],
            "prices": [round(v, 4) for v in df["Close"]],
        }
        data["evidence"][ticker] = {
            strategy: build_evidence(df, strategy) for strategy in STRATEGY_GRIDS
        }
        print(f"  {ticker}: {len(df)} days, evidence for "
              f"{', '.join(sorted(STRATEGY_GRIDS))}")

    return data


def render(data: dict) -> str:
    html = TEMPLATE.read_text(encoding="utf-8")
    if "/*ENGINE*/" not in html:
        raise ValueError(f"{TEMPLATE} has no /*ENGINE*/ marker to splice the engine into")
    html = html.replace("/*ENGINE*/", ENGINE.read_text(encoding="utf-8"), 1)

    payload = json.dumps(data, separators=(",", ":"))
    start, end = html.index("/*DATA*/"), html.index("/*END*/") + len("/*END*/")
    return html[:start] + payload + html[end:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dashboard")
    parser.add_argument("tickers", nargs="*", default=list(UNIVERSE))
    add_data_args(parser)
    args = parser.parse_args()

    tickers = args.tickers or list(UNIVERSE)
    print(f"Building dashboard for {len(tickers)} tickers")
    data = build(tickers, load=loader(args))

    if not data["tickers"]:
        raise SystemExit("no tickers produced results")

    html = render(data)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    size = OUTPUT.stat().st_size / 1024
    print(f"\nWrote {OUTPUT} ({size:.0f} KB, {len(data['tickers'])} tickers)")


if __name__ == "__main__":
    main()
