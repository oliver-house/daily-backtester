# Daily Backtesting Engine

[![CI](https://github.com/oliver-house/daily-backtester/actions/workflows/ci.yml/badge.svg)](https://github.com/oliver-house/daily-backtester/actions/workflows/ci.yml)

Daily backtester for SMA-crossover and volatility-targeting rules, benchmarked against buy-and-hold with data-snooping-aware significance testing. 18 tickers, 2018-2026, daily bars from Yahoo Finance.

**[Live dashboard](https://oliver-house.github.io/daily-backtester/)**

## Result

Neither strategy family shows a detectable edge over buy-and-hold. Across all 18 tickers x 2 strategies (36 pairs):

| Test | Result |
|------|--------|
| White's Reality Check, p < 0.05 | **0 / 36** (min p 0.37, median 0.89) |
| Pooled walk-forward difference negative | 33 / 36 |
| Pooled difference significant at p < 0.05 | 12 / 36 - **all negative** |
| Walk-forward folds won | 91 / 270 (34%) |

Not one parameter set survives correction for having searched the grid, and every difference that does reach significance goes the wrong way. That is the finding: the in-sample edge these rules appear to have is selection, and it does not survive out of sample.

SPY under the dashboard defaults (SMA 50/200, 2 bps costs, 4% risk-free) as a worked example: Sharpe 0.52 against 0.61 for buy-and-hold, -3.80%/yr, Reality Check p = 0.805, 0 of 3 walk-forward folds won. On this one ticker the -3.80%/yr gap is not itself statistically significant (t = -1.02, p = 0.309, Newey-West); the aggregate above is what carries the conclusion, not SPY alone.

The universe is survivorship-biased - these tickers are liquid today, so delisted and bankrupt names are absent by construction.

## Methodology

- **White's Reality Check** over a Politis-Romano stationary bootstrap, 1,000 draws, correcting for having searched a 26-point SMA grid (25 for vol-targeting) rather than for testing a single rule
- **Anchored walk-forward** with an embargo equal to the longest warm-up in the grid (250 days), so no test window scores a parameter fitted on data adjacent to it
- **Newey-West** paired t-test, Bartlett kernel, automatic lag selection
- **Best-of-N noise ceiling** - the Sharpe the best of N grid points reaches on noise alone, plotted against the in-sample/out-of-sample Sharpe scatter per ticker

## Engineering

- The Python engine and the JavaScript engine the dashboard actually runs are asserted equal to 1e-12 across 13 cases, including NaN warm-up, signed-infinity Sharpe, and the equity-wipeout error path
- A further test asserts the engine file under test is the one `report.py` inlines into the published dashboard, so the two cannot silently diverge
- 94 tests, green in CI on Python 3.13 and 3.14, which also smoke-tests all three CLIs fully offline; the parity tests need Node installed
- The bootstrap is seeded, so every figure above reproduces exactly from those fixtures, with no network access

## Reproduce

```bash
pip install -r requirements-dev.txt
python -m pytest -q

# regenerates every figure above, offline, from committed fixtures
python report.py --cache-dir tests/fixtures/prices --offline

# a single backtest, or a per-strategy sweep across the universe
python run.py SPY --strategy sma --cache-dir tests/fixtures/prices --offline
python sweep.py --universe --strategy vol --cache-dir tests/fixtures/prices --offline
```

`report.py` defaults to the full 18-ticker universe. `sweep.py` covers one strategy per run, so the 36-pair table above is two `sweep.py` runs or a single `report.py`.

## Layout

```
backtest/engine.py      returns, equity curve, summary stats
backtest/strategy.py    sma_crossover, vol_target, buy_and_hold
backtest/bootstrap.py   stationary bootstrap, White's Reality Check
backtest/inference.py   Newey-West standard errors, paired test
backtest/validation.py  anchored walk-forward with embargo
backtest/data.py        Yahoo fetch with CSV cache
run.py / sweep.py / report.py    one backtest / universe sweep / dashboard build
templates/              dashboard.html and engine.js
```
