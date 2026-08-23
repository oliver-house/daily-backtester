# Daily Backtesting Engine

Daily backtester for SMA crossover and volatility-targeting rules, benchmarked against buy-and-hold, with statistical corrections. Data: Yahoo Finance.

[CI](https://github.com/oliver-house/daily-backtester/actions/workflows/ci.yml)

**[Live dashboard](https://oliver-house.github.io/daily-backtester/)**

## Methodology

- In-sample vs out-of-sample Sharpe scatter
- White's Reality Check
- Walk-forward validation with embargo

## Engineering

- The Python engine is tested against known correct outputs; the JavaScript engine the dashboard actually runs is checked to agree with it to 1e-12 on every test case.
- Fully offline test suite, run in CI on every push.
