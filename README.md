# Daily Backtesting Engine

Daily backtester. SMA crossover and volatility-targeting strategies,
benchmarked against buy-and-hold with significance testing.

**[Live dashboard](https://oliver-house.github.io/daily-backtester/)**

## Usage

```
python run.py SPY --strategy sma --fast 50 --slow 200
python sweep.py --universe --strategy vol
python report.py
```

## Tests

```
python -m pytest
```
