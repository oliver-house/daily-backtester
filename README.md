# Daily Backtesting Engine

Minimal daily backtesting engine for equities/ETFs.

## Usage

```
python run.py SPY --strategy sma --fast 50 --slow 200
python run.py SPY --strategy hold --start 2010-01-01
```

Add `--plot` to save an equity curve to `figures/`.

## Tests

```
python -m pytest
```
