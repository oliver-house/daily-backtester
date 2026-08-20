import json
import ssl
import urllib.request
from pathlib import Path

import pandas as pd

CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/"
    "{ticker}?period1=0&period2=9999999999&interval=1d"
)


def _fetch_json(url: str) -> dict:
    ctx = ssl.create_default_context()
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        return json.load(resp)


def load_daily(
    ticker: str, cache_dir: str = "data_cache", allow_download: bool = True
) -> pd.DataFrame:
    ticker = ticker.strip().upper()
    cache = Path(cache_dir) / f"{ticker}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col="Date", parse_dates=True)

    if not allow_download:
        raise ValueError(
            f"no cached data for {ticker!r} in {cache_dir!r} and downloads are disabled"
        )

    try:
        payload = _fetch_json(CHART_URL.format(ticker=ticker))
        result = payload["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        adjclose = result["indicators"]["adjclose"][0]["adjclose"]
    except Exception as exc:
        raise ValueError(f"No data for ticker {ticker!r}: {exc}") from exc

    df = pd.DataFrame(
        {
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": quote["close"],
            "AdjClose": adjclose,
            "Volume": quote["volume"],
        },
        index=pd.to_datetime(result["timestamp"], unit="s", utc=True)
        .tz_convert(result["meta"].get("exchangeTimezoneName", "UTC"))
        .normalize()
        .tz_localize(None),
    ).dropna(subset=["Close"])

    ratio = df["AdjClose"] / df["Close"]
    for col in ("Open", "High", "Low", "Close"):
        df[col] = df[col] * ratio
    df = df.drop(columns="AdjClose").sort_index()
    df.index.name = "Date"

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


def add_data_args(parser) -> None:
    parser.add_argument("--cache-dir", default="data_cache",
                        help="directory of cached price CSVs")
    parser.add_argument("--offline", action="store_true",
                        help="fail on a cache miss instead of downloading")


def loader(args):
    def load(ticker: str) -> pd.DataFrame:
        return load_daily(ticker, args.cache_dir, allow_download=not args.offline)

    return load
