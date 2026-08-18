import argparse
import json
from datetime import date
from pathlib import Path

from backtest import load_daily
from sweep import UNIVERSE

TEMPLATE = Path(__file__).parent / "templates" / "dashboard.html"
OUTPUT = Path("docs") / "index.html"

DEFAULTS = {"cost_bps": 5.0, "rf": 0.04, "fast": 50, "slow": 200,
            "target_vol": 0.10, "lookback": 20}


def build(tickers) -> dict:
    data = {
        "generated": date.today().isoformat(),
        "defaults": DEFAULTS,
        "tickers": [],
        "series": {},
    }

    for ticker in tickers:
        try:
            df = load_daily(ticker)
        except (ValueError, RuntimeError) as exc:
            print(f"  {ticker}: skipped ({exc})")
            continue

        data["tickers"].append(ticker)
        data["series"][ticker] = {
            "dates": [d.strftime("%Y-%m-%d") for d in df.index],
            "prices": [round(v, 4) for v in df["Close"]],
        }
        print(f"  {ticker}: prices embedded ({len(df)} days)")

    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dashboard")
    parser.add_argument("tickers", nargs="*", default=list(UNIVERSE))
    args = parser.parse_args()

    tickers = args.tickers or list(UNIVERSE)
    print(f"Building dashboard for {len(tickers)} tickers")
    data = build(tickers)

    if not data["tickers"]:
        raise SystemExit("no tickers produced results")

    payload = json.dumps(data, separators=(",", ":"))
    html = TEMPLATE.read_text(encoding="utf-8")
    start, end = html.index("/*DATA*/"), html.index("/*END*/") + len("/*END*/")
    html = html[:start] + payload + html[end:]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    size = OUTPUT.stat().st_size / 1024
    print(f"\nWrote {OUTPUT} ({size:.0f} KB, {len(data['tickers'])} tickers)")


if __name__ == "__main__":
    main()
