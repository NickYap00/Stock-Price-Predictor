"""CLI: fetch one or more symbols and persist to SQLite.

Usage:
    python fetch_cli.py AAPL MSFT NVDA --period 6mo
"""
from __future__ import annotations

import argparse

from data.fetcher import FetchRequest, fetch_history
from data.storage import get_engine, init_db, save_prices


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch stock prices from Yahoo Finance.")
    p.add_argument("symbols", nargs="+", help="Ticker symbols, e.g. AAPL MSFT")
    p.add_argument("--period", default="1y", help="Period (e.g. 1mo, 6mo, 1y, 5y, max)")
    p.add_argument("--interval", default="1d", help="Interval (e.g. 1d, 1h, 5m)")
    args = p.parse_args()

    engine = get_engine()
    init_db(engine)

    for sym in args.symbols:
        try:
            df = fetch_history(
                FetchRequest(symbol=sym, period=args.period, interval=args.interval)
            )
            n = save_prices(df, engine)
            print(f"{sym.upper():<6} saved {n} rows  (latest close: ${df['close'].iloc[-1]:.2f})")
        except Exception as e:
            print(f"{sym.upper():<6} ERROR: {e}")


if __name__ == "__main__":
    main()
