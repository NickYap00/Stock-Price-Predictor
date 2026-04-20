"""Fetch stock price data from Yahoo Finance."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd
import yfinance as yf


@dataclass
class FetchRequest:
    symbol: str
    period: str = "1y"      # e.g. "1d", "5d", "1mo", "1y", "5y", "max"
    interval: str = "1d"    # e.g. "1m", "5m", "1h", "1d", "1wk"
    start: str | datetime | None = None
    end: str | datetime | None = None


def fetch_history(req: FetchRequest) -> pd.DataFrame:
    """Fetch OHLCV history for a single symbol.

    Returns a DataFrame indexed by timestamp with columns:
    open, high, low, close, volume, symbol.
    """
    ticker = yf.Ticker(req.symbol)

    if req.start or req.end:
        df = ticker.history(start=req.start, end=req.end, interval=req.interval)
    else:
        df = ticker.history(period=req.period, interval=req.interval)

    if df.empty:
        raise ValueError(f"No data returned for symbol {req.symbol!r}")

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )[["open", "high", "low", "close", "volume"]]

    df.index.name = "timestamp"
    df["symbol"] = req.symbol.upper()
    return df


def fetch_many(symbols: Iterable[str], **kwargs) -> dict[str, pd.DataFrame]:
    """Fetch history for multiple symbols. Returns {symbol: df}."""
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            out[sym.upper()] = fetch_history(FetchRequest(symbol=sym, **kwargs))
        except Exception as e:
            print(f"[warn] failed to fetch {sym}: {e}")
    return out


def latest_price(symbol: str) -> float:
    """Return the most recent close price for a symbol."""
    df = fetch_history(FetchRequest(symbol=symbol, period="5d", interval="1d"))
    return float(df["close"].iloc[-1])


if __name__ == "__main__":
    # Quick smoke test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fetcher.py <TICKER>", file=sys.stderr)
        sys.exit(1)
    sym = sys.argv[1]
    df = fetch_history(FetchRequest(symbol=sym, period="1mo"))
    print(df.tail())
    print(f"\nLatest {sym} close: ${latest_price(sym):.2f}")
