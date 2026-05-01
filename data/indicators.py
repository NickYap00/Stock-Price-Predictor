"""Technical indicators computed from OHLCV price data."""
from __future__ import annotations

import pandas as pd


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute MACD for a price DataFrame.

    Returns a DataFrame with columns:
        macd       — MACD line (fast EMA - slow EMA)
        signal     — signal line (EMA of MACD)
        histogram  — macd - signal
        crossover  — 1 = bullish cross, -1 = bearish cross, 0 = none
    """
    close = df[price_col]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    above = macd_line > signal_line
    crossover = above.astype(int).diff()  # +1 or -1 on cross, 0 otherwise

    result = pd.DataFrame(
        {
            "symbol": df["symbol"] if "symbol" in df.columns else None,
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
            "crossover": crossover.fillna(0).astype(int),
        },
        index=df.index,
    )
    return result


def rsi(
    df: pd.DataFrame,
    period: int = 14,
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute RSI for a price DataFrame.

    Returns a DataFrame with columns:
        rsi       — RSI value (0–100)
        zone      — 'overbought' (>70), 'oversold' (<30), or 'neutral'
    """
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi_line = 100 - (100 / (1 + rs))

    zone = pd.cut(
        rsi_line,
        bins=[-float("inf"), 30, 70, float("inf")],
        labels=["oversold", "neutral", "overbought"],
    )

    return pd.DataFrame(
        {
            "symbol": df["symbol"] if "symbol" in df.columns else None,
            "rsi": rsi_line,
            "zone": zone,
        },
        index=df.index,
    )
