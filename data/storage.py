"""Persist fetched price data to SQLite."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "prices.db"


def get_engine(db_path: Path | str = DEFAULT_DB_PATH) -> Engine:
    return create_engine(f"sqlite:///{db_path}")


def init_db(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS prices (
                    symbol     TEXT    NOT NULL,
                    timestamp  TEXT    NOT NULL,
                    open       REAL,
                    high       REAL,
                    low        REAL,
                    close      REAL,
                    volume     INTEGER,
                    PRIMARY KEY (symbol, timestamp)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS macd (
                    symbol     TEXT    NOT NULL,
                    timestamp  TEXT    NOT NULL,
                    macd       REAL,
                    signal     REAL,
                    histogram  REAL,
                    crossover  INTEGER,
                    PRIMARY KEY (symbol, timestamp)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS rsi (
                    symbol     TEXT    NOT NULL,
                    timestamp  TEXT    NOT NULL,
                    rsi        REAL,
                    zone       TEXT,
                    PRIMARY KEY (symbol, timestamp)
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    symbol        TEXT    NOT NULL,
                    predicted_for TEXT    NOT NULL,
                    predicted_at  TEXT    NOT NULL,
                    direction     INTEGER NOT NULL,
                    confidence    REAL,
                    window_size   INTEGER,
                    PRIMARY KEY (symbol, predicted_for, predicted_at)
                )
                """
            )
        )


def save_prices(df: pd.DataFrame, engine: Engine) -> int:
    """Upsert price rows. Returns number of rows written."""
    if df.empty:
        return 0

    rows = df.reset_index().to_dict(orient="records")
    with engine.begin() as conn:
        for r in rows:
            conn.execute(
                text(
                    """
                    INSERT INTO prices (symbol, timestamp, open, high, low, close, volume)
                    VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume)
                    ON CONFLICT(symbol, timestamp) DO UPDATE SET
                        open=excluded.open,
                        high=excluded.high,
                        low=excluded.low,
                        close=excluded.close,
                        volume=excluded.volume
                    """
                ),
                {
                    "symbol": r["symbol"],
                    "timestamp": str(r["timestamp"]),
                    "open": r["open"],
                    "high": r["high"],
                    "low": r["low"],
                    "close": r["close"],
                    "volume": int(r["volume"]) if pd.notna(r["volume"]) else None,
                },
            )
    return len(rows)


def load_prices(symbol: str, engine: Engine) -> pd.DataFrame:
    """Load all stored price history for a symbol."""
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT * FROM prices WHERE symbol = :s ORDER BY timestamp"),
            conn,
            params={"s": symbol.upper()},
            parse_dates=["timestamp"],
        )
    return df.set_index("timestamp") if not df.empty else df


def save_macd(df: pd.DataFrame, engine: Engine) -> int:
    """Upsert MACD indicator rows. Returns number of rows written."""
    if df.empty:
        return 0

    rows = df.reset_index().to_dict(orient="records")
    with engine.begin() as conn:
        for r in rows:
            conn.execute(
                text(
                    """
                    INSERT INTO macd (symbol, timestamp, macd, signal, histogram, crossover)
                    VALUES (:symbol, :timestamp, :macd, :signal, :histogram, :crossover)
                    ON CONFLICT(symbol, timestamp) DO UPDATE SET
                        macd=excluded.macd,
                        signal=excluded.signal,
                        histogram=excluded.histogram,
                        crossover=excluded.crossover
                    """
                ),
                {
                    "symbol": r["symbol"],
                    "timestamp": str(r["timestamp"]),
                    "macd": r["macd"],
                    "signal": r["signal"],
                    "histogram": r["histogram"],
                    "crossover": int(r["crossover"]),
                },
            )
    return len(rows)


def load_macd(symbol: str, engine: Engine) -> pd.DataFrame:
    """Load stored MACD data for a symbol."""
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT * FROM macd WHERE symbol = :s ORDER BY timestamp"),
            conn,
            params={"s": symbol.upper()},
            parse_dates=["timestamp"],
        )
    return df.set_index("timestamp") if not df.empty else df


def save_rsi(df: pd.DataFrame, engine: Engine) -> int:
    """Upsert RSI indicator rows. Returns number of rows written."""
    if df.empty:
        return 0

    rows = df.reset_index().to_dict(orient="records")
    with engine.begin() as conn:
        for r in rows:
            conn.execute(
                text(
                    """
                    INSERT INTO rsi (symbol, timestamp, rsi, zone)
                    VALUES (:symbol, :timestamp, :rsi, :zone)
                    ON CONFLICT(symbol, timestamp) DO UPDATE SET
                        rsi=excluded.rsi,
                        zone=excluded.zone
                    """
                ),
                {
                    "symbol": r["symbol"],
                    "timestamp": str(r["timestamp"]),
                    "rsi": r["rsi"],
                    "zone": str(r["zone"]),
                },
            )
    return len(rows)


def load_rsi(symbol: str, engine: Engine) -> pd.DataFrame:
    """Load stored RSI data for a symbol."""
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT * FROM rsi WHERE symbol = :s ORDER BY timestamp"),
            conn,
            params={"s": symbol.upper()},
            parse_dates=["timestamp"],
        )
    return df.set_index("timestamp") if not df.empty else df


def save_predictions(predictions: list[dict], engine: Engine) -> int:
    """Upsert prediction rows. Returns number of rows written."""
    if not predictions:
        return 0
    with engine.begin() as conn:
        for r in predictions:
            conn.execute(
                text(
                    """
                    INSERT INTO predictions
                        (symbol, predicted_for, predicted_at, direction, confidence, window_size)
                    VALUES
                        (:symbol, :predicted_for, :predicted_at, :direction, :confidence, :window_size)
                    ON CONFLICT(symbol, predicted_for, predicted_at) DO UPDATE SET
                        direction=excluded.direction,
                        confidence=excluded.confidence,
                        window_size=excluded.window_size
                    """
                ),
                r,
            )
    return len(predictions)


def load_predictions(engine: Engine, symbol: str | None = None) -> pd.DataFrame:
    """Load stored predictions, optionally filtered by symbol."""
    with engine.connect() as conn:
        if symbol:
            df = pd.read_sql(
                text("SELECT * FROM predictions WHERE symbol = :s ORDER BY predicted_for"),
                conn,
                params={"s": symbol.upper()},
            )
        else:
            df = pd.read_sql(
                text("SELECT * FROM predictions ORDER BY predicted_for, symbol"),
                conn,
            )
    return df
