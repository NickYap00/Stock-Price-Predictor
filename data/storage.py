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
