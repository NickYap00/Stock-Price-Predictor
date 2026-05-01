"""CLI: fetch one or more symbols and persist to SQLite.

Usage:
    python fetch_cli.py AAPL MSFT NVDA --period 6mo
    python fetch_cli.py AAPL --macd --rsi
    python fetch_cli.py AAPL GOOG MSFT NVDA TSLA --macd --rsi --predict
"""
from __future__ import annotations

import argparse
from pathlib import Path

from data.fetcher import FetchRequest, fetch_history
from data.indicators import macd as compute_macd
from data.indicators import rsi as compute_rsi
from data.storage import get_engine, init_db, load_prices, save_macd, save_predictions, save_prices, save_rsi


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch stock prices from Yahoo Finance.")
    p.add_argument("symbols", nargs="+", help="Ticker symbols, e.g. AAPL MSFT")
    p.add_argument("--period", default="1y", help="Period (e.g. 1mo, 6mo, 1y, 5y, max)")
    p.add_argument("--interval", default="1d", help="Interval (e.g. 1d, 1h, 5m)")
    p.add_argument("--macd", action="store_true", help="Compute and save MACD indicators")
    p.add_argument("--rsi", action="store_true", help="Compute and save RSI indicator")
    p.add_argument("--predict", action="store_true", help="Train model and predict next-day direction")
    p.add_argument("--window", type=int, default=20, help="Sliding window size in days (default: 20)")
    p.add_argument("--epochs", type=int, default=200, help="Max training epochs (default: 200)")
    p.add_argument("--model-dir", default=None, help="Directory to save/load model artifacts")
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

            if args.macd:
                macd_df = compute_macd(df)
                save_macd(macd_df, engine)
                latest = macd_df.iloc[-1]
                cross = {1: "BULLISH CROSS", -1: "BEARISH CROSS"}.get(latest["crossover"], "")
                print(
                    f"       MACD {latest['macd']:+.4f}  signal {latest['signal']:+.4f}"
                    f"  hist {latest['histogram']:+.4f}"
                    + (f"  ** {cross} **" if cross else "")
                )

            if args.rsi:
                rsi_df = compute_rsi(df)
                save_rsi(rsi_df, engine)
                latest = rsi_df.iloc[-1]
                zone = latest["zone"]
                flag = " **" if zone in ("overbought", "oversold") else ""
                print(f"       RSI  {latest['rsi']:.2f}  [{zone}{flag}]")
        except Exception as e:
            print(f"{sym.upper():<6} ERROR: {e}")

    if args.predict:
        from data.model import predict_next_day, train

        model_dir = Path(args.model_dir) if args.model_dir else Path("models")
        model_dir.mkdir(exist_ok=True)
        symbols = [s.upper() for s in args.symbols]

        print("\nTraining model on all symbols...")
        try:
            metrics = train(symbols, engine, window=args.window, epochs=args.epochs, model_dir=model_dir)
            print(f"  Test accuracy: {metrics['test_accuracy']:.3f}  AUC: {metrics['test_auc']:.3f}")
        except Exception as e:
            print(f"  Training ERROR: {e}")
            return

        print("\nNext-day predictions:")
        all_preds = []
        for sym in symbols:
            try:
                result = predict_next_day(sym, engine, model_dir=model_dir, window=args.window)
                direction = "UP  " if result["direction"] == 1 else "DOWN"
                conf = result["confidence"]
                print(f"  {sym:<6} -> {direction}  (confidence: {conf:.1%})")
                all_preds.append(result)
            except Exception as e:
                print(f"  {sym:<6} ERROR: {e}")

        if all_preds:
            save_predictions(all_preds, engine)
            print("\nPredictions saved to DB.")


if __name__ == "__main__":
    main()
