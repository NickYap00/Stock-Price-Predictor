# moneymoneymoney

Fetch stock price history, compute technical indicators (MACD, RSI), and predict next-day price direction with a 1D CNN.

## What it does

- **Fetch** OHLCV data from Yahoo Finance via `yfinance` and persist it to SQLite.
- **Compute** MACD (with bullish/bearish crossover detection) and RSI (with overbought/oversold zones).
- **Predict** next-day direction (UP / DOWN) using a TensorFlow/Keras 1D-CNN trained on a sliding window of price + indicator features.

All operations are driven from a single CLI: [fetch_cli.py](fetch_cli.py).

## Project layout

```
.
├── fetch_cli.py            # CLI entry point
├── data/
│   ├── fetcher.py          # Yahoo Finance fetching
│   ├── indicators.py       # MACD and RSI
│   ├── model.py            # 1D-CNN: features, training, prediction
│   └── storage.py          # SQLite schema + upserts (prices, macd, rsi, predictions)
├── models/                 # Saved model + scaler (ignored by git)
├── models_highvol/         # Model artifacts trained on high-volatility tickers
├── models_lowvol/          # Model artifacts trained on low-volatility tickers
├── prices.db               # SQLite database
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ (uses PEP 604 union syntax). TensorFlow ≥ 2.16 is needed for the model.

## Usage

Everything runs through [fetch_cli.py](fetch_cli.py). You give it one or more ticker symbols and turn features on with flags. Symbols are case-insensitive (`aapl` and `AAPL` both work).

### 1. Fetch prices only

The simplest workflow — pull OHLCV history for one or more tickers and save it to SQLite:

```bash
python fetch_cli.py AAPL MSFT NVDA --period 6mo
```

Output:
```
AAPL   saved 126 rows  (latest close: $189.84)
MSFT   saved 126 rows  (latest close: $415.26)
NVDA   saved 126 rows  (latest close: $878.37)
```

Re-running the same command is safe — rows are upserted by `(symbol, timestamp)`, so existing bars get refreshed and new ones are appended. To grow your history, just increase `--period` (`1mo` → `6mo` → `1y` → `5y` → `max`).

### 2. Fetch + compute indicators

Add `--macd` and/or `--rsi` to compute and store technical indicators alongside the prices:

```bash
python fetch_cli.py AAPL --macd --rsi
```

Output:
```
AAPL   saved 252 rows  (latest close: $189.84)
       MACD +1.2453  signal +0.9821  hist +0.2632  ** BULLISH CROSS **
       RSI  68.42  [neutral]
```

The `** BULLISH CROSS **` / `** BEARISH CROSS **` flags appear only on the day the MACD line crosses its signal line. RSI prints `**` when it enters the overbought (>70) or oversold (<30) zones.

### 3. Fetch + train + predict next-day direction

The full pipeline. `--predict` requires `--macd` and `--rsi` (the model needs those features):

```bash
python fetch_cli.py AAPL GOOG MSFT NVDA TSLA --macd --rsi --predict
```

What happens, in order:
1. Each symbol's prices are fetched and saved.
2. MACD and RSI are computed and saved per symbol.
3. **One model** is trained across **all** symbols passed in. More symbols = more training data = generally a better model. You typically want at least 5+ tickers and 1+ year of history.
4. The trained model predicts next-business-day direction for each symbol:
   ```
   AAPL   -> UP    (confidence: 62.4%)
   GOOG   -> DOWN  (confidence: 71.2%)
   MSFT   -> UP    (confidence: 53.8%)
   ```
5. Predictions are written to the `predictions` table along with the test-set accuracy/AUC printed at the end of training.

`confidence` is the raw sigmoid output. ≥ 0.5 ⇒ UP, < 0.5 ⇒ DOWN. Values close to 0.5 mean the model is unsure — treat low-confidence calls as noise.

### Typical workflows

**Daily routine — refresh today's data and get a fresh prediction:**
```bash
python fetch_cli.py AAPL MSFT NVDA GOOG TSLA META AMZN --period 2y --macd --rsi --predict
```
Run once a day after market close. The `2y` period gives the model enough history; the model retrains from scratch each call (fast — usually under a minute on CPU).

**Backfill a new ticker with maximum history:**
```bash
python fetch_cli.py PLTR --period max --macd --rsi
```

**Train on high-volatility names, save to a separate model dir:**
```bash
python fetch_cli.py TSLA NVDA COIN MSTR --period 2y --macd --rsi --predict --model-dir models_highvol
```
The repo already contains `models_highvol/` and `models_lowvol/` from previous runs — useful if you want to keep separate models for different volatility regimes rather than mixing them in one.

**Tune the window or training length:**
```bash
python fetch_cli.py AAPL MSFT NVDA --macd --rsi --predict --window 30 --epochs 300
```
Larger `--window` = more historical context per prediction (but fewer training samples). Early stopping (patience 25 on val AUC) usually halts well before `--epochs` is exhausted.

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--period` | `1y` | Yahoo Finance period (`1mo`, `6mo`, `1y`, `5y`, `max`) |
| `--interval` | `1d` | Bar interval (`1d`, `1h`, `5m`, …) |
| `--macd` | off | Compute and store MACD |
| `--rsi` | off | Compute and store RSI |
| `--predict` | off | Train model and predict next-day direction (requires `--macd` and `--rsi`) |
| `--window` | `20` | Sliding window size in days |
| `--epochs` | `200` | Max training epochs (early stopping on val AUC) |
| `--model-dir` | `models/` | Where to save model + scaler |

### Inspecting the database

Everything is in `prices.db` — a plain SQLite file. Easiest way to poke at it:

```bash
sqlite3 prices.db
```
```sql
.tables
SELECT * FROM predictions ORDER BY predicted_at DESC LIMIT 10;
SELECT symbol, timestamp, close FROM prices WHERE symbol = 'AAPL' ORDER BY timestamp DESC LIMIT 5;
SELECT symbol, timestamp, rsi, zone FROM rsi WHERE zone != 'neutral' ORDER BY timestamp DESC LIMIT 20;
```

Or load it from Python:

```python
from data.storage import get_engine, load_prices, load_predictions

engine = get_engine()
df = load_prices("AAPL", engine)
preds = load_predictions(engine, symbol="AAPL")
```

### Using the model from Python

If you've already trained and just want predictions without retraining:

```python
from data.storage import get_engine
from data.model import predict_next_day

engine = get_engine()
result = predict_next_day("AAPL", engine, model_dir="models")
print(result)
# {'symbol': 'AAPL', 'predicted_for': '2026-05-09', 'predicted_at': '2026-05-08',
#  'direction': 1, 'confidence': 0.624, 'window_size': 20}
```

Pass a pre-loaded `model` and `scaler` (via `load_artifacts()`) when predicting many symbols in a row to avoid reloading from disk each call.

## Model

A small 1D-CNN trained on 20-day windows of 11 features:

- Price/volume: `close`, `volume`
- MACD: `macd`, `signal`, `histogram`
- RSI: `rsi`
- Momentum: `momentum_5`, `momentum_10`
- Mean reversion: `sma_ratio`, `bb_pct` (Bollinger %B)
- Volume regime: `volume_ratio`

Architecture (see [data/model.py:127-143](data/model.py#L127-L143)):

```
Conv1D(32) → Dropout → Conv1D(16) → GlobalAveragePooling → Dense(32) → Dropout → Dense(1, sigmoid)
```

Training uses L2 regularisation (`0.01`), class weighting to handle UP/DOWN imbalance, early stopping on validation AUC (patience 25), and `ReduceLROnPlateau`. Splits are 65 / 20 / 15 train/val/test, sequential per symbol (no shuffling across time).

Saved artifacts: `direction_model.keras` and `scaler.pkl` (a `MinMaxScaler` fit on the training set only).

## Database

SQLite file at `prices.db` with four tables:

- `prices` — OHLCV per `(symbol, timestamp)`
- `macd` — MACD line / signal / histogram / crossover flag
- `rsi` — RSI value and zone (`oversold` / `neutral` / `overbought`)
- `predictions` — model output per `(symbol, predicted_for, predicted_at)`

All tables use upserts (`ON CONFLICT … DO UPDATE`), so re-running fetches is safe.

## Disclaimer

This project is a personal experiment. Predictions are not investment advice — next-day direction is a notoriously hard target and accuracy will be close to chance for most regimes. Don't trade real money on it.
