# Summary of Work Done — 20/04/2026

## Project: moneymoneymoney
Stock price fetcher and ML prediction system using Yahoo Finance, SQLite, and TensorFlow.

---

## 1. MACD Indicator

**New file:** `data/indicators.py`
- `macd(df, fast=12, slow=26, signal=9)` — computes MACD line, signal line, histogram, and crossover detection (1=bullish, -1=bearish, 0=none)

**Modified:** `data/storage.py`
- Added `macd` table to SQLite
- Added `save_macd(df, engine)` and `load_macd(symbol, engine)`

**Modified:** `fetch_cli.py`
- Added `--macd` flag — computes and saves MACD, prints latest values and crossover signals

---

## 2. RSI Indicator

**Modified:** `data/indicators.py`
- `rsi(df, period=14)` — computes RSI (0–100), zone classification: `oversold` (<30), `neutral` (30–70), `overbought` (>70)

**Modified:** `data/storage.py`
- Added `rsi` table to SQLite
- Added `save_rsi(df, engine)` and `load_rsi(symbol, engine)`

**Modified:** `fetch_cli.py`
- Added `--rsi` flag — computes and saves RSI, flags overbought/oversold

---

## 3. TensorFlow Prediction Model

### Architecture: 1D-CNN + GlobalAveragePooling
Chosen over LSTM/GRU due to small dataset (~124 rows/symbol). Low parameter count (~3,000–5,000) prevents overfitting.

```
Input: (window=20, features=6)
  -> Conv1D(32, kernel=3, relu, causal padding)
  -> Dropout(0.3)
  -> Conv1D(16, kernel=3, relu, causal padding)
  -> GlobalAveragePooling1D()
  -> Dense(32, relu)
  -> Dropout(0.3)
  -> Dense(1, sigmoid)   # 1=UP, 0=DOWN
```

### Features (6)
| Feature | Source |
|---------|--------|
| close | prices |
| volume | prices |
| macd | macd |
| signal | macd |
| histogram | macd |
| rsi | rsi |

### Training Details
- **Target:** next-day price direction (binary: up/down)
- **Split:** 70% train / 15% val / 15% test (time-ordered, no shuffle)
- **Pooling:** all 5 symbols trained together
- **Normalisation:** single global MinMaxScaler fitted on training split only
- **Window size:** 20 days
- **Callbacks:** EarlyStopping on val_auc (patience=15), ReduceLROnPlateau

### New file: `data/model.py`
- `build_features(symbol, engine)` — joins prices + macd + rsi, computes direction label
- `prepare_dataset(symbols, engine, window, ...)` — splits, scales, forms sliding windows
- `build_model(window_size, n_features)` — returns compiled Keras model
- `train(symbols, engine, window, epochs, model_dir)` — trains and saves model + scaler
- `predict_next_day(symbol, engine, model_dir, window)` — loads model and returns prediction dict

### Modified: `data/storage.py`
- Added `predictions` table to SQLite
- Added `save_predictions(predictions, engine)` and `load_predictions(engine, symbol)`

```sql
CREATE TABLE predictions (
    symbol        TEXT    NOT NULL,
    predicted_for TEXT    NOT NULL,
    predicted_at  TEXT    NOT NULL,
    direction     INTEGER NOT NULL,
    confidence    REAL,
    window_size   INTEGER,
    PRIMARY KEY (symbol, predicted_for, predicted_at)
)
```

### Modified: `fetch_cli.py`
Added flags:
- `--predict` — trains model across all symbols, prints next-day predictions, saves to DB
- `--window INT` — sliding window size (default: 20)
- `--epochs INT` — max training epochs (default: 200)
- `--model-dir PATH` — directory to save/load model artifacts (default: `./models`)

### Modified: `requirements.txt`
Added: `tensorflow>=2.16.0`, `scikit-learn>=1.4.0`, `numpy>=1.26.0`, `joblib>=1.3.0`

### Model artifacts saved to `models/`
- `models/direction_model.keras`
- `models/scaler.pkl`

---

## First Run Results

```
Test accuracy: 0.533  AUC: 0.586

Next-day predictions (2026-04-20):
  AAPL   -> DOWN  (confidence: 49.7%)
  GOOG   -> DOWN  (confidence: 49.4%)
  MSFT   -> DOWN  (confidence: 49.1%)
  NVDA   -> DOWN  (confidence: 49.9%)
  TSLA   -> DOWN  (confidence: 48.9%)
```

---

## Full CLI Usage

```bash
# Fetch + indicators only
python3 fetch_cli.py AAPL GOOG MSFT NVDA TSLA --period 6mo --macd --rsi

# Fetch + indicators + train + predict
python3 fetch_cli.py AAPL GOOG MSFT NVDA TSLA --period 1y --macd --rsi --predict

# Predict with custom settings
python3 fetch_cli.py AAPL GOOG MSFT NVDA TSLA --macd --rsi --predict --window 30 --epochs 300
```

---

## File Structure

```
moneymoneymoney/
├── fetch_cli.py          # CLI entrypoint
├── prices.db             # SQLite database (prices, macd, rsi, predictions)
├── requirements.txt
├── plan_Implement_ML.md  # TensorFlow implementation plan
├── summary_workdone_20042026.md
├── models/
│   ├── direction_model.keras
│   └── scaler.pkl
└── data/
    ├── fetcher.py        # Yahoo Finance data fetching
    ├── indicators.py     # MACD + RSI computation
    ├── model.py          # TensorFlow ML pipeline
    └── storage.py        # SQLite persistence
```
