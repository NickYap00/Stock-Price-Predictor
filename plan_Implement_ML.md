# TensorFlow Prediction Model — Implementation Plan

## Context
The moneymoneymoney project fetches daily OHLCV stock data and computes MACD + RSI indicators. The goal is to add a TensorFlow model that trains on this historical data and predicts next-day price direction (up/down) for each symbol. The user wants to use MACD and RSI indicators as features alongside price data to make momentum-based predictions.

---

## Architecture Decision: 1D-CNN + GlobalAveragePooling

**Why not LSTM/GRU:** With only ~124 rows per symbol (6 months daily), recurrent models overfit immediately. LSTM has too many parameters for this dataset size.

**Why 1D-CNN:** Convolutional filters detect local time patterns (e.g. "RSI spike after gap up") with far fewer parameters. GlobalAveragePooling is a strong structural regularizer. Trains in seconds on CPU.

```
Input: (window=20, features=6)
  -> Conv1D(32, kernel=3, activation='relu', padding='causal')
  -> Dropout(0.3)
  -> Conv1D(16, kernel=3, activation='relu', padding='causal')
  -> GlobalAveragePooling1D()
  -> Dense(32, activation='relu')
  -> Dropout(0.3)
  -> Dense(1, activation='sigmoid')   # 1=up, 0=down
```

~3,000–5,000 total parameters — appropriate for ~400 training samples.

---

## Features (6 total)

| Feature | Source | Notes |
|---------|--------|-------|
| `close` | prices | normalized |
| `volume` | prices | normalized |
| `macd` | macd | differenced signal |
| `signal` | macd | EMA of MACD |
| `histogram` | macd | macd - signal |
| `rsi` | rsi | 0–100, normalized to 0–1 |

Dropped: `zone`, `crossover` (categorical/sparse), `open`, `high`, `low` (correlated with close, add noise).

**Window size:** 20 days (one trading month)

**Normalization:** Single global `MinMaxScaler` fitted on training split only, saved to disk. Prevents data leakage. One scaler across all symbols is more stable than per-symbol at this dataset size.

**Target:** `(close.shift(-1) > close).astype(int)` — binary next-day direction

---

## Train / Val / Test Split

Time-ordered split per symbol — no shuffling across boundaries (prevents future leakage):
- **70% train** | **15% val** | **15% test**
- With ~104 usable rows/symbol × 5 symbols = ~520 total samples after pooling
- All 5 symbols are pooled for training; model learns cross-symbol patterns

---

## File Changes

### New: `data/model.py`
Core ML pipeline (~250–300 lines):
- `build_features(symbol, engine)` — joins prices + macd + rsi, computes direction label, drops NaNs
- `prepare_dataset(symbols, engine, window, test_frac, val_frac)` — splits, scales, forms sliding windows
- `build_model(window_size, n_features)` — returns compiled Keras model
- `train(symbols, engine, window, epochs, model_dir)` — full training loop, saves model + scaler, returns test metrics
- `predict_next_day(symbol, engine, model_dir, window)` — loads model + scaler, returns prediction dict
- Saved artifacts: `models/direction_model.keras`, `models/scaler.pkl`

### Modified: `data/storage.py`
- Extend `init_db()` to create `predictions` table
- Add `save_predictions(predictions, engine)` — upserts list of prediction dicts

New table:
```sql
CREATE TABLE IF NOT EXISTS predictions (
    symbol        TEXT  NOT NULL,
    predicted_for TEXT  NOT NULL,   -- next trading day date
    predicted_at  TEXT  NOT NULL,   -- date prediction was made
    direction     INTEGER NOT NULL, -- 1=up, 0=down
    confidence    REAL,             -- sigmoid output 0.0–1.0
    window_size   INTEGER,
    PRIMARY KEY (symbol, predicted_for, predicted_at)
)
```

### Modified: `fetch_cli.py`
Add flags (called once after the per-symbol loop):
- `--predict` — train model and predict next-day direction for all symbols
- `--window INT` — sliding window size (default: 20)
- `--epochs INT` — max training epochs (default: 200, uses early stopping)
- `--model-dir PATH` — where to save/load model artifacts (default: `./models`)

### Modified: `requirements.txt`
Add:
```
tensorflow>=2.16.0
scikit-learn>=1.4.0
numpy>=1.26.0
```

---

## Training Details

```python
callbacks = [
    EarlyStopping(monitor="val_auc", patience=15, restore_best_weights=True, mode="max"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7),
]
model.fit(..., epochs=200, batch_size=32, class_weight=..., callbacks=callbacks)
```

- `EarlyStopping` on `val_auc` (more reliable than val_loss for binary classification)
- `class_weight` passed if class balance is worse than 60/40

---

## CLI Usage After Implementation

```bash
# Fetch data + compute indicators + train model + predict
python3 fetch_cli.py AAPL GOOG MSFT NVDA TSLA --macd --rsi --predict

# Expected output:
# Training model on all symbols...
#   Test accuracy: 0.538  AUC: 0.571
#
# Next-day predictions:
#   AAPL   -> UP    (confidence: 63.2%)
#   GOOG   -> DOWN  (confidence: 58.1%)
#   MSFT   -> UP    (confidence: 71.4%)
#   NVDA   -> DOWN  (confidence: 55.9%)
#   TSLA   -> UP    (confidence: 60.3%)
# Predictions saved to DB.
```

---

## Verification Steps

1. **Install deps:** `pip3 install tensorflow scikit-learn numpy`
2. **Populate indicators:** `python3 fetch_cli.py AAPL GOOG MSFT NVDA TSLA --period 6mo --macd --rsi`
3. **Smoke-test feature builder:**
   ```python
   from data.storage import get_engine
   from data.model import build_features
   df = build_features("AAPL", get_engine())
   print(df.shape, df["direction"].value_counts())
   ```
4. **Run full training + prediction:** `python3 fetch_cli.py AAPL GOOG MSFT NVDA TSLA --predict`
5. **Check model files:** `ls models/` → `direction_model.keras`, `scaler.pkl`
6. **Check predictions table:** query `SELECT * FROM predictions` in prices.db

---

## Critical Files

| File | Action |
|------|--------|
| `/Users/nick/Projects/moneymoneymoney/data/model.py` | **Create** — entire ML pipeline |
| `/Users/nick/Projects/moneymoneymoney/data/storage.py` | **Modify** — add predictions table + save_predictions |
| `/Users/nick/Projects/moneymoneymoney/fetch_cli.py` | **Modify** — add --predict, --window, --epochs, --model-dir |
| `/Users/nick/Projects/moneymoneymoney/requirements.txt` | **Modify** — add tensorflow, scikit-learn, numpy |
