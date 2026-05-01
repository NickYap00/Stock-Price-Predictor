"""TensorFlow 1D-CNN model for next-day price direction prediction."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy.engine import Engine

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data.storage import load_macd, load_prices, load_rsi

FEATURE_COLS = ["close", "volume", "macd", "signal", "histogram", "rsi"]
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def build_features(symbol: str, engine: Engine) -> pd.DataFrame:
    """Merge prices + MACD + RSI into one aligned DataFrame with a direction label."""
    prices = load_prices(symbol, engine)[["close", "volume"]]
    macd_df = load_macd(symbol, engine)[["macd", "signal", "histogram"]]
    rsi_df = load_rsi(symbol, engine)[["rsi"]]

    df = prices.join(macd_df, how="inner").join(rsi_df, how="inner")
    df["symbol"] = symbol.upper()
    df["direction"] = (df["close"].shift(-1) > df["close"]).astype("Int64")
    df = df.dropna()
    return df


def _make_windows(arr: np.ndarray, labels: np.ndarray, window: int):
    X, y = [], []
    for i in range(window - 1, len(arr)):
        X.append(arr[i - window + 1 : i + 1])
        y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_dataset(
    symbols: list[str],
    engine: Engine,
    window: int = 20,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
):
    """Build train/val/test splits and the predict window (last N rows per symbol).

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, predict_X (dict symbol->array)
    """
    train_frames, val_frames, test_frames = [], [], []
    predict_rows: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        df = build_features(sym, engine)
        n = len(df)
        train_end = int(n * (1 - val_frac - test_frac))
        val_end = int(n * (1 - test_frac))

        train_frames.append(df.iloc[:train_end])
        val_frames.append(df.iloc[train_end:val_end])
        test_frames.append(df.iloc[val_end:])
        predict_rows[sym] = df.iloc[-window:]

    scaler = MinMaxScaler()
    scaler.fit(pd.concat(train_frames)[FEATURE_COLS])

    def _scale_and_window(frames):
        X_all, y_all = [], []
        for frame in frames:
            scaled = scaler.transform(frame[FEATURE_COLS])
            labels = frame["direction"].to_numpy(dtype=np.float32)
            X_w, y_w = _make_windows(scaled, labels, window)
            X_all.append(X_w)
            y_all.append(y_w)
        if not X_all:
            return np.empty((0, window, len(FEATURE_COLS))), np.empty(0)
        return np.concatenate(X_all), np.concatenate(y_all)

    X_train, y_train = _scale_and_window(train_frames)
    X_val, y_val = _scale_and_window(val_frames)
    X_test, y_test = _scale_and_window(test_frames)

    predict_X = {}
    for sym, frame in predict_rows.items():
        scaled = scaler.transform(frame[FEATURE_COLS])
        predict_X[sym] = scaled[np.newaxis, :, :]  # shape (1, window, features)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, predict_X


def build_model(window_size: int = 20, n_features: int = 6) -> keras.Model:
    inputs = keras.Input(shape=(window_size, n_features))
    x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="causal")(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(16, kernel_size=3, activation="relu", padding="causal")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def train(
    symbols: list[str],
    engine: Engine,
    window: int = 20,
    epochs: int = 200,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> dict:
    """Train the model on all symbols and save artifacts. Returns test metrics."""
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, _ = prepare_dataset(
        symbols, engine, window=window
    )

    if len(X_train) == 0:
        raise ValueError("Not enough data to train. Fetch more history first.")

    counts = np.bincount(y_train.astype(int))
    total = counts.sum()
    class_weight = {i: total / (len(counts) * c) for i, c in enumerate(counts)}

    model = build_model(window_size=window, n_features=len(FEATURE_COLS))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=15, restore_best_weights=True, mode="max"
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
    metrics = {"test_loss": loss, "test_accuracy": accuracy, "test_auc": auc}

    model.save(model_dir / "direction_model.keras")
    joblib.dump(scaler, model_dir / "scaler.pkl")

    return metrics


def predict_next_day(
    symbol: str,
    engine: Engine,
    model_dir: Path = DEFAULT_MODEL_DIR,
    window: int = 20,
) -> dict:
    """Load saved model and predict next-day direction for a single symbol."""
    model_dir = Path(model_dir)
    model = keras.models.load_model(model_dir / "direction_model.keras")
    scaler = joblib.load(model_dir / "scaler.pkl")

    df = build_features(symbol, engine)
    if len(df) < window:
        raise ValueError(f"Not enough rows for {symbol} (need {window}, have {len(df)})")

    last_window = df.iloc[-window:]
    scaled = scaler.transform(last_window[FEATURE_COLS])
    X = scaled[np.newaxis, :, :]

    confidence = float(model.predict(X, verbose=0)[0][0])
    direction = 1 if confidence >= 0.5 else 0

    last_ts = df.index[-1]
    predicted_for = str((last_ts + pd.tseries.offsets.BDay(1)).date())
    predicted_at = str(pd.Timestamp.now().date())

    return {
        "symbol": symbol.upper(),
        "predicted_for": predicted_for,
        "predicted_at": predicted_at,
        "direction": direction,
        "confidence": confidence,
        "window_size": window,
    }
