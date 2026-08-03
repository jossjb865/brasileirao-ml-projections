"""
LSTM_Result – secuencia de resultados codificados (W/D/L).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from src.utils import get_models_dir, ensure_dirs

logger = logging.getLogger(__name__)

SEQ_LEN = 5


class LSTMResult:
    def __init__(self, seq_len: int = SEQ_LEN, units: int = 48):
        self.seq_len = seq_len
        self.units = units
        self.model: Optional[keras.Model] = None
        self.is_fitted = False

    def _build(self) -> keras.Model:
        inp = keras.Input(shape=(self.seq_len, 6))  # 3 one-hot home + 3 away
        x = layers.LSTM(self.units)(inp)
        x = layers.Dropout(0.25)(x)
        out = layers.Dense(3, activation="softmax")(x)
        model = keras.Model(inp, out)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _build_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Proxy: usamos pts_roll y gd_roll normalizados como señal de forma
        cols = [
            "home_pts_roll5", "home_gd_roll5", "home_gf_roll5",
            "away_pts_roll5", "away_gd_roll5", "away_gf_roll5",
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
        X = df[cols].fillna(0.0).values.astype(np.float32)
        X_seq = np.repeat(X[:, np.newaxis, :], self.seq_len, axis=1)
        y = df["result"].astype(int).values
        return X_seq, y

    def fit(self, df: pd.DataFrame, epochs: int = 20, batch_size: int = 32) -> "LSTMResult":
        train = df[df["status"].str.lower().isin(["finished", "ft", "complete"])].copy()
        X, y = self._build_sequences(train)
        self.model = self._build()
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.15, verbose=0)
        self.is_fitted = True
        logger.info("LSTMResult entrenado con %s muestras", len(X))
        return self

    def predict_proba_1x2(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Modelo no entrenado")
        X, _ = self._build_sequences(df)
        return self.model.predict(X, verbose=0)

    def save(self, path: Optional[Path] = None) -> Path:
        ensure_dirs(get_models_dir())
        path = path or get_models_dir() / "lstm_result.keras"
        self.model.save(path)
        joblib.dump({"seq_len": self.seq_len, "units": self.units}, get_models_dir() / "lstm_result_meta.joblib")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "LSTMResult":
        path = path or get_models_dir() / "lstm_result.keras"
        meta = joblib.load(get_models_dir() / "lstm_result_meta.joblib")
        obj = cls(seq_len=meta["seq_len"], units=meta["units"])
        obj.model = keras.models.load_model(path)
        obj.is_fitted = True
        return obj
