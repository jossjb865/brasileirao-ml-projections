"""
LSTM_Model – versión más completa con más features de contexto.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from src.utils import get_models_dir, ensure_dirs

logger = logging.getLogger(__name__)

SEQ_LEN = 5


class LSTMModel:
    def __init__(self, seq_len: int = SEQ_LEN, units: int = 80):
        self.seq_len = seq_len
        self.units = units
        self.model: Optional[keras.Model] = None
        self.feature_cols: List[str] = []
        self.is_fitted = False

    def _feature_cols(self, df: pd.DataFrame) -> List[str]:
        preferred = [
            "home_gf_roll5", "home_ga_roll5", "home_pts_roll5", "home_gd_roll5",
            "away_gf_roll5", "away_ga_roll5", "away_pts_roll5", "away_gd_roll5",
            "gf_diff_roll5", "pts_diff_roll5", "gd_diff_roll5",
            "home_possession", "away_possession",
            "home_shots", "away_shots",
            "home_xg", "away_xg",
        ]
        return [c for c in preferred if c in df.columns]

    def _build(self, n_feat: int) -> keras.Model:
        inp = keras.Input(shape=(self.seq_len, n_feat))
        x = layers.LSTM(self.units, return_sequences=True)(inp)
        x = layers.LSTM(self.units // 2)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(3, activation="softmax")(x)
        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(8e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _build_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        cols = self.feature_cols or self._feature_cols(df)
        self.feature_cols = cols
        X = df[cols].fillna(0.0).values.astype(np.float32)
        X_seq = np.repeat(X[:, np.newaxis, :], self.seq_len, axis=1)
        y = df["result"].astype(int).values
        return X_seq, y

    def fit(self, df: pd.DataFrame, epochs: int = 30, batch_size: int = 32) -> "LSTMModel":
        train = df[df["status"].str.lower().isin(["finished", "ft", "complete"])].copy()
        self.feature_cols = self._feature_cols(train)
        X, y = self._build_sequences(train)
        self.model = self._build(X.shape[-1])
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.15, verbose=0)
        self.is_fitted = True
        logger.info("LSTMModel entrenado con %s muestras, %s features", len(X), len(self.feature_cols))
        return self

    def predict_proba_1x2(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Modelo no entrenado")
        X, _ = self._build_sequences(df)
        return self.model.predict(X, verbose=0)

    def save(self, path: Optional[Path] = None) -> Path:
        ensure_dirs(get_models_dir())
        path = path or get_models_dir() / "lstm_model.keras"
        self.model.save(path)
        joblib.dump(
            {"seq_len": self.seq_len, "units": self.units, "feature_cols": self.feature_cols},
            get_models_dir() / "lstm_model_meta.joblib",
        )
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "LSTMModel":
        path = path or get_models_dir() / "lstm_model.keras"
        meta = joblib.load(get_models_dir() / "lstm_model_meta.joblib")
        obj = cls(seq_len=meta["seq_len"], units=meta["units"])
        obj.feature_cols = meta["feature_cols"]
        obj.model = keras.models.load_model(path)
        obj.is_fitted = True
        return obj
