"""
Modelo Poisson independiente (Dixon-Coles simplificado) para goles.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from src.utils import get_models_dir, ensure_dirs

logger = logging.getLogger(__name__)


class PoissonModel:
    def __init__(self, alpha: float = 1.0):
        self.home_model = PoissonRegressor(alpha=alpha, max_iter=300)
        self.away_model = PoissonRegressor(alpha=alpha, max_iter=300)
        self.feature_cols: list[str] = []
        self.is_fitted = False

    def _prepare(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        cols = [
            c
            for c in [
                "home_gf_roll5",
                "home_ga_roll5",
                "away_gf_roll5",
                "away_ga_roll5",
                "home_pts_roll5",
                "away_pts_roll5",
                "gf_diff_roll5",
                "pts_diff_roll5",
                "gd_diff_roll5",
            ]
            if c in df.columns
        ]
        self.feature_cols = cols
        X = df[cols].fillna(0.0)
        y_home = df["home_score"].astype(float)
        y_away = df["away_score"].astype(float)
        return X, y_home, y_away

    def fit(self, df: pd.DataFrame) -> "PoissonModel":
        train = df[df["status"].str.lower().isin(["finished", "ft", "complete"])].copy()

        # Eliminar filas con marcadores NaN (evita ValueError: Input y contains NaN)
        train = train.dropna(subset=["home_score", "away_score"])
        train["home_score"] = pd.to_numeric(train["home_score"], errors="coerce")
        train["away_score"] = pd.to_numeric(train["away_score"], errors="coerce")
        train = train.dropna(subset=["home_score", "away_score"])

        if len(train) < 20:
            raise RuntimeError(
                f"PoissonModel: solo {len(train)} partidos con marcador válido. "
                "Revisa el fetch de datos."
            )

        X, y_home, y_away = self._prepare(train)
        self.home_model.fit(X, y_home)
        self.away_model.fit(X, y_away)
        self.is_fitted = True
        logger.info("PoissonModel entrenado con %s partidos", len(train))
        return self

    def predict_lambdas(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado")
        X = df[self.feature_cols].fillna(0.0)
        lam_home = self.home_model.predict(X)
        lam_away = self.away_model.predict(X)
        return np.clip(lam_home, 0.05, 6.0), np.clip(lam_away, 0.05, 6.0)

    def predict_proba_1x2(self, df: pd.DataFrame, max_goals: int = 8) -> np.ndarray:
        """Devuelve matriz (n, 3) → [P(Home), P(Draw), P(Away)]."""
        lam_h, lam_a = self.predict_lambdas(df)
        n = len(df)
        probs = np.zeros((n, 3))

        for i in range(n):
            ph = poisson.pmf(np.arange(0, max_goals + 1), lam_h[i])
            pa = poisson.pmf(np.arange(0, max_goals + 1), lam_a[i])
            matrix = np.outer(ph, pa)
            p_home = matrix[np.tril_indices_from(matrix, k=-1)].sum()
            p_draw = np.trace(matrix)
            p_away = matrix[np.triu_indices_from(matrix, k=1)].sum()
            total = p_home + p_draw + p_away
            if total > 0:
                probs[i] = [p_home / total, p_draw / total, p_away / total]
            else:
                probs[i] = [1 / 3, 1 / 3, 1 / 3]
        return probs

    def save(self, path: Optional[Path] = None) -> Path:
        ensure_dirs(get_models_dir())
        path = path or get_models_dir() / "poisson.joblib"
        joblib.dump(self, path)
        logger.info("PoissonModel guardado → %s", path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PoissonModel":
        path = path or get_models_dir() / "poisson.joblib"
        return joblib.load(path)
