"""
XGBoost multiclass (1X2) + regresión de goles.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from src.utils import get_models_dir, ensure_dirs

logger = logging.getLogger(__name__)


class XGBoostModel:
    def __init__(self, **kwargs):
        default = dict(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        default.update(kwargs)
        self.clf = XGBClassifier(**default)
        self.reg_home = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1
        )
        self.reg_away = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1
        )
        self.feature_cols: List[str] = []
        self.is_fitted = False

    def _feature_cols(self, df: pd.DataFrame) -> List[str]:
        exclude = {
            "match_id", "competition_id", "season_id", "season_name", "status",
            "kickoff_utc", "matchday", "home_team_id", "home_team_name",
            "away_team_id", "away_team_name", "home_score", "away_score",
            "home_ht_score", "away_ht_score", "result", "total_goals", "btts",
            "xg_available", "odds_available",
        }
        cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        return cols

    def fit(self, df: pd.DataFrame) -> "XGBoostModel":
        train = df[df["status"].str.lower().isin(["finished", "ft", "complete"])].copy()
        self.feature_cols = self._feature_cols(train)
        X = train[self.feature_cols].fillna(0.0)
        y = train["result"].astype(int)

        self.clf.fit(X, y)
        self.reg_home.fit(X, train["home_score"].astype(float))
        self.reg_away.fit(X, train["away_score"].astype(float))
        self.is_fitted = True
        logger.info("XGBoostModel entrenado con %s partidos, %s features", len(train), len(self.feature_cols))
        return self

    def predict_proba_1x2(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado")
        X = df[self.feature_cols].fillna(0.0)
        return self.clf.predict_proba(X)

    def predict_goals(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df[self.feature_cols].fillna(0.0)
        return self.reg_home.predict(X), self.reg_away.predict(X)

    def save(self, path: Optional[Path] = None) -> Path:
        ensure_dirs(get_models_dir())
        path = path or get_models_dir() / "xgboost.joblib"
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "XGBoostModel":
        path = path or get_models_dir() / "xgboost.joblib"
        return joblib.load(path)
