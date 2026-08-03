"""
Árbol de decisión (baseline interpretable).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.utils import get_models_dir, ensure_dirs

logger = logging.getLogger(__name__)


class DecisionTreeModel:
    def __init__(self, max_depth: int = 8, min_samples_leaf: int = 20):
        self.clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
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
        return [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

    def fit(self, df: pd.DataFrame) -> "DecisionTreeModel":
        train = df[df["status"].str.lower().isin(["finished", "ft", "complete"])].copy()
        self.feature_cols = self._feature_cols(train)
        X = train[self.feature_cols].fillna(0.0)
        y = train["result"].astype(int)
        self.clf.fit(X, y)
        self.is_fitted = True
        logger.info("DecisionTreeModel entrenado con %s partidos", len(train))
        return self

    def predict_proba_1x2(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado")
        X = df[self.feature_cols].fillna(0.0)
        return self.clf.predict_proba(X)

    def save(self, path: Optional[Path] = None) -> Path:
        ensure_dirs(get_models_dir())
        path = path or get_models_dir() / "decision_tree.joblib"
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DecisionTreeModel":
        path = path or get_models_dir() / "decision_tree.joblib"
        return joblib.load(path)
