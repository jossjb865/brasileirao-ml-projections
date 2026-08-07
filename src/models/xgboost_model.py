"""
XGBoost multiclass classification model for 1X2 match outcomes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Any, Dict

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.data.features import FEATURE_COLUMNS
from src.models.base_model import Base1X2Model
from src.utils import ensure_dirs, get_models_dir, load_yaml, ROOT

logger = logging.getLogger(__name__)


def _load_xgboost_params() -> Dict[str, Any]:
    cfg_path = ROOT / "configs" / "model_hyperparams.yaml"
    if cfg_path.exists():
        cfg = load_yaml(cfg_path)
        return cfg.get("xgboost", {})
    return {}


class XGBoostModel(Base1X2Model):
    """
    XGBoost multi-class classifier predicting 1X2 outcomes (0=Home, 1=Draw, 2=Away).
    """

    def __init__(self, **kwargs):
        params = _load_xgboost_params()
        default_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "n_jobs": -1,
        }
        default_params.update(params)
        default_params.update(kwargs)

        self.clf = XGBClassifier(**default_params)
        self.feature_cols: List[str] = []
        self.is_fitted = False

    @property
    def name(self) -> str:
        return "xgboost"

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in FEATURE_COLUMNS if c in X.columns]
        if not cols:
            exclude_cols = {
                "match_id",
                "kickoff_utc",
                "status",
                "home_team_id",
                "away_team_id",
                "home_team_name",
                "away_team_name",
                "home_match_count",
                "away_match_count",
                "home_score",
                "away_score",
                "result",
            }
            cols = [
                c for c in X.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(X[c])
            ]
        return X[cols].fillna(0.0)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        X_clean = self._select_features(X)
        self.feature_cols = list(X_clean.columns)
        y_clean = y.astype(int)

        self.clf.fit(X_clean, y_clean)
        self.is_fitted = True
        logger.info("XGBoostModel fitted on %d samples with %d features", len(X_clean), len(self.feature_cols))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("XGBoostModel is not fitted yet.")
        X_clean = self._select_features(X)
        X_clean = X_clean.reindex(columns=self.feature_cols, fill_value=0.0)
        return self.clf.predict_proba(X_clean)

    def save(self, path: Optional[Path] = None) -> Path:
        save_path = path or (get_models_dir() / "xgboost.joblib")
        ensure_dirs(save_path.parent)
        joblib.dump(self, save_path)
        logger.info("XGBoostModel saved to %s", save_path)
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "XGBoostModel":
        load_path = path or (get_models_dir() / "xgboost.joblib")
        obj = joblib.load(load_path)
        if not isinstance(obj, XGBoostModel):
            raise TypeError(f"Loaded object is {type(obj)}, expected XGBoostModel")
        return obj
