"""
Ensemble final: combinación ponderada de todos los modelos.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.models.poisson import PoissonModel
from src.models.xgboost_model import XGBoostModel
from src.models.catboost_model import CatBoostModel
from src.models.decision_tree import DecisionTreeModel
from src.models.lstm_momentum import LSTMMomentum
from src.models.lstm_result import LSTMResult
from src.models.lstm_model import LSTMModel
from src.utils import get_models_dir, ensure_dirs

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "poisson": 0.18,
    "xgboost": 0.22,
    "catboost": 0.22,
    "decision_tree": 0.08,
    "lstm_momentum": 0.10,
    "lstm_result": 0.08,
    "lstm_model": 0.12,
}


class Ensemble:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.models: Dict[str, object] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "Ensemble":
        logger.info("Entrenando ensemble completo…")

        self.models["poisson"] = PoissonModel().fit(df)
        self.models["xgboost"] = XGBoostModel().fit(df)
        self.models["catboost"] = CatBoostModel().fit(df)
        self.models["decision_tree"] = DecisionTreeModel().fit(df)
        self.models["lstm_momentum"] = LSTMMomentum().fit(df)
        self.models["lstm_result"] = LSTMResult().fit(df)
        self.models["lstm_model"] = LSTMModel().fit(df)

        self.is_fitted = True
        logger.info("Ensemble entrenado.")
        return self

    def predict_proba_1x2(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Ensemble no entrenado")

        probs_list = []
        weight_list = []

        for name, model in self.models.items():
            try:
                p = model.predict_proba_1x2(df)
                probs_list.append(p)
                weight_list.append(self.weights.get(name, 0.1))
            except Exception as e:
                logger.warning("Modelo %s falló en predicción: %s", name, e)

        if not probs_list:
            raise RuntimeError("Ningún modelo generó predicciones")

        weights = np.array(weight_list)
        weights = weights / weights.sum()
        stacked = np.stack(probs_list, axis=0)  # (n_models, n_samples, 3)
        final = np.tensordot(weights, stacked, axes=(0, 0))
        # Renormalizar por seguridad
        final = final / final.sum(axis=1, keepdims=True)
        return final

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        proba = self.predict_proba_1x2(df)
        out = df[["match_id", "home_team_name", "away_team_name", "kickoff_utc"]].copy()
        out["prob_home"] = proba[:, 0]
        out["prob_draw"] = proba[:, 1]
        out["prob_away"] = proba[:, 2]
        out["pred_result"] = np.argmax(proba, axis=1)  # 0=H, 1=D, 2=A
        out["pred_label"] = out["pred_result"].map({0: "H", 1: "D", 2: "A"})
        return out

    def save(self) -> None:
        ensure_dirs(get_models_dir())
        for name, model in self.models.items():
            model.save()
        joblib.dump(self.weights, get_models_dir() / "ensemble_weights.joblib")
        logger.info("Ensemble guardado en %s", get_models_dir())

    @classmethod
    def load(cls) -> "Ensemble":
        obj = cls()
        obj.weights = joblib.load(get_models_dir() / "ensemble_weights.joblib")
        obj.models["poisson"] = PoissonModel.load()
        obj.models["xgboost"] = XGBoostModel.load()
        obj.models["catboost"] = CatBoostModel.load()
        obj.models["decision_tree"] = DecisionTreeModel.load()
        obj.models["lstm_momentum"] = LSTMMomentum.load()
        obj.models["lstm_result"] = LSTMResult.load()
        obj.models["lstm_model"] = LSTMModel.load()
        obj.is_fitted = True
        return obj
