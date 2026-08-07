"""
Ensemble module combining 1X2 match outcome models with time-series CV weight optimization and Platt scaling calibration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

from src.models.base_model import Base1X2Model
from src.models.catboost_model import CatBoostModel
from src.models.decision_tree import DecisionTreeModel
from src.models.logistic_model import LogisticModel
from src.models.poisson import PoissonModel
from src.models.xgboost_model import XGBoostModel
from src.utils import ensure_dirs, get_models_dir

logger = logging.getLogger(__name__)


def _get_default_model_dict() -> Dict[str, Base1X2Model]:
    return {
        "poisson": PoissonModel(),
        "xgboost": XGBoostModel(),
        "catboost": CatBoostModel(),
        "decision_tree": DecisionTreeModel(),
        "logistic": LogisticModel(),
    }


class Ensemble:
    """
    Weighted ensemble of multi-class 1X2 prediction models.
    Supports probability calibration via Platt scaling and optimal weight estimation via time-series CV.
    """

    def __init__(self, models: Optional[Dict[str, Base1X2Model]] = None):
        self.models: Dict[str, Base1X2Model] = models or _get_default_model_dict()
        self.calibrators: Dict[str, LogisticRegression] = {}
        self.weights: Dict[str, float] = {name: 1.0 / len(self.models) for name in self.models}
        self.is_fitted = False

    def _prepare_inputs(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Helper to extract X and y cleanly if a single DataFrame with 'result' target is passed."""
        if y is None:
            if "result" in X.columns:
                y_series = X["result"].astype(int)
                X_df = X.drop(columns=["result"])
            else:
                raise ValueError("Target 'y' not provided and 'result' column not in X.")
        else:
            X_df = X
            y_series = y.astype(int)
        return X_df, y_series

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, n_splits: int = 5) -> "Ensemble":
        """
        Train all base models, fit per-model probability calibrators, and optimize ensemble weights
        using time-series cross-validation.
        """
        X_df, y_series = self._prepare_inputs(X, y)
        n_samples = len(X_df)

        if n_samples < 30:
            logger.warning("Dataset small (%d samples); using simplified fit.", n_samples)
            for name, model in self.models.items():
                model.fit(X_df, y_series)
            self.is_fitted = True
            return self

        logger.info("Performing time-series cross-validation for ensemble calibration & weight optimization...")

        # Collect out-of-fold predictions
        tscv = TimeSeriesSplit(n_splits=min(n_splits, n_samples // 10))
        oof_preds: Dict[str, List[np.ndarray]] = {name: [] for name in self.models}
        oof_targets: List[np.ndarray] = []

        for train_idx, val_idx in tscv.split(X_df):
            X_tr, y_tr = X_df.iloc[train_idx], y_series.iloc[train_idx]
            X_val, y_val = X_df.iloc[val_idx], y_series.iloc[val_idx]

            oof_targets.append(y_val.values)

            for name, model in self.models.items():
                try:
                    # Create a clean temp copy of model with same parameters
                    temp_model = model.__class__()
                    temp_model.fit(X_tr, y_tr)
                    p_val = temp_model.predict_proba(X_val)
                    oof_preds[name].append(p_val)
                except Exception as e:
                    logger.warning("Temp model %s failed in CV split: %s", name, e)
                    oof_preds[name].append(np.ones((len(val_idx), 3)) / 3.0)

        # Concatenate out-of-fold predictions
        y_oof = np.concatenate(oof_targets, axis=0)
        oof_probs_calibrated: Dict[str, np.ndarray] = {}

        for name in self.models:
            raw_oof = np.concatenate(oof_preds[name], axis=0)
            # Fit Platt scaling calibrator on raw probabilities
            try:
                calibrator = LogisticRegression(
                    multi_class="multinomial", solver="lbfgs", max_iter=500, random_state=42
                )
                calibrator.fit(raw_oof, y_oof)
                self.calibrators[name] = calibrator
                cal_oof = calibrator.predict_proba(raw_oof)
            except Exception as e:
                logger.warning("Calibration failed for model %s: %s", name, e)
                cal_oof = raw_oof

            oof_probs_calibrated[name] = cal_oof

        # Optimize ensemble weights on calibrated OOF predictions
        self._optimize_weights_from_probs(oof_probs_calibrated, y_oof)

        # Fit all final base models on FULL dataset
        logger.info("Fitting all base models on full dataset (%d samples)...", n_samples)
        for name, model in self.models.items():
            model.fit(X_df, y_series)

        self.is_fitted = True
        logger.info("Ensemble fit complete. Learned weights: %s", self.weights)
        return self

    def _optimize_weights_from_probs(
        self, probs_dict: Dict[str, np.ndarray], y_true: np.ndarray
    ) -> Dict[str, float]:
        """Find optimal convex combination weights minimizing multi-class log loss."""
        names = list(probs_dict.keys())
        m_count = len(names)

        if m_count == 0:
            return {}

        probs_tensor = np.stack([probs_dict[name] for name in names], axis=0)  # (M, N, 3)

        def objective(weights: np.ndarray) -> float:
            w = weights / (np.sum(weights) + 1e-12)
            blend = np.tensordot(w, probs_tensor, axes=(0, 0))  # (N, 3)
            blend = np.clip(blend, 1e-12, 1.0 - 1e-12)
            blend /= blend.sum(axis=1, keepdims=True)
            return log_loss(y_true, blend, labels=[0, 1, 2])

        init_w = np.ones(m_count) / m_count
        bounds = [(0.0, 1.0) for _ in range(m_count)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(
            objective, init_w, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if res.success:
            opt_w = np.clip(res.x, 0.0, 1.0)
            opt_w /= np.sum(opt_w)
        else:
            opt_w = init_w

        self.weights = {name: float(opt_w[i]) for i, name in enumerate(names)}
        return self.weights

    def optimize_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """
        Explicitly optimize weights on a validation dataset.
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble models must be fitted before optimizing weights.")

        X_v, y_v = self._prepare_inputs(X_val, y_val)
        probs_dict = {}

        for name, model in self.models.items():
            raw_p = model.predict_proba(X_v)
            if name in self.calibrators:
                p = self.calibrators[name].predict_proba(raw_p)
            else:
                p = raw_p
            probs_dict[name] = p

        return self._optimize_weights_from_probs(probs_dict, y_v.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated, weighted multi-class probabilities (0=Home, 1=Draw, 2=Away).
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble is not fitted.")

        X_df = X.drop(columns=["result"], errors="ignore") if "result" in X.columns else X
        probs_list = []
        weight_list = []

        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w <= 1e-6:
                continue
            try:
                raw_p = model.predict_proba(X_df)
                if name in self.calibrators:
                    p = self.calibrators[name].predict_proba(raw_p)
                else:
                    p = raw_p
                probs_list.append(p)
                weight_list.append(w)
            except Exception as e:
                logger.warning("Model %s prediction failed: %s", name, e)

        if not probs_list:
            raise RuntimeError("All ensemble models failed during predict_proba.")

        weights_arr = np.array(weight_list)
        weights_arr /= weights_arr.sum()

        stacked = np.stack(probs_list, axis=0)  # (M_active, N, 3)
        final_probs = np.tensordot(weights_arr, stacked, axes=(0, 0))  # (N, 3)

        final_probs = np.clip(final_probs, 1e-12, 1.0)
        final_probs /= final_probs.sum(axis=1, keepdims=True)
        return final_probs

    def predict_proba_1x2(self, X: pd.DataFrame) -> np.ndarray:
        """Backwards compatibility alias for predict_proba."""
        return self.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return structured prediction DataFrame with class probabilities and predicted winner.
        """
        proba = self.predict_proba(X)
        pred_class = proba.argmax(axis=1)

        result_map = {0: "Home", 1: "Draw", 2: "Away"}

        out = pd.DataFrame(
            {
                "prob_home": proba[:, 0],
                "prob_draw": proba[:, 1],
                "prob_away": proba[:, 2],
                "pred_result": pred_class,
                "pred_label": [result_map[c] for c in pred_class],
            }
        )

        if "match_id" in X.columns:
            out.insert(0, "match_id", X["match_id"].values)
        return out

    def save(self, path: Optional[Path] = None) -> Path:
        save_path = path or (get_models_dir() / "ensemble.joblib")
        ensure_dirs(save_path.parent)
        joblib.dump(self, save_path)
        logger.info("Ensemble saved to %s", save_path)
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Ensemble":
        load_path = path or (get_models_dir() / "ensemble.joblib")
        obj = joblib.load(load_path)
        if not isinstance(obj, Ensemble):
            raise TypeError(f"Loaded object is {type(obj)}, expected Ensemble")
        return obj
