"""
Poisson Regression model with Dixon-Coles adjustment for 1X2 match outcome predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from src.data.features import FEATURE_COLUMNS
from src.models.base_model import Base1X2Model
from src.utils import ensure_dirs, get_models_dir

logger = logging.getLogger(__name__)


def dixon_coles_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """
    Dixon & Coles (1997) adjustment factor for low-scoring match outcomes.
    """
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    elif x == 1 and y == 0:
        return 1.0 + mu * rho
    elif x == 0 and y == 1:
        return 1.0 + lam * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0


class PoissonModel(Base1X2Model):
    """
    Independent Poisson regression for home/away goal expectations with Dixon-Coles draw adjustment.
    """

    def __init__(self, alpha: float = 1.0, max_goals: int = 8, rho: float = -0.05):
        self.alpha = alpha
        self.max_goals = max_goals
        self.rho = rho
        self.home_model = PoissonRegressor(alpha=self.alpha, max_iter=500)
        self.away_model = PoissonRegressor(alpha=self.alpha, max_iter=500)
        self.feature_cols: List[str] = []
        self.is_fitted = False

    @property
    def name(self) -> str:
        return "poisson"

    def _clean_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
        """Extract features and optional home/away goal targets if present in X."""
        X_df = X.copy()
        y_home = None
        y_away = None

        if "home_score" in X_df.columns:
            y_home = pd.to_numeric(X_df.pop("home_score"), errors="coerce")
        if "away_score" in X_df.columns:
            y_away = pd.to_numeric(X_df.pop("away_score"), errors="coerce")

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
            "result",
        }
        cols = [c for c in FEATURE_COLUMNS if c in X_df.columns]
        if not cols:
            cols = [c for c in X_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(X_df[c])]

        return X_df[cols].fillna(0.0), y_home, y_away

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PoissonModel":
        X_clean, y_home, y_away = self._clean_features(X)
        self.feature_cols = list(X_clean.columns)

        # If goal targets were not in X, construct goal proxies from y (0=Home, 1=Draw, 2=Away)
        if y_home is None or y_away is None or y_home.isna().all() or y_away.isna().all():
            y_val = y.astype(int).values
            y_home = pd.Series(np.where(y_val == 0, 1.8, np.where(y_val == 1, 1.1, 0.6)), index=X_clean.index)
            y_away = pd.Series(np.where(y_val == 2, 1.8, np.where(y_val == 1, 1.1, 0.6)), index=X_clean.index)

        mask = y_home.notna() & y_away.notna()
        X_fit = X_clean[mask]
        yh_fit = y_home[mask]
        ya_fit = y_away[mask]

        self.home_model.fit(X_fit, yh_fit)
        self.away_model.fit(X_fit, ya_fit)
        self.is_fitted = True
        logger.info("PoissonModel fitted on %d samples with %d features", len(X_fit), len(self.feature_cols))
        return self

    def predict_lambdas(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("PoissonModel is not fitted yet.")
        X_clean, _, _ = self._clean_features(X)
        X_clean = X_clean.reindex(columns=self.feature_cols, fill_value=0.0)

        lam_home = self.home_model.predict(X_clean)
        lam_away = self.away_model.predict(X_clean)
        return np.clip(lam_home, 0.1, 6.0), np.clip(lam_away, 0.1, 6.0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        lam_h, lam_a = self.predict_lambdas(X)
        n = len(X)
        probs = np.zeros((n, 3))
        max_g = self.max_goals

        goals_arr = np.arange(0, max_g + 1)

        for i in range(n):
            ph = poisson.pmf(goals_arr, lam_h[i])
            pa = poisson.pmf(goals_arr, lam_a[i])
            matrix = np.outer(ph, pa)

            # Apply Dixon-Coles adjustment for low scores (0-0, 1-0, 0-1, 1-1)
            for x in range(min(2, max_g + 1)):
                for y in range(min(2, max_g + 1)):
                    tau = dixon_coles_tau(x, y, lam_h[i], lam_a[i], self.rho)
                    matrix[x, y] = max(0.0, matrix[x, y] * tau)

            total_p = matrix.sum()
            if total_p > 0:
                matrix /= total_p
            else:
                matrix = np.ones_like(matrix) / matrix.size

            p_home = matrix[np.tril_indices(max_g + 1, k=-1)].sum()
            p_draw = np.trace(matrix)
            p_away = matrix[np.triu_indices(max_g + 1, k=1)].sum()

            total_1x2 = p_home + p_draw + p_away
            probs[i] = [p_home / total_1x2, p_draw / total_1x2, p_away / total_1x2]

        return probs

    def save(self, path: Optional[Path] = None) -> Path:
        save_path = path or (get_models_dir() / "poisson.joblib")
        ensure_dirs(save_path.parent)
        joblib.dump(self, save_path)
        logger.info("PoissonModel saved to %s", save_path)
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PoissonModel":
        load_path = path or (get_models_dir() / "poisson.joblib")
        obj = joblib.load(load_path)
        if not isinstance(obj, PoissonModel):
            raise TypeError(f"Loaded object is {type(obj)}, expected PoissonModel")
        return obj
