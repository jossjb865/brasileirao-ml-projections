"""
Evaluation and backtesting module for 1X2 match outcome ML models.
Includes chronological time-series splitting, walk-forward backtesting, metric calculation, and comparison reporting.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any, Generator

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, classification_report as sk_classification_report

from src.data.features import FEATURE_COLUMNS
from src.models.base_model import Base1X2Model
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def calculate_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate multi-class Brier score: (1/N) * sum_i sum_k (p_ik - y_ik)^2.
    """
    n_samples, n_classes = y_proba.shape
    y_onehot = np.zeros((n_samples, n_classes))
    for i, label in enumerate(y_true):
        y_onehot[i, int(label)] = 1.0
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


def calculate_expected_calibration_error(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE) for multi-class predictions.
    """
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin.astype(float))

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def time_series_split(
    df: pd.DataFrame, n_splits: int = 5, test_size: int = 50
) -> Generator[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], None, None]:
    """
    Yield chronological train/test splits for walk-forward evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset sorted chronologically by kickoff date, containing features and target 'result'.
    n_splits : int
        Number of walk-forward folds.
    test_size : int
        Number of matches per test set fold.
    """
    data = df.copy()
    if "kickoff_utc" in data.columns:
        data["kickoff_utc"] = pd.to_datetime(data["kickoff_utc"], utc=True, errors="coerce")
        data = data.sort_values("kickoff_utc").reset_index(drop=True)

    n_samples = len(data)
    total_test = n_splits * test_size

    if n_samples <= total_test + 20:
        # Dynamic fallback for smaller datasets
        test_size = max(10, (n_samples - 20) // n_splits)
        total_test = n_splits * test_size

    for i in range(n_splits):
        test_end = n_samples - (n_splits - 1 - i) * test_size
        test_start = test_end - test_size
        train_end = test_start

        train_df = data.iloc[:train_end]
        test_df = data.iloc[test_start:test_end]

        feature_cols = [c for c in FEATURE_COLUMNS if c in data.columns]
        if not feature_cols:
            exclude = {
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
            feature_cols = [
                c for c in data.columns if c not in exclude and pd.api.types.is_numeric_dtype(data[c])
            ]

        X_train = train_df[feature_cols].fillna(0.0)
        y_train = train_df["result"].astype(int)
        X_test = test_df[feature_cols].fillna(0.0)
        y_test = test_df["result"].astype(int)

        yield X_train, y_train, X_test, y_test


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Fit a model on train set and compute accuracy, log_loss, brier_score, and calibration_error on test set.
    """
    # Fit model if not already fitted or if standard interface model
    if hasattr(model, "fit"):
        model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, "predict_proba_1x2"):
        y_proba = model.predict_proba_1x2(X_test)
    else:
        raise ValueError("Model does not implement predict_proba or predict_proba_1x2.")

    y_true = y_test.values.astype(int)
    y_pred = y_proba.argmax(axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    loss = float(log_loss(y_true, y_proba, labels=[0, 1, 2]))
    brier = calculate_brier_score(y_true, y_proba)
    ece = calculate_expected_calibration_error(y_true, y_proba)

    return {
        "accuracy": acc,
        "log_loss": loss,
        "brier_score": brier,
        "calibration_error": ece,
    }


def backtest(ensemble: Any, df: pd.DataFrame, n_splits: int = 5) -> Dict[str, Any]:
    """
    Perform walk-forward backtesting for the ensemble model.
    """
    fold_results = []

    for fold_idx, (X_tr, y_tr, X_te, y_te) in enumerate(time_series_split(df, n_splits=n_splits)):
        metrics = evaluate_model(ensemble, X_tr, y_tr, X_te, y_te)
        metrics["fold"] = fold_idx + 1
        fold_results.append(metrics)

    res_df = pd.DataFrame(fold_results)
    mean_metrics = {
        "mean_accuracy": float(res_df["accuracy"].mean()),
        "mean_log_loss": float(res_df["log_loss"].mean()),
        "mean_brier_score": float(res_df["brier_score"].mean()),
        "mean_calibration_error": float(res_df["calibration_error"].mean()),
    }

    logger.info("Backtest results across %d folds: %s", n_splits, mean_metrics)
    return {
        "summary": mean_metrics,
        "fold_details": res_df,
    }


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Generate detailed classification performance report dictionary and string.
    """
    target_names = ["Home (0)", "Draw (1)", "Away (2)"]
    rep_str = sk_classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )

    acc = float(accuracy_score(y_true, y_pred))
    report = {
        "accuracy": acc,
        "report_str": rep_str,
    }

    if y_proba is not None:
        report["log_loss"] = float(log_loss(y_true, y_proba, labels=[0, 1, 2]))
        report["brier_score"] = calculate_brier_score(y_true, y_proba)
        report["calibration_error"] = calculate_expected_calibration_error(y_true, y_proba)

    return report


def compare_models(
    models: List[Any], X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> pd.DataFrame:
    """
    Compare multiple models across time-series cross-validation splits and print formatted results table.
    """
    df = X.copy()
    df["result"] = y.values

    results = []

    for model in models:
        model_name = getattr(model, "name", model.__class__.__name__)
        fold_metrics = []

        for X_tr, y_tr, X_te, y_te in time_series_split(df, n_splits=n_splits):
            m_copy = model.__class__() if hasattr(model, "__class__") else model
            metrics = evaluate_model(m_copy, X_tr, y_tr, X_te, y_te)
            fold_metrics.append(metrics)

        m_df = pd.DataFrame(fold_metrics)
        results.append(
            {
                "Model": model_name,
                "Accuracy": f"{m_df['accuracy'].mean():.4f} +/- {m_df['accuracy'].std():.4f}",
                "Log Loss": f"{m_df['log_loss'].mean():.4f} +/- {m_df['log_loss'].std():.4f}",
                "Brier Score": f"{m_df['brier_score'].mean():.4f} +/- {m_df['brier_score'].std():.4f}",
                "ECE": f"{m_df['calibration_error'].mean():.4f} +/- {m_df['calibration_error'].std():.4f}",
                "_raw_acc": m_df["accuracy"].mean(),
                "_raw_loss": m_df["log_loss"].mean(),
            }
        )

    summary_df = pd.DataFrame(results).sort_values("_raw_loss").reset_index(drop=True)
    display_df = summary_df.drop(columns=["_raw_acc", "_raw_loss"])

    print("\n" + "=" * 70)
    print(f" MODEL COMPARISON SUMMARY ({n_splits}-fold Time-Series Cross-Validation)")
    print("=" * 70)
    print(display_df.to_string(index=False))
    print("=" * 70 + "\n")

    return display_df
