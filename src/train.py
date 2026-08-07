"""
Training and evaluation pipeline for the Brasileirão ML Ensemble.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

from src.data.features import build_features
from src.data.store import DataStore
from src.ensemble import Ensemble
from src.utils import load_yaml, setup_logging, ensure_dirs, get_models_dir, get_outputs_dir

logger = logging.getLogger(__name__)


def brier_score_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Computes multiclass Brier score."""
    n_samples, n_classes = y_proba.shape
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


def calculate_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) for multiclass probabilities.
    ECE measures the difference between predicted confidence and actual accuracy across probability bins.
    """
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def extract_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Filters finished matches, prepares numerical feature matrix X and target y.
    Target: 0 = Home Win (H), 1 = Draw (D), 2 = Away Win (A).
    """
    train_df = df[
        df["status"].astype(str).str.lower().isin(["finished", "ft", "complete"])
        & df["result"].notna()
    ].copy()

    train_df = train_df.dropna(subset=["home_score", "away_score"])
    train_df["home_score"] = pd.to_numeric(train_df["home_score"], errors="coerce")
    train_df["away_score"] = pd.to_numeric(train_df["away_score"], errors="coerce")
    train_df = train_df.dropna(subset=["home_score", "away_score"])

    # Ensure chronological order
    if "kickoff_utc" in train_df.columns:
        train_df["kickoff_utc"] = pd.to_datetime(train_df["kickoff_utc"], utc=True, errors="coerce")
        train_df = train_df.sort_values("kickoff_utc").reset_index(drop=True)

    train_df["result"] = np.where(
        train_df["home_score"] > train_df["away_score"],
        0,
        np.where(train_df["home_score"] == train_df["away_score"], 1, 2),
    ).astype(int)

    exclude_cols = {
        "match_id", "competition_id", "season_id", "season_name", "status",
        "kickoff_utc", "matchday", "home_team_id", "home_team_name",
        "away_team_id", "away_team_name", "home_score", "away_score",
        "home_ht_score", "away_ht_score", "result", "total_goals", "btts",
        "xg_available", "odds_available", "odds_home", "odds_draw", "odds_away"
    }

    feature_cols = [
        c for c in train_df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    X = train_df[feature_cols].fillna(0.0)
    y = train_df["result"].astype(int)

    return X, y, feature_cols


def optimize_ensemble_weights(
    model_probas: Dict[str, np.ndarray],
    y_true: np.ndarray,
) -> Dict[str, float]:
    """
    Optimizes ensemble model weights using scipy.optimize.minimize to minimize log_loss.
    """
    model_names = list(model_probas.keys())
    n_models = len(model_names)
    if n_models == 0:
        return {}

    probas_stack = np.stack([model_probas[name] for name in model_names], axis=0)  # (M, N, 3)

    def loss_func(weights: np.ndarray) -> float:
        w = weights / np.sum(weights)
        blended = np.tensordot(w, probas_stack, axes=(0, 0))  # (N, 3)
        blended = np.clip(blended, 1e-15, 1 - 1e-15)
        blended = blended / blended.sum(axis=1, keepdims=True)
        return log_loss(y_true, blended, labels=[0, 1, 2])

    init_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    if res.success and np.sum(res.x) > 0:
        opt_w = res.x / np.sum(res.x)
    else:
        opt_w = init_weights

    return {name: float(round(w, 4)) for name, w in zip(model_names, opt_w)}


def run_walk_forward_backtest(
    df: pd.DataFrame,
    cv_splits: int = 5,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluates models using time-series split walk-forward backtesting.
    """
    X, y, _ = extract_features_and_target(df)
    n_samples = len(X)

    if n_samples < 30:
        logger.warning("Not enough samples (%d) for meaningful backtesting", n_samples)
        return {}, pd.DataFrame()

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    oof_predictions = []
    oof_targets = []
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        y_test = y.iloc[test_idx]

        ensemble = Ensemble()
        ensemble.fit(train_df)

        proba = ensemble.predict_proba_1x2(test_df)
        preds = proba.argmax(axis=1)

        acc = float(accuracy_score(y_test, preds))
        ll = float(log_loss(y_test, proba, labels=[0, 1, 2]))
        bs = brier_score_loss(y_test.values, proba)
        ece = calculate_ece(y_test.values, proba)

        fold_metrics.append({
            "fold": fold + 1,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": round(acc, 4),
            "log_loss": round(ll, 4),
            "brier_score": round(bs, 4),
            "calibration_ece": round(ece, 4),
        })

        oof_predictions.append(proba)
        oof_targets.append(y_test.values)

    all_oof_proba = np.vstack(oof_predictions)
    all_oof_y = np.concatenate(oof_targets)
    all_oof_preds = all_oof_proba.argmax(axis=1)

    overall_metrics = {
        "accuracy": float(round(accuracy_score(all_oof_y, all_oof_preds), 4)),
        "log_loss": float(round(log_loss(all_oof_y, all_oof_proba, labels=[0, 1, 2]), 4)),
        "brier_score": float(round(brier_score_loss(all_oof_y, all_oof_proba), 4)),
        "calibration_ece": float(round(calculate_ece(all_oof_y, all_oof_proba), 4)),
    }

    return overall_metrics, pd.DataFrame(fold_metrics)


def train_pipeline(
    force_features: bool = False,
    no_backtest: bool = False,
    cv_splits: int = 5,
) -> None:
    """Executes the full training, validation, ensemble optimization, and backtesting pipeline."""
    setup_logging()
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "model_hyperparams.yaml"
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}

    ensure_dirs(get_models_dir(), get_outputs_dir())

    store = DataStore()
    features = store.load_features()

    if force_features or features.empty:
        logger.info("Building features from raw data...")
        window = cfg.get("features", {}).get("rolling_window", 5)
        features = build_features()
    else:
        logger.info("Using cached features.parquet")

    if features.empty:
        raise RuntimeError("No feature dataset available. Please fetch data first.")

    train_df = features[
        features["status"].astype(str).str.lower().isin(["finished", "ft", "complete"])
        & features["result"].notna()
    ].copy()

    if len(train_df) < 30:
        raise RuntimeError(f"Insufficient training records ({len(train_df)}). Need at least 30.")

    logger.info("Training dataset: %d finished matches", len(train_df))

    X, y, feature_cols = extract_features_and_target(train_df)
    n_samples = len(X)
    val_size = min(int(n_samples * 0.2), 100)
    train_split_idx = n_samples - val_size

    train_sub = train_df.iloc[:train_split_idx]
    val_sub = train_df.iloc[train_split_idx:]

    ensemble = Ensemble()
    ensemble.fit(train_sub)

    val_probas = {}
    for name, model in ensemble.models.items():
        try:
            if hasattr(model, "predict_proba_1x2"):
                p = model.predict_proba_1x2(val_sub)
            elif hasattr(model, "predict_proba"):
                val_X = val_sub[feature_cols].fillna(0.0)
                p = model.predict_proba(val_X)
            else:
                continue
            val_probas[name] = p
        except Exception as e:
            logger.warning("Could not compute validation probabilities for %s: %s", name, e)

    y_val = y.iloc[train_split_idx:].values
    optimized_weights = optimize_ensemble_weights(val_probas, y_val)
    logger.info("Optimized ensemble weights: %s", optimized_weights)

    final_ensemble = Ensemble(weights=optimized_weights)
    final_ensemble.fit(train_df)
    final_ensemble.save()

    backtest_metrics = {}
    fold_df = pd.DataFrame()
    if not no_backtest:
        logger.info("Running walk-forward backtesting with %d CV splits...", cv_splits)
        backtest_metrics, fold_df = run_walk_forward_backtest(train_df, cv_splits=cv_splits)

    val_probas_ensemble = final_ensemble.predict_proba_1x2(val_sub)
    val_y = y.iloc[train_split_idx:].values
    val_preds = val_probas_ensemble.argmax(axis=1)

    summary_metrics = {
        "val_accuracy": float(round(accuracy_score(val_y, val_preds), 4)),
        "val_log_loss": float(round(log_loss(val_y, val_probas_ensemble, labels=[0, 1, 2]), 4)),
        "val_brier_score": float(round(brier_score_loss(val_y, val_probas_ensemble), 4)),
        "val_calibration_ece": float(round(calculate_ece(val_y, val_probas_ensemble), 4)),
        "backtest": backtest_metrics,
        "ensemble_weights": optimized_weights,
    }

    metrics_path = get_outputs_dir() / "metrics.json"
    models_metrics_path = get_models_dir() / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    with open(models_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info("Saved metrics to %s and %s", metrics_path, models_metrics_path)

    print("\n" + "=" * 60)
    print("        BRASILEIRÃO ML PIPELINE SUMMARY        ")
    print("=" * 60)
    print(f"Total Matches Trained : {len(train_df)}")
    print(f"Validation Set Size   : {len(val_sub)}")
    print("-" * 60)
    print("Validation Metrics:")
    print(f"  Accuracy        : {summary_metrics['val_accuracy']:.4f}")
    print(f"  Log Loss        : {summary_metrics['val_log_loss']:.4f}")
    print(f"  Brier Score     : {summary_metrics['val_brier_score']:.4f}")
    print(f"  Calibration ECE : {summary_metrics['val_calibration_ece']:.4f}")

    if backtest_metrics:
        print("-" * 60)
        print("Walk-Forward Backtest Metrics:")
        print(f"  Accuracy        : {backtest_metrics.get('accuracy', 0.0):.4f}")
        print(f"  Log Loss        : {backtest_metrics.get('log_loss', 0.0):.4f}")
        print(f"  Brier Score     : {backtest_metrics.get('brier_score', 0.0):.4f}")
        print(f"  Calibration ECE : {backtest_metrics.get('calibration_ece', 0.0):.4f}")

    print("-" * 60)
    print("Optimized Model Weights:")
    for name, w in optimized_weights.items():
        print(f"  {name:<16}: {w:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brasileirão ML Ensemble Pipeline")
    parser.add_argument("--force-features", action="store_true", help="Force rebuilding features")
    parser.add_argument("--no-backtest", action="store_true", help="Skip walk-forward backtesting")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of CV splits for backtesting")
    args = parser.parse_args()

    train_pipeline(
        force_features=args.force_features,
        no_backtest=args.no_backtest,
        cv_splits=args.cv_splits,
    )
