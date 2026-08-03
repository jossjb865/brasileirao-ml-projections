"""
Entrenamiento completo del ensemble sobre datos reales.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features import build_features
from src.data.store import DataStore
from src.ensemble import Ensemble
from src.utils import load_yaml, setup_logging, ensure_dirs, get_models_dir

logger = logging.getLogger(__name__)


def train_pipeline(force_rebuild_features: bool = True) -> None:
    setup_logging()
    cfg = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "model_hyperparams.yaml")
    ensure_dirs(get_models_dir())

    store = DataStore()
    features = store.load_features()

    if force_rebuild_features or features.empty:
        logger.info("Construyendo features desde datos reales…")
        window = cfg.get("features", {}).get("rolling_window", 5)
        features = build_features(window=window)

    if features.empty:
        raise RuntimeError(
            "No hay features disponibles. "
            "Ejecuta primero el fetch de datos (src.data.fetch)."
        )

    # Solo partidos finalizados con target válido
    train_df = features[
        features["status"].str.lower().isin(["finished", "ft", "complete"])
        & features["result"].notna()
    ].copy()

    # Eliminar cualquier NaN residual en scores
    train_df = train_df.dropna(subset=["home_score", "away_score"])
    train_df["home_score"] = pd.to_numeric(train_df["home_score"], errors="coerce")
    train_df["away_score"] = pd.to_numeric(train_df["away_score"], errors="coerce")
    train_df = train_df.dropna(subset=["home_score", "away_score"])

    # Recalcular result de forma segura
    train_df["result"] = np.where(
        train_df["home_score"] > train_df["away_score"],
        0,
        np.where(train_df["home_score"] == train_df["away_score"], 1, 2),
    ).astype(int)

    logger.info("Partidos de entrenamiento (con marcador válido): %s", len(train_df))
    if len(train_df) < 50:
        raise RuntimeError(
            f"Pocos partidos para entrenar ({len(train_df)}). "
            "Descarga más historial o revisa el fetch."
        )

    weights = cfg.get("ensemble", {}).get("weights")
    ensemble = Ensemble(weights=weights)
    ensemble.fit(train_df)
    ensemble.save()

    # Evaluación rápida in-sample (solo referencia)
    proba = ensemble.predict_proba_1x2(train_df)
    pred = proba.argmax(axis=1)
    acc = (pred == train_df["result"].astype(int).values).mean()
    logger.info("Accuracy in-sample (referencia): %.3f", acc)
    logger.info("Entrenamiento finalizado. Modelos guardados en %s", get_models_dir())


if __name__ == "__main__":
    train_pipeline(force_rebuild_features=True)
