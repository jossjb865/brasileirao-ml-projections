"""
Utilidades generales del proyecto brasileirao-ml-projections.
"""

from __future__ import annotations

import functools
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def setup_logging(level: str = "INFO") -> None:
    """Configura el sistema de logging estándar para el proyecto."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Carga y parsea un archivo YAML desde disco."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs(*paths: str | Path) -> None:
    """Crea los directorios indicados si no existen."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def get_data_dir() -> Path:
    """Retorna la ruta absoluta al directorio de datos."""
    p = ROOT / "data"
    ensure_dirs(p)
    return p


def get_models_dir() -> Path:
    """Retorna la ruta absoluta al directorio de modelos."""
    p = ROOT / "models"
    ensure_dirs(p)
    return p


def get_outputs_dir() -> Path:
    """Retorna la ruta absoluta al directorio de outputs/predicciones."""
    p = ROOT / "outputs"
    ensure_dirs(p)
    return p


@functools.lru_cache(maxsize=4)
def _load_config_cached(str_path: str) -> Dict[str, Any]:
    return load_yaml(Path(str_path))


def get_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Carga y mantiene en caché la configuración general (competition.yaml)."""
    target_path = Path(path) if path else ROOT / "configs" / "competition.yaml"
    return _load_config_cached(str(target_path.resolve()))


def get_cache_dir() -> Path:
    """Retorna la ruta absoluta al directorio de caché interno (.cache)."""
    p = ROOT / ".cache"
    ensure_dirs(p)
    return p


def validate_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspecciona un DataFrame de características verificando valores NaN o Inf.
    Retorna un reporte detallado e imprime advertencias si existen problemas.
    """
    if df.empty:
        return {
            "total_rows": 0,
            "total_cols": 0,
            "nan_counts": {},
            "inf_counts": {},
            "has_issues": False,
        }

    nan_counts: Dict[str, int] = {}
    inf_counts: Dict[str, int] = {}

    for col in df.columns:
        s = df[col]
        nan_c = int(s.isna().sum())
        if nan_c > 0:
            nan_counts[col] = nan_c

        if pd.api.types.is_numeric_dtype(s):
            inf_c = int(np.isinf(s).sum())
            if inf_c > 0:
                inf_counts[col] = inf_c

    has_issues = len(nan_counts) > 0 or len(inf_counts) > 0
    report = {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "nan_counts": nan_counts,
        "inf_counts": inf_counts,
        "has_issues": has_issues,
    }

    if has_issues:
        logging.getLogger(__name__).warning(
            "validate_features: se detectaron anomalías en las características. "
            "Columnas con NaN: %s | Columnas con Inf: %s",
            list(nan_counts.keys()),
            list(inf_counts.keys()),
        )

    return report


def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    """Convierte una Serie a numérica y reemplaza valores NaN o Inf por un valor por defecto."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.fillna(default)


def get_timestamp() -> str:
    """Retorna una cadena de timestamp ordenable (YYYYMMDD_HHMMSS)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
