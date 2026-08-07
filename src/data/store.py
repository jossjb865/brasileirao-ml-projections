"""
Persistencia en Parquet y metadatos JSON.
Soporta versionado, lecturas/escrituras incrementales y almacenamiento de métricas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.utils import ensure_dirs, get_config, get_data_dir, get_outputs_dir, get_timestamp

logger = logging.getLogger(__name__)


class DataStore:
    """
    Capa de almacenamiento de datos para parquets, metadatos y respuestas de API.
    """

    def __init__(self, data_dir: Optional[Path | str] = None):
        cfg = get_config()
        self.data_dir = Path(data_dir) if data_dir else get_data_dir()
        ensure_dirs(self.data_dir)

        storage = cfg.get("storage", {})
        self.matches_path = self.data_dir / storage.get("matches_file", "matches.parquet")
        self.stats_path = self.data_dir / storage.get("stats_file", "match_stats.parquet")
        self.odds_path = self.data_dir / storage.get("odds_file", "odds.parquet")
        self.features_path = self.data_dir / storage.get("features_file", "features.parquet")
        self.standings_path = self.data_dir / storage.get("standings_file", "standings.parquet")

    def _save_parquet_with_meta(
        self,
        df: pd.DataFrame,
        file_path: Path,
        key_cols: Optional[List[str]] = None,
        incremental: bool = False,
        sort_col: Optional[str] = None,
    ) -> None:
        if df.empty:
            logger.warning("DataFrame vacío – no se guarda en %s", file_path)
            return

        data_to_save = df.copy()

        if incremental and file_path.exists():
            try:
                existing_df = pd.read_parquet(file_path)
                data_to_save = pd.concat([existing_df, data_to_save], ignore_index=True)
                if key_cols:
                    valid_keys = [c for c in key_cols if c in data_to_save.columns]
                    if valid_keys:
                        data_to_save = data_to_save.drop_duplicates(
                            subset=valid_keys, keep="last"
                        )
                else:
                    data_to_save = data_to_save.drop_duplicates()
            except Exception as e:
                logger.warning(
                    "Error al realizar lectura incremental desde %s: %s", file_path, e
                )

        if sort_col and sort_col in data_to_save.columns:
            data_to_save = data_to_save.sort_values(sort_col).reset_index(drop=True)

        data_to_save.to_parquet(file_path, index=False)

        # Archivo sidecar con metadatos y versionado
        meta_path = file_path.with_name(f"{file_path.name}.meta.json")
        metadata = {
            "version": "1.0",
            "saved_at": get_timestamp(),
            "file_name": file_path.name,
            "row_count": len(data_to_save),
            "column_count": len(data_to_save.columns),
            "columns": list(data_to_save.columns),
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("No se pudieron guardar metadatos en %s: %s", meta_path, e)

        logger.info("Parquet guardado (%s filas) → %s", len(data_to_save), file_path)

    # --- MATCHES ---
    def save_matches(self, df: pd.DataFrame, incremental: bool = False) -> None:
        self._save_parquet_with_meta(
            df,
            self.matches_path,
            key_cols=["match_id"],
            incremental=incremental,
            sort_col="kickoff_utc",
        )

    def load_matches(self) -> pd.DataFrame:
        if not self.matches_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.matches_path)

    # --- STATS ---
    def save_stats(self, df: pd.DataFrame, incremental: bool = False) -> None:
        self._save_parquet_with_meta(
            df,
            self.stats_path,
            key_cols=["match_id"],
            incremental=incremental,
        )

    def load_stats(self) -> pd.DataFrame:
        if not self.stats_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.stats_path)

    # --- ODDS ---
    def save_odds(self, df: pd.DataFrame, incremental: bool = False) -> None:
        self._save_parquet_with_meta(
            df,
            self.odds_path,
            key_cols=["match_id"],
            incremental=incremental,
        )

    def load_odds(self) -> pd.DataFrame:
        if not self.odds_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.odds_path)

    # --- FEATURES ---
    def save_features(self, df: pd.DataFrame, incremental: bool = False) -> None:
        self._save_parquet_with_meta(
            df,
            self.features_path,
            key_cols=["match_id"],
            incremental=incremental,
            sort_col="kickoff_utc",
        )

    def load_features(self) -> pd.DataFrame:
        if not self.features_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.features_path)

    # --- STANDINGS ---
    def save_standings(self, df: pd.DataFrame, incremental: bool = False) -> None:
        self._save_parquet_with_meta(
            df,
            self.standings_path,
            key_cols=["season_id", "team_id"],
            incremental=incremental,
        )

    def load_standings(self) -> pd.DataFrame:
        if not self.standings_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.standings_path)

    # --- PREDICTIONS ---
    def save_predictions(
        self,
        df: pd.DataFrame,
        name: Optional[str] = None,
        incremental: bool = False,
    ) -> Path:
        target_dir = get_outputs_dir()
        ensure_dirs(target_dir)

        if not name:
            file_name = "predictions.parquet"
        elif name.endswith(".parquet"):
            file_name = name
        else:
            file_name = f"predictions_{name}.parquet"

        target_path = target_dir / file_name
        self._save_parquet_with_meta(
            df,
            target_path,
            key_cols=["match_id"],
            incremental=incremental,
        )
        return target_path

    def load_predictions(self, name: Optional[str] = None) -> pd.DataFrame:
        target_dir = get_outputs_dir()
        if not name:
            file_path = target_dir / "predictions.parquet"
        elif name.endswith(".parquet"):
            file_path = target_dir / name
        else:
            file_path = target_dir / f"predictions_{name}.parquet"

        if not file_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(file_path)

    # --- METADATA & HELPER METHODS ---
    def get_last_fetch_date(self) -> Optional[str]:
        """Retorna la fecha más reciente de partido descargado en formato ISO."""
        matches = self.load_matches()
        if matches.empty:
            return None

        date_col = None
        for col in ["kickoff_utc", "date", "utc_date"]:
            if col in matches.columns:
                date_col = col
                break

        if not date_col:
            return None

        valid_dates = pd.to_datetime(matches[date_col], errors="coerce").dropna()
        if valid_dates.empty:
            return None

        return valid_dates.max().isoformat()

    # --- RAW RESPONSES ---
    def save_raw_response(
        self, name: str, data: Union[Dict[str, Any], List[Any]]
    ) -> Path:
        """Guarda respuestas sin procesar de la API en el subdirectorio 'raw'."""
        raw_dir = self.data_dir / "raw"
        ensure_dirs(raw_dir)

        file_name = name if name.endswith(".json") else f"{name}.json"
        target_path = raw_dir / file_name

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Respuesta RAW guardada → %s", target_path)
        return target_path

    def load_raw_response(self, name: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """Carga respuestas RAW guardadas previamente."""
        raw_dir = self.data_dir / "raw"
        file_name = name if name.endswith(".json") else f"{name}.json"
        target_path = raw_dir / file_name

        if not target_path.exists():
            return None

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Error leyendo respuesta RAW de %s: %s", target_path, e)
            return None

    # --- MODEL METRICS ---
    def save_model_metrics(
        self, metrics: Dict[str, Any], model_name: str
    ) -> Path:
        """Guarda las métricas de rendimiento de un modelo ML."""
        metrics_dir = self.data_dir / "metrics"
        ensure_dirs(metrics_dir)

        file_name = (
            model_name
            if model_name.endswith("_metrics.json")
            else f"{model_name}_metrics.json"
        )
        target_path = metrics_dir / file_name

        payload = {
            "model_name": model_name,
            "saved_at": get_timestamp(),
            "metrics": metrics,
        }

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("Métricas de modelo guardadas → %s", target_path)
        return target_path

    def load_model_metrics(
        self, model_name: Optional[str] = None
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Carga las métricas de un modelo específico o de todos los modelos."""
        metrics_dir = self.data_dir / "metrics"
        if not metrics_dir.exists():
            return {}

        if model_name:
            file_name = (
                model_name
                if model_name.endswith("_metrics.json")
                else f"{model_name}_metrics.json"
            )
            target_path = metrics_dir / file_name
            if not target_path.exists():
                return {}
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Error leyendo métricas de %s: %s", target_path, e)
                return {}

        all_metrics: Dict[str, Dict[str, Any]] = {}
        for p in metrics_dir.glob("*_metrics.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    m_name = data.get("model_name", p.stem.replace("_metrics", ""))
                    all_metrics[m_name] = data
            except Exception as e:
                logger.warning("Error leyendo %s: %s", p, e)

        return all_metrics
