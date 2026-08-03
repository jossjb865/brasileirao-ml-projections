"""
Persistencia en Parquet. Todo se genera a partir de respuestas reales de la API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils import ensure_dirs, get_data_dir, load_yaml

logger = logging.getLogger(__name__)


class DataStore:
    def __init__(self, data_dir: Optional[Path] = None):
        cfg = load_yaml(Path(__file__).resolve().parents[2] / "configs" / "competition.yaml")
        self.data_dir = data_dir or get_data_dir()
        ensure_dirs(self.data_dir)

        storage = cfg.get("storage", {})
        self.matches_path = self.data_dir / storage.get("matches_file", "matches.parquet")
        self.stats_path = self.data_dir / storage.get("stats_file", "match_stats.parquet")
        self.odds_path = self.data_dir / storage.get("odds_file", "odds.parquet")
        self.features_path = self.data_dir / storage.get("features_file", "features.parquet")
        self.standings_path = self.data_dir / storage.get("standings_file", "standings.parquet")

    def save_matches(self, df: pd.DataFrame) -> None:
        if df.empty:
            logger.warning("DataFrame de matches vacío – no se guarda")
            return
        df.to_parquet(self.matches_path, index=False)
        logger.info("Matches guardados: %s filas → %s", len(df), self.matches_path)

    def load_matches(self) -> pd.DataFrame:
        if not self.matches_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.matches_path)

    def save_stats(self, df: pd.DataFrame) -> None:
        if df.empty:
            logger.warning("DataFrame de stats vacío – no se guarda")
            return
        df.to_parquet(self.stats_path, index=False)
        logger.info("Stats guardados: %s filas → %s", len(df), self.stats_path)

    def load_stats(self) -> pd.DataFrame:
        if not self.stats_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.stats_path)

    def save_odds(self, df: pd.DataFrame) -> None:
        if df.empty:
            logger.warning("DataFrame de odds vacío – no se guarda")
            return
        df.to_parquet(self.odds_path, index=False)
        logger.info("Odds guardados: %s filas → %s", len(df), self.odds_path)

    def load_odds(self) -> pd.DataFrame:
        if not self.odds_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.odds_path)

    def save_features(self, df: pd.DataFrame) -> None:
        if df.empty:
            logger.warning("DataFrame de features vacío – no se guarda")
            return
        df.to_parquet(self.features_path, index=False)
        logger.info("Features guardados: %s filas → %s", len(df), self.features_path)

    def load_features(self) -> pd.DataFrame:
        if not self.features_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.features_path)

    def save_standings(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df.to_parquet(self.standings_path, index=False)
        logger.info("Standings guardados: %s filas → %s", len(df), self.standings_path)

    def load_standings(self) -> pd.DataFrame:
        if not self.standings_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.standings_path)
