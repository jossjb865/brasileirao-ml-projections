"""
Feature engineering a partir de datos reales descargados.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data.store import DataStore
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def _rolling_team_stats(
    matches: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Calcula features de forma rolling por equipo (home + away).
    Usa solo partidos finalizados ordenados por fecha.
    """
    df = matches.copy()
    df = df[df["status"].str.lower().isin(["finished", "ft", "complete"])].copy()

    # Solo partidos con marcador válido
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])

    df = df.sort_values("kickoff_utc").reset_index(drop=True)

    # Expandir a filas por equipo
    home = df[
        [
            "match_id",
            "kickoff_utc",
            "home_team_id",
            "home_team_name",
            "home_score",
            "away_score",
            "season_id",
        ]
    ].rename(
        columns={
            "home_team_id": "team_id",
            "home_team_name": "team_name",
            "home_score": "goals_for",
            "away_score": "goals_against",
        }
    )
    home["is_home"] = 1

    away = df[
        [
            "match_id",
            "kickoff_utc",
            "away_team_id",
            "away_team_name",
            "away_score",
            "home_score",
            "season_id",
        ]
    ].rename(
        columns={
            "away_team_id": "team_id",
            "away_team_name": "team_name",
            "away_score": "goals_for",
            "home_score": "goals_against",
        }
    )
    away["is_home"] = 0

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team_id", "kickoff_utc"])

    long["points"] = np.where(
        long["goals_for"] > long["goals_against"],
        3,
        np.where(long["goals_for"] == long["goals_against"], 1, 0),
    )
    long["goal_diff"] = long["goals_for"] - long["goals_against"]

    # Rolling (shift para no usar el partido actual)
    g = long.groupby("team_id", group_keys=False)
    for col in ["goals_for", "goals_against", "points", "goal_diff"]:
        long[f"roll_{col}_{window}"] = g[col].apply(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    long[f"roll_matches_{window}"] = g["match_id"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).count()
    )

    return long


def build_features(
    window: int = 5,
    include_stats: bool = True,
    include_odds: bool = True,
) -> pd.DataFrame:
    setup_logging()
    store = DataStore()

    matches = store.load_matches()
    if matches.empty:
        raise RuntimeError("No hay matches. Ejecuta primero src.data.fetch")

    stats = store.load_stats() if include_stats else pd.DataFrame()
    odds = store.load_odds() if include_odds else pd.DataFrame()

    # Rolling form
    long = _rolling_team_stats(matches, window=window)

    home_cols = {
        "match_id": "match_id",
        "roll_goals_for_5": "home_gf_roll5",
        "roll_goals_against_5": "home_ga_roll5",
        "roll_points_5": "home_pts_roll5",
        "roll_goal_diff_5": "home_gd_roll5",
        "roll_matches_5": "home_matches_roll5",
    }
    away_cols = {
        "match_id": "match_id",
        "roll_goals_for_5": "away_gf_roll5",
        "roll_goals_against_5": "away_ga_roll5",
        "roll_points_5": "away_pts_roll5",
        "roll_goal_diff_5": "away_gd_roll5",
        "roll_matches_5": "away_matches_roll5",
    }

    # Adaptar nombres de columnas rolling al window real
    home_cols = {k.replace("_5", f"_{window}"): v for k, v in home_cols.items()}
    away_cols = {k.replace("_5", f"_{window}"): v for k, v in away_cols.items()}
    # Corregir las claves que ya tenían el número
    home_cols = {
        "match_id": "match_id",
        f"roll_goals_for_{window}": "home_gf_roll5",
        f"roll_goals_against_{window}": "home_ga_roll5",
        f"roll_points_{window}": "home_pts_roll5",
        f"roll_goal_diff_{window}": "home_gd_roll5",
        f"roll_matches_{window}": "home_matches_roll5",
    }
    away_cols = {
        "match_id": "match_id",
        f"roll_goals_for_{window}": "away_gf_roll5",
        f"roll_goals_against_{window}": "away_ga_roll5",
        f"roll_points_{window}": "away_pts_roll5",
        f"roll_goal_diff_{window}": "away_gd_roll5",
        f"roll_matches_{window}": "away_matches_roll5",
    }

    home_f = long[long["is_home"] == 1][list(home_cols.keys())].rename(columns=home_cols)
    away_f = long[long["is_home"] == 0][list(away_cols.keys())].rename(columns=away_cols)

    features = matches.merge(home_f, on="match_id", how="left")
    features = features.merge(away_f, on="match_id", how="left")

    # Diferencias
    features["gf_diff_roll5"] = features["home_gf_roll5"] - features["away_gf_roll5"]
    features["pts_diff_roll5"] = features["home_pts_roll5"] - features["away_pts_roll5"]
    features["gd_diff_roll5"] = features["home_gd_roll5"] - features["away_gd_roll5"]

    # Targets solo donde hay scores válidos
    valid_score = features["home_score"].notna() & features["away_score"].notna()
    features["result"] = np.nan
    features["total_goals"] = np.nan
    features["btts"] = np.nan

    features.loc[valid_score, "result"] = np.where(
        features.loc[valid_score, "home_score"] > features.loc[valid_score, "away_score"],
        0,
        np.where(
            features.loc[valid_score, "home_score"] == features.loc[valid_score, "away_score"],
            1,
            2,
        ),
    )
    features.loc[valid_score, "total_goals"] = (
        features.loc[valid_score, "home_score"] + features.loc[valid_score, "away_score"]
    )
    features.loc[valid_score, "btts"] = (
        (features.loc[valid_score, "home_score"] > 0)
        & (features.loc[valid_score, "away_score"] > 0)
    ).astype(int)

    # Stats si existen
    if not stats.empty:
        features = features.merge(stats, on="match_id", how="left")

    # Odds si existen
    if not odds.empty:
        features = features.merge(odds, on="match_id", how="left")

    # Filtrar partidos con suficiente historial
    features = features[
        (features["home_matches_roll5"].fillna(0) >= 3)
        & (features["away_matches_roll5"].fillna(0) >= 3)
    ].copy()

    store.save_features(features)
    logger.info("Features generados: %s filas", len(features))
    return features


if __name__ == "__main__":
    build_features(window=5)
