"""
Generación de predicciones para próximos partidos (datos reales).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.api_client import get_client
from src.data.features import build_features
from src.data.store import DataStore
from src.ensemble import Ensemble
from src.utils import setup_logging, ensure_dirs, get_outputs_dir, load_yaml

logger = logging.getLogger(__name__)
COMPETITION_ID = "comp_4795"


def get_upcoming_matches(client, days_ahead: int = 14) -> pd.DataFrame:
    """Descarga partidos futuros reales de la API."""
    from datetime import timedelta

    today = datetime.now(timezone.utc).date()
    date_to = today + timedelta(days=days_ahead)

    all_rows = []
    page = 1
    while True:
        resp = client.get_matches(
            competition_id=COMPETITION_ID,
            status="scheduled",
            date_from=str(today),
            date_to=str(date_to),
            page=page,
            per_page=100,
        )
        items = resp.get("data", [])
        if not items:
            resp = client.get_matches(
                competition_id=COMPETITION_ID,
                date_from=str(today),
                date_to=str(date_to),
                page=page,
                per_page=100,
            )
            items = resp.get("data", [])

        if not items:
            break

        for m in items:
            home = m.get("home_team") or m.get("home") or {}
            away = m.get("away_team") or m.get("away") or {}
            status = (m.get("status") or "").lower()
            if status in ("finished", "ft", "complete", "cancelled", "postponed"):
                continue
            all_rows.append(
                {
                    "match_id": m.get("id") or m.get("match_id"),
                    "competition_id": COMPETITION_ID,
                    "status": m.get("status"),
                    "kickoff_utc": m.get("utc_date") or m.get("kickoff_utc") or m.get("date"),
                    "home_team_id": home.get("id"),
                    "home_team_name": home.get("name"),
                    "away_team_id": away.get("id"),
                    "away_team_name": away.get("name"),
                    "home_score": None,
                    "away_score": None,
                    "result": None,
                }
            )

        meta = resp.get("meta", {})
        if page >= meta.get("total_pages", 1):
            break
        page += 1

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
        df = df.sort_values("kickoff_utc").reset_index(drop=True)
    return df


def attach_rolling_features(upcoming: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features de forma usando el historial real ya descargado
    y las une a los partidos futuros (incluyendo volumen de corners y remates).
    """
    from src.data.features import _rolling_team_stats

    if historical.empty:
        raise RuntimeError("No hay historial. Ejecuta fetch + features primero.")

    long = _rolling_team_stats(historical, window=5)

    # Último estado conocido por equipo incluyendo variables de volumen
    last_state = (
        long.sort_values("kickoff_utc")
        .groupby("team_id")
        .tail(1)[
            [
                "team_id",
                "roll_goals_for_5",
                "roll_goals_against_5",
                "roll_points_5",
                "roll_goal_diff_5",
                "roll_matches_5",
                "roll_corners_for_5",
                "roll_corners_against_5",
                "roll_shots_target_for_5",
                "roll_shots_target_against_5",
            ]
        ]
    )

    home_state = last_state.rename(
        columns={
            "team_id": "home_team_id",
            "roll_goals_for_5": "home_gf_roll5",
            "roll_goals_against_5": "home_ga_roll5",
            "roll_points_5": "home_pts_roll5",
            "roll_goal_diff_5": "home_gd_roll5",
            "roll_matches_5": "home_matches_roll5",
            "roll_corners_for_5": "home_corners_for_roll",
            "roll_corners_against_5": "home_corners_against_roll",
            "roll_shots_target_for_5": "home_shots_target_for_roll",
            "roll_shots_target_against_5": "home_shots_target_against_roll",
        }
    )
    away_state = last_state.rename(
        columns={
            "team_id": "away_team_id",
            "roll_goals_for_5": "away_gf_roll5",
            "roll_goals_against_5": "away_ga_roll5",
            "roll_points_5": "away_pts_roll5",
            "roll_goal_diff_5": "away_gd_roll5",
            "roll_matches_5": "away_matches_roll5",
            "roll_corners_for_5": "away_corners_for_roll",
            "roll_corners_against_5": "away_corners_against_roll",
            "roll_shots_target_for_5": "away_shots_target_for_roll",
            "roll_shots_target_against_5": "away_shots_target_against_roll",
        }
    )

    up = upcoming.merge(home_state, on="home_team_id", how="left")
    up = up.merge(away_state, on="away_team_id", how="left")

    # Diferencias clásicas y de volumen para el modelo
    up["gf_diff_roll5"] = up["home_gf_roll5"] - up["away_gf_roll5"]
    up["pts_diff_roll5"] = up["home_pts_roll5"] - up["away_pts_roll5"]
    up["gd_diff_roll5"] = up["home_gd_roll5"] - up["away_gd_roll5"]
    up["corners_diff_roll5"] = up["home_corners_for_roll"] - up["away_corners_for_roll"]
    up["shots_target_diff_roll5"] = up["home_shots_target_for_roll"] - up["away_shots_target_for_roll"]

    # Rellenar NaN (equipos nuevos o sin historial)
    roll_cols = [c for c in up.columns if "roll" in c or "diff" in c]
    up[roll_cols] = up[roll_cols].fillna(0.0)

    return up


def run_predict(days_ahead: int = 14) -> pd.DataFrame:
    setup_logging()
    ensure_dirs(get_outputs_dir())

    client = get_client()
    store = DataStore()

    # Historial real
    historical = store.load_features()
    if historical.empty:
        historical = store.load_matches()
        if historical.empty:
            raise RuntimeError("Sin datos históricos. Ejecuta fetch primero.")

    # Próximos partidos reales
    upcoming = get_upcoming_matches(client, days_ahead=days_ahead)
    if upcoming.empty:
        logger.warning("No se encontraron partidos futuros en los próximos %s días", days_ahead)
        return pd.DataFrame()

    logger.info("Partidos futuros encontrados: %s", len(upcoming))

    # Features de forma
    features_up = attach_rolling_features(upcoming, historical)

    # Cargar ensemble
    ensemble = Ensemble.load()
    preds = ensemble.predict(features_up)

    # Guardar
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = get_outputs_dir() / f"predictions_{ts}.parquet"
    preds.to_parquet(out_path, index=False)
    csv_path = get_outputs_dir() / f"predictions_{ts}.csv"
    preds.to_csv(csv_path, index=False)

    logger.info("Predicciones guardadas → %s (%s filas)", out_path, len(preds))
    print(preds.to_string(index=False))
    return preds


if __name__ == "__main__":
    run_predict(days_ahead=14)
