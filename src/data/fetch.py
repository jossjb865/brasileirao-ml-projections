"""
Descarga completa de datos reales de TheStatsAPI para Brasileirão Série A.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.api_client import TheStatsAPIClient, get_client
from src.data.store import DataStore
from src.utils import get_config, setup_logging

logger = logging.getLogger(__name__)


def resolve_seasons(
    client: TheStatsAPIClient, competition_id: str
) -> List[Dict[str, Any]]:
    """Obtiene la lista real de temporadas de la competición desde la API."""
    resp = client.get_seasons(competition_id)
    seasons = resp.get("data", []) if isinstance(resp, dict) else []
    if not seasons and isinstance(resp, list):
        seasons = resp

    if not seasons:
        raise RuntimeError(f"No se encontraron temporadas para la competición {competition_id}")

    logger.info(
        "Temporadas disponibles: %s",
        [s.get("name") for s in seasons if isinstance(s, dict)],
    )
    return seasons


def select_target_seasons(
    seasons: List[Dict[str, Any]], wanted: List[str]
) -> List[Dict[str, Any]]:
    """Filtra y selecciona temporadas especificadas (current, previous, años o IDs)."""
    seasons_sorted = sorted(
        seasons,
        key=lambda s: (s.get("start_year") or 0, s.get("end_year") or 0),
        reverse=True,
    )
    if not seasons_sorted:
        return []

    targets: List[Dict[str, Any]] = []
    current_season = next(
        (s for s in seasons_sorted if s.get("is_current")), seasons_sorted[0]
    )

    current_idx = seasons_sorted.index(current_season) if current_season in seasons_sorted else 0

    for item in wanted:
        item_str = str(item).strip().lower()
        if item_str == "current":
            targets.append(current_season)
        elif item_str == "previous":
            prev = next(
                (s for s in seasons_sorted if s.get("id") != current_season.get("id")),
                None,
            )
            if prev:
                targets.append(prev)
        elif item_str.startswith("previous-"):
            # Handle "previous-2", "previous-3", etc.
            try:
                offset = int(item_str.split("-", 1)[1])
                if offset < len(seasons_sorted):
                    targets.append(seasons_sorted[offset])
            except (ValueError, IndexError):
                logger.warning("No se pudo resolver la temporada %s", item_str)
        else:
            matched = next(
                (
                    s
                    for s in seasons_sorted
                    if str(s.get("id")).lower() == item_str
                    or str(s.get("name")).lower() == item_str
                    or str(s.get("start_year")) == item_str
                    or str(s.get("end_year")) == item_str
                ),
                None,
            )
            if matched:
                targets.append(matched)

    seen = set()
    unique: List[Dict[str, Any]] = []
    for s in targets:
        sid = s.get("id")
        if sid and sid not in seen:
            seen.add(sid)
            unique.append(s)

    if not unique:
        unique = [current_season]

    return unique


def flatten_match(
    m: Dict[str, Any], competition_id: Optional[str] = None
) -> Dict[str, Any]:
    """Normaliza un objeto match de la API a una estructura plana y robusta."""
    if not isinstance(m, dict):
        return {}

    home = m.get("home_team") or m.get("home") or m.get("teams", {}).get("home") or {}
    away = m.get("away_team") or m.get("away") or m.get("teams", {}).get("away") or {}
    score = m.get("score") or {}

    if not isinstance(home, dict):
        home = {"name": str(home)}
    if not isinstance(away, dict):
        away = {"name": str(away)}
    if not isinstance(score, dict):
        score = {}

    home_score = home.get("score")
    away_score = away.get("score")
    if home_score is None:
        home_score = score.get("home") if "home" in score else score.get("full_time", {}).get("home")
    if away_score is None:
        away_score = score.get("away") if "away" in score else score.get("full_time", {}).get("away")

    return {
        "match_id": m.get("id") or m.get("match_id"),
        "competition_id": m.get("competition_id") or competition_id,
        "season_id": m.get("season_id"),
        "status": m.get("status"),
        "kickoff_utc": m.get("utc_date") or m.get("kickoff_utc") or m.get("date"),
        "matchday": m.get("matchday") or m.get("round"),
        "home_team_id": home.get("id"),
        "home_team_name": home.get("name"),
        "away_team_id": away.get("id"),
        "away_team_name": away.get("name"),
        "home_score": home_score,
        "away_score": away_score,
        "home_ht_score": home.get("ht_score") or score.get("half_time", {}).get("home"),
        "away_ht_score": away.get("ht_score") or score.get("half_time", {}).get("away"),
        "xg_available": m.get("xg_available"),
        "odds_available": m.get("odds_available"),
    }


def flatten_stats(match_id: str, stats_resp: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae estadísticas de equipo del endpoint /stats con tolerancia a esquemas variables."""
    data = (
        stats_resp.get("data")
        if isinstance(stats_resp, dict) and "data" in stats_resp
        else stats_resp
    )
    if not isinstance(data, dict):
        data = {}

    home = data.get("home") or {}
    away = data.get("away") or {}

    if "overview" in data and isinstance(data["overview"], dict):
        overview = data["overview"]
        home = overview.get("home") or home
        away = overview.get("away") or away

    if not isinstance(home, dict):
        home = {}
    if not isinstance(away, dict):
        away = {}

    def safe_get(d: Dict[str, Any], *keys: str):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    return {
        "match_id": match_id,
        "home_possession": safe_get(home, "possession", "ball_possession"),
        "away_possession": safe_get(away, "possession", "ball_possession"),
        "home_shots": safe_get(home, "shots", "total_shots"),
        "away_shots": safe_get(away, "shots", "total_shots"),
        "home_shots_on_target": safe_get(
            home, "shots_on_target", "shots_on_goal", "shots_target"
        ),
        "away_shots_on_target": safe_get(
            away, "shots_on_target", "shots_on_goal", "shots_target"
        ),
        "home_corners": safe_get(home, "corners", "corner_kicks"),
        "away_corners": safe_get(away, "corners", "corner_kicks"),
        "home_fouls": safe_get(home, "fouls"),
        "away_fouls": safe_get(away, "fouls"),
        "home_yellow_cards": safe_get(home, "yellow_cards", "yellow"),
        "away_yellow_cards": safe_get(away, "yellow_cards", "yellow"),
        "home_red_cards": safe_get(home, "red_cards", "red"),
        "away_red_cards": safe_get(away, "red_cards", "red"),
        "home_xg": safe_get(home, "xg", "expected_goals"),
        "away_xg": safe_get(away, "xg", "expected_goals"),
        "home_passes": safe_get(home, "passes", "total_passes"),
        "away_passes": safe_get(away, "passes", "total_passes"),
    }


def flatten_odds(match_id: str, odds_resp: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae cuotas de apuestas tolerando múltiples formatos."""
    data = (
        odds_resp.get("data")
        if isinstance(odds_resp, dict) and "data" in odds_resp
        else odds_resp
    )
    if not isinstance(data, dict):
        return {"match_id": match_id}

    row: Dict[str, Any] = {"match_id": match_id}

    bookmakers = data.get("bookmakers") or data.get("odds") or []
    if isinstance(bookmakers, dict):
        bookmakers = [bookmakers]
    elif not isinstance(bookmakers, list):
        bookmakers = []

    for bm in bookmakers:
        if not isinstance(bm, dict):
            continue
        name = str(bm.get("name") or bm.get("bookmaker") or "unknown").lower()
        markets = bm.get("markets") or bm.get("bets") or []
        if isinstance(markets, dict):
            markets = [markets]
        elif not isinstance(markets, list):
            markets = []

        for market in markets:
            if not isinstance(market, dict):
                continue
            mname = str(market.get("name") or market.get("key") or "").lower()
            if (
                "1x2" in mname
                or "match winner" in mname
                or mname in ("h2h", "match_odds")
            ):
                outcomes = market.get("outcomes") or market.get("odds") or []
                if isinstance(outcomes, dict):
                    outcomes = [outcomes]
                elif not isinstance(outcomes, list):
                    outcomes = []

                for o in outcomes:
                    if not isinstance(o, dict):
                        continue
                    oname = str(o.get("name") or o.get("label") or "").lower()
                    price = o.get("price") or o.get("odd") or o.get("value")
                    if "home" in oname or oname in ("1", "home"):
                        row[f"{name}_home"] = price
                    elif "draw" in oname or oname in ("x", "draw"):
                        row[f"{name}_draw"] = price
                    elif "away" in oname or oname in ("2", "away"):
                        row[f"{name}_away"] = price

    for key in ("home", "draw", "away", "1", "x", "2"):
        if key in data and data[key] is not None:
            row[f"odds_{key}"] = data[key]

    return row


def fetch_team_form(matches_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Calcula la forma/racha reciente por equipo basada en el historial de partidos."""
    if matches_df.empty:
        return pd.DataFrame()

    df = matches_df.copy()
    if "status" in df.columns:
        df = df[df["status"].str.lower().isin(["finished", "ft", "complete"])].copy()

    df = df.dropna(subset=["home_score", "away_score"])
    if df.empty:
        return pd.DataFrame()

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])

    if "kickoff_utc" in df.columns:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], errors="coerce")
        df = df.sort_values("kickoff_utc")

    home = df[
        ["match_id", "kickoff_utc", "home_team_id", "home_team_name", "home_score", "away_score"]
    ].rename(
        columns={
            "home_team_id": "team_id",
            "home_team_name": "team_name",
            "home_score": "goals_for",
            "away_score": "goals_against",
        }
    )
    away = df[
        ["match_id", "kickoff_utc", "away_team_id", "away_team_name", "away_score", "home_score"]
    ].rename(
        columns={
            "away_team_id": "team_id",
            "away_team_name": "team_name",
            "away_score": "goals_for",
            "home_score": "goals_against",
        }
    )

    long = pd.concat([home, away], ignore_index=True)
    long = long.dropna(subset=["team_id"]).sort_values(["team_id", "kickoff_utc"])

    long["points"] = np.where(
        long["goals_for"] > long["goals_against"],
        3,
        np.where(long["goals_for"] == long["goals_against"], 1, 0),
    )
    long["result_char"] = np.where(
        long["goals_for"] > long["goals_against"],
        "W",
        np.where(long["goals_for"] == long["goals_against"], "D", "L"),
    )

    form_rows = []
    for team_id, group in long.groupby("team_id"):
        recent = group.tail(window)
        names = group["team_name"].dropna()
        team_name = names.iloc[-1] if not names.empty else str(team_id)
        form_str = "-".join(recent["result_char"].tolist())

        form_rows.append(
            {
                "team_id": team_id,
                "team_name": team_name,
                "matches_played": len(group),
                "recent_matches": len(recent),
                "roll_points": float(recent["points"].sum()),
                "roll_goals_for": float(recent["goals_for"].sum()),
                "roll_goals_against": float(recent["goals_against"].sum()),
                "roll_goal_diff": float((recent["goals_for"] - recent["goals_against"]).sum()),
                "form_string": form_str,
            }
        )

    return pd.DataFrame(form_rows)


def fetch_all_matches(
    client: TheStatsAPIClient, seasons: List[Dict[str, Any]], competition_id: str
) -> pd.DataFrame:
    rows = []
    for season in seasons:
        sid = season["id"]
        sname = season.get("name")
        logger.info("Descargando partidos de la temporada %s (%s)...", sname, sid)
        matches = client.get_all_matches(
            competition_id=competition_id,
            season_id=sid,
            per_page=100,
        )
        for m in matches:
            flat = flatten_match(m, competition_id=competition_id)
            flat["season_id"] = sid
            flat["season_name"] = sname
            rows.append(flat)
        logger.info("  → %s partidos obtenidos para temporada %s", len(matches), sid)

    df = pd.DataFrame(rows)
    if not df.empty and "kickoff_utc" in df.columns:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
        df = df.sort_values("kickoff_utc").reset_index(drop=True)
    return df


def fetch_match_stats(
    client: TheStatsAPIClient,
    match_ids: List[str],
    max_matches: Optional[int] = None,
    sleep: float = 0.0,
) -> pd.DataFrame:
    target_ids = match_ids[:max_matches] if max_matches else match_ids
    rows = []
    logger.info("Descargando estadísticas para %s partidos...", len(target_ids))
    for mid in tqdm(target_ids, desc="Stats"):
        try:
            resp = client.get_match_stats(mid)
            rows.append(flatten_stats(mid, resp))
        except Exception as e:
            logger.warning("Error descargando stats de partido %s: %s", mid, e)
        time.sleep(sleep)
    return pd.DataFrame(rows)


def fetch_match_odds(
    client: TheStatsAPIClient,
    match_ids: List[str],
    sleep: float = 0.0,
) -> pd.DataFrame:
    rows = []
    logger.info("Descargando cuotas para %s partidos...", len(match_ids))
    for mid in tqdm(match_ids, desc="Odds"):
        try:
            resp = client.get_match_odds(mid)
            rows.append(flatten_odds(mid, resp))
        except Exception as e:
            logger.debug("Odds no disponibles para partido %s: %s", mid, e)
        time.sleep(sleep)
    return pd.DataFrame(rows)


def fetch_standings(
    client: TheStatsAPIClient, seasons: List[Dict[str, Any]], competition_id: str
) -> pd.DataFrame:
    rows = []
    for season in seasons:
        sid = season["id"]
        sname = season.get("name")
        try:
            resp = client.get_standings(competition_id, sid)
            data = resp.get("data", []) if isinstance(resp, dict) else []
            if not data and isinstance(resp, list):
                data = resp

            for r in data:
                if not isinstance(r, dict):
                    continue
                team = r.get("team") or {}
                if not isinstance(team, dict):
                    team = {"name": str(team)}

                rows.append(
                    {
                        "season_id": sid,
                        "season_name": sname,
                        "team_id": team.get("id"),
                        "team_name": team.get("name"),
                        "position": r.get("position"),
                        "matches_played": r.get("matches_played"),
                        "wins": r.get("wins"),
                        "draws": r.get("draws"),
                        "losses": r.get("losses"),
                        "goals_for": r.get("goals_for"),
                        "goals_against": r.get("goals_against"),
                        "goal_difference": r.get("goal_difference"),
                        "points": r.get("points"),
                    }
                )
        except Exception as e:
            logger.warning("Error descargando standings para temporada %s: %s", sid, e)
    return pd.DataFrame(rows)


def run_full_fetch(
    seasons: Optional[List[str]] = None,
    include_stats: bool = True,
    include_odds: bool = True,
    max_stats_matches: Optional[int] = None,
    incremental: bool = False,
    config_path: Optional[str | Path] = None,
) -> None:
    """Punto de entrada principal para el pipeline de descarga de datos."""
    setup_logging()
    cfg = get_config(config_path)

    comp_cfg = cfg.get("competition", {})
    competition_id = comp_cfg.get("id", "comp_4795")

    wanted_seasons = seasons or cfg.get("seasons_to_fetch", ["current", "previous"])

    try:
        client = get_client()
        store = DataStore()

        health = client.health()
        logger.info("Estado de salud de la API: %s", health)

        all_seasons = resolve_seasons(client, competition_id)
        target_seasons = select_target_seasons(all_seasons, wanted_seasons)
        logger.info(
            "Temporadas seleccionadas: %s",
            [s.get("name") for s in target_seasons],
        )

        # 1. Matches
        matches_df = fetch_all_matches(client, target_seasons, competition_id)
        if not matches_df.empty:
            store.save_matches(matches_df, incremental=incremental)
            logger.info("Partidos procesados correctamente: %s filas", len(matches_df))

            # Calcular y reportar forma reciente
            form_df = fetch_team_form(matches_df)
            if not form_df.empty:
                logger.info("Resumen de forma reciente calculado para %s equipos", len(form_df))

            match_ids = matches_df["match_id"].dropna().unique().tolist()

            # 2. Stats
            if include_stats and match_ids:
                stats_df = fetch_match_stats(
                    client, match_ids, max_matches=max_stats_matches
                )
                if not stats_df.empty:
                    store.save_stats(stats_df, incremental=incremental)

            # 3. Odds
            if include_odds and match_ids:
                odds_df = fetch_match_odds(client, match_ids)
                if not odds_df.empty:
                    store.save_odds(odds_df, incremental=incremental)

        # 4. Standings
        standings_df = fetch_standings(client, target_seasons, competition_id)
        if not standings_df.empty:
            store.save_standings(standings_df, incremental=incremental)

        logger.info("Descarga completa finalizada exitosamente.")

    except RuntimeError as e:
        err_msg = str(e)
        if "401" in err_msg or "Unauthorized" in err_msg or "Circuit breaker" in err_msg:
            logger.error("Error de autenticación crítico en la API (401): %s", err_msg)
            print(f"CRITICAL ERROR: API 401 Unauthorized - {err_msg}", file=sys.stderr)
            sys.exit(1)
        else:
            logger.exception("Error en tiempo de ejecución durante la descarga: %s", e)
            sys.exit(1)
    except Exception as e:
        logger.exception("Error inesperado en la descarga de datos: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga datos reales desde TheStatsAPI para Brasileirão Série A"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Lista de temporadas a descargar (ej: current previous 2024)",
    )
    parser.add_argument(
        "--stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Descargar estadísticas de partidos",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Omitir descarga de estadísticas de partidos",
    )
    parser.add_argument(
        "--odds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Descargar cuotas de apuestas de partidos",
    )
    parser.add_argument(
        "--skip-odds",
        action="store_true",
        help="Omitir descarga de cuotas de apuestas",
    )
    parser.add_argument(
        "--max-stats",
        type=int,
        default=None,
        help="Límite máximo de partidos para los que descargar estadísticas",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Activar actualización incremental sin sobrescribir historial",
    )

    args = parser.parse_args()

    fetch_stats = args.stats and not args.skip_stats
    fetch_odds = args.odds and not args.skip_odds

    run_full_fetch(
        seasons=args.seasons,
        include_stats=fetch_stats,
        include_odds=fetch_odds,
        max_stats_matches=args.max_stats,
        incremental=args.incremental,
    )
