"""
Descarga completa de datos reales de TheStatsAPI para Brasileirão Série A (comp_4795).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.api_client import TheStatsAPIClient, get_client
from src.data.store import DataStore
from src.utils import load_yaml, setup_logging
from pathlib import Path

logger = logging.getLogger(__name__)

COMPETITION_ID = "comp_4795"


def resolve_seasons(client: TheStatsAPIClient) -> List[Dict[str, Any]]:
    """Obtiene la lista real de temporadas de la competición."""
    resp = client.get_seasons(COMPETITION_ID)
    seasons = resp.get("data", [])
    if not seasons:
        raise RuntimeError(f"No se encontraron temporadas para {COMPETITION_ID}")
    logger.info("Temporadas disponibles: %s", [s.get("name") for s in seasons])
    return seasons


def select_target_seasons(seasons: List[Dict], cfg: Dict) -> List[Dict]:
    """Selecciona current + previous según competition.yaml."""
    seasons_sorted = sorted(
        seasons,
        key=lambda s: (s.get("start_year", 0), s.get("end_year", 0)),
        reverse=True,
    )
    targets = []
    wanted = cfg.get("seasons_to_fetch", ["current", "previous"])

    current = next((s for s in seasons_sorted if s.get("is_current")), seasons_sorted[0])
    if "current" in wanted:
        targets.append(current)

    if "previous" in wanted and len(seasons_sorted) > 1:
        prev = next((s for s in seasons_sorted if s["id"] != current["id"]), None)
        if prev:
            targets.append(prev)

    # Evitar duplicados
    seen = set()
    unique = []
    for s in targets:
        if s["id"] not in seen:
            seen.add(s["id"])
            unique.append(s)
    return unique


def flatten_match(m: Dict) -> Dict[str, Any]:
    """Normaliza un objeto match de la API a fila plana."""
    home = m.get("home_team") or m.get("home") or {}
    away = m.get("away_team") or m.get("away") or {}
    score = m.get("score") or {}

    home_score = home.get("score")
    away_score = away.get("score")
    if home_score is None:
        home_score = score.get("home") or score.get("full_time", {}).get("home")
    if away_score is None:
        away_score = score.get("away") or score.get("full_time", {}).get("away")

    return {
        "match_id": m.get("id") or m.get("match_id"),
        "competition_id": m.get("competition_id") or COMPETITION_ID,
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


def flatten_stats(match_id: str, stats_resp: Dict) -> Dict[str, Any]:
    """Extrae estadísticas de equipo del endpoint /stats incluyendo esquinas y tiros a puerta."""
    data = stats_resp.get("data") or stats_resp
    home = data.get("home") or {}
    away = data.get("away") or {}

    if "overview" in data:
        overview = data["overview"]
        home = overview.get("home") or home
        away = overview.get("away") or away

    def safe_get(d: Dict, *keys, default=None):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default

    return {
        "match_id": match_id,
        "home_possession": safe_get(home, "possession", "ball_possession"),
        "away_possession": safe_get(away, "possession", "ball_possession"),
        "home_shots": safe_get(home, "shots", "total_shots"),
        "away_shots": safe_get(away, "shots", "total_shots"),
        "home_shots_on_target": safe_get(home, "shots_on_target", "shots_on_goal", "shots_target"),
        "away_shots_on_target": safe_get(away, "shots_on_target", "shots_on_goal", "shots_target"),
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


def flatten_odds(match_id: str, odds_resp: Dict) -> Dict[str, Any]:
    """Extrae cuotas 1X2 principales."""
    data = odds_resp.get("data") or odds_resp
    row: Dict[str, Any] = {"match_id": match_id}

    bookmakers = data.get("bookmakers") or data.get("odds") or []
    if isinstance(bookmakers, dict):
        bookmakers = [bookmakers]

    for bm in bookmakers:
        name = (bm.get("name") or bm.get("bookmaker") or "unknown").lower()
        markets = bm.get("markets") or bm.get("bets") or []
        for market in markets:
            mname = (market.get("name") or market.get("key") or "").lower()
            if "1x2" in mname or "match winner" in mname or mname in ("h2h", "match_odds"):
                outcomes = market.get("outcomes") or market.get("odds") or []
                for o in outcomes:
                    oname = (o.get("name") or o.get("label") or "").lower()
                    price = o.get("price") or o.get("odd") or o.get("value")
                    if "home" in oname or oname in ("1", "home"):
                        row[f"{name}_home"] = price
                    elif "draw" in oname or oname in ("x", "draw"):
                        row[f"{name}_draw"] = price
                    elif "away" in oname or oname in ("2", "away"):
                        row[f"{name}_away"] = price

    for key in ("home", "draw", "away", "1", "x", "2"):
        if key in data:
            row[f"odds_{key}"] = data[key]

    return row


def fetch_all_matches(client: TheStatsAPIClient, seasons: List[Dict]) -> pd.DataFrame:
    rows = []
    for season in seasons:
        sid = season["id"]
        logger.info("Descargando partidos de temporada %s (%s)", season.get("name"), sid)
        matches = client.get_all_matches(
            competition_id=COMPETITION_ID,
            season_id=sid,
            per_page=100,
        )
        for m in matches:
            flat = flatten_match(m)
            flat["season_id"] = sid
            flat["season_name"] = season.get("name")
            rows.append(flat)
        logger.info("  → %s partidos obtenidos", len(matches))
    df = pd.DataFrame(rows)
    if not df.empty and "kickoff_utc" in df.columns:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
        df = df.sort_values("kickoff_utc").reset_index(drop=True)
    return df


def fetch_match_stats(
    client: TheStatsAPIClient,
    match_ids: List[str],
    sleep: float = 0.15,
) -> pd.DataFrame:
    rows = []
    for mid in tqdm(match_ids, desc="Stats"):
        try:
            resp = client.get_match_stats(mid)
            rows.append(flatten_stats(mid, resp))
        except Exception as e:
            logger.warning("Stats falló para %s: %s", mid, e)
        time.sleep(sleep)
    return pd.DataFrame(rows)


def fetch_match_odds(
    client: TheStatsAPIClient,
    match_ids: List[str],
    sleep: float = 0.15,
) -> pd.DataFrame:
    rows = []
    for mid in tqdm(match_ids, desc="Odds"):
        try:
            resp = client.get_match_odds(mid)
            rows.append(flatten_odds(mid, resp))
        except Exception as e:
            logger.debug("Odds no disponibles para %s: %s", mid, e)
        time.sleep(sleep)
    return pd.DataFrame(rows)


def fetch_standings(client: TheStatsAPIClient, seasons: List[Dict]) -> pd.DataFrame:
    rows = []
    for season in seasons:
        sid = season["id"]
        try:
            resp = client.get_standings(COMPETITION_ID, sid)
            for r in resp.get("data", []):
                team = r.get("team") or {}
                rows.append(
                    {
                        "season_id": sid,
                        "season_name": season.get("name"),
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
            logger.warning("Standings falló para %s: %s", sid, e)
    return pd.DataFrame(rows)


def run_full_fetch(
    include_stats: bool = True,
    include_odds: bool = True,
    max_stats_matches: Optional[int] = None,
) -> None:
    """Punto de entrada principal de descarga."""
    setup_logging()
    cfg = load_yaml(Path(__file__).resolve().parents[2] / "configs" / "competition.yaml")
    client = get_client()
    store = DataStore()

    try:
        health = client.health()
        logger.info("API health: %s", health)
    except Exception as e:
        logger.error("API no responde: %s", e)
        raise

    seasons = resolve_seasons(client)
    target_seasons = select_target_seasons(seasons, cfg)
    logger.info("Temporadas objetivo: %s", [s.get("name") for s in target_seasons])

    # 1. Matches
    matches_df = fetch_all_matches(client, target_seasons)
    store.save_matches(matches_df)

    # 2. Standings
    standings_df = fetch_standings(client, target_seasons)
    store.save_standings(standings_df)

    finished = matches_df[matches_df["status"].str.lower().isin(["finished", "ft", "complete"])]
    match_ids = finished["match_id"].dropna().unique().tolist()
    if max_stats_matches:
        match_ids = match_ids[-max_stats_matches:]

    # 3. Stats (solo partidos finalizados)
    if include_stats and match_ids:
        stats_df = fetch_match_stats(client, match_ids)
        store.save_stats(stats_df)

    # 4. Odds
    if include_odds and match_ids:
        odds_df = fetch_match_odds(client, match_ids)
        store.save_odds(odds_df)

    logger.info("Fetch completo finalizado.")


if __name__ == "__main__":
    run_full_fetch(include_stats=True, include_odds=True)
