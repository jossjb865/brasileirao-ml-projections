"""
Feature engineering module for Brazilian football ML predictions with strict ANTI-LEAKAGE design.

Rules:
- Features use ONLY data available BEFORE kickoff: past goals, past points, past form, ELO, rest days, H2H.
- NEVER include match-day stats (possession, shots, xG, corners from that match) as features.
- Strict chronological processing ensures zero look-ahead bias.
- Filter training set to matches where both teams have >= 3 prior historical matches.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from src.data.store import DataStore
from src.utils import setup_logging

logger = logging.getLogger(__name__)

INITIAL_ELO = 1500.0
HOME_ADVANTAGE_ELO = 60.0
K_FACTOR = 20.0

FEATURE_COLUMNS = [
    # ELO ratings
    "home_elo",
    "away_elo",
    "elo_diff",
    # Rest days
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    # Overall rolling stats (5 games)
    "home_roll_gf_5",
    "home_roll_ga_5",
    "home_roll_pts_5",
    "away_roll_gf_5",
    "away_roll_ga_5",
    "away_roll_pts_5",
    "gf_diff_roll5",
    "ga_diff_roll5",
    "pts_diff_roll5",
    # Overall rolling stats (10 games)
    "home_roll_gf_10",
    "home_roll_ga_10",
    "home_roll_pts_10",
    "away_roll_gf_10",
    "away_roll_ga_10",
    "away_roll_pts_10",
    "gf_diff_roll10",
    "ga_diff_roll10",
    "pts_diff_roll10",
    # Venue-specific rolling stats (5 games)
    "home_specific_pts_5",
    "home_specific_gf_5",
    "home_specific_ga_5",
    "away_specific_pts_5",
    "away_specific_gf_5",
    "away_specific_ga_5",
    # Form streaks
    "home_win_streak",
    "home_unbeaten_streak",
    "away_win_streak",
    "away_unbeaten_streak",
    # Head-to-Head history
    "h2h_matches_count",
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_away_win_rate",
    "h2h_home_pts_avg",
    "h2h_home_gf_avg",
    "h2h_away_gf_avg",
]


def _calc_rolling_metrics(history: List[Dict[str, Any]], window: int) -> Tuple[float, float, float]:
    """Calculate average goals for, goals against, and total points in the last `window` matches."""
    if not history:
        return 0.0, 0.0, 0.0
    slice_matches = history[-window:]
    n = len(slice_matches)
    gf_sum = sum(m["gf"] for m in slice_matches)
    ga_sum = sum(m["ga"] for m in slice_matches)
    pts_sum = sum(m["pts"] for m in slice_matches)
    return gf_sum / n, ga_sum / n, float(pts_sum)


def _calc_streaks(history: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Calculate current win streak and unbeaten streak prior to this match."""
    win_streak = 0.0
    unbeaten_streak = 0.0
    for m in reversed(history):
        if m["pts"] == 3:
            win_streak += 1.0
            unbeaten_streak += 1.0
        elif m["pts"] == 1:
            win_streak = 0.0  # draw breaks win streak
            unbeaten_streak += 1.0
        else:
            break
    return win_streak, unbeaten_streak


def _calc_h2h_stats(
    h2h_history: List[Dict[str, Any]], home_team_id: Any, away_team_id: Any
) -> Dict[str, float]:
    """Calculate Head-to-Head statistics between home and away team strictly from past matches."""
    n_matches = len(h2h_history)
    if n_matches == 0:
        return {
            "h2h_matches_count": 0.0,
            "h2h_home_win_rate": 0.0,
            "h2h_draw_rate": 0.0,
            "h2h_away_win_rate": 0.0,
            "h2h_home_pts_avg": 0.0,
            "h2h_home_gf_avg": 0.0,
            "h2h_away_gf_avg": 0.0,
        }

    home_wins = 0
    draws = 0
    away_wins = 0
    home_pts = 0
    home_gf = 0
    away_gf = 0

    for m in h2h_history:
        if m["home_id"] == home_team_id:
            h_score, a_score = m["home_score"], m["away_score"]
        else:
            h_score, a_score = m["away_score"], m["home_score"]

        home_gf += h_score
        away_gf += a_score

        if h_score > a_score:
            home_wins += 1
            home_pts += 3
        elif h_score == a_score:
            draws += 1
            home_pts += 1
        else:
            away_wins += 1

    return {
        "h2h_matches_count": float(n_matches),
        "h2h_home_win_rate": home_wins / n_matches,
        "h2h_draw_rate": draws / n_matches,
        "h2h_away_win_rate": away_wins / n_matches,
        "h2h_home_pts_avg": home_pts / n_matches,
        "h2h_home_gf_avg": home_gf / n_matches,
        "h2h_away_gf_avg": away_gf / n_matches,
    }


def _process_chronological_matches(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Process matches in strict chronological order.
    Returns:
      - Processed feature DataFrame
      - Final state dictionary for initializing upcoming match feature calculations
    """
    matches = df.copy()
    matches["kickoff_utc"] = pd.to_datetime(matches["kickoff_utc"], utc=True, errors="coerce")
    matches = matches.sort_values("kickoff_utc").reset_index(drop=True)

    team_elo: Dict[Any, float] = defaultdict(lambda: INITIAL_ELO)
    team_last_date: Dict[Any, pd.Timestamp] = {}
    team_matches: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    h2h_store: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)

    feature_rows = []

    for idx, row in matches.iterrows():
        match_id = row.get("match_id")
        k_date = row["kickoff_utc"]
        h_id = row["home_team_id"]
        a_id = row["away_team_id"]

        # 1. Pre-match ELO
        h_elo = team_elo[h_id]
        a_elo = team_elo[a_id]
        elo_diff = h_elo - a_elo

        # 2. Pre-match Rest Days
        h_last = team_last_date.get(h_id)
        a_last = team_last_date.get(a_id)

        if h_last is not None and pd.notna(k_date) and pd.notna(h_last):
            h_rest = max(1.0, min(30.0, (k_date - h_last).total_seconds() / 86400.0))
        else:
            h_rest = 7.0

        if a_last is not None and pd.notna(k_date) and pd.notna(a_last):
            a_rest = max(1.0, min(30.0, (k_date - a_last).total_seconds() / 86400.0))
        else:
            a_rest = 7.0

        rest_diff = h_rest - a_rest

        # 3. Rolling metrics (Home team past matches)
        h_hist = team_matches[h_id]
        h_count = len(h_hist)
        h_gf5, h_ga5, h_pts5 = _calc_rolling_metrics(h_hist, 5)
        h_gf10, h_ga10, h_pts10 = _calc_rolling_metrics(h_hist, 10)
        h_win_strk, h_unb_strk = _calc_streaks(h_hist)

        h_home_hist = [m for m in h_hist if m["is_home"]]
        h_spec_gf5, h_spec_ga5, h_spec_pts5 = _calc_rolling_metrics(h_home_hist, 5)

        # 4. Rolling metrics (Away team past matches)
        a_hist = team_matches[a_id]
        a_count = len(a_hist)
        a_gf5, a_ga5, a_pts5 = _calc_rolling_metrics(a_hist, 5)
        a_gf10, a_ga10, a_pts10 = _calc_rolling_metrics(a_hist, 10)
        a_win_strk, a_unb_strk = _calc_streaks(a_hist)

        a_away_hist = [m for m in a_hist if not m["is_home"]]
        a_spec_gf5, a_spec_ga5, a_spec_pts5 = _calc_rolling_metrics(a_away_hist, 5)

        # 5. Differentials
        gf_diff_roll5 = h_gf5 - a_gf5
        ga_diff_roll5 = h_ga5 - a_ga5
        pts_diff_roll5 = h_pts5 - a_pts5

        gf_diff_roll10 = h_gf10 - a_gf10
        ga_diff_roll10 = h_ga10 - a_ga10
        pts_diff_roll10 = h_pts10 - a_pts10

        # 6. Head-to-Head
        h2h_key = tuple(sorted([h_id, a_id]))
        h2h_stats = _calc_h2h_stats(h2h_store[h2h_key], h_id, a_id)

        feat = {
            "match_id": match_id,
            "kickoff_utc": k_date,
            "status": row.get("status"),
            "home_team_id": h_id,
            "home_team_name": row.get("home_team_name"),
            "away_team_id": a_id,
            "away_team_name": row.get("away_team_name"),
            "home_match_count": h_count,
            "away_match_count": a_count,
            "home_score": row.get("home_score"),
            "away_score": row.get("away_score"),
            "home_elo": h_elo,
            "away_elo": a_elo,
            "elo_diff": elo_diff,
            "home_rest_days": h_rest,
            "away_rest_days": a_rest,
            "rest_diff": rest_diff,
            "home_roll_gf_5": h_gf5,
            "home_roll_ga_5": h_ga5,
            "home_roll_pts_5": h_pts5,
            "away_roll_gf_5": a_gf5,
            "away_roll_ga_5": a_ga5,
            "away_roll_pts_5": a_pts5,
            "gf_diff_roll5": gf_diff_roll5,
            "ga_diff_roll5": ga_diff_roll5,
            "pts_diff_roll5": pts_diff_roll5,
            "home_roll_gf_10": h_gf10,
            "home_roll_ga_10": h_ga10,
            "home_roll_pts_10": h_pts10,
            "away_roll_gf_10": a_gf10,
            "away_roll_ga_10": a_ga10,
            "away_roll_pts_10": a_pts10,
            "gf_diff_roll10": gf_diff_roll10,
            "ga_diff_roll10": ga_diff_roll10,
            "pts_diff_roll10": pts_diff_roll10,
            "home_specific_pts_5": h_spec_pts5,
            "home_specific_gf_5": h_spec_gf5,
            "home_specific_ga_5": h_spec_ga5,
            "away_specific_pts_5": a_spec_pts5,
            "away_specific_gf_5": a_spec_gf5,
            "away_specific_ga_5": a_spec_ga5,
            "home_win_streak": h_win_strk,
            "home_unbeaten_streak": h_unb_strk,
            "away_win_streak": a_win_strk,
            "away_unbeaten_streak": a_unb_strk,
            **h2h_stats,
        }

        # Calculate result target if scores are available
        h_score = row.get("home_score")
        a_score = row.get("away_score")
        if pd.notna(h_score) and pd.notna(a_score):
            h_s = float(h_score)
            a_s = float(a_score)
            if h_s > a_s:
                feat["result"] = 0
                h_pts, a_pts = 3, 0
                s_h = 1.0
            elif h_s == a_s:
                feat["result"] = 1
                h_pts, a_pts = 1, 1
                s_h = 0.5
            else:
                feat["result"] = 2
                h_pts, a_pts = 0, 3
                s_h = 0.0

            # Update ELO incrementally POST-MATCH
            e_h = 1.0 / (1.0 + 10.0 ** ((a_elo - (h_elo + HOME_ADVANTAGE_ELO)) / 400.0))
            g_diff = abs(h_s - a_s)
            k_adj = K_FACTOR * (1.0 + 0.5 * max(0.0, g_diff - 1.0))

            team_elo[h_id] += k_adj * (s_h - e_h)
            team_elo[a_id] += k_adj * ((1.0 - s_h) - (1.0 - e_h))

            # Update last dates
            if pd.notna(k_date):
                team_last_date[h_id] = k_date
                team_last_date[a_id] = k_date

            # Update histories
            team_matches[h_id].append({"gf": h_s, "ga": a_s, "pts": h_pts, "is_home": True})
            team_matches[a_id].append({"gf": a_s, "ga": h_s, "pts": a_pts, "is_home": False})
            h2h_store[h2h_key].append(
                {"home_id": h_id, "away_id": a_id, "home_score": h_s, "away_score": a_s}
            )

        feature_rows.append(feat)

    features_df = pd.DataFrame(feature_rows)
    final_state = {
        "team_elo": team_elo,
        "team_last_date": team_last_date,
        "team_matches": team_matches,
        "h2h_store": h2h_store,
    }
    return features_df, final_state


def build_features(
    matches_df: Optional[pd.DataFrame] = None,
    stats_df: Optional[pd.DataFrame] = None,
    odds_df: Optional[pd.DataFrame] = None,
    min_history_matches: int = 3,
) -> pd.DataFrame:
    """
    Build feature matrix from match history with strict anti-leakage.

    Returns DataFrame with `result` column (0=Home, 1=Draw, 2=Away) and `FEATURE_COLUMNS`.
    Filters out matches where either team has fewer than `min_history_matches` prior matches.
    """
    if matches_df is None or matches_df.empty:
        store = DataStore()
        matches_df = store.load_matches()

    if matches_df.empty:
        logger.warning("build_features called with empty matches_df")
        return pd.DataFrame()

    # Filter to finished matches with valid scores
    finished = matches_df[
        matches_df["status"].str.lower().isin(["finished", "ft", "complete"])
    ].copy()
    finished = finished.dropna(subset=["home_score", "away_score"])

    if finished.empty:
        logger.warning("No finished matches found to build features.")
        return pd.DataFrame()

    features_df, _ = _process_chronological_matches(finished)

    # Filter to matches with sufficient history (>= min_history_matches for both teams)
    valid_df = features_df[
        (features_df["home_match_count"] >= min_history_matches)
        & (features_df["away_match_count"] >= min_history_matches)
        & (features_df["result"].notna())
    ].copy()

    valid_df["result"] = valid_df["result"].astype(int)
    logger.info(
        "Built features for %d matches (filtered from %d total, min_history=%d)",
        len(valid_df),
        len(features_df),
        min_history_matches,
    )
    return valid_df


def build_prediction_features(
    upcoming_df: pd.DataFrame, historical_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build features for upcoming matches using latest known form/ELO from historical matches.
    """
    if upcoming_df.empty:
        return pd.DataFrame()

    if historical_df.empty:
        raise ValueError("historical_df cannot be empty when constructing prediction features.")

    # Process all historical finished matches to arrive at latest team states
    finished = historical_df[
        historical_df["status"].str.lower().isin(["finished", "ft", "complete"])
    ].copy()
    finished = finished.dropna(subset=["home_score", "away_score"])

    _, state = _process_chronological_matches(finished)

    team_elo = state["team_elo"]
    team_last_date = state["team_last_date"]
    team_matches = state["team_matches"]
    h2h_store = state["h2h_store"]

    upcoming = upcoming_df.copy()
    upcoming["kickoff_utc"] = pd.to_datetime(upcoming["kickoff_utc"], utc=True, errors="coerce")
    upcoming = upcoming.sort_values("kickoff_utc").reset_index(drop=True)

    feature_rows = []

    for idx, row in upcoming.iterrows():
        match_id = row.get("match_id")
        k_date = row["kickoff_utc"]
        h_id = row["home_team_id"]
        a_id = row["away_team_id"]

        h_elo = team_elo[h_id]
        a_elo = team_elo[a_id]
        elo_diff = h_elo - a_elo

        h_last = team_last_date.get(h_id)
        a_last = team_last_date.get(a_id)

        if h_last is not None and pd.notna(k_date) and pd.notna(h_last):
            h_rest = max(1.0, min(30.0, (k_date - h_last).total_seconds() / 86400.0))
        else:
            h_rest = 7.0

        if a_last is not None and pd.notna(k_date) and pd.notna(a_last):
            a_rest = max(1.0, min(30.0, (k_date - a_last).total_seconds() / 86400.0))
        else:
            a_rest = 7.0

        rest_diff = h_rest - a_rest

        h_hist = team_matches[h_id]
        h_gf5, h_ga5, h_pts5 = _calc_rolling_metrics(h_hist, 5)
        h_gf10, h_ga10, h_pts10 = _calc_rolling_metrics(h_hist, 10)
        h_win_strk, h_unb_strk = _calc_streaks(h_hist)
        h_home_hist = [m for m in h_hist if m["is_home"]]
        h_spec_gf5, h_spec_ga5, h_spec_pts5 = _calc_rolling_metrics(h_home_hist, 5)

        a_hist = team_matches[a_id]
        a_gf5, a_ga5, a_pts5 = _calc_rolling_metrics(a_hist, 5)
        a_gf10, a_ga10, a_pts10 = _calc_rolling_metrics(a_hist, 10)
        a_win_strk, a_unb_strk = _calc_streaks(a_hist)
        a_away_hist = [m for m in a_hist if not m["is_home"]]
        a_spec_gf5, a_spec_ga5, a_spec_pts5 = _calc_rolling_metrics(a_away_hist, 5)

        gf_diff_roll5 = h_gf5 - a_gf5
        ga_diff_roll5 = h_ga5 - a_ga5
        pts_diff_roll5 = h_pts5 - a_pts5

        gf_diff_roll10 = h_gf10 - a_gf10
        ga_diff_roll10 = h_ga10 - a_ga10
        pts_diff_roll10 = h_pts10 - a_pts10

        h2h_key = tuple(sorted([h_id, a_id]))
        h2h_stats = _calc_h2h_stats(h2h_store[h2h_key], h_id, a_id)

        feat = {
            "match_id": match_id,
            "kickoff_utc": k_date,
            "status": row.get("status", "scheduled"),
            "home_team_id": h_id,
            "home_team_name": row.get("home_team_name"),
            "away_team_id": a_id,
            "away_team_name": row.get("away_team_name"),
            "home_match_count": len(h_hist),
            "away_match_count": len(a_hist),
            "home_score": None,
            "away_score": None,
            "result": None,
            "home_elo": h_elo,
            "away_elo": a_elo,
            "elo_diff": elo_diff,
            "home_rest_days": h_rest,
            "away_rest_days": a_rest,
            "rest_diff": rest_diff,
            "home_roll_gf_5": h_gf5,
            "home_roll_ga_5": h_ga5,
            "home_roll_pts_5": h_pts5,
            "away_roll_gf_5": a_gf5,
            "away_roll_ga_5": a_ga5,
            "away_roll_pts_5": a_pts5,
            "gf_diff_roll5": gf_diff_roll5,
            "ga_diff_roll5": ga_diff_roll5,
            "pts_diff_roll5": pts_diff_roll5,
            "home_roll_gf_10": h_gf10,
            "home_roll_ga_10": h_ga10,
            "home_roll_pts_10": h_pts10,
            "away_roll_gf_10": a_gf10,
            "away_roll_ga_10": a_ga10,
            "away_roll_pts_10": a_pts10,
            "gf_diff_roll10": gf_diff_roll10,
            "ga_diff_roll10": ga_diff_roll10,
            "pts_diff_roll10": pts_diff_roll10,
            "home_specific_pts_5": h_spec_pts5,
            "home_specific_gf_5": h_spec_gf5,
            "home_specific_ga_5": h_spec_ga5,
            "away_specific_pts_5": a_spec_pts5,
            "away_specific_gf_5": a_spec_gf5,
            "away_specific_ga_5": a_spec_ga5,
            "home_win_streak": h_win_strk,
            "home_unbeaten_streak": h_unb_strk,
            "away_win_streak": a_win_strk,
            "away_unbeaten_streak": a_unb_strk,
            **h2h_stats,
        }
        feature_rows.append(feat)

    return pd.DataFrame(feature_rows)


if __name__ == "__main__":
    setup_logging()
    df = build_features()
    print("Features summary:", df.shape if not df.empty else "Empty")
