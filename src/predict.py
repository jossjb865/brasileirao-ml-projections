"""
Prediction pipeline for upcoming Brasileirão matches.
Uses strictly pre-match data, generates 1X2 probabilities,
estimates confidence intervals / model agreement, and identifies value bets.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.api_client import get_client
from src.data.features import build_prediction_features, FEATURE_COLUMNS
from src.data.store import DataStore
from src.ensemble import Ensemble
from src.utils import load_yaml, setup_logging, ensure_dirs, get_outputs_dir, get_config

logger = logging.getLogger(__name__)


def get_upcoming_matches(client, competition_id: str = "comp_4795", days_ahead: int = 14) -> pd.DataFrame:
    """Fetch scheduled upcoming matches from API."""
    today = datetime.now(timezone.utc).date()
    date_to = today + timedelta(days=days_ahead)

    all_rows = []
    page = 1
    while True:
        resp = client.get_matches(
            competition_id=competition_id,
            status="scheduled",
            date_from=str(today),
            date_to=str(date_to),
            page=page,
            per_page=100,
        )
        items = resp.get("data", [])
        if not items:
            resp = client.get_matches(
                competition_id=competition_id,
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

            odds_data = m.get("odds") or {}
            all_rows.append(
                {
                    "match_id": m.get("id") or m.get("match_id"),
                    "competition_id": competition_id,
                    "status": m.get("status"),
                    "kickoff_utc": m.get("utc_date") or m.get("kickoff_utc") or m.get("date"),
                    "home_team_id": home.get("id"),
                    "home_team_name": home.get("name"),
                    "away_team_id": away.get("id"),
                    "away_team_name": away.get("name"),
                    "odds_home": odds_data.get("home") or odds_data.get("1"),
                    "odds_draw": odds_data.get("draw") or odds_data.get("X"),
                    "odds_away": odds_data.get("away") or odds_data.get("2"),
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


def run_predict(competition_id: str = "comp_4795", days_ahead: int = 14) -> pd.DataFrame:
    setup_logging()
    outputs_dir = get_outputs_dir()
    ensure_dirs(outputs_dir)

    client = get_client()
    store = DataStore()

    # Load historical matches for feature building
    historical = store.load_matches()
    if historical.empty:
        raise RuntimeError("No historical data available in DataStore. Run data fetch first.")

    # Fetch upcoming scheduled matches
    upcoming = get_upcoming_matches(client, competition_id=competition_id, days_ahead=days_ahead)
    if upcoming.empty:
        logger.warning("No upcoming matches found for competition %s in next %d days.", competition_id, days_ahead)
        return pd.DataFrame()

    logger.info("Found %d upcoming matches to project", len(upcoming))

    # Build pre-match features using the anti-leakage feature builder
    pre_match_df = build_prediction_features(upcoming, historical)
    if pre_match_df.empty:
        logger.warning("No features could be built for upcoming matches.")
        return pd.DataFrame()

    # Load trained ensemble
    ensemble = Ensemble.load()

    # Get per-model predictions for uncertainty estimation
    model_probas = []
    for name, model in ensemble.models.items():
        try:
            p = model.predict_proba(pre_match_df)
            model_probas.append(p)
        except Exception as e:
            logger.warning("Model %s prediction error: %s", name, e)

    if not model_probas:
        raise RuntimeError("No models generated predictions")

    stacked = np.stack(model_probas, axis=0)  # (n_models, n_samples, 3)
    prob_ensemble = ensemble.predict_proba(pre_match_df)
    pred_indices = prob_ensemble.argmax(axis=1)

    n_samples = len(pre_match_df)
    std_uncertainty = np.array([
        np.std(stacked[:, i, pred_indices[i]]) for i in range(n_samples)
    ])
    individual_preds = stacked.argmax(axis=2)  # (n_models, n_samples)
    model_agreement = np.array([
        np.mean(individual_preds[:, i] == pred_indices[i]) for i in range(n_samples)
    ])

    max_prob = np.max(prob_ensemble, axis=1)
    confidence = np.clip(max_prob * (0.6 + 0.4 * model_agreement) - 0.5 * std_uncertainty, 0.0, 1.0)

    results = pd.DataFrame({
        "match_id": pre_match_df["match_id"],
        "kickoff_utc": pre_match_df["kickoff_utc"],
        "home_team_name": pre_match_df["home_team_name"],
        "away_team_name": pre_match_df["away_team_name"],
        "prob_home": prob_ensemble[:, 0].round(4),
        "prob_draw": prob_ensemble[:, 1].round(4),
        "prob_away": prob_ensemble[:, 2].round(4),
        "pred_result": pred_indices,
        "pred_label": pd.Series(pred_indices).map({0: "H", 1: "D", 2: "A"}),
        "confidence": confidence.round(4),
        "model_agreement": model_agreement.round(4),
        "uncertainty_std": std_uncertainty.round(4),
    })

    # Attach odds if available for value detection
    if "odds_home" in pre_match_df.columns:
        results["odds_home"] = pd.to_numeric(pre_match_df["odds_home"], errors="coerce")
        results["odds_draw"] = pd.to_numeric(pre_match_df["odds_draw"], errors="coerce")
        results["odds_away"] = pd.to_numeric(pre_match_df["odds_away"], errors="coerce")

        has_odds = results["odds_home"].notna() & results["odds_draw"].notna() & results["odds_away"].notna()
        results.loc[has_odds, "implied_home"] = 1.0 / results.loc[has_odds, "odds_home"]
        results.loc[has_odds, "implied_draw"] = 1.0 / results.loc[has_odds, "odds_draw"]
        results.loc[has_odds, "implied_away"] = 1.0 / results.loc[has_odds, "odds_away"]

        # Value bet: model probability significantly higher than implied probability
        results.loc[has_odds, "value_home"] = results.loc[has_odds, "prob_home"] - results.loc[has_odds, "implied_home"]
        results.loc[has_odds, "value_draw"] = results.loc[has_odds, "prob_draw"] - results.loc[has_odds, "implied_draw"]
        results.loc[has_odds, "value_away"] = results.loc[has_odds, "prob_away"] - results.loc[has_odds, "implied_away"]

        best_value = results.loc[has_odds, ["value_home", "value_draw", "value_away"]].idxmax(axis=1)
        best_value_val = results.loc[has_odds, ["value_home", "value_draw", "value_away"]].max(axis=1)
        results.loc[has_odds, "value_bet"] = best_value
        results.loc[has_odds, "value_edge"] = best_value_val.round(4)
        results.loc[has_odds & (best_value_val > 0.05), "value_signal"] = "VALUE"
        results.loc[has_odds & (best_value_val <= 0.05), "value_signal"] = "-"
    else:
        for col in ["odds_home", "odds_draw", "odds_away", "value_bet", "value_edge", "value_signal"]:
            results[col] = np.nan

    # Save predictions
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parquet_path = outputs_dir / f"predictions_{ts}.parquet"
    csv_path = outputs_dir / f"predictions_{ts}.csv"
    results.to_parquet(parquet_path, index=False)
    results.to_csv(csv_path, index=False)

    logger.info("Predictions saved -> %s (%d matches)", parquet_path, len(results))
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 1X2 predictions for upcoming Brasileirao matches")
    parser.add_argument("--days-ahead", type=int, default=14, help="Days ahead to look for upcoming matches")
    parser.add_argument("--competition", type=str, default="comp_4795", help="Competition ID")
    args = parser.parse_args()

    run_predict(competition_id=args.competition, days_ahead=args.days_ahead)
