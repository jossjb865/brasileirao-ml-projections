"""
Cliente oficial TheStatsAPI – datos reales únicamente.
Requiere THESTATS_API_KEY en el entorno (GitHub Secrets o .env local).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://api.thestatsapi.com/api"


class TheStatsAPIClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        backoff: float = 2.0,
    ):
        self.api_key = api_key or os.environ.get("THESTATS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "THESTATS_API_KEY no encontrada. "
                "Configúrala en GitHub Secrets o como variable de entorno."
            )

        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": "brasileirao-ml-projections/1.0",
            }
        )

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict:
        url = f"{BASE_URL}{endpoint}"
        response = self.session.get(url, params=params or {}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_competitions(
        self,
        search: Optional[str] = None,
        country: Optional[str] = None,
        country_code: Optional[str] = None,
        page: int = 1,
        per_page: int = 100,
    ) -> Dict:
        params = {"page": page, "per_page": per_page}
        if search:
            params["search"] = search
        if country:
            params["country"] = country
        if country_code:
            params["country_code"] = country_code
        return self._get("/football/competitions", params)

    def get_competition(self, competition_id: str) -> Dict:
        return self._get(f"/football/competitions/{competition_id}")

    def get_seasons(self, competition_id: str) -> Dict:
        return self._get(f"/football/competitions/{competition_id}/seasons")

    def get_standings(self, competition_id: str, season_id: str) -> Dict:
        return self._get(
            f"/football/competitions/{competition_id}/seasons/{season_id}/standings"
        )

    def get_matches(
        self,
        competition_id: Optional[str] = None,
        season_id: Optional[str] = None,
        team_id: Optional[str] = None,
        status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: int = 1,
        per_page: int = 100,
    ) -> Dict:
        params: Dict[str, Any] = {"page": page, "per_page": per_page}
        if competition_id:
            params["competition_id"] = competition_id
        if season_id:
            params["season_id"] = season_id
        if team_id:
            params["team_id"] = team_id
        if status:
            params["status"] = status
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        return self._get("/football/matches", params)

    def get_all_matches(
        self,
        competition_id: str,
        season_id: Optional[str] = None,
        status: Optional[str] = None,
        per_page: int = 100,
        sleep_between_pages: float = 0.25,
    ) -> List[Dict]:
        """Paginación completa – datos reales."""
        all_matches: List[Dict] = []
        page = 1
        while True:
            data = self.get_matches(
                competition_id=competition_id,
                season_id=season_id,
                status=status,
                page=page,
                per_page=per_page,
            )
            items = data.get("data", [])
            if not items:
                break
            all_matches.extend(items)
            meta = data.get("meta", {})
            total_pages = meta.get("total_pages", 1)
            if page >= total_pages:
                break
            page += 1
            time.sleep(sleep_between_pages)
        return all_matches

    def get_match_stats(self, match_id: str) -> Dict:
        return self._get(f"/football/matches/{match_id}/stats")

    def get_match_odds(self, match_id: str) -> Dict:
        return self._get(f"/football/matches/{match_id}/odds")

    def get_team_stats(self, team_id: str, season_id: str) -> Dict:
        return self._get(
            f"/football/teams/{team_id}/stats",
            params={"season_id": season_id},
        )

    def health(self) -> Dict:
        return self._get("/health")


# Singleton conveniente
def get_client() -> TheStatsAPIClient:
    return TheStatsAPIClient()
