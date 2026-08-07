"""
Cliente oficial TheStatsAPI – datos reales únicamente.
Requiere THESTATS_API_KEY en el entorno (GitHub Secrets o .env local).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils import ensure_dirs, get_cache_dir

logger = logging.getLogger(__name__)

BASE_URL = "https://api.thestatsapi.com/api"


class TheStatsAPIClient:
    """
    Cliente robusto para la API de TheStatsAPI con:
    - Autenticación con únicamente el encabezado Authorization: Bearer
    - Disipador de tasa (rate limiting sleep)
    - Caché local en disco para evitar consultas redundantes
    - Paginación integrada
    - Circuit Breaker ante errores de autenticación repetidos (401)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        backoff: float = 2.0,
        rate_limit_sleep: float = 0.1,
        use_cache: bool = True,
        cache_dir: Optional[str | Path] = None,
    ):
        self.api_key = api_key or os.environ.get("THESTATS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "THESTATS_API_KEY no encontrada. "
                "Configúrala en GitHub Secrets o como variable de entorno."
            )

        self.timeout = timeout
        self.rate_limit_sleep = rate_limit_sleep
        self.use_cache = use_cache

        cache_path = Path(cache_dir) if cache_dir else get_cache_dir() / "api"
        ensure_dirs(cache_path)
        self.cache_dir = cache_path

        self._consecutive_401s = 0

        self.session = requests.Session()
        # IMPORTANTE: Se usa EXCLUSIVAMENTE el encabezado Authorization: Bearer.
        # Enviar x-api-key simultáneamente provoca conflicto y respuestas 401 en la API.
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

    def _get_cache_key(self, endpoint: str, params: Optional[Dict[str, Any]]) -> str:
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        sorted_params = sorted(clean_params.items())
        raw_key = f"{endpoint}?{urlencode(sorted_params)}"
        return hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Error leyendo caché de %s: %s", cache_file, e)
        return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning("Error guardando en caché %s: %s", cache_file, e)

    def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
    ) -> Dict[str, Any]:
        should_cache = self.use_cache if use_cache is None else use_cache
        cache_key = self._get_cache_key(endpoint, params)

        if should_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

        # Circuit breaker check
        if self._consecutive_401s >= 3:
            raise RuntimeError(
                "Circuit breaker activado: Se recibieron 3 respuestas 401 consecutivas "
                "desde api.thestatsapi.com. Verifica la validez de THESTATS_API_KEY."
            )

        if self.rate_limit_sleep > 0:
            time.sleep(self.rate_limit_sleep)

        url = f"{BASE_URL}{endpoint}"
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}

        try:
            response = self.session.get(url, params=clean_params, timeout=self.timeout)

            if response.status_code == 401:
                self._consecutive_401s += 1
                if self._consecutive_401s >= 3:
                    raise RuntimeError(
                        "Circuit breaker activado: Se recibieron 3 respuestas 401 consecutivas. "
                        "Verifica que THESTATS_API_KEY sea correcta."
                    )
                raise RuntimeError(
                    "Unauthorized (401): THESTATS_API_KEY está ausente, es inválida o no tiene permisos. "
                    "Asegúrate de que la clave enviada en el header 'Authorization: Bearer <key>' sea la correcta."
                )

            response.raise_for_status()
            self._consecutive_401s = 0  # Reset circuit breaker tras éxito
            data = response.json()

            if should_cache and isinstance(data, dict):
                self._save_to_cache(cache_key, data)

            return data

        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 401:
                self._consecutive_401s += 1
                if self._consecutive_401s >= 3:
                    raise RuntimeError(
                        "Circuit breaker activado: 3 errores 401 consecutivos desde la API."
                    ) from e
                raise RuntimeError(
                    "Unauthorized (401): Error de autenticación con TheStatsAPI."
                ) from e
            raise

    def get_competitions(
        self,
        search: Optional[str] = None,
        country: Optional[str] = None,
        country_code: Optional[str] = None,
        page: int = 1,
        per_page: int = 100,
    ) -> Dict[str, Any]:
        params = {"page": page, "per_page": per_page}
        if search:
            params["search"] = search
        if country:
            params["country"] = country
        if country_code:
            params["country_code"] = country_code
        return self._get("/football/competitions", params)

    def get_competition(self, competition_id: str) -> Dict[str, Any]:
        return self._get(f"/football/competitions/{competition_id}")

    def get_seasons(self, competition_id: str) -> Dict[str, Any]:
        return self._get(f"/football/competitions/{competition_id}/seasons")

    def get_standings(self, competition_id: str, season_id: str) -> Dict[str, Any]:
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
    ) -> Dict[str, Any]:
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
        sleep_between_pages: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Paginación completa de partidos con rate limiting."""
        all_matches: List[Dict[str, Any]] = []
        page = 1
        sleep_time = (
            sleep_between_pages if sleep_between_pages is not None else self.rate_limit_sleep
        )

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
                if isinstance(data, list):
                    items = data
                else:
                    break

            all_matches.extend(items)
            meta = data.get("meta", {}) if isinstance(data, dict) else {}
            total_pages = meta.get("total_pages") or meta.get("last_page") or 1
            if page >= total_pages:
                break

            page += 1
            if sleep_time > 0:
                time.sleep(sleep_time)

        return all_matches

    def get_all_seasons_matches(
        self,
        competition_id: str,
        season_ids: List[str],
        status: Optional[str] = None,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtiene de forma eficiente todos los partidos para múltiples temporadas."""
        combined_matches: List[Dict[str, Any]] = []
        for sid in season_ids:
            logger.info("Descargando partidos de la temporada %s...", sid)
            matches = self.get_all_matches(
                competition_id=competition_id,
                season_id=sid,
                status=status,
                per_page=per_page,
            )
            combined_matches.extend(matches)
        return combined_matches

    def get_match_stats(self, match_id: str) -> Dict[str, Any]:
        return self._get(f"/football/matches/{match_id}/stats")

    def get_match_odds(self, match_id: str) -> Dict[str, Any]:
        return self._get(f"/football/matches/{match_id}/odds")

    def get_team_stats(self, team_id: str, season_id: str) -> Dict[str, Any]:
        return self._get(
            f"/football/teams/{team_id}/stats",
            params={"season_id": season_id},
        )

    def health(self) -> Dict[str, Any]:
        return self._get("/health", use_cache=False)


def get_client(
    api_key: Optional[str] = None,
    rate_limit_sleep: float = 0.1,
    use_cache: bool = True,
) -> TheStatsAPIClient:
    """Función de conveniencia para instanciar el cliente."""
    return TheStatsAPIClient(
        api_key=api_key,
        rate_limit_sleep=rate_limit_sleep,
        use_cache=use_cache,
    )
