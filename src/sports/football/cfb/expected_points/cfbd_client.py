"""Small REST client for the College Football Data API."""

from __future__ import annotations

from typing import Any

import requests


class CFBDClient:
    """Context-managed CFBD REST client returning unmodified list payloads."""

    BASE_URL = "https://api.collegefootballdata.com"

    def __init__(
        self, api_key: str, timeout: int = 30, session: requests.Session | None = None
    ) -> None:
        if not api_key:
            raise ValueError("CFBD_API_KEY is required")
        self._session = session or requests.Session()
        self._timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }

    def __enter__(self) -> "CFBDClient":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        self._session.close()

    def get_lines(self, year: int, season_type: str) -> list[dict[str, Any]]:
        return self._get("/lines", year, season_type)

    def get_games(self, year: int, season_type: str) -> list[dict[str, Any]]:
        return self._get("/games", year, season_type)

    def get_predicted_points_added_by_game(
        self, year: int, season_type: str
    ) -> list[dict[str, Any]]:
        return self._get("/ppa/games", year, season_type)

    def get_advanced_game_stats(self, year: int, season_type: str) -> list[dict[str, Any]]:
        return self._get("/stats/game/advanced", year, season_type)

    def _get(self, path: str, year: int, season_type: str) -> list[dict[str, Any]]:
        response = self._session.get(
            f"{self.BASE_URL}{path}",
            headers=self._headers,
            params={"year": year, "seasonType": season_type},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list from {path}, received {type(payload).__name__}")
        return payload
