#!/usr/bin/env python3
"""Build frontend football team manifests and cache their logos locally."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Iterable


CFB_TEAMS_URL = "https://api.collegefootballdata.com/teams/fbs"
NFL_TEAMS_URL = "https://github.com/nflverse/nflverse-data/releases/download/teams/teams_colors_logos.csv"
USER_AGENT = "sports-models-team-sync/1.0"
FRONTEND_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = FRONTEND_ROOT / "app" / "generated"
PUBLIC_TEAMS_DIR = FRONTEND_ROOT / "public" / "teams"


def _request_bytes(url: str, *, authorization: str | None = None, attempts: int = 3) -> bytes:
    headers = {"Accept": "*/*", "User-Agent": USER_AGENT}
    if authorization:
        headers["Authorization"] = authorization

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except (TimeoutError, urllib.error.URLError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))

    raise RuntimeError(f"Request failed after {attempts} attempts: {url}") from last_error


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "team"


def _unique_strings(values: Iterable[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            unique.append(cleaned)
            seen.add(key)
    return unique


def select_cfb_logo_url(logos: Any) -> str | None:
    if not isinstance(logos, list):
        return None
    urls = [value for value in logos if isinstance(value, str) and value.startswith("https://")]
    standard = [value for value in urls if "/logos-dark/" not in value]
    for size in ("/128/", "/256/", "/512/"):
        match = next((value for value in standard if size in value), None)
        if match:
            return match
    return standard[0] if standard else (urls[0] if urls else None)


def normalize_cfb_team(team: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    external_id = team.get("id")
    school = str(team.get("school") or "").strip()
    if not school or not isinstance(external_id, int):
        raise ValueError("Every CFBD team must include an integer id and school name")

    abbreviation = str(team.get("abbreviation") or "").strip().upper()
    mascot = str(team.get("mascot") or "").strip()
    aliases = _unique_strings(
        [school, abbreviation, mascot, f"{school} {mascot}".strip(), *(team.get("alternateNames") or [])]
    )
    logo_url = select_cfb_logo_url(team.get("logos"))
    entry = {
        "id": f"cfb-{external_id}",
        "externalId": external_id,
        "displayName": school,
        "abbreviation": abbreviation,
        "mascot": mascot or None,
        "aliases": aliases,
        "conference": str(team.get("conference") or "").strip() or None,
        "color": str(team.get("color") or "").strip() or None,
        "alternateColor": str(team.get("alternateColor") or "").strip() or None,
        "logoPath": f"/teams/cfb/{external_id}.png" if logo_url else None,
    }
    return entry, logo_url


def normalize_nfl_team(row: dict[str, str]) -> tuple[dict[str, Any], str | None] | None:
    abbreviation = (row.get("team_abbr") or "").strip().upper()
    display_name = (row.get("team_name") or "").strip()
    logo_url = (row.get("team_logo_espn") or "").strip() or None
    if not abbreviation or not display_name or abbreviation in {"LAR", "OAK", "SD", "STL"}:
        return None

    team_id = _slug(abbreviation)
    entry = {
        "id": team_id,
        "displayName": display_name,
        "abbreviation": abbreviation,
        "aliases": _unique_strings([display_name, row.get("team_nick"), abbreviation]),
        "conference": (row.get("team_conf") or "").strip() or None,
        "division": (row.get("team_division") or "").strip() or None,
        "color": (row.get("team_color") or "").strip() or None,
        "alternateColor": (row.get("team_color2") or "").strip() or None,
        "logoPath": f"/teams/nfl/{team_id}.png" if logo_url else None,
    }
    return entry, logo_url


def _download_logo(url: str, destination: Path) -> None:
    image = _request_bytes(url)
    if not image.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Logo response is not a PNG: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(image)
    temporary.replace(destination)


def _sync_logos(logo_jobs: list[tuple[str, str]], league: str) -> set[str]:
    destination_dir = PUBLIC_TEAMS_DIR / league
    destination_dir.mkdir(parents=True, exist_ok=True)
    available: set[str] = set()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_download_logo, url, destination_dir / filename): (filename, url)
            for filename, url in logo_jobs
        }
        for future in as_completed(futures):
            filename, url = futures[future]
            try:
                future.result()
                available.add(filename)
            except Exception as error:  # Keep a previously cached asset on transient failures.
                if (destination_dir / filename).is_file():
                    available.add(filename)
                    print(f"warning: kept cached {league.upper()} logo after download failed: {url}", file=sys.stderr)
                else:
                    print(f"warning: omitted unavailable {league.upper()} logo: {error}", file=sys.stderr)
    return available


def _write_manifest(filename: str, payload: dict[str, Any]) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    destination = GENERATED_DIR / filename
    content = json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") == content:
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)


def sync_cfb(year: int) -> int:
    api_key = os.environ.get("CFBD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("CFBD_API_KEY is required. Load it from the repository .env before running the sync.")

    source_url = f"{CFB_TEAMS_URL}?{urllib.parse.urlencode({'year': year})}"
    raw = json.loads(_request_bytes(source_url, authorization=f"Bearer {api_key}"))
    if not isinstance(raw, list):
        raise ValueError("CFBD returned an unexpected teams payload")

    normalized = [normalize_cfb_team(team) for team in raw if isinstance(team, dict)]
    normalized.sort(key=lambda item: item[0]["displayName"].casefold())
    team_ids = {entry["id"] for entry, _ in normalized}
    if len(normalized) < 100 or len(team_ids) != len(normalized):
        raise ValueError(f"Refusing to replace the CFB catalog with {len(normalized)} invalid or duplicate rows")
    jobs = [(f"{entry['externalId']}.png", url) for entry, url in normalized if url]
    available = _sync_logos(jobs, "cfb")
    teams = [
        {**entry, "logoPath": entry["logoPath"] if f"{entry['externalId']}.png" in available else None}
        for entry, _ in normalized
    ]
    _write_manifest(
        "cfb-teams.json",
        {"league": "cfb", "season": year, "source": source_url, "teams": teams},
    )
    print(f"Synced {len(teams)} CFB teams and {len(available)} logos for {year}.")
    return len(teams)


def sync_nfl() -> int:
    body = _request_bytes(NFL_TEAMS_URL).decode("utf-8-sig")
    normalized = [item for row in csv.DictReader(io.StringIO(body)) if (item := normalize_nfl_team(row))]
    normalized.sort(key=lambda item: item[0]["displayName"].casefold())
    team_ids = {entry["id"] for entry, _ in normalized}
    if len(normalized) < 30 or len(team_ids) != len(normalized):
        raise ValueError(f"Refusing to replace the NFL catalog with {len(normalized)} invalid or duplicate rows")
    jobs = [(f"{entry['id']}.png", url) for entry, url in normalized if url]
    available = _sync_logos(jobs, "nfl")
    teams = [
        {**entry, "logoPath": entry["logoPath"] if f"{entry['id']}.png" in available else None}
        for entry, _ in normalized
    ]
    _write_manifest(
        "nfl-teams.json",
        {"league": "nfl", "source": NFL_TEAMS_URL, "teams": teams},
    )
    print(f"Synced {len(teams)} NFL teams and {len(available)} logos.")
    return len(teams)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("league", choices=("cfb", "nfl", "all"), help="Team catalog(s) to refresh")
    parser.add_argument("--year", type=int, default=date.today().year, help="CFB season (default: current year)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.league in {"nfl", "all"}:
            sync_nfl()
        if args.league in {"cfb", "all"}:
            sync_cfb(args.year)
    except (RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
