"""NFL data ingestion boundary for the expected-points workflow."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Iterable

import nflreadpy as nfl
from nflreadpy.config import update_config
import pandas as pd
import polars as pl


LOGGER = logging.getLogger(__name__)

PBP_REQUIRED_COLUMNS = (
    "game_id", "season", "week", "season_type", "home_team", "away_team",
    "home_score", "away_score", "posteam", "defteam", "side_of_field",
    "yardline_100", "rush_attempt", "pass_attempt", "rush", "pass", "down",
    "yards_gained", "ydstogo", "epa", "id", "name", "qb_epa",
)
PLAYER_STATS_REQUIRED_COLUMNS = (
    "player_id", "player_name", "position_group", "team", "season", "week",
    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions",
)
SCHEDULE_REQUIRED_COLUMNS = (
    "game_id", "season", "week", "home_team", "away_team", "weekday",
    "home_qb_id", "away_qb_id", "home_moneyline", "away_moneyline",
    "spread_line", "home_spread_odds", "away_spread_odds", "total_line",
    "over_odds", "roof", "away_rest", "home_rest", "stadium_id", "div_game",
    "gametime", "gameday",
)
TEAM_REQUIRED_COLUMNS = ("team_abbr", "team_color", "team_color2", "team_logo_espn")

_LEGACY_TEAM_ABBREVIATIONS = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
_PBP_TEAM_COLUMNS = ("home_team", "away_team", "posteam", "defteam")
_SCHEDULE_TEAM_COLUMNS = ("home_team", "away_team")


@dataclass(frozen=True)
class NFLExpectedPointsInputs:
    """Pandas dataframes consumed by the NFL expected-points notebook."""

    pbp: pd.DataFrame
    player_stats: pd.DataFrame
    schedules: pd.DataFrame
    teams: pd.DataFrame


def load_expected_points_inputs(
    start_year: int, current_year: int, current_week: int
) -> NFLExpectedPointsInputs:
    """Load the selected NFL data needed by the expected-points workflow.

    nflreadpy returns Polars frames. Selecting at the source boundary keeps the
    notebook's pandas feature engineering unchanged without retaining full PBP
    frames in the nflreadpy in-memory cache.
    """
    update_config(cache_mode="off", verbose=False)
    pbp_years = list(range(start_year - 1, current_year + 1))
    schedule_years = list(range(start_year, current_year + 1))
    LOGGER.info(
        "Loading NFL expected-points data: pbp/player seasons=%s-%s, schedules=%s-%s",
        pbp_years[0], pbp_years[-1], schedule_years[0], schedule_years[-1],
    )

    pbp = _load_season_frames(
        "PBP", pbp_years, current_year, current_week, nfl.load_pbp, PBP_REQUIRED_COLUMNS
    )
    player_stats = _load_season_frames(
        "player stats",
        pbp_years,
        current_year,
        current_week,
        lambda season: nfl.load_player_stats(season, summary_level="week"),
        PLAYER_STATS_REQUIRED_COLUMNS,
    )
    schedules = _to_pandas(
        "schedule",
        _select_required("schedule", nfl.load_schedules(schedule_years), SCHEDULE_REQUIRED_COLUMNS),
    )
    teams = _to_pandas("team", _select_required("team", nfl.load_teams(), TEAM_REQUIRED_COLUMNS))

    player_stats = player_stats.rename(
        columns={"team": "recent_team", "passing_interceptions": "interceptions"}
    )
    _downcast_pbp_floats(pbp)
    _normalize_team_abbreviations(pbp, _PBP_TEAM_COLUMNS)
    _normalize_team_abbreviations(schedules, _SCHEDULE_TEAM_COLUMNS)

    LOGGER.info(
        "Loaded NFL expected-points data: pbp_rows=%s pbp_columns=%s, "
        "player_stat_rows=%s player_stat_columns=%s, schedule_rows=%s "
        "schedule_columns=%s, team_rows=%s team_columns=%s",
        len(pbp), len(pbp.columns), len(player_stats), len(player_stats.columns),
        len(schedules), len(schedules.columns), len(teams), len(teams.columns),
    )
    return NFLExpectedPointsInputs(pbp, player_stats, schedules, teams)


def _load_season_frames(
    dataset_name: str,
    seasons: Iterable[int],
    current_year: int,
    current_week: int,
    loader: Callable[[int], pl.DataFrame],
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    frames: list[pl.DataFrame] = []
    loaded_seasons: list[int] = []
    for season in seasons:
        try:
            frame = loader(season)
        except Exception as exc:
            if _may_skip_current_season(season, current_year, current_week, exc):
                LOGGER.info(
                    "Skipping unavailable current-season %s for %s during week 1: %s",
                    season, dataset_name, exc,
                )
                continue
            raise
        frames.append(_select_required(dataset_name, frame, required_columns))
        loaded_seasons.append(season)

    if not frames:
        raise ValueError(f"No {dataset_name} data was loaded for requested seasons")
    LOGGER.info("Loaded %s seasons: %s", dataset_name, loaded_seasons)
    return _to_pandas(dataset_name, pl.concat(frames, how="diagonal_relaxed"))


def _select_required(
    dataset_name: str, frame: pl.DataFrame, required_columns: tuple[str, ...]
) -> pl.DataFrame:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"{dataset_name} data is missing required columns: {missing_list}")
    return frame.select(required_columns)


def _to_pandas(dataset_name: str, frame: pl.DataFrame) -> pd.DataFrame:
    try:
        return frame.to_pandas()
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"PyArrow is required to convert {dataset_name} data from Polars to pandas"
        ) from exc


def _downcast_pbp_floats(pbp: pd.DataFrame) -> None:
    for column in pbp.select_dtypes(include=["float64"]).columns:
        pbp[column] = pd.to_numeric(pbp[column], downcast="float")


def _normalize_team_abbreviations(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = frame[column].replace(_LEGACY_TEAM_ABBREVIATIONS)


def _may_skip_current_season(
    season: int, current_year: int, current_week: int, error: Exception
) -> bool:
    if season != current_year or current_week != 1:
        return False
    return isinstance(error, ValueError) or (
        isinstance(error, ConnectionError) and "404" in str(error)
    )
