"""NFL-wide structural contracts reusable by model workflows."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.sports.data_validation import validate_feature_coverage, validate_frame


def validate_expected_points_inputs(
    *,
    pbp: pd.DataFrame,
    player_stats: pd.DataFrame,
    schedules: pd.DataFrame,
    teams: pd.DataFrame,
    pbp_seasons: Iterable[int],
    player_seasons: Iterable[int],
    schedule_seasons: Iterable[int],
    strict: bool = True,
) -> None:
    validate_frame(
        pbp,
        label="NFL play-by-play",
        non_null_columns=("game_id", "season", "week"),
        expected_seasons=pbp_seasons,
        strict=strict,
    )
    quarterback_stats = player_stats.loc[player_stats["position_group"] == "QB"]
    validate_frame(
        quarterback_stats,
        label="NFL weekly quarterback stats",
        non_null_columns=("player_id", "season", "week"),
        unique_keys=(("season", "week", "player_id"),),
        expected_seasons=player_seasons,
        strict=strict,
    )
    validate_frame(
        schedules,
        label="NFL schedules",
        non_null_columns=("game_id", "season", "week", "home_team", "away_team"),
        unique_keys=(("game_id",),),
        expected_seasons=schedule_seasons,
        strict=strict,
    )
    validate_frame(
        teams,
        label="NFL teams",
        non_null_columns=("team_abbr", "team_color", "team_color2", "team_logo_espn"),
        unique_keys=(("team_abbr",),),
        strict=strict,
    )


def validate_expected_points_frame(frame: pd.DataFrame, *, strict: bool = True) -> bool:
    return validate_frame(
        frame,
        label="NFL expected-points model frame",
        required_columns=(
            "game_id", "season", "week", "home_team", "away_team",
            "home_score", "away_score", "date_time",
        ),
        non_null_columns=("game_id", "season", "week", "home_team", "away_team", "date_time"),
        unique_keys=(("game_id",),),
        strict=strict,
    )


def validate_expected_points_features(
    frame: pd.DataFrame,
    *,
    current_year: int,
    current_week: int,
    feature_columns: tuple[str, ...] | list[str],
    strict: bool = True,
) -> bool:
    mask = (frame["season"] == current_year) & (frame["week"] == current_week)
    return validate_feature_coverage(
        frame,
        label="NFL expected-points feature frame",
        feature_columns=feature_columns,
        prediction_mask=mask,
        strict=strict,
    )


__all__ = [
    "validate_expected_points_features",
    "validate_expected_points_frame",
    "validate_expected_points_inputs",
]
