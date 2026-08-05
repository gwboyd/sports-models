"""CFB-wide structural contracts reusable by model workflows."""

from __future__ import annotations

import pandas as pd

from src.sports.data_validation import validate_feature_coverage, validate_frame


def validate_cfbd_year(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    year: int,
    key_columns: tuple[str, ...],
    allow_empty: bool = False,
    strict: bool = True,
) -> bool:
    expected_seasons = None if frame.empty and allow_empty else (year,)
    return validate_frame(
        frame,
        label=f"CFB {dataset_name} for {year}",
        required_columns=(*key_columns, "season"),
        non_null_columns=(*key_columns, "season"),
        unique_keys=(key_columns,),
        expected_seasons=expected_seasons,
        allow_empty=allow_empty,
        strict=strict,
    )


def validate_expected_points_frame(frame: pd.DataFrame, *, strict: bool = True) -> bool:
    return validate_frame(
        frame,
        label="CFB expected-points model frame",
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
        label="CFB expected-points feature frame",
        feature_columns=feature_columns,
        prediction_mask=mask,
        strict=strict,
    )


__all__ = [
    "validate_cfbd_year",
    "validate_expected_points_features",
    "validate_expected_points_frame",
]
