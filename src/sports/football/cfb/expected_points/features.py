"""Stable, leakage-safe CFB expected-points feature construction."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.sports.data_validation import validate_frame
from src.sports.football.transforms import build_lagged_team_metrics


TEAM_GAME_KEYS = ("game_id", "season", "week", "team")


def build_pregame_advanced_stats(
    advanced_stats: pd.DataFrame,
    schedule: pd.DataFrame,
    columns: Sequence[str],
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Carry completed-game metrics onto a full team-game schedule before lagging.

    Advanced-stat feeds only contain completed games. Adding scheduled team-game
    rows first lets an upcoming game receive the EWMA of strictly earlier games
    instead of falling through to model-wide median imputation.
    """
    validate_frame(
        advanced_stats,
        label="CFB advanced stats feature source",
        required_columns=(*TEAM_GAME_KEYS, "start_date", *columns),
        non_null_columns=(*TEAM_GAME_KEYS, "start_date"),
        unique_keys=(TEAM_GAME_KEYS,),
        strict=strict,
    )
    validate_frame(
        schedule,
        label="CFB advanced stats schedule",
        required_columns=(
            "game_id", "season", "week", "home_team", "away_team", "start_date",
        ),
        non_null_columns=(
            "game_id", "season", "week", "home_team", "away_team", "start_date",
        ),
        unique_keys=(("game_id",),),
        strict=strict,
    )

    base_columns = ["game_id", "season", "week", "start_date"]
    home = schedule[base_columns + ["home_team"]].rename(columns={"home_team": "team"})
    away = schedule[base_columns + ["away_team"]].rename(columns={"away_team": "team"})
    scheduled_team_games = pd.concat([home, away], ignore_index=True)
    validate_frame(
        scheduled_team_games,
        label="CFB scheduled team games",
        unique_keys=(TEAM_GAME_KEYS,),
        strict=strict,
    )

    metric_source = advanced_stats[[*TEAM_GAME_KEYS, "start_date", *columns]].copy()
    scheduled_metrics = scheduled_team_games.merge(
        metric_source.drop(columns="start_date"),
        on=list(TEAM_GAME_KEYS),
        how="left",
        validate="one_to_one",
    )

    history_only = metric_source.merge(
        scheduled_team_games[list(TEAM_GAME_KEYS)],
        on=list(TEAM_GAME_KEYS),
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    history_only = history_only.loc[
        history_only["_merge"] == "left_only",
        [*TEAM_GAME_KEYS, "start_date", *columns],
    ]
    timeline = pd.concat([history_only, scheduled_metrics], ignore_index=True, sort=False)
    return build_lagged_team_metrics(timeline, columns)


__all__ = ["build_pregame_advanced_stats"]
