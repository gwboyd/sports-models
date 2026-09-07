"""Stable, leakage-safe CFB expected-points feature construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from src.model_patterns.expected_points.chronology import RESULT_AVAILABILITY_LAG
from src.sports.data_validation import validate_frame
from src.sports.football.transforms import (
    OpponentAdjustmentConfig,
    OpponentMetricSpec,
    build_lagged_team_metrics,
    build_opponent_adjusted_team_metrics,
    resolve_opponent_adjustment_config,
)


TEAM_GAME_KEYS = ("game_id", "season", "week", "team")

# Each entry is a per-play measure, not a game total. Only the offensive
# provider row is consumed; the defense is its scheduled opponent.
CFB_EFFICIENCY_METRICS = (
    "explosiveness", "ppa", "success_rate", "rushing_plays_ppa",
    "passing_plays_ppa", "rushing_plays_success_rate", "passing_plays_success_rate",
)
DEFAULT_CFB_EFFICIENCY_METRICS = ("explosiveness", "ppa", "success_rate")
DEFAULT_CFB_OPPONENT_ADJUSTMENT = OpponentAdjustmentConfig(
    adjustment_strength=0.5, fcs_policy="pooled", excluded_seasons=(2020,),
)


def build_weekly_pregame_elo(schedule: pd.DataFrame) -> pd.DataFrame:
    """Freeze pregame Elo before each week; retain chronological postgame diagnostics."""
    required = (
        "game_id", "season", "week", "start_date", "home_team", "away_team",
        "home_classification", "away_classification", "home_points", "away_points",
    )
    validate_frame(
        schedule, label="CFB Elo schedule", required_columns=required,
        non_null_columns=required[:6], unique_keys=(("game_id",),), strict=True,
    )
    output = schedule.reset_index(drop=True).copy()
    for side in ("home", "away"):
        output[f"{side}_pregame_elo"] = float("nan")
        output[f"{side}_postgame_elo"] = float("nan")
    ordered = schedule.reset_index(drop=True).copy()
    ordered["__kickoff"] = pd.to_datetime(ordered["start_date"], utc=True, errors="raise")
    if ordered["__kickoff"].isna().any():
        raise ValueError("CFB Elo schedule has missing kickoff times")
    ordered = ordered.sort_values(["__kickoff", "game_id"], kind="stable")
    completed = ordered.loc[ordered[["home_points", "away_points"]].notna().all(axis=1)]
    events = list(completed.iterrows())
    ratings: dict[str, int] = {}
    position = 0

    def rating(row: pd.Series, side: str) -> int:
        return ratings.get(row[f"{side}_team"], 1500 if row[f"{side}_classification"] == "fbs" else 1200)

    def update(index, row: pd.Series) -> None:
        home, away = rating(row, "home"), rating(row, "away")
        margin = float(row["home_points"]) - float(row["away_points"])
        outcome = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        expected_home = 1.0 / (1.0 + 10 ** ((away - home) / 400))
        expected_away = 1.0 / (1.0 + 10 ** ((home - away) / 400))
        new_home = round(home + 25 * (outcome - expected_home))
        new_away = round(away + 25 * ((1.0 - outcome) - expected_away))
        ratings[row["home_team"]], ratings[row["away_team"]] = new_home, new_away
        output.loc[index, ["home_postgame_elo", "away_postgame_elo"]] = [new_home, new_away]

    periods = sorted(
        ordered.groupby(["season", "week"], sort=False),
        key=lambda item: item[1]["__kickoff"].min(),
    )
    for _, period in periods:
        cutoff = period["__kickoff"].min()
        while position < len(events) and events[position][1]["__kickoff"] + RESULT_AVAILABILITY_LAG < cutoff:
            update(*events[position])
            position += 1
        for index, row in period.iterrows():
            output.loc[index, ["home_pregame_elo", "away_pregame_elo"]] = [rating(row, "home"), rating(row, "away")]
    # Postgame columns are inspection-only. Filling them after the last snapshot
    # cannot alter any stored pregame value, including scheduled future games.
    for index, row in events[position:]:
        update(index, row)
    output.index = schedule.index
    return output


def build_cfb_efficiency_schedule(
    schedule: pd.DataFrame,
    historical_games: pd.DataFrame,
    *,
    history_start_season: int,
) -> pd.DataFrame:
    """Supply historical opponents for feature warmup without extending score rows.

    The score schedule remains unchanged. Older advanced-stat observations need
    their own dated opponents/classifications before the shared adjustment can
    use them; kickoff timestamps alone are insufficient.
    """
    first_season = int(pd.to_numeric(schedule["season"], errors="raise").min())
    if history_start_season > first_season:
        raise ValueError("Efficiency history cannot start after the score schedule")
    history = historical_games.copy()
    if "game_id" not in history:
        history = history.rename(columns={"id": "game_id"})
    seasons = pd.to_numeric(history["season"], errors="raise")
    history = history.loc[(seasons >= history_start_season) & (seasons < first_season)]
    columns = ["game_id", "season", "week", "home_team", "away_team", "start_date",
               "home_classification", "away_classification"]
    validate_frame(history, label="CFB efficiency warmup schedule", required_columns=columns,
                   non_null_columns=columns[:6], unique_keys=(("game_id",),), strict=True)
    if "neutral_site" in history:
        columns.append("neutral_site")
    result = pd.concat([history[columns], schedule.copy()], ignore_index=True, sort=False)
    result["start_date"] = pd.to_datetime(result["start_date"], utc=True, errors="raise")
    return result


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


def build_opponent_adjusted_advanced_stats(
    advanced_stats: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    config: OpponentAdjustmentConfig | Mapping[str, object] | None = None,
    metrics: Sequence[str] | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    """Build selected CFB efficiency features with stable descriptive names.

    CFBD reports the offensive and defensive side of the same game metric.  We
    intentionally use the offensive row as the one canonical observation and
    infer the defensive observation from its opponent in the schedule.
    """
    metrics = DEFAULT_CFB_EFFICIENCY_METRICS if metrics is None else tuple(metrics)
    if not metrics or len(set(metrics)) != len(metrics) or set(metrics) - set(CFB_EFFICIENCY_METRICS):
        raise ValueError("CFB metrics must be unique registered efficiency metrics")
    required = (*TEAM_GAME_KEYS, "start_date", *[f"offense_{metric}" for metric in metrics])
    validate_frame(
        advanced_stats,
        label="CFB opponent-adjusted advanced stats",
        required_columns=required,
        non_null_columns=(*TEAM_GAME_KEYS, "start_date"),
        unique_keys=(TEAM_GAME_KEYS,),
        strict=strict,
    )
    observations = pd.concat([
        advanced_stats.loc[:, [*TEAM_GAME_KEYS, f"offense_{metric}"]]
        .rename(columns={f"offense_{metric}": "value"}).assign(metric=metric)
        for metric in metrics
    ], ignore_index=True)
    return build_opponent_adjusted_team_metrics(
        schedule,
        observations,
        tuple(
            OpponentMetricSpec(
                key=metric,
                offense_output=f"offense_{metric}",
                defense_output=f"defense_{metric}",
                smoothing="dynamic",
            ) for metric in metrics
        ),
        config=resolve_opponent_adjustment_config(config, defaults=DEFAULT_CFB_OPPONENT_ADJUSTMENT),
        strict=strict,
    )


__all__ = [
    "CFB_EFFICIENCY_METRICS",
    "DEFAULT_CFB_EFFICIENCY_METRICS",
    "DEFAULT_CFB_OPPONENT_ADJUSTMENT",
    "build_cfb_efficiency_schedule",
    "build_opponent_adjusted_advanced_stats",
    "build_pregame_advanced_stats",
    "build_weekly_pregame_elo",
]
