"""Stable NFL expected-points feature calculations."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

from src.sports.football.transforms import (
    OpponentAdjustmentConfig,
    OpponentMetricSpec,
    build_opponent_adjusted_team_metrics,
    resolve_opponent_adjustment_config,
)
from src.sports.football.kickoff import parse_eastern_kickoffs

DEFAULT_NFL_OPPONENT_ADJUSTMENT = OpponentAdjustmentConfig(adjustment_strength=0.5)


def calculate_nfl_passer_rating(
    attempts: float,
    completions: float,
    passing_yards: float,
    passing_touchdowns: float,
    interceptions: float,
) -> float:
    """Calculate the official NFL passer rating, bounded to 0–158.3."""
    try:
        attempts, completions, passing_yards, passing_touchdowns, interceptions = (
            float(value)
            for value in (
                attempts,
                completions,
                passing_yards,
                passing_touchdowns,
                interceptions,
            )
        )
    except (TypeError, ValueError):
        return float("nan")
    if not all(
        math.isfinite(value)
        for value in (
            attempts,
            completions,
            passing_yards,
            passing_touchdowns,
            interceptions,
        )
    ):
        return float("nan")
    if attempts <= 0:
        return 0.0

    components = (
        ((completions / attempts) - 0.3) * 5,
        ((passing_yards / attempts) - 3) * 0.25,
        (passing_touchdowns / attempts) * 20,
        2.375 - ((interceptions / attempts) * 25),
    )
    bounded = [min(2.375, max(0.0, component)) for component in components]
    return float(sum(bounded) / 6 * 100)


def build_pregame_quarterback_metrics(starting_qbs: pd.DataFrame) -> pd.DataFrame:
    """Smooth only earlier starts; unknown/debut quarterbacks remain missing.

    The caller joins scheduled starters to game-level passer ratings. Never fill
    a debut from a full-dataset average: that would incorporate future outcomes.
    Missing inputs are handled natively by the score estimator inside each fit.
    """
    required = {"player_id", "season", "week", "gameday", "gametime", "qbr", "passer_rating", "player_name"}
    missing = sorted(required - set(starting_qbs.columns))
    if missing:
        raise ValueError(f"Quarterback history is missing columns: {missing}")
    output = starting_qbs.copy()
    output["__kickoff"] = parse_eastern_kickoffs(
        output["gameday"].astype(str) + "-" + output["gametime"].astype(str)
    )
    if output["__kickoff"].isna().any():
        raise ValueError("Quarterback history has invalid kickoff times")
    known = output.loc[output["player_id"].notna()]
    if known.duplicated(["player_id", "__kickoff"]).any():
        raise ValueError("A quarterback cannot have multiple starts at the same kickoff")
    output = output.sort_values(["__kickoff", "player_id"], kind="stable").reset_index(drop=True)
    for metric in ("passer_rating", "qbr"):
        shifted = f"{metric}_shifted"
        output[shifted] = output.groupby("player_id")[metric].shift()
        output[f"ewma_{metric}"] = output.groupby("player_id")[shifted].transform(
            lambda values: values.ewm(min_periods=1, span=10).mean()
        )
    output["player_name"] = output.groupby("player_id")["player_name"].shift()
    return output.drop(columns=["__kickoff", "gameday", "gametime"])


def build_nfl_opponent_adjusted_metrics(
    pbp: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    config: OpponentAdjustmentConfig | Mapping[str, object] | None = None,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the current NFL EPA/success features with opponent adjustment.

    The return values retain the legacy column names consumed by the notebook,
    recipe, score model, and confidence model.  Only the values change.
    """
    required = {
        "game_id", "season", "week", "posteam", "defteam", "rush_attempt",
        "pass_attempt", "epa", "down", "yards_gained", "ydstogo",
    }
    missing = sorted(required - set(pbp.columns))
    if missing:
        raise ValueError(f"NFL opponent adjustment missing PBP columns: {', '.join(missing)}")
    schedule_required = {"game_id", "season", "week", "home_team", "away_team", "gameday", "gametime"}
    schedule_missing = sorted(schedule_required - set(schedule.columns))
    if schedule_missing:
        raise ValueError(
            "NFL opponent adjustment missing schedule columns: " + ", ".join(schedule_missing)
        )
    games = schedule.loc[:, list(schedule_required)].copy()
    games["start_date"] = parse_eastern_kickoffs(
        games["gameday"].astype(str) + "-" + games["gametime"].astype(str)
    )
    games = games.drop(columns=["gameday", "gametime"])

    source = pbp.copy()
    source["custom_success"] = (
        ((source["down"] == 1) & (source["yards_gained"] >= 0.4 * source["ydstogo"]))
        | ((source["down"] == 2) & (source["yards_gained"] >= 0.6 * source["ydstogo"]))
        | ((source["down"].isin([3, 4])) & (source["yards_gained"] >= source["ydstogo"]))
    )
    observations = []
    for key, mask, value in (
        ("rush_epa", source["rush_attempt"] == 1, "epa"),
        ("pass_epa", source["pass_attempt"] == 1, "epa"),
        ("rush_success", source["rush_attempt"] == 1, "custom_success"),
        ("pass_success", source["pass_attempt"] == 1, "custom_success"),
    ):
        grouped = source.loc[mask].groupby(["game_id", "posteam"], as_index=False)[value].mean()
        grouped = grouped.rename(columns={"posteam": "team", value: "value"})
        grouped["metric"] = key
        observations.append(grouped)
    metrics = build_opponent_adjusted_team_metrics(
        games,
        pd.concat(observations, ignore_index=True),
        (
            OpponentMetricSpec("rush_epa", "epa_rushing_offense", "epa_rushing_defense", "dynamic"),
            OpponentMetricSpec("pass_epa", "epa_passing_offense", "epa_passing_defense", "dynamic"),
            OpponentMetricSpec("rush_success", "success_rate_rushing_offense", "success_rate_rushing_defense", "static"),
            OpponentMetricSpec("pass_success", "success_rate_passing_offense", "success_rate_passing_defense", "static"),
        ),
        config=resolve_opponent_adjustment_config(config, defaults=DEFAULT_NFL_OPPONENT_ADJUSTMENT),
        strict=strict,
    )
    metrics = metrics.rename(columns={
        "epa_rushing_offense_shifted": "epa_shifted_rushing_offense",
        "epa_rushing_defense_shifted": "epa_shifted_rushing_defense",
        "epa_passing_offense_shifted": "epa_shifted_passing_offense",
        "epa_passing_defense_shifted": "epa_shifted_passing_defense",
        "epa_rushing_offense_ewma": "ewma_rushing_offense",
        "epa_rushing_defense_ewma": "ewma_rushing_defense",
        "epa_passing_offense_ewma": "ewma_passing_offense",
        "epa_passing_defense_ewma": "ewma_passing_defense",
        "epa_rushing_offense_ewma_dynamic_window": "ewma_dynamic_window_rushing_offense",
        "epa_rushing_defense_ewma_dynamic_window": "ewma_dynamic_window_rushing_defense",
        "epa_passing_offense_ewma_dynamic_window": "ewma_dynamic_window_passing_offense",
        "epa_passing_defense_ewma_dynamic_window": "ewma_dynamic_window_passing_defense",
    })
    epa_columns = [
        "game_id", "season", "week", "team",
        *[
            column
            for column in metrics.columns
            if column.startswith("epa_") or column.startswith("ewma_rushing_")
            or column.startswith("ewma_passing_") or column.startswith("ewma_dynamic_window_")
        ],
    ]
    success = metrics.loc[:, [
        "game_id", "season", "week", "team",
        *[column for column in metrics.columns if column.startswith("success_rate_")],
    ]].copy()
    success = success.rename(columns={
        "success_rate_rushing_offense": "custom_success_rushing_offense",
        "success_rate_rushing_defense": "custom_success_rushing_defense",
        "success_rate_passing_offense": "custom_success_passing_offense",
        "success_rate_passing_defense": "custom_success_passing_defense",
        "success_rate_rushing_offense_shifted": "success_shifted_rushing_offense",
        "success_rate_rushing_defense_shifted": "success_shifted_rushing_defense",
        "success_rate_passing_offense_shifted": "success_shifted_passing_offense",
        "success_rate_passing_defense_shifted": "success_shifted_passing_defense",
        "success_rate_rushing_offense_ewma": "ewma_success_rate_rushing_offense",
        "success_rate_rushing_defense_ewma": "ewma_success_rate_rushing_defense",
        "success_rate_passing_offense_ewma": "ewma_success_rate_passing_offense",
        "success_rate_passing_defense_ewma": "ewma_success_rate_passing_defense",
    })
    return metrics.loc[:, epa_columns], success


__all__ = ["DEFAULT_NFL_OPPONENT_ADJUSTMENT", "build_nfl_opponent_adjusted_metrics", "calculate_nfl_passer_rating"]
