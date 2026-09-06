"""Reproducible NFL expected-points recipe configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from src.model_patterns.expected_points.types import (
    ExpectedPointsConfig,
    ExpectedPointsLeague,
)
from src.sports.football.nfl.data_validation import validate_expected_points_frame
from src.sports.football.nfl.expected_points.utils import (
    nflverse_kickoffs_to_eastern_strings,
)
from src.sports.football.nfl.expected_points.features import build_nfl_opponent_adjusted_metrics
from src.sports.football.transforms import OpponentAdjustmentConfig


@dataclass(frozen=True)
class NFLModelFrameStages:
    scores: pd.DataFrame
    schedule_scores: pd.DataFrame
    model_frame: pd.DataFrame


def assemble_nfl_model_frame(
    *,
    pbp: pd.DataFrame,
    schedule: pd.DataFrame,
    epa: pd.DataFrame,
    success: pd.DataFrame,
    starting_qbs: pd.DataFrame,
    opponent_adjustment_config: OpponentAdjustmentConfig | Mapping[str, object] | None = None,
    strict: bool = True,
) -> NFLModelFrameStages:
    """Assemble the versioned NFL model frame while retaining notebook stages."""
    # Existing callers may still provide the pre-v2, raw intermediate frames.
    # Regenerate them when that happens; the v2 notebook supplies the adjusted
    # frames directly so the comparatively expensive ratings are fit only once.
    required_epa = {"game_id", "team", "ewma_dynamic_window_rushing_offense"}
    required_success = {"game_id", "team", "ewma_success_rate_rushing_offense"}
    if not required_epa.issubset(epa.columns) or not required_success.issubset(success.columns):
        epa, success = build_nfl_opponent_adjusted_metrics(
            pbp, schedule, config=opponent_adjustment_config, strict=strict,
        )
    scores = (
        pbp[["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(home_team_win=lambda value: (value.home_score > value.away_score).astype(int))
    )
    schedule_scores = schedule.merge(
        scores[["game_id", "home_score", "away_score"]],
        on="game_id", how="left", validate="one_to_one",
    )
    frame = (
        schedule_scores
        .merge(
            epa.rename(columns={"team": "home_team"}),
            on=["home_team", "game_id", "season", "week"], how="left", validate="one_to_one",
        )
        .merge(
            epa.rename(columns={"team": "away_team"}),
            on=["away_team", "game_id", "season", "week"], how="left",
            suffixes=("_home", "_away"), validate="one_to_one",
        )
        .merge(
            success.rename(columns={"team": "home_team"}),
            on=["home_team", "game_id", "season", "week"], how="left", validate="one_to_one",
        )
        .merge(
            success.rename(columns={"team": "away_team"}),
            on=["away_team", "game_id", "season", "week"], how="left",
            suffixes=("_home", "_away"), validate="one_to_one",
        )
        .merge(
            starting_qbs.rename(columns={"team": "home_team"}),
            on=["home_team", "season", "week"], how="left", validate="one_to_one",
        )
        .merge(
            starting_qbs.rename(columns={"team": "away_team"}),
            on=["away_team", "season", "week"], how="left",
            suffixes=("_home", "_away"), validate="one_to_one",
        )
        .rename(columns={
            "home_rest": "rest_home",
            "away_rest": "rest_away",
            "home_moneyline": "moneyline_home",
            "away_moneyline": "moneyline_away",
            "home_spread_odds": "spread_odds_home",
            "away_spread_odds": "spread_odds_away",
        })
    )
    frame = frame.loc[~((frame["season"] == 2010) & (frame["week"] == 1))].copy()
    frame["spread_line"] = frame["spread_line"] * -1
    frame["year_week"] = frame["season"].astype(str) + "_" + frame["week"].astype(str)
    frame["date_time"] = nflverse_kickoffs_to_eastern_strings(frame["gameday"], frame["gametime"])
    frame["pred_team"] = "undefined"
    validate_expected_points_frame(frame, strict=strict)
    return NFLModelFrameStages(scores=scores, schedule_scores=schedule_scores, model_frame=frame)


@dataclass(frozen=True)
class NFLExpectedPointsRecipe:
    name: str = "nfl_expected_points"
    version: str = "working-tree"
    protocol_version: str = "2"
    league: ExpectedPointsLeague = ExpectedPointsLeague.NFL
    opponent_adjustment_config: OpponentAdjustmentConfig = OpponentAdjustmentConfig()

    def prepare_frame(
        self,
        sources: Mapping[str, pd.DataFrame],
        *,
        strict: bool = True,
    ) -> pd.DataFrame:
        required = {"pbp", "schedule", "epa", "success", "starting_qbs"}
        missing = sorted(required - set(sources))
        if missing:
            raise ValueError(f"NFL recipe sources are missing: {', '.join(missing)}")
        return assemble_nfl_model_frame(
            pbp=sources["pbp"],
            schedule=sources["schedule"],
            epa=sources["epa"],
            success=sources["success"],
            starting_qbs=sources["starting_qbs"],
            opponent_adjustment_config=self.opponent_adjustment_config,
            strict=strict,
        ).model_frame

    def build_config(
        self,
        frame: pd.DataFrame,
        *,
        season: int,
        week: int,
        prediction_now: pd.Timestamp | None = None,
    ) -> ExpectedPointsConfig:
        # These names intentionally match the pre-v2 recipe.  Their values are
        # now opponent-adjusted, but downstream score/confidence schemas remain
        # stable and raw diagnostic columns cannot slip into a model input.
        ewma_bases = (
            "ewma_dynamic_window_rushing_offense",
            "ewma_dynamic_window_passing_offense",
            "ewma_dynamic_window_rushing_defense",
            "ewma_dynamic_window_passing_defense",
            "ewma_success_rate_rushing_offense",
            "ewma_success_rate_passing_offense",
            "ewma_success_rate_rushing_defense",
            "ewma_success_rate_passing_defense",
        )
        ewma_features = [
            feature
            for base in ewma_bases
            if f"{base}_home" in frame and f"{base}_away" in frame
            for feature in (f"{base}_home", f"{base}_away")
        ]
        # Keep protocol-1 frame inspection/backtest fixtures readable.  A v2
        # prepared frame always contains the explicit names above.
        if not ewma_features:
            ewma_features = [
                column for column in frame.columns if "ewma" in column and "dynamic" in column
            ] + [
                column for column in frame.columns if "ewma" in column and "success_rate" in column
            ]
        cat_features = ["roof", "weekday"]
        betting_features = [
            "moneyline_home", "spread_line", "spread_odds_home", "total_line", "over_odds",
        ]
        other_features = [
            "rest_away", "rest_home", "div_game", "implied_points_home",
            "implied_points_away", "ewma_qbr_home", "ewma_qbr_away",
        ]
        features = other_features + cat_features + ewma_features + betting_features
        input_features = features + ["moneyline_away", "spread_odds_away"]
        return ExpectedPointsConfig(
            current_year=int(season),
            current_week=int(week),
            targets=["home_score", "away_score"],
            features=features,
            input_features=input_features,
            spread_class_features=ewma_features + betting_features + other_features + ["spread_diff"],
            total_class_features=ewma_features + betting_features + other_features + ["total_diff"],
            cat_features=cat_features,
            spread_class_cat_features=cat_features,
            total_class_cat_features=cat_features,
            home_prediction_features=features,
            away_prediction_features=input_features,
            prediction_now=prediction_now,
        )


__all__ = ["NFLExpectedPointsRecipe", "NFLModelFrameStages", "assemble_nfl_model_frame"]
