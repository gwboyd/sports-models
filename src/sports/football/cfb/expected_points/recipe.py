"""Reproducible CFB expected-points recipe configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from src.model_patterns.expected_points.types import (
    ExpectedPointsConfig,
    ExpectedPointsLeague,
    LockHeadConfig,
    PlayThresholds,
)
from src.sports.football.cfb.data_validation import validate_expected_points_frame
from src.sports.football.cfb.expected_points.betting_lines import scores_to_cfb_bets
from src.sports.football.cfb.expected_points.utils import prepare_cfb_expected_points_df
from src.sports.football.cfb.expected_points.features import CFB_EFFICIENCY_METRICS, DEFAULT_CFB_EFFICIENCY_METRICS


@dataclass(frozen=True)
class CFBModelFrameStages:
    joined_features: pd.DataFrame
    model_frame: pd.DataFrame


def assemble_cfb_model_frame(
    *,
    schedule: pd.DataFrame,
    epa: pd.DataFrame,
    game_stats: pd.DataFrame,
    market_lines: pd.DataFrame,
    strict: bool = True,
) -> CFBModelFrameStages:
    """Assemble and validate the versioned CFB model frame."""
    epa_for_merge = epa.drop(columns=["opponent"], errors="ignore")
    game_stats_for_merge = game_stats.drop(columns=["opponent"], errors="ignore")
    joined = (
        schedule
        .merge(
            epa_for_merge.rename(columns={"team": "home_team"}),
            on=["home_team", "game_id", "season", "week"],
            how="left", validate="one_to_one",
        )
        .merge(
            epa_for_merge.rename(columns={"team": "away_team"}),
            on=["away_team", "game_id", "season", "week"],
            how="left", suffixes=("_home", "_away"), validate="one_to_one",
        )
        .merge(
            game_stats_for_merge.rename(columns={"team": "home_team"}),
            on=["home_team", "game_id", "season", "week"],
            how="left", validate="one_to_one",
        )
        .merge(
            game_stats_for_merge.rename(columns={"team": "away_team"}),
            on=["away_team", "game_id", "season", "week"],
            how="left", suffixes=("_home", "_away"), validate="one_to_one",
        )
        .merge(market_lines, on="id", how="left", validate="one_to_one")
    )
    frame = joined.dropna(
        subset=["home_team", "spread_reference_line", "total_reference_line"]
    )
    frame = prepare_cfb_expected_points_df(frame)
    frame["pred_team"] = "undefined"
    validate_expected_points_frame(frame, strict=strict)
    return CFBModelFrameStages(joined_features=joined, model_frame=frame)


@dataclass(frozen=True)
class CFBExpectedPointsRecipe:
    name: str = "cfb_expected_points"
    version: str = "working-tree"
    protocol_version: str = "2"
    league: ExpectedPointsLeague = ExpectedPointsLeague.CFB
    efficiency_metrics: tuple[str, ...] = DEFAULT_CFB_EFFICIENCY_METRICS

    def __post_init__(self) -> None:
        if (not self.efficiency_metrics or len(set(self.efficiency_metrics)) != len(self.efficiency_metrics)
                or set(self.efficiency_metrics) - set(CFB_EFFICIENCY_METRICS)):
            raise ValueError("CFB efficiency_metrics must be unique registered metrics")

    def prepare_frame(
        self,
        sources: Mapping[str, pd.DataFrame],
        *,
        strict: bool = True,
    ) -> pd.DataFrame:
        required = {"schedule", "epa", "game_stats", "market_lines"}
        missing = sorted(required - set(sources))
        if missing:
            raise ValueError(f"CFB recipe sources are missing: {', '.join(missing)}")
        return assemble_cfb_model_frame(
            schedule=sources["schedule"],
            epa=sources["epa"],
            game_stats=sources["game_stats"],
            market_lines=sources["market_lines"],
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
        # Select only registered model inputs; diagnostic/future columns must
        # never become features through substring matching or missing columns.
        ewma_features = [
            f"{side}_{metric}_ewma_dynamic_window_{venue}"
            for metric in self.efficiency_metrics
            for venue in ("home", "away")
            for side in ("offense", "defense")
        ]
        missing = sorted(set(ewma_features) - set(frame.columns))
        if missing:
            raise ValueError(f"CFB selected efficiency features are missing: {missing}")
        cat_features = [column for column in ("weekday",) if column in frame.columns]
        other_features = [
            column
            for column in (
                "conference_game", "implied_points_home", "implied_points_away",
                "home_pregame_elo", "away_pregame_elo",
            )
            if column in frame.columns
        ]
        features = other_features + cat_features + ewma_features
        confidence_market_features = [
            "spread_reference_line", "total_reference_line", "spread_line", "total_line",
        ]
        return ExpectedPointsConfig(
            current_year=int(season),
            current_week=int(week),
            targets=["home_score", "away_score"],
            features=features,
            input_features=features,
            spread_class_features=features + confidence_market_features + ["spread_diff"],
            total_class_features=features + confidence_market_features + ["total_diff"],
            cat_features=cat_features,
            spread_class_cat_features=cat_features,
            total_class_cat_features=cat_features,
            home_prediction_features=features,
            away_prediction_features=features,
            score_n_jobs=1,
            confidence_n_jobs=1,
            confidence_scoring="neg_log_loss",
            betting_transform=scores_to_cfb_bets,
            prediction_now=prediction_now,
            league=self.league,
            spread_lock_head=LockHeadConfig(
                family="symmetric_residual",
                parameters={"trust": 0.2},
                locks_enabled=True,
            ),
            total_lock_head=LockHeadConfig(family="base_rate", locks_enabled=False),
            play_thresholds=PlayThresholds(
                max_spreads_plays=None,
                max_total_plays=0,
                min_spread_diff=0.0,
                min_spread_win_prob=52.5,
                min_total_diff=0.0,
                max_combined_plays=3,
                extra_lock_min_probability=0.55,
            ),
        )


__all__ = ["CFBExpectedPointsRecipe", "CFBModelFrameStages", "assemble_cfb_model_frame"]
