from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd


class ExpectedPointsLeague(str, Enum):
    NFL = "nfl"
    CFB = "cfb"


@dataclass
class PlayThresholds:
    max_spreads_plays: int = 5
    max_total_plays: int = 5
    min_spread_diff: float = 0.5
    min_total_diff: float = 0.5
    min_spread_win_prob: float = 55.0
    min_total_win_prob: float = 55.0


@dataclass
class ExpectedPointsTrackingConfig:
    lock_started_games: bool = True
    lock_window_minutes: int = 30
    play_thresholds: PlayThresholds = field(default_factory=PlayThresholds)
    pick_metadata_columns: Tuple[str, ...] = ()


@dataclass
class ExpectedPointsConfig:
    current_year: int
    current_week: int
    targets: List[str]
    features: List[str]
    input_features: List[str]
    spread_class_features: List[str]
    total_class_features: List[str]
    cat_features: List[str]
    spread_class_cat_features: List[str] = field(default_factory=list)
    total_class_cat_features: List[str] = field(default_factory=list)
    season_col: str = "season"
    week_col: str = "week"
    split_strategy: str = "chronological"
    time_col: str = "date_time"
    test_size: float = 0.2
    inner_validation_size: float = 0.2
    confidence_validation_size: float = 0.2
    score_n_jobs: int = -1
    confidence_n_jobs: int = -1
    prediction_now: Optional[pd.Timestamp] = None
    score_param_grid: Dict[str, List] = field(
        default_factory=lambda: {
            "lgbmregressor__n_estimators": [300, 400],
            "lgbmregressor__max_depth": [8, 12],
            "lgbmregressor__learning_rate": [0.05, 0.1],
        }
    )
    confidence_param_grid: Dict[str, List] = field(
        default_factory=lambda: {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
        }
    )
    play_thresholds: PlayThresholds = field(default_factory=PlayThresholds)
    home_prediction_features: Optional[List[str]] = None
    away_prediction_features: Optional[List[str]] = None


@dataclass
class ExpectedPointsRunResult:
    score_model: object
    spread_clf: object
    total_clf: object
    eval_results: pd.DataFrame
    this_week: pd.DataFrame
    plays: pd.DataFrame
    train_df: pd.DataFrame
    metrics: Dict[str, float]
