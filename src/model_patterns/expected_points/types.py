from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class PlayThresholds:
    max_spreads_plays: int = 5
    max_total_plays: int = 5
    min_spread_diff: float = 0.5
    min_total_diff: float = 0.5
    min_spread_win_prob: float = 55.0
    min_total_win_prob: float = 55.0


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
    season_col: str = "season"
    week_col: str = "week"
    split_strategy: str = "random"
    test_size: float = 0.2
    random_state: int = 42
    score_cv: int = 2
    confidence_cv: int = 5
    score_n_jobs: int = -1
    confidence_n_jobs: int = -1
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
            "min_samples_split": [2, 5, 10],
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
