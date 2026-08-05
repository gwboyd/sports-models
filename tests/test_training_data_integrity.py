import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from src.model_patterns.expected_points.chronology import (
    chronological_train_test_split,
    predefined_chronological_split,
)
from src.model_patterns.expected_points.trainer import (
    _duplicate_predefined_split,
    _fit_final_score_model,
)
from src.sports.data_validation import DataContractError, validate_frame
from src.sports.data_validation import validate_feature_coverage
from src.sports.football.kickoff import (
    exclude_started_games,
    parse_eastern_kickoffs,
)
from src.sports.football.cfb.expected_points.features import build_pregame_advanced_stats
from src.sports.football.cfb.expected_points.utils import (
    cfbd_kickoffs_to_eastern_strings,
)
from src.sports.football.nfl.expected_points.features import calculate_nfl_passer_rating
from src.sports.football.nfl.expected_points.utils import (
    nflverse_kickoffs_to_eastern_strings,
)


def test_chronological_splits_put_every_validation_game_after_training_games():
    frame = pd.DataFrame(
        {
            "game_id": ["late", "first", "middle", "last"],
            "date_time": [
                "2026-09-20-13:00",
                "2026-09-01-20:00",
                "2026-09-10-13:00",
                "2026-09-21-20:00",
            ],
        },
        index=[4, 1, 3, 8],
    )

    train, test = chronological_train_test_split(frame, time_col="date_time", test_size=0.5)
    assert train["game_id"].tolist() == ["first", "middle"]
    assert test["game_id"].tolist() == ["late", "last"]
    assert parse_eastern_kickoffs(train["date_time"]).max() < parse_eastern_kickoffs(
        test["date_time"]
    ).min()


def test_score_split_keeps_home_and_away_representations_together():
    games = pd.DataFrame(
        {
            "game_id": ["1", "2", "3", "4"],
            "date_time": [f"2026-09-0{day}-13:00" for day in range(1, 5)],
        }
    )
    game_split = predefined_chronological_split(games, time_col="date_time", test_size=0.5)
    duplicated = _duplicate_predefined_split(game_split).test_fold

    np.testing.assert_array_equal(duplicated[: len(games)], duplicated[len(games) :])
    np.testing.assert_array_equal(duplicated, [-1, -1, 0, 0, -1, -1, 0, 0])


def test_chronological_split_does_not_separate_simultaneous_games():
    games = pd.DataFrame(
        {
            "game_id": ["1", "2", "3", "4", "5"],
            "date_time": [
                "2026-09-01-13:00",
                "2026-09-02-13:00",
                "2026-09-03-13:00",
                "2026-09-03-13:00",
                "2026-09-03-13:00",
            ],
        }
    )
    train, test = chronological_train_test_split(games, time_col="date_time", test_size=0.2)
    assert train["game_id"].tolist() == ["1", "2"]
    assert test["game_id"].tolist() == ["3", "4", "5"]


def test_final_score_refit_uses_all_games_without_reusing_search_object():
    class RecordingEstimator(BaseEstimator):
        def fit(self, X, y):
            self.fit_rows_ = len(X)
            self.pred_teams_ = set(X["pred_team"])
            return self

    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y = pd.DataFrame({"home_score": [21, 24, 17], "away_score": [14, 20, 10]})
    selected = RecordingEstimator()

    final = _fit_final_score_model(selected, X, y)

    assert final is not selected
    assert final.fit_rows_ == 2 * len(X)
    assert final.pred_teams_ == {"home", "away"}


def test_football_kickoff_contract_converts_cfb_utc_to_eastern_and_respects_dst():
    converted = cfbd_kickoffs_to_eastern_strings(
        pd.Series(["2026-09-05T17:00:00Z", "2026-01-04T18:00:00Z"])
    )
    assert converted.tolist() == ["2026-09-05-13:00", "2026-01-04-13:00"]

    games = pd.DataFrame(
        {
            "game_id": ["started", "future"],
            "date_time": ["2026-09-05-13:00", "2026-09-05-13:01"],
        }
    )
    eligible = exclude_started_games(games, now=pd.Timestamp("2026-09-05T17:00:00Z"))
    assert eligible["game_id"].tolist() == ["future"]


def test_nflverse_kickoff_adapter_preserves_eastern_wall_time():
    converted = nflverse_kickoffs_to_eastern_strings(
        pd.Series(["2026-09-05", "2026-01-04"]),
        pd.Series(["13:00", "13:00"]),
    )
    assert converted.tolist() == ["2026-09-05-13:00", "2026-01-04-13:00"]


def test_official_nfl_passer_rating_is_bounded_and_uses_attempts():
    assert calculate_nfl_passer_rating(0, 0, 0, 0, 0) == 0.0
    assert calculate_nfl_passer_rating(10, 10, 500, 10, 0) == pytest.approx(158.3333333)
    assert calculate_nfl_passer_rating(10, 0, -100, 0, 10) == 0.0
    assert calculate_nfl_passer_rating(35, 20, 250, 2, 1) == pytest.approx(86.6071429)


def test_cfb_upcoming_game_uses_only_prior_advanced_stats():
    advanced = pd.DataFrame(
        {
            "game_id": ["1", "2"],
            "season": [2026, 2026],
            "week": [1, 2],
            "team": ["A", "A"],
            "start_date": ["2026-09-01T17:00:00Z", "2026-09-08T17:00:00Z"],
            "explosiveness": [1.0, 3.0],
        }
    )
    schedule = pd.DataFrame(
        {
            "game_id": ["1", "2", "3"],
            "season": [2026, 2026, 2026],
            "week": [1, 2, 3],
            "home_team": ["A", "B", "A"],
            "away_team": ["B", "A", "B"],
            "start_date": [
                "2026-09-01T17:00:00Z",
                "2026-09-08T17:00:00Z",
                "2026-09-15T17:00:00Z",
            ],
        }
    )

    features = build_pregame_advanced_stats(advanced, schedule, ["explosiveness"])
    upcoming = features.loc[
        (features["game_id"] == "3") & (features["team"] == "A")
    ].iloc[0]

    assert pd.isna(upcoming["explosiveness"])
    assert upcoming["explosiveness_shifted"] == 3.0
    assert upcoming["explosiveness_ewma_dynamic_window"] == pytest.approx(2.1)


def test_data_contract_strict_mode_raises_and_warning_mode_reports(caplog):
    duplicate = pd.DataFrame({"game_id": ["1", "1"], "season": [2025, 2025]})
    with pytest.raises(DataContractError, match="duplicate key"):
        validate_frame(
            duplicate,
            label="test feed",
            unique_keys=(("game_id",),),
            strict=True,
        )

    with caplog.at_level(logging.WARNING):
        valid = validate_frame(
            duplicate,
            label="test feed",
            unique_keys=(("game_id",),),
            strict=False,
        )
    assert valid is False
    assert "Data contract warning" in caplog.text


def test_prediction_feature_contract_rejects_silent_all_null_imputation():
    frame = pd.DataFrame(
        {
            "season": [2026, 2026],
            "week": [1, 1],
            "feature_a": [1.0, np.nan],
            "feature_b": [np.nan, np.nan],
        }
    )
    with pytest.raises(DataContractError, match="all monitored features null"):
        validate_feature_coverage(
            frame,
            label="prediction frame",
            feature_columns=("feature_a", "feature_b"),
            prediction_mask=pd.Series([True, True]),
        )
