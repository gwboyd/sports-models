import numpy as np
import pandas as pd
import pytest

from src.model_patterns.expected_points.betting import determine_plays, scores_to_bets
from src.model_patterns.expected_points.trainer import _build_train_df, run_expected_points
from src.model_patterns.expected_points.types import ExpectedPointsConfig, PlayThresholds
from src.sports.football.transforms import build_lagged_team_metrics


def test_build_lagged_team_metrics_generates_all_requested_columns():
    df = pd.DataFrame(
        {
            "team": ["A", "A", "A", "B", "B", "B"],
            "game_id": ["3", "1", "2", "6", "4", "5"],
            "week": [1, 2, 3, 1, 2, 3],
            "start_date": [
                "2024-09-15T17:00:00Z",
                "2024-09-01T17:00:00Z",
                "2024-09-08T17:00:00Z",
                "2024-09-15T20:00:00Z",
                "2024-09-01T20:00:00Z",
                "2024-09-08T20:00:00Z",
            ],
            "col1": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "col2": [2.0, 3.0, 4.0, 4.0, 3.0, 2.0],
        }
    )

    out = build_lagged_team_metrics(df, ["col1", "col2"])

    assert "col1_ewma_dynamic_window" in out.columns
    assert "col2_ewma_dynamic_window" in out.columns
    assert out.loc[(out["team"] == "A") & (out["game_id"] == "1"), "col1_shifted"].isna().all()
    assert out.loc[(out["team"] == "A") & (out["game_id"] == "2"), "col1_shifted"].item() == 2.0


def test_build_lagged_team_metrics_rejects_duplicate_team_games():
    frame = pd.DataFrame(
        {
            "team": ["A", "A"],
            "game_id": ["1", "1"],
            "week": [1, 1],
            "start_date": ["2024-09-01T17:00:00Z"] * 2,
            "metric": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate key"):
        build_lagged_team_metrics(frame, ["metric"])


def test_build_lagged_team_metrics_accepts_league_neutral_column_names():
    frame = pd.DataFrame(
        {
            "club": ["A", "A"],
            "event_id": ["1", "2"],
            "round": [1, 2],
            "kickoff": ["2026-09-01T17:00:00Z", "2026-09-08T17:00:00Z"],
            "efficiency": [1.0, 3.0],
        }
    )

    output = build_lagged_team_metrics(
        frame,
        ["efficiency"],
        group_col="club",
        game_col="event_id",
        period_col="round",
        time_col="kickoff",
    )

    assert output.loc[output["event_id"] == "2", "efficiency_shifted"].item() == 1.0


def test_determine_plays_thresholds_apply():
    df = pd.DataFrame(
        {
            "game_id": ["1", "2"],
            "spread_pred": [3.0, -1.0],
            "spread_line": [0.0, -0.5],
            "total_pred": [49.0, 44.0],
            "total_line": [45.0, 43.5],
            "spread_win_prob": [60.0, 50.0],
            "total_win_prob": [57.0, 51.0],
        }
    )

    out = determine_plays(df, thresholds=PlayThresholds())
    assert out.loc[0, "spread_lock"] == 1
    assert out.loc[0, "total_lock"] == 1


def test_train_frame_excludes_rows_missing_any_score_target():
    config = ExpectedPointsConfig(
        current_year=2026,
        current_week=2,
        targets=["home_score", "away_score"],
        features=[],
        input_features=[],
        spread_class_features=[],
        total_class_features=[],
        cat_features=[],
    )
    df = pd.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025, 2026],
            "week": [1, 2, 3, 4, 1],
            "home_score": ["21", "17", np.nan, "not-a-score", "24"],
            "away_score": ["14", np.nan, "10", "7", "20"],
        }
    )

    train_df = _build_train_df(df, config)

    assert train_df.index.tolist() == [0, 4]
    assert train_df["home_score"].dtype.kind == "f"
    assert train_df["away_score"].dtype.kind == "f"


def test_run_expected_points_end_to_end():
    rng = np.random.default_rng(42)
    rows = []

    teams = ["A", "B", "C", "D"]
    for week in range(1, 9):
        for i in range(0, len(teams), 2):
            home = teams[i]
            away = teams[i + 1]
            spread_line = float(rng.normal(0, 5))
            total_line = float(rng.normal(45, 5))
            home_score = float(rng.normal(24, 7))
            away_score = float(rng.normal(21, 7))
            rows.append(
                {
                    "season": 2024,
                    "week": week,
                    "game_id": f"2024_{week}_{i}",
                    "date_time": f"2024-01-{week * 3:02d}-{13 + i}:00",
                    "home_team": home,
                    "away_team": away,
                    "weekday": "Sunday",
                    "pred_team": "undefined",
                    "metric_home": float(rng.normal(0, 1)),
                    "metric_away": float(rng.normal(0, 1)),
                    "spread_line": spread_line,
                    "total_line": total_line,
                    "moneyline_home": float(rng.normal(-120, 80)),
                    "moneyline_away": float(rng.normal(100, 80)),
                    "spread_odds_home": -110.0,
                    "spread_odds_away": -110.0,
                    "over_odds": -110.0,
                    "rest_home": 7.0,
                    "rest_away": 7.0,
                    "div_game": False,
                    "implied_points_home": (total_line / 2) + (spread_line / 2),
                    "implied_points_away": (total_line / 2) - (spread_line / 2),
                    "home_score": home_score,
                    "away_score": away_score,
                }
            )

    # week to predict
    rows.append(
        {
            "season": 2024,
            "week": 9,
            "game_id": "2024_9_0",
            "date_time": "2024-02-28-13:00",
            "home_team": "A",
            "away_team": "B",
            "weekday": "Sunday",
            "pred_team": "undefined",
            "metric_home": 0.5,
            "metric_away": -0.2,
            "spread_line": -2.5,
            "total_line": 46.5,
            "moneyline_home": -130.0,
            "moneyline_away": 110.0,
            "spread_odds_home": -110.0,
            "spread_odds_away": -110.0,
            "over_odds": -110.0,
            "rest_home": 7.0,
            "rest_away": 7.0,
            "div_game": False,
            "implied_points_home": 22.0,
            "implied_points_away": 24.5,
            "home_score": np.nan,
            "away_score": np.nan,
        }
    )

    df = pd.DataFrame(rows)

    features = [
        "rest_away",
        "rest_home",
        "div_game",
        "implied_points_home",
        "implied_points_away",
        "pred_team",
        "weekday",
        "metric_home",
        "metric_away",
        "moneyline_home",
        "spread_line",
        "spread_odds_home",
        "total_line",
        "over_odds",
    ]
    input_features = features + ["moneyline_away", "spread_odds_away"]

    config = ExpectedPointsConfig(
        current_year=2024,
        current_week=9,
        targets=["home_score", "away_score"],
        features=features,
        input_features=input_features,
        spread_class_features=[
            "metric_home",
            "metric_away",
            "moneyline_home",
            "spread_line",
            "total_line",
            "spread_diff",
        ],
        total_class_features=[
            "metric_home",
            "metric_away",
            "moneyline_home",
            "spread_line",
            "total_line",
            "total_diff",
        ],
        cat_features=["pred_team", "weekday"],
        score_param_grid={
            "lgbmregressor__n_estimators": [20],
            "lgbmregressor__max_depth": [4],
            "lgbmregressor__learning_rate": [0.1],
        },
        confidence_param_grid={
            "n_estimators": [20],
            "max_depth": [3],
        },
        score_n_jobs=1,
        confidence_n_jobs=1,
        prediction_now=pd.Timestamp("2024-02-27T12:00:00Z"),
    )

    result = run_expected_points(df, config)
    assert len(result.plays) == 1
    assert "spread_win_prob" in result.plays.columns
    assert "total_win_prob" in result.plays.columns
    assert result.metrics["train_rows"] > 0
    assert result.eval_results["week"].min() == 7


def test_run_expected_points_excludes_games_at_or_after_kickoff():
    # The full model behavior is covered above; this boundary helper is tested
    # directly in the kickoff tests to avoid another expensive model fit.
    from src.sports.football.kickoff import exclude_started_games

    games = pd.DataFrame(
        {
            "game_id": ["started", "future"],
            "date_time": ["2026-09-05-13:00", "2026-09-05-16:00"],
        }
    )
    eligible = exclude_started_games(games, now=pd.Timestamp("2026-09-05T18:00:00Z"))
    assert eligible["game_id"].tolist() == ["future"]
