import pandas as pd
import pytest

from src.sports.football.transforms import (
    OpponentAdjustmentConfig,
    OpponentMetricSpec,
    build_opponent_adjusted_team_metrics,
)
from src.sports.football.nfl.expected_points.features import (
    build_nfl_opponent_adjusted_metrics,
)


SPEC = OpponentMetricSpec("efficiency", "off_efficiency", "def_efficiency", "static")


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["1", "2", "3", "4", "5"],
            "season": [2025] * 5,
            "week": [1, 1, 2, 2, 3],
            "home_team": ["A", "C", "B", "C", "A"],
            "away_team": ["B", "B", "A", "A", "C"],
            "start_date": [
                "2025-09-01T12:00:00Z",
                "2025-09-01T13:00:00Z",
                "2025-09-08T12:00:00Z",
                "2025-09-08T13:00:00Z",
                "2025-09-15T12:00:00Z",
            ],
        }
    )


def _observations() -> pd.DataFrame:
    # B allows two strong offensive performances; A's performance against B
    # should therefore receive less credit than its raw value of ten.
    return pd.DataFrame(
        {
            "game_id": ["1", "2", "3", "4"],
            "team": ["A", "C", "B", "C"],
            "metric": ["efficiency"] * 4,
            "value": [10.0, 10.0, 0.0, 0.0],
        }
    )


def test_opponent_adjustment_reduces_credit_against_permissive_defense():
    frame = build_opponent_adjusted_team_metrics(_schedule(), _observations(), [SPEC])
    target = frame.loc[(frame["game_id"] == "5") & (frame["team"] == "A")].iloc[0]

    assert target["off_efficiency_ewma"] < 10.0
    assert target["off_efficiency_ewma"] == pytest.approx(8.658536585, abs=1e-8)


def test_future_observation_cannot_change_an_earlier_target():
    schedule = _schedule()
    baseline = build_opponent_adjusted_team_metrics(schedule, _observations(), [SPEC])
    future_schedule = pd.concat(
        [
            schedule,
            pd.DataFrame(
                {
                    "game_id": ["6"], "season": [2025], "week": [4],
                    "home_team": ["B"], "away_team": ["C"],
                    "start_date": ["2025-09-22T12:00:00Z"],
                }
            ),
        ],
        ignore_index=True,
    )
    future_observations = pd.concat(
        [
            _observations(),
            pd.DataFrame({"game_id": ["6"], "team": ["B"], "metric": ["efficiency"], "value": [999.0]}),
        ],
        ignore_index=True,
    )
    with_future = build_opponent_adjusted_team_metrics(future_schedule, future_observations, [SPEC])

    old = baseline.loc[(baseline["game_id"] == "5") & (baseline["team"] == "A"), "off_efficiency_ewma"].iloc[0]
    new = with_future.loc[(with_future["game_id"] == "5") & (with_future["team"] == "A"), "off_efficiency_ewma"].iloc[0]
    assert new == pytest.approx(old)


def test_fcs_pooled_policy_is_a_valid_configurable_alternative():
    schedule = _schedule().assign(
        home_classification=["fbs", "fbs", "fcs", "fbs", "fbs"],
        away_classification=["fcs", "fcs", "fbs", "fcs", "fbs"],
    )
    frame = build_opponent_adjusted_team_metrics(
        schedule,
        _observations(),
        [SPEC],
        config=OpponentAdjustmentConfig(fcs_policy="pooled"),
    )
    assert frame.loc[(frame["game_id"] == "5") & (frame["team"] == "A"), "off_efficiency_ewma"].notna().all()


def test_config_mapping_supports_notebook_parameterization():
    frame = build_opponent_adjusted_team_metrics(
        _schedule(),
        _observations(),
        [SPEC],
        config={"ridge_alpha": 10.0, "season_carryover": 0.25},
    )

    value = frame.loc[
        (frame["game_id"] == "5") & (frame["team"] == "A"), "off_efficiency_ewma"
    ].iloc[0]
    assert pd.notna(value)


def test_nfl_adapter_keeps_the_legacy_model_feature_names():
    schedule = _schedule().rename(columns={"start_date": "kickoff"})
    schedule["gameday"] = pd.to_datetime(schedule["kickoff"]).dt.strftime("%Y-%m-%d")
    schedule["gametime"] = pd.to_datetime(schedule["kickoff"]).dt.strftime("%H:%M")
    pbp = pd.DataFrame(
        {
            "game_id": ["1", "2", "3", "4"],
            "season": [2025] * 4,
            "week": [1, 1, 2, 2],
            "posteam": ["A", "C", "B", "C"],
            "defteam": ["B", "B", "A", "A"],
            "rush_attempt": [1, 1, 1, 1],
            "pass_attempt": [0, 0, 0, 0],
            "epa": [0.2, 0.1, 0.0, -0.1],
            "down": [1, 1, 1, 1],
            "yards_gained": [5, 5, 0, 0],
            "ydstogo": [10, 10, 10, 10],
        }
    )

    epa, success = build_nfl_opponent_adjusted_metrics(pbp, schedule)

    assert "ewma_dynamic_window_rushing_offense" in epa
    assert "ewma_success_rate_rushing_offense" in success
