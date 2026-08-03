from datetime import datetime, timezone

import pandas as pd
import pytest

from src.model_patterns.expected_points.tracking import (
    build_pick_records,
    build_result_records,
    grade_completed_picks,
    prepare_tracking_run,
    serialize_pick_frame,
    validate_pick_frame,
)
from src.model_patterns.expected_points.types import ExpectedPointsTrackingConfig


def pick(game_id="1", *, date_time="2099-09-01-17:00", spread_play="A", spread_lock=1):
    return {
        "season": 2026,
        "week": 1,
        "year_week": "2026_1",
        "game_id": game_id,
        "home_team": "A",
        "away_team": "B",
        "home_score_pred": 27.0,
        "away_score_pred": 20.0,
        "spread_pred": -7.0,
        "spread_line": -3.5,
        "spread_play": spread_play,
        "spread_win_prob": 61.0,
        "spread_lock": spread_lock,
        "total_pred": 47.0,
        "total_line": 44.5,
        "total_play": "over",
        "total_win_prob": 58.0,
        "total_lock": 1,
        "date_time": date_time,
    }


def test_first_tracking_run_counts_new_games_and_builds_records():
    predicted = pd.DataFrame([pick("1"), pick("2")])
    run = prepare_tracking_run(
        predicted,
        pd.DataFrame(),
        "2026_1",
        ExpectedPointsTrackingConfig(),
        now=pd.Timestamp("2026-08-01T00:00:00Z"),
    )

    assert run.pick_changes_games == ["1", "2"]
    assert run.play_changes_games == ["1", "2"]
    assert set(run.differences["source"]) == {"predictions"}
    assert all(value is not None for value in serialize_pick_frame(run.differences)[0].values())
    records = build_pick_records(run.picks, datetime(2026, 8, 1, tzinfo=timezone.utc))
    assert records[0]["week"] == "1"
    assert records[0]["write_time"].tzinfo is timezone.utc


def test_locked_pick_is_preserved():
    existing = pick(date_time="2026-08-01-00:15", spread_play="B", spread_lock=1)
    existing["write_time"] = "2026-07-31 00:00:00"
    predicted = pick(date_time="2026-08-01-00:15", spread_play="A", spread_lock=0)
    run = prepare_tracking_run(
        pd.DataFrame([predicted]),
        pd.DataFrame([existing]),
        "2026_1",
        ExpectedPointsTrackingConfig(),
        now=pd.Timestamp("2026-08-01T00:00:00Z"),
    )

    assert run.locked_game_ids == ["1"]
    assert run.picks.iloc[0]["spread_play"] == "B"
    assert run.picks.iloc[0]["spread_lock"] == 1


def test_cfb_metadata_is_preserved_through_tracking_and_grading():
    metadata_columns = ("home_conference", "away_conference")
    predicted = pick()
    predicted.update({"home_conference": "SEC", "away_conference": "Big Ten"})
    config = ExpectedPointsTrackingConfig(pick_metadata_columns=metadata_columns)

    run = prepare_tracking_run(
        pd.DataFrame([predicted]),
        pd.DataFrame(),
        "2026_1",
        config,
        now=pd.Timestamp("2026-08-01T00:00:00Z"),
    )
    pick_records = build_pick_records(
        run.picks,
        datetime(2026, 8, 1, tzinfo=timezone.utc),
        metadata_columns,
    )
    scores = pd.DataFrame([{"game_id": "1", "home_score": 24, "away_score": 21}])
    graded = grade_completed_picks(run.picks, pd.DataFrame(), scores)
    result_records = build_result_records(graded, metadata_columns)

    assert pick_records[0]["home_conference"] == "SEC"
    assert pick_records[0]["away_conference"] == "Big Ten"
    assert result_records[0]["home_conference"] == "SEC"
    assert result_records[0]["away_conference"] == "Big Ten"


def test_diff_snapshots_replace_missing_write_times_with_null():
    existing = pick(date_time="2099-08-01-00:15", spread_play="B")
    existing["write_time"] = "2026-07-31 00:00:00"
    run = prepare_tracking_run(
        pd.DataFrame([pick(date_time="2099-08-01-00:15", spread_play="A")]),
        pd.DataFrame([existing]),
        "2026_1",
        ExpectedPointsTrackingConfig(),
        now=pd.Timestamp("2026-08-01T00:00:00Z"),
    )
    serialized = serialize_pick_frame(run.differences)
    predicted = next(record for record in serialized if record["source"] == "predictions")
    assert predicted["write_time"] is None


def test_tracking_ignores_duplicate_auxiliary_model_columns():
    predicted = pd.concat(
        [
            pd.DataFrame([pick(spread_play="A")]),
            pd.DataFrame({"joined_feature": [1.0]}),
            pd.DataFrame({"joined_feature": [2.0]}),
        ],
        axis=1,
    )
    existing = pick(spread_play="B")
    existing["write_time"] = "2026-07-31 00:00:00"

    run = prepare_tracking_run(
        predicted,
        pd.DataFrame([existing]),
        "2026_1",
        ExpectedPointsTrackingConfig(),
        now=pd.Timestamp("2026-08-01T00:00:00Z"),
    )

    assert "joined_feature" not in run.picks.columns
    assert set(run.differences["source"]) == {"database", "predictions"}


def test_validation_rejects_duplicate_and_missing_games():
    with pytest.raises(ValueError, match="duplicate"):
        validate_pick_frame(pd.DataFrame([pick(), pick()]))
    with pytest.raises(ValueError, match="do not match"):
        validate_pick_frame(pd.DataFrame([pick()]), expected_game_ids=["1", "2"])


def test_grade_completed_picks_only_returns_ungraded_games():
    existing = pd.DataFrame([pick("1"), pick("2")])
    existing["write_time"] = "2026-08-01 00:00:00"
    prior = pd.DataFrame([{"game_id": "1"}])
    scores = pd.DataFrame(
        [
            {"game_id": "1", "home_score": 28, "away_score": 17},
            {"game_id": "2", "home_score": 24, "away_score": 21},
        ]
    )

    graded = grade_completed_picks(existing, prior, scores)
    assert graded["game_id"].tolist() == ["2"]
    assert graded.iloc[0]["true_spread"] == -3
    assert len(build_result_records(graded)) == 1
