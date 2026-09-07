"""Unfinished CFB games stay visible without supplying final scores or features."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.model_patterns.expected_points.tracking import grade_completed_picks
from src.model_patterns.expected_points.trainer import _build_train_df
from src.sports.football.cfb.expected_points.completion import (
    cfb_completed_game_mask,
    mask_incomplete_cfb_scores,
)
from src.sports.football.cfb.expected_points.features import (
    build_cfb_efficiency_schedule,
    build_opponent_adjusted_advanced_stats,
    build_weekly_pregame_elo,
)
from src.sports.football.cfb.expected_points.utils import prepare_cfb_expected_points_df
from src.sports.football.cfb.expected_points.recipe import CFBExpectedPointsRecipe


def _schedule():
    return pd.DataFrame([
        dict(game_id="finished", season=2025, week=1, start_date="2025-09-01T17:00Z",
             home_team="A", away_team="B", home_points=21., away_points=14., completed=True),
        dict(game_id="suspended", season=2025, week=2, start_date="2025-09-08T17:00Z",
             home_team="A", away_team="C", home_points=70., away_points=3., completed=False),
        dict(game_id="future", season=2025, week=3, start_date="2025-09-15T17:00Z",
             home_team="A", away_team="B", home_points=np.nan, away_points=np.nan, completed=False),
    ]).assign(home_classification="fbs", away_classification="fbs")


def test_completion_flags_and_csv_status_preserve_source_precedence():
    frame = pd.DataFrame({
        "completed": [True, False, None, None, None, "false"],
        "status": [None, "completed", "completed", "scheduled", None, "completed"],
    })
    assert cfb_completed_game_mask(frame).tolist() == [True, False, True, False, False, False]


def test_unfinished_scores_are_masked_without_removing_rows_or_mutating_input():
    frame = _schedule()
    original = frame.copy(deep=True)
    result = mask_incomplete_cfb_scores(frame)
    assert result.game_id.tolist() == frame.game_id.tolist()
    assert result.loc[0, "home_points"] == 21.
    assert result.loc[1:, ["home_points", "away_points"]].isna().all().all()
    assert_frame_equal(frame, original)
    assert_frame_equal(mask_incomplete_cfb_scores(result), result)


def test_legacy_frame_without_final_status_is_unchanged():
    frame = _schedule().drop(columns="completed")
    assert cfb_completed_game_mask(frame).all()
    assert_frame_equal(mask_incomplete_cfb_scores(frame), frame)


def test_elo_ignores_partial_scores_even_after_availability_lag():
    frame = _schedule()
    changed = frame.copy()
    changed.loc[1, ["home_points", "away_points"]] = [0., 100.]
    before = build_weekly_pregame_elo(frame)
    after = build_weekly_pregame_elo(changed)
    assert_frame_equal(before, after)
    assert before.loc[1, "home_pregame_elo"] == before.loc[2, "home_pregame_elo"]
    assert before.loc[1, ["home_postgame_elo", "away_postgame_elo"]].isna().all()


def test_model_frame_and_grading_cannot_admit_unfinished_scores():
    frame = prepare_cfb_expected_points_df(_schedule())
    assert len(frame) == 3
    assert frame.loc[frame.game_id.eq("suspended"), ["home_score", "away_score"]].isna().all().all()
    picks = frame[["game_id", "home_team", "away_team"]].assign(
        spread_line=-3.5, spread_play="A", total_line=44.5, total_play="over",
    )
    graded = grade_completed_picks(picks, pd.DataFrame(), frame)
    assert graded.game_id.tolist() == ["finished"]

    feature_values = {
        f"{side}_{metric}_ewma_dynamic_window_{venue}": 0.
        for side in ("offense", "defense")
        for metric in ("explosiveness", "ppa", "success_rate")
        for venue in ("home", "away")
    }
    frame = frame.assign(**feature_values)
    config = CFBExpectedPointsRecipe().build_config(
        frame, season=2025, week=3, prediction_now=pd.Timestamp("2025-09-15T16:00Z"),
    )
    assert _build_train_df(frame, config).game_id.tolist() == ["finished"]


def test_unfinished_advanced_stats_cannot_change_opponent_features():
    schedule = _schedule()
    source = pd.DataFrame([
        dict(game_id="finished", season=2025, week=1, team="A", start_date=schedule.iloc[0].start_date, offense_ppa=.2),
        dict(game_id="finished", season=2025, week=1, team="B", start_date=schedule.iloc[0].start_date, offense_ppa=.1),
        dict(game_id="suspended", season=2025, week=2, team="A", start_date=schedule.iloc[1].start_date, offense_ppa=100.),
        dict(game_id="suspended", season=2025, week=2, team="C", start_date=schedule.iloc[1].start_date, offense_ppa=200.),
    ])
    before = build_opponent_adjusted_advanced_stats(source, schedule, metrics=("ppa",))
    changed = source.copy()
    changed.loc[changed.game_id.eq("suspended"), "offense_ppa"] *= -100
    after = build_opponent_adjusted_advanced_stats(changed, schedule, metrics=("ppa",))
    assert_frame_equal(before, after)
    assert set(before.game_id) == set(schedule.game_id)
    assert before.loc[before.game_id.eq("future") & before.team.eq("A"), "offense_ppa_ewma_dynamic_window"].notna().all()


def test_efficiency_warmup_retains_csv_completion_after_mixed_sources():
    target = _schedule()
    history = target.iloc[:2].copy().assign(season=2024, completed=None, status=["completed", "suspended"])
    history["game_id"] = ["old_finished", "old_suspended"]
    result = build_cfb_efficiency_schedule(target, history, history_start_season=2024)
    indexed = result.set_index("game_id")
    assert bool(indexed.loc["old_finished", "completed"])
    assert not bool(indexed.loc["old_suspended", "completed"])
    assert bool(indexed.loc["finished", "completed"])


@pytest.mark.parametrize("legacy_input", ["history", "schedule"])
def test_efficiency_schedule_preserves_legacy_completion_before_concatenation(legacy_input):
    common = dict(week=1, home_team="A", away_team="B", home_classification="fbs", away_classification="fbs")
    history = pd.DataFrame([
        dict(common, game_id="old_finished", season=2024, start_date="2024-09-01T12:00Z", status="completed"),
        dict(common, game_id="old_suspended", season=2024, start_date="2024-09-02T12:00Z", status="suspended"),
    ])
    schedule = pd.DataFrame([
        dict(common, game_id="new_finished", season=2025, start_date="2025-09-01T12:00Z", completed=True),
        dict(common, game_id="new_suspended", season=2025, start_date="2025-09-02T12:00Z", completed=False),
    ])
    if legacy_input == "history":
        history = history.drop(columns="status")
    else:
        schedule = schedule.drop(columns="completed")
    history_original, schedule_original = history.copy(deep=True), schedule.copy(deep=True)
    expected = cfb_completed_game_mask(history).tolist() + cfb_completed_game_mask(schedule).tolist()

    result = build_cfb_efficiency_schedule(schedule, history, history_start_season=2024)

    assert result.game_id.tolist() == history.game_id.tolist() + schedule.game_id.tolist()
    assert cfb_completed_game_mask(result).tolist() == expected
    pd.testing.assert_frame_equal(history, history_original)
    pd.testing.assert_frame_equal(schedule, schedule_original)
