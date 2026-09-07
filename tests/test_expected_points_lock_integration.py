from dataclasses import replace

import pandas as pd
import pytest

from src.model_patterns.expected_points.confidence import fit_configured_classifiers
from src.model_patterns.expected_points.inspection import inspect_game
from src.model_patterns.expected_points.tracking import prepare_tracking_run
from src.model_patterns.expected_points.types import (
    ExpectedPointsConfig,
    ExpectedPointsLeague,
    ExpectedPointsTrackingConfig,
    LockHeadConfig,
    PlayThresholds,
)


def _config() -> ExpectedPointsConfig:
    return ExpectedPointsConfig(
        current_year=2026,
        current_week=1,
        targets=["home_score", "away_score"],
        features=[],
        input_features=[],
        spread_class_features=[],
        total_class_features=[],
        cat_features=[],
        league=ExpectedPointsLeague.NFL,
        spread_lock_head=LockHeadConfig("symmetric_residual", locks_enabled=True),
        total_lock_head=LockHeadConfig("symmetric_residual", locks_enabled=True),
        play_thresholds=PlayThresholds(max_combined_plays=3),
    )


def _prediction() -> pd.DataFrame:
    return pd.DataFrame([{
        "season": 2026, "week": 1, "year_week": "2026_1", "game_id": "game",
        "home_team": "A", "away_team": "B", "date_time": "2026-09-13-13:00",
        "home_score_pred": 27., "away_score_pred": 20.,
        "spread_pred": -7., "spread_line": -3.5, "spread_diff": 3.5,
        "spread_play": "A", "spread_win_prob": 54., "spread_lock": 0,
        "total_pred": 47., "total_line": 44.5, "total_diff": 2.5,
        "total_play": "over", "total_win_prob": 53., "total_lock": 0,
        "spread_locks_enabled": False, "total_locks_enabled": True,
    }])


def test_lock_inspection_retains_exact_edges_and_runtime_readiness_after_tracking():
    config = _config()
    prediction = _prediction()
    tracked = prepare_tracking_run(
        prediction, pd.DataFrame(), "2026_1",
        ExpectedPointsTrackingConfig(play_thresholds=config.play_thresholds),
        now=pd.Timestamp("2026-09-12T12:00:00Z"),
    ).picks
    assert "spread_diff" not in tracked
    before = tracked.copy(deep=True)

    result = inspect_game(prediction, config, game_id="game", plays=tracked)

    assert result.spread_confidence_features["edge"].item() == 3.5
    assert result.total_confidence_features["edge"].item() == 2.5
    assert not result.spread_confidence_features["locks_enabled"].item()
    assert result.spread_confidence_features["locks_configured"].item()
    assert result.total_confidence_features["locks_enabled"].item()
    pd.testing.assert_frame_equal(tracked, before)


def test_lock_inspection_does_not_infer_readiness_from_configuration():
    prediction = _prediction().drop(columns=["spread_locks_enabled", "total_locks_enabled"])
    result = inspect_game(prediction, _config(), game_id="game", plays=prediction)
    assert pd.isna(result.spread_confidence_features["locks_enabled"].item())
    assert pd.isna(result.total_confidence_features["locks_enabled"].item())


@pytest.mark.parametrize("missing_head", ["spread_lock_head", "total_lock_head"])
def test_partial_lock_configuration_fails_instead_of_silently_training_legacy(missing_head):
    config = replace(_config(), **{missing_head: None})
    with pytest.raises(ValueError, match="both spread and total Lock heads"):
        fit_configured_classifiers(pd.DataFrame(), config)


def test_legacy_configuration_still_dispatches_to_legacy_classifiers(monkeypatch):
    expected = (object(), object())
    calls = []

    def fit_legacy(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr("src.model_patterns.expected_points.confidence.fit_classifiers", fit_legacy)
    config = replace(_config(), spread_lock_head=None, total_lock_head=None)
    assert fit_configured_classifiers(pd.DataFrame(), config) is expected
    assert len(calls) == 1


def test_configured_heads_disable_locks_when_history_is_not_ready():
    history = _prediction().assign(home_score=28., away_score=20., spread_win=1., total_win=1.)
    spread, total = fit_configured_classifiers(history, _config())
    assert not spread.locks_enabled
    assert not total.locks_enabled
    probability, push = spread.predict_outcomes(_prediction())
    assert 0.5 < probability[0] < 0.525
    assert push[0] == 0.
