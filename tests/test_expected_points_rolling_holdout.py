import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from types import SimpleNamespace

from src.model_patterns.expected_points.trainer import _walk_forward_score_holdout
from src.model_patterns.expected_points.types import ExpectedPointsConfig


class MeanScoreEstimator(BaseEstimator):
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict_scores(self, X, team):
        return np.full(len(X), self.mean_)


def _inputs():
    frame = pd.DataFrame({
        "game_id": np.arange(8), "season": 2023, "week": np.repeat([1, 2, 3, 4], 2),
        "date_time": [f"2023-09-{day:02d}-{hour}:00" for day in (1, 8, 15, 22) for hour in (13, 16)],
        "home_team": "A", "away_team": "B", "home_score": np.arange(8) + 20,
        "away_score": np.arange(8) + 10, "signal_home": 1.0, "signal_away": 2.0,
        "spread_line": -3.0, "total_line": 40.0,
    })
    config = ExpectedPointsConfig(
        current_year=2023, current_week=5, targets=["home_score", "away_score"],
        features=["signal_home", "signal_away"], input_features=["signal_home", "signal_away"],
        cat_features=[], spread_class_features=[], total_class_features=[],
    )
    return frame, config


def test_rolling_holdout_uses_only_earlier_available_labels_and_keeps_games_together():
    frame, config = _inputs()
    estimator = MeanScoreEstimator()
    result = _walk_forward_score_holdout(frame, frame.iloc[4:], config, best_estimator=estimator)
    assert result.index.tolist() == [4, 5, 6, 7]
    np.testing.assert_allclose(result.home_score_pred, [16.5, 16.5, 17.5, 17.5])
    assert result.score_training_rows.tolist() == [4, 4, 6, 6]
    changed = frame.copy()
    changed.loc[6:, ["home_score", "away_score"]] += 1000
    unchanged_earlier = _walk_forward_score_holdout(changed, changed.iloc[4:], config, best_estimator=estimator)
    np.testing.assert_array_equal(result.home_score_pred, unchanged_earlier.home_score_pred)
    changed.loc[4:5, ["home_score", "away_score"]] += 100
    later = _walk_forward_score_holdout(changed, changed.iloc[4:], config, best_estimator=estimator)
    np.testing.assert_array_equal(result.home_score_pred.iloc[:2], later.home_score_pred.iloc[:2])
    assert (later.home_score_pred.iloc[2:] > result.home_score_pred.iloc[2:]).all()
    assert not hasattr(estimator, "mean_")


def test_rolling_holdout_excludes_a_recent_but_unavailable_result():
    frame, config = _inputs()
    recent = frame.iloc[[0]].assign(game_id=9, date_time="2023-09-15-12:00", home_score=1000, away_score=1000)
    expanded = pd.concat([frame, recent], ignore_index=True)
    result = _walk_forward_score_holdout(expanded, expanded.iloc[4:8], config, best_estimator=MeanScoreEstimator())
    assert result.iloc[0].score_training_rows == 4
    assert result.iloc[0].home_score_pred == 16.5
    assert result.iloc[2].score_training_rows == 7


def test_rolling_holdout_keeps_one_parameter_search_and_original_market_inputs(monkeypatch):
    from src.model_patterns.expected_points import trainer

    frame, config = _inputs()
    config.score_holdout_method = "walk_forward"
    searches = []

    def tune_once(*args, **kwargs):
        searches.append(args[1].index.tolist())
        stale = frame.loc[args[2].index].assign(spread_line=1000.0)
        return stale, SimpleNamespace(best_estimator_=MeanScoreEstimator())

    monkeypatch.setattr(trainer, "fit_eval", tune_once)
    monkeypatch.setattr(trainer, "fit_configured_classifiers", lambda *args: (None, None))
    monkeypatch.setattr(trainer, "safe_confidence_health_evaluation", lambda *args: (pd.DataFrame(), {}))
    model, _, _, results, _, _ = trainer._fit_expected_points(frame, config)
    assert len(searches) == 1
    assert max(searches[0]) < min(results.index)
    assert results.spread_line.eq(-3.0).all()
    assert model.mean_ == np.mean(frame[["home_score", "away_score"]].to_numpy())
