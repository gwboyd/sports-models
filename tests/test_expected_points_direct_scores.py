"""Direct-score fits can remove sportsbook inputs without changing their target."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src.model_patterns.expected_points.pipeline import make_score_pipeline
from src.model_patterns.expected_points.trainer import _fit_final_score_model
from src.sports.football.cfb.expected_points.recipe import CFBExpectedPointsRecipe
from src.sports.football.nfl.expected_points.recipe import NFLExpectedPointsRecipe


def _frame(league):
    rng = np.random.default_rng(41)
    n = 40
    if league == "nfl":
        bases = [
            "ewma_dynamic_window_rushing_offense", "ewma_dynamic_window_passing_offense",
            "ewma_dynamic_window_rushing_defense", "ewma_dynamic_window_passing_defense",
            "ewma_success_rate_rushing_offense", "ewma_success_rate_passing_offense",
            "ewma_success_rate_rushing_defense", "ewma_success_rate_passing_defense",
        ]
        columns = [f"{base}_{side}" for base in bases for side in ("home", "away")]
        recipe_type = NFLExpectedPointsRecipe
    else:
        recipe_type = CFBExpectedPointsRecipe
        columns = [f"{side}_{metric}_ewma_dynamic_window_{venue}"
                   for metric in recipe_type().efficiency_metrics
                   for venue in ("home", "away") for side in ("offense", "defense")]
        columns += ["home_pregame_elo", "away_pregame_elo"]
    frame = pd.DataFrame(rng.normal(size=(n, len(columns))), columns=columns)
    frame = frame.assign(weekday="Sunday", pred_team="undefined", rest_home=7., rest_away=7.,
                         div_game=0., conference_game=0.)
    return frame, recipe_type


@pytest.mark.parametrize("league", ["nfl", "cfb"])
def test_analytics_only_fit_requires_no_market_columns_and_ignores_extra_quotes(league):
    frame, recipe_type = _frame(league)
    config = recipe_type(include_sportsbook_features=False).build_config(frame, season=2025, week=1)
    assert config.score_target == "points"
    assert all(not any(word in column for word in ("implied", "moneyline", "odds", "spread_line", "total_line"))
               for column in config.input_features)
    targets = pd.DataFrame({"home_score": 24. + frame.iloc[:, 0], "away_score": 19. + frame.iloc[:, 1]})
    pipeline = make_score_pipeline(config.features, config.cat_features,
                                   score_target=config.score_target, score_estimator=config.score_estimator,
                                   team_features=config.score_team_features)
    model = _fit_final_score_model(pipeline, frame[config.input_features], targets)
    arbitrary_quotes = frame.assign(implied_points_home=9999., implied_points_away=-9999.,
                                    spread_line=123., total_line=456., moneyline_home=100000.)
    for team in ("home", "away"):
        expected = model.predict_scores(frame, team)
        np.testing.assert_allclose(model.predict_scores(arbitrary_quotes, team), expected)
        assert np.isfinite(expected).all() and (expected >= 0).all()
    assert not np.allclose(model.predict_scores(frame, "home"), model.predict_scores(frame, "away"))


def test_direct_target_is_actual_points_without_a_fixed_market_offset():
    frame = pd.DataFrame({"pred_team": ["home", "away"], "implied_points_home": [20., 40.]})
    model = make_score_pipeline(["implied_points_home"], [], score_target="points")
    model.steps[-1] = ("lgbmregressor", DummyRegressor(strategy="mean"))
    model.fit(frame, [10., 30.])
    np.testing.assert_allclose(model.predict(frame), [20., 20.])
    np.testing.assert_allclose(model.predict(frame.assign(implied_points_home=1000.)), [20., 20.])


def test_direct_scores_cannot_be_negative():
    frame = pd.DataFrame({"metric": [0., 1.]})
    model = make_score_pipeline(["metric"], [], score_target="points")
    model.steps[-1] = ("lgbmregressor", DummyRegressor(strategy="constant", constant=-2.))
    model.fit(frame, [0., 3.])
    np.testing.assert_allclose(model.predict_scores(frame, "home"), [0., 0.])
