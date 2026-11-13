"""Research estimators preserve rotation, isolation and an exact ridge fallback."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from scripts.experiment_score_models import (
    MatchupTransformer, ResearchRecipe, RidgeTreeBlend, matchup_spec, research_factory,
)
from scripts.experiment_residual_score_tree import RidgeResidualTree, residual_factory
from scripts.score_research_robustness import paired_year_sensitivity
from src.model_patterns.expected_points.pipeline import HomeAwayTransformer, make_score_pipeline
from src.sports.football.nfl.expected_points.recipe import NFLExpectedPointsRecipe


def test_matchup_rotates_before_combining_and_preserves_missing():
    features = [f"ewma_{side}_{venue}" for side in ("offense", "defense")
                for venue in ("home", "away")]
    frame = pd.DataFrame({"ewma_offense_home": [1., np.nan], "ewma_offense_away": [3., 3.],
                          "ewma_defense_home": [5., 5.], "ewma_defense_away": [7., 7.],
                          "pred_team": ["away", "home"]})
    selected, pairs, elo = matchup_spec(features)
    result = MatchupTransformer(pairs, elo).transform(HomeAwayTransformer().transform(frame))
    assert result[selected[0]].iloc[0] == 4.
    assert np.isnan(result[selected[0]].iloc[1])
    assert frame.ewma_offense_home.iloc[0] == 1.


def test_blend_zero_is_exact_ridge_and_does_not_fit_tree():
    frame = pd.DataFrame({"metric": np.arange(60.) / 10, "context": np.sin(np.arange(60.)),
                          "pred_team": ["home", "away"] * 30})
    features = ["metric", "context", "pred_team"]
    y = 20 + frame.metric + frame.context ** 2
    blend = clone(RidgeTreeBlend(features, ["pred_team"], ["metric"],
                                alpha=10., team_weight=.25, tree_weight=0.)).fit(frame, y)
    ridge = make_score_pipeline(features, ["pred_team"], score_estimator="ridge",
                                team_features=["metric"]).set_params(
        ridge__alpha=10., columntransformer__transformer_weights={"team": .25}).fit(frame, y)
    np.testing.assert_array_equal(blend.predict(frame), ridge.predict(frame))
    assert blend.tree_ is None


def test_residual_tree_can_make_negative_corrections_and_has_exact_zero_fallback():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"x": rng.uniform(-3, 3, 1200)})
    y = 20 + frame.x + 3 * (frame.x ** 2 - 3)
    residual = RidgeResidualTree(["x"], [], [], alpha=10., tree_weight=.5).fit(frame, y)
    zero = RidgeResidualTree(["x"], [], [], alpha=10., tree_weight=0.).fit(frame, y)
    adjustment = residual.predict(frame) - zero.predict(frame)
    assert adjustment.min() < -.1 and adjustment.max() > .1
    np.testing.assert_array_equal(zero.predict(frame), zero.ridge_.predict(frame))
    assert zero.tree_ is None


def test_year_sensitivity_preserves_pairing_and_direction():
    base = pd.DataFrame({"season": [2023, 2023, 2024, 2024], "week": [1, 2, 1, 2],
                         "game_id": [1, 2, 3, 4], "home_score": [20.] * 4,
                         "away_score": [10.] * 4, "home_score_pred": [22.] * 4,
                         "away_score_pred": [12.] * 4})
    candidate = base.assign(home_score_pred=21., away_score_pred=11.)
    result = paired_year_sensitivity(base, candidate, samples=100)
    assert result["improvement"] == 1.
    assert result["hierarchical_95_ci"] == [1., 1.]
    assert set(result["leave_one_season_out_improvement"].values()) == {1.}
    changing = base.assign(home_score_pred=[20., 20., 24., 24.],
                           away_score_pred=[10., 10., 14., 14.])
    sensitivity = paired_year_sensitivity(base, changing, samples=100)
    assert sensitivity["improvement"] == 0.
    assert sensitivity["hierarchical_95_ci"] == [-2., 2.]
    assert sensitivity["leave_one_season_out_improvement"] == {"2023": -2., "2024": 2.}
    with pytest.raises(ValueError, match="paired games"):
        paired_year_sensitivity(base, candidate.iloc[:3], samples=100)


@pytest.mark.parametrize("variant", ["tree", "compact", "blend", "residual"])
def test_research_models_ignore_unselected_future_and_book_columns(variant):
    rng = np.random.default_rng(3)
    features = ["ewma_offense_home", "ewma_offense_away", "ewma_defense_home",
                "ewma_defense_away", "pred_team"]
    frame = pd.DataFrame(rng.normal(size=(450, 4)), columns=features[:4])
    frame["pred_team"] = ["home", "away"] * 225
    factory = residual_factory if variant == "residual" else research_factory
    model = factory(features, ["pred_team"], score_estimator="research-" + variant,
                    team_features=features[:4])
    if variant == "blend":
        model.set_params(blend__tree_weight=.25)
    if variant == "residual":
        model.set_params(residual__tree_weight=.25)
    model.fit(frame, 20 + frame.ewma_offense_home)
    quotes = frame.assign(implied_points_home=1000., home_score=999., future_result=-100.)
    np.testing.assert_array_equal(model.predict(quotes), model.predict(frame))
    np.testing.assert_array_equal(clone(model).fit(frame, 20 + frame.ewma_offense_home).predict(frame),
                                  model.predict(frame))
