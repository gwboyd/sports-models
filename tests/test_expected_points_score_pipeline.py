import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import PredefinedSplit, cross_val_score

from src.model_patterns.expected_points.pipeline import HomeAwayTransformer, make_score_pipeline


def test_nfl_recipe_excludes_game_observed_starters_and_roof_status():
    from src.sports.football.nfl.expected_points.recipe import NFLExpectedPointsRecipe

    bases = [
        "ewma_dynamic_window_rushing_offense", "ewma_dynamic_window_passing_offense",
        "ewma_dynamic_window_rushing_defense", "ewma_dynamic_window_passing_defense",
        "ewma_success_rate_rushing_offense", "ewma_success_rate_passing_offense",
        "ewma_success_rate_rushing_defense", "ewma_success_rate_passing_defense",
    ]
    frame = pd.DataFrame({f"{base}_{side}": [0.0] for base in bases for side in ("home", "away")})
    frame = frame.assign(ewma_qbr_home=150.0, ewma_qbr_away=20.0, roof="closed")
    config = NFLExpectedPointsRecipe().build_config(frame, season=2025, week=1)
    observed_only = {"ewma_qbr_home", "ewma_qbr_away", "home_qb_id", "away_qb_id", "roof"}
    for columns in [
        config.features, config.input_features, config.home_prediction_features,
        config.away_prediction_features, config.spread_class_features,
        config.total_class_features, config.score_team_features,
    ]:
        assert observed_only.isdisjoint(columns)


def test_away_score_inputs_follow_the_predicted_team_for_both_naming_conventions():
    frame = pd.DataFrame(
        {
            "pred_team": ["home", "away"],
            "home_pregame_elo": [1600.0, 1700.0],
            "away_pregame_elo": [1400.0, 1300.0],
            "implied_points_home": [28.0, 31.0],
            "implied_points_away": [21.0, 17.0],
            "spread_line": [-7.0, -14.0],
            "neutral_site": [False, True],
            "home_unpaired_context": [1.0, 2.0],
        }
    )
    original = frame.copy(deep=True)

    result = HomeAwayTransformer().transform(frame)

    assert result.loc[1, "home_pregame_elo"] == 1300.0
    assert result.loc[1, "away_pregame_elo"] == 1700.0
    assert result.loc[1, "implied_points_home"] == 17.0
    assert result.loc[1, "implied_points_away"] == 31.0
    assert result.loc[1, "spread_line"] == 14.0
    assert_frame_equal(result.loc[[0]], original.loc[[0]])
    assert result["neutral_site"].equals(original["neutral_site"])
    assert result["home_unpaired_context"].equals(original["home_unpaired_context"])
    assert_frame_equal(frame, original)


def test_team_orientation_is_an_involution_for_paired_inputs():
    frame = pd.DataFrame(
        {
            "pred_team": ["away"],
            "home_pregame_elo": [1600.0],
            "away_pregame_elo": [1400.0],
            "efficiency_home": [0.2],
            "efficiency_away": [-0.1],
            "spread_line": [-3.5],
        }
    )
    transformer = HomeAwayTransformer()
    assert_frame_equal(transformer.transform(transformer.transform(frame)), frame)


def _market_residual_pipeline():
    pipeline = make_score_pipeline(
        ["implied_points_home", "implied_points_away"], [], score_target="market_residual",
    )
    pipeline.steps[-1] = ("lgbmregressor", DummyRegressor(strategy="mean"))
    return pipeline


def test_market_residual_predictions_restore_each_teams_own_market_baseline():
    X = pd.DataFrame({
        "pred_team": ["home", "away", "home", "away"],
        "implied_points_home": [24., 24., 30., 30.],
        "implied_points_away": [17., 17., 21., 21.],
    })
    y = np.array([27., 20., 33., 24.])  # Every team scores three above its own baseline.
    original = X.copy(deep=True)
    pipeline = clone(_market_residual_pipeline()).fit(X, y)
    np.testing.assert_allclose(pipeline.predict(X), y)
    np.testing.assert_allclose(pipeline.predict_scores(X, "home"), X.implied_points_home + 3)
    np.testing.assert_allclose(pipeline.predict_scores(X, "away"), X.implied_points_away + 3)
    assert pipeline.score(X, y) == pytest.approx(1.0)
    assert_frame_equal(X, original)


def test_market_residual_grid_scoring_evaluates_restored_scores():
    X = pd.DataFrame({
        "pred_team": ["home", "away", "home", "away"],
        "implied_points_home": [20., 20., 35., 35.],
        "implied_points_away": [14., 14., 28., 28.],
    })
    y = np.array([22., 16., 37., 30.])
    scores = cross_val_score(
        _market_residual_pipeline(), X, y,
        cv=PredefinedSplit([-1, -1, 0, 0]), scoring="neg_mean_squared_error",
    )
    np.testing.assert_allclose(scores, [0.])


def test_market_residual_prediction_is_nonnegative_and_requires_valid_market_inputs():
    X = pd.DataFrame({"pred_team": ["home", "away"], "implied_points_home": [30., 30.], "implied_points_away": [30., 30.]})
    pipeline = _market_residual_pipeline().fit(X, [0., 0.])
    low_market = X.assign(implied_points_home=10., implied_points_away=10.)
    np.testing.assert_allclose(pipeline.predict(low_market), [0., 0.])
    with pytest.raises(ValueError, match="market"):
        pipeline.predict(low_market.assign(implied_points_home=np.nan))


def test_linear_score_odds_transform_uses_probability_scale_without_mutating_inputs():
    from src.model_patterns.expected_points.pipeline import AmericanOddsProbabilityTransformer

    frame = pd.DataFrame({
        "moneyline_home": [-200., 100., 200., 0., np.nan],
        "spread_line": [-3., 0., 3., 0., 7.],
    })
    original = frame.copy(deep=True)
    result = AmericanOddsProbabilityTransformer().transform(frame)
    np.testing.assert_allclose(result.moneyline_home, [2 / 3, .5, 1 / 3, np.nan, np.nan])
    assert result.spread_line.equals(frame.spread_line)
    assert_frame_equal(frame, original)


def test_ridge_score_preprocessing_is_fitted_only_on_training_rows():
    features = ["signal_home", "signal_away", "roof", "moneyline_home"]
    train = pd.DataFrame({
        "pred_team": ["home"] * 4,
        "implied_points_home": [24.] * 4,
        "implied_points_away": [20.] * 4,
        "signal_home": [1., np.nan, 3., 5.],
        "signal_away": [np.nan] * 4,
        "roof": ["outdoors", None, "dome", "outdoors"],
        "moneyline_home": [-200., 100., 200., np.nan],
    })
    original = train.copy(deep=True)
    model = clone(make_score_pipeline(
        features, ["roof"], score_target="market_residual", score_estimator="ridge",
    )).fit(train, np.full(4, 25.))
    imputer = model.named_steps["columntransformer"].named_transformers_["numeric"].named_steps["simpleimputer"]
    before = imputer.statistics_.copy()
    assert before[0] == 3.0
    future = train.iloc[:1].assign(signal_home=100000., signal_away=99., roof="unseen", implied_points_home=31.)
    np.testing.assert_allclose(model.predict(future), [32.])
    np.testing.assert_array_equal(imputer.statistics_, before)
    assert_frame_equal(train, original)


def test_ridge_zero_team_weight_removes_team_signal_and_missingness_effects():
    features = ["implied_points_home", "implied_points_away", "signal_home", "signal_away"]
    frame = pd.DataFrame({
        "pred_team": ["home"] * 4,
        "implied_points_home": [24.] * 4, "implied_points_away": [20.] * 4,
        "signal_home": [-3., -1., 1., 3.], "signal_away": [np.nan, 1., -1., -3.],
    })
    template = make_score_pipeline(
        features, [], score_target="market_residual", score_estimator="ridge",
        team_features=["signal_home", "signal_away"],
    ).set_params(ridge__alpha=.01)
    y = 24. + frame.signal_home
    full = clone(template).set_params(columntransformer__transformer_weights={"team": 1.}).fit(frame, y)
    suppressed = clone(template).set_params(columntransformer__transformer_weights={"team": 0.}).fit(frame, y)
    assert np.ptp(full.predict(frame)) > 5.
    np.testing.assert_allclose(suppressed.predict(frame.assign(signal_home=100000., signal_away=np.nan)), 24.)
    with pytest.raises(ValueError, match="selected numeric"):
        make_score_pipeline(features, [], score_estimator="ridge", team_features=["unselected"])


def test_ridge_matches_direct_solution_and_is_stable_under_training_row_order():
    rng = np.random.default_rng(31)
    values = rng.normal(size=(100, 8)) @ np.diag(np.geomspace(1., 1e4, 8))
    columns = [f"signal_{i}" for i in range(8)]
    frame = pd.DataFrame(values, columns=columns).assign(
        pred_team="home", implied_points_home=24., implied_points_away=21.,
    )
    target = 24. + values[:, 0] - values[:, 1] / 4 + rng.normal(size=100)
    template = make_score_pipeline(columns, [], score_target="market_residual", score_estimator="ridge")
    iterative = clone(template).fit(frame, target)
    direct = clone(template).set_params(ridge__solver="svd").fit(frame, target)
    order = rng.permutation(len(frame))
    reordered = clone(template).fit(frame.iloc[order], target[order])
    np.testing.assert_allclose(iterative.predict(frame), direct.predict(frame), atol=1e-8, rtol=0)
    np.testing.assert_allclose(iterative.predict(frame), reordered.predict(frame), atol=1e-8, rtol=0)
