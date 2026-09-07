from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.model_patterns.expected_points.backtest_metrics import (
    _BOOTSTRAP_METRICS,
    _bootstrap_metric_values,
    _period_metric_components,
)
from src.model_patterns.expected_points.backtesting import (
    BacktestSpec,
    bootstrap_metric_intervals,
    compare_backtest_runs,
    load_backtest_frame,
    load_backtest_run,
    recipe_code_identity,
    run_walk_forward,
    save_backtest_frame,
    select_backtest_seasons,
    source_bundle_fingerprint,
    summarize_predictions,
    write_comparison_artifacts,
    write_run_artifacts,
)
from src.model_patterns.expected_points.evaluation import (
    determine_plays_by_period,
    prediction_metrics,
)
from src.model_patterns.expected_points.inspection import (
    inspect_game,
    inspect_team_history,
    load_backtest_comparison,
)
from src.model_patterns.expected_points.trainer import (
    _build_train_df_at_cutoff,
    run_expected_points_at_cutoff,
)
from src.model_patterns.expected_points.types import (
    ExpectedPointsConfig,
    ExpectedPointsLeague,
    PlayThresholds,
)
from src.sports.football.cfb.expected_points.recipe import CFBExpectedPointsRecipe
from src.sports.football.nfl.expected_points.recipe import NFLExpectedPointsRecipe


@dataclass(frozen=True)
class TinyRecipe:
    version: str
    league: ExpectedPointsLeague = ExpectedPointsLeague.NFL
    name: str = "tiny"
    protocol_version: str = "test"

    def build_config(self, frame, *, season, week, prediction_now=None):
        return ExpectedPointsConfig(
            current_year=season,
            current_week=week,
            targets=["home_score", "away_score"],
            features=["metric_home", "metric_away"],
            input_features=["metric_home", "metric_away"],
            spread_class_features=["spread_diff"],
            total_class_features=["total_diff"],
            cat_features=[],
            prediction_now=prediction_now,
        )


@dataclass(frozen=True)
class WeekVaryingTinyRecipe(TinyRecipe):
    final_week_minimum: float = 0.5

    def build_config(self, frame, *, season, week, prediction_now=None):
        config = super().build_config(
            frame,
            season=season,
            week=week,
            prediction_now=prediction_now,
        )
        if week == 4:
            config.play_thresholds.min_total_diff = self.final_week_minimum
        return config


def _historical_frame():
    rows = []
    for season in (2022, 2023, 2024):
        for week in range(1, 5):
            for game in range(2):
                home_score = 20 + week + game
                away_score = 17 + (game * 2)
                rows.append({
                    "season": season,
                    "week": week,
                    "game_id": f"{season}_{week}_{game}",
                    "date_time": f"{season}-09-{week * 3:02d}-{13 + game}:00",
                    "home_team": f"H{game}",
                    "away_team": f"A{game}",
                    "home_score": home_score,
                    "away_score": away_score,
                    "metric_home": float(week),
                    "metric_away": float(game),
                    "spread_line": -2.5,
                    "total_line": 44.5,
                })
    return pd.DataFrame(rows)


def _fake_cutoff_run(_frame, _config, *, cutoff, prediction_frame):
    plays = prediction_frame.copy()
    plays["home_score_pred"] = plays["home_score"] - 1.0
    plays["away_score_pred"] = plays["away_score"] + 1.0
    plays["spread_pred"] = plays["away_score_pred"] - plays["home_score_pred"]
    plays["total_pred"] = plays["home_score_pred"] + plays["away_score_pred"]
    plays["spread_play"] = np.where(
        plays["spread_pred"] < plays["spread_line"], plays["home_team"], plays["away_team"]
    )
    plays["total_play"] = np.where(plays["total_pred"] < plays["total_line"], "under", "over")
    plays["spread_diff"] = (plays["spread_pred"] - plays["spread_line"]).abs()
    plays["total_diff"] = (plays["total_pred"] - plays["total_line"]).abs()
    plays["spread_win_prob"] = [60.0, 50.0]
    plays["total_win_prob"] = [50.0, 60.0]
    plays["spread_lock"] = [1, 0]
    plays["total_lock"] = [0, 1]
    return SimpleNamespace(plays=plays, metrics={})


def test_cutoff_training_excludes_complete_games_at_or_after_cutoff():
    frame = _historical_frame()
    config = TinyRecipe("1.0").build_config(frame, season=2024, week=2)
    cutoff = pd.Timestamp("2024-09-06T16:59:59Z")
    training = _build_train_df_at_cutoff(frame, config, cutoff=cutoff)
    assert "2024_2_0" not in set(training["game_id"])
    assert "2024_4_0" not in set(training["game_id"])


def test_cutoff_prediction_hides_target_outcomes_until_grading(monkeypatch):
    frame = _historical_frame()
    target = frame.loc[(frame["season"] == 2024) & (frame["week"] == 2)].copy()
    config = TinyRecipe("1.0").build_config(frame, season=2024, week=2)
    cutoff = pd.Timestamp("2024-09-06T16:59:59Z")
    observed: dict[str, pd.DataFrame] = {}

    monkeypatch.setattr(
        "src.model_patterns.expected_points.trainer._fit_expected_points",
        lambda _train, _config: (
            object(),
            object(),
            object(),
            pd.DataFrame(),
            pd.DataFrame(),
            {},
        ),
    )

    def fake_predict(prediction_frame, prediction_config, **_models):
        observed["prediction_frame"] = prediction_frame.copy()
        assert prediction_config.prediction_now == cutoff
        plays = prediction_frame[["game_id"]].copy()
        return prediction_frame.copy(), plays

    monkeypatch.setattr(
        "src.model_patterns.expected_points.trainer._predict_period",
        fake_predict,
    )
    monkeypatch.setattr(
        "src.model_patterns.expected_points.trainer._build_metrics",
        lambda **_kwargs: {},
    )

    result = run_expected_points_at_cutoff(
        frame,
        config,
        cutoff=cutoff,
        prediction_frame=target,
    )

    assert observed["prediction_frame"][["home_score", "away_score"]].isna().all().all()
    expected = target.set_index("game_id")[["home_score", "away_score"]].sort_index()
    actual = result.plays.set_index("game_id")[["home_score", "away_score"]].sort_index()
    pd.testing.assert_frame_equal(actual, expected)


def test_profiles_require_a_prior_complete_history_season():
    frame = _historical_frame()
    assert select_backtest_seasons(frame, BacktestSpec(profile="quick")) == (2024,)
    assert select_backtest_seasons(
        frame,
        BacktestSpec(profile="standard", seasons=(2023, 2024)),
    ) == (2023, 2024)


def test_cutoff_cache_reuses_rows_and_invalidates_after_history_correction(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.model_patterns.expected_points.backtesting.run_expected_points_at_cutoff",
        _fake_cutoff_run,
    )
    frame = _historical_frame()
    frame["trace_payload"] = [{"source": "test"} for _ in range(len(frame))]
    spec = BacktestSpec(profile="quick", cache_root=tmp_path)
    first = run_walk_forward(frame, TinyRecipe("1.0"), spec)
    second = run_walk_forward(frame, TinyRecipe("1.0"), spec)
    assert first.trained_cutoffs == 2
    assert second.cache_hits == 2
    assert second.predictions.iloc[0]["trace_payload"] == {"source": "test"}

    corrected = frame.copy()
    corrected.loc[corrected["game_id"] == "2022_1_0", "home_score"] += 1
    third = run_walk_forward(corrected, TinyRecipe("1.0"), spec)
    assert third.trained_cutoffs == 2
    assert third.cache_hits == 0


def test_working_tree_cache_rolls_in_one_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.model_patterns.expected_points.backtesting.run_expected_points_at_cutoff",
        _fake_cutoff_run,
    )
    identity = {"value": "first"}
    monkeypatch.setattr(
        "src.model_patterns.expected_points.backtesting.recipe_code_identity",
        lambda _league: identity["value"],
    )
    frame = _historical_frame()
    spec = BacktestSpec(profile="quick", cache_root=tmp_path, cache_working_tree=True)
    recipe = TinyRecipe("working-tree")

    first = run_walk_forward(frame, recipe, spec)
    identity["value"] = "second"
    changed = run_walk_forward(frame, recipe, spec)
    repeated = run_walk_forward(frame, recipe, spec)

    assert first.trained_cutoffs == 2
    assert changed.trained_cutoffs == 2
    assert changed.cache_hits == 0
    assert repeated.trained_cutoffs == 0
    assert repeated.cache_hits == 2
    cache_directories = [path.name for path in (tmp_path / "cache" / "nfl").iterdir()]
    assert cache_directories == ["working-tree"]


def test_working_tree_cache_is_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.model_patterns.expected_points.backtesting.run_expected_points_at_cutoff",
        _fake_cutoff_run,
    )
    frame = _historical_frame()
    spec = BacktestSpec(profile="quick", cache_root=tmp_path)
    recipe = TinyRecipe("working-tree")

    first = run_walk_forward(frame, recipe, spec)
    repeated = run_walk_forward(frame, recipe, spec)

    assert first.trained_cutoffs == 2
    assert repeated.trained_cutoffs == 2
    assert repeated.cache_hits == 0
    assert not (tmp_path / "cache").exists()


def test_recipe_code_identity_discovers_new_shared_python_files(tmp_path, monkeypatch):
    shared = tmp_path / "src/model_patterns/expected_points"
    shared.mkdir(parents=True)
    helper = shared / "new_prediction_helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "src.model_patterns.expected_points.backtesting.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="test-sha\n"),
    )

    first = recipe_code_identity(ExpectedPointsLeague.NFL, root=tmp_path)
    helper.write_text("VALUE = 2\n", encoding="utf-8")
    second = recipe_code_identity(ExpectedPointsLeague.NFL, root=tmp_path)

    assert first != second


def test_released_version_cache_rolls_in_readable_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.model_patterns.expected_points.backtesting.run_expected_points_at_cutoff",
        _fake_cutoff_run,
    )
    identity = {"value": "release-sha-one"}
    monkeypatch.setattr(
        "src.model_patterns.expected_points.backtesting.recipe_code_identity",
        lambda _league: identity["value"],
    )
    frame = _historical_frame()
    spec = BacktestSpec(profile="quick", cache_root=tmp_path)
    recipe = TinyRecipe("version:1.0")

    first = run_walk_forward(frame, recipe, spec)
    identity["value"] = "release-sha-two"
    changed_sha = run_walk_forward(frame, recipe, spec)
    repeated = run_walk_forward(frame, recipe, spec)

    assert first.trained_cutoffs == 2
    assert changed_sha.trained_cutoffs == 2
    assert changed_sha.cache_hits == 0
    assert repeated.trained_cutoffs == 0
    assert repeated.cache_hits == 2
    cache_directories = [path.name for path in (tmp_path / "cache" / "nfl").iterdir()]
    assert cache_directories == ["version-1.0"]


def test_identical_runs_have_zero_candidate_delta(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.model_patterns.expected_points.backtesting.run_expected_points_at_cutoff",
        _fake_cutoff_run,
    )
    frame = _historical_frame()
    spec = BacktestSpec(profile="quick", cache_root=tmp_path)
    baseline = run_walk_forward(frame, TinyRecipe("1.0"), spec)
    candidate = run_walk_forward(frame, TinyRecipe("candidate"), spec)
    comparison = compare_backtest_runs(baseline, candidate, bootstrap_samples=20, random_seed=4)
    overall = comparison.deltas.loc[
        (comparison.deltas["cut_type"] == "overall")
        & (comparison.deltas["metric"] == "score_mae")
    ].iloc[0]
    assert overall["raw_delta"] == 0
    assert overall["improvement"] == 0
    assert overall["status"] == "inconclusive"
    assert comparison.deltas[["raw_delta", "improvement"]].eq(0).all().all()
    assert "latest_season" in set(comparison.summary["cut_type"])
    overall_metrics = comparison.deltas.loc[comparison.deltas["cut_type"] == "overall"]
    for metric in ("score_rmse", "score_bias", "spread_calibration_error", "spread_locks"):
        row = overall_metrics.loc[overall_metrics["metric"] == metric].iloc[0]
        assert row["improvement_ci_low"] == pytest.approx(0)
        assert row["improvement_ci_high"] == pytest.approx(0)


def test_cache_identity_includes_each_cutoffs_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.model_patterns.expected_points.backtesting.run_expected_points_at_cutoff",
        _fake_cutoff_run,
    )
    frame = _historical_frame()
    spec = BacktestSpec(profile="quick", cache_root=tmp_path)
    first = run_walk_forward(frame, WeekVaryingTinyRecipe("1.0", final_week_minimum=0.5), spec)
    changed = run_walk_forward(frame, WeekVaryingTinyRecipe("1.0", final_week_minimum=2.0), spec)
    assert first.trained_cutoffs == 2
    assert changed.cache_hits == 1
    assert changed.trained_cutoffs == 1


def test_comparison_rejects_partial_or_differently_profiled_universes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.model_patterns.expected_points.backtesting.run_expected_points_at_cutoff",
        _fake_cutoff_run,
    )
    run = run_walk_forward(
        _historical_frame(), TinyRecipe("1.0"), BacktestSpec(profile="quick", cache_root=tmp_path)
    )
    partial = replace(run, predictions=run.predictions.iloc[:-1].copy())
    with pytest.raises(ValueError, match="identical evaluated game universes"):
        compare_backtest_runs(run, partial, bootstrap_samples=0)
    with pytest.raises(ValueError, match="same profile"):
        compare_backtest_runs(run, replace(run, profile="standard"), bootstrap_samples=0)


def test_bias_improvement_means_closer_to_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.model_patterns.expected_points.backtesting.run_expected_points_at_cutoff",
        _fake_cutoff_run,
    )
    run = run_walk_forward(
        _historical_frame(), TinyRecipe("1.0"), BacktestSpec(profile="quick", cache_root=tmp_path)
    )
    baseline_predictions = run.predictions.copy()
    candidate_predictions = run.predictions.copy()
    for predictions, residual in ((baseline_predictions, 2.0), (candidate_predictions, 1.0)):
        predictions["home_score_pred"] = predictions["home_score"] + residual
        predictions["away_score_pred"] = predictions["away_score"] + residual
    baseline = replace(
        run,
        predictions=baseline_predictions,
        summary=summarize_predictions(baseline_predictions),
    )
    candidate = replace(
        run,
        predictions=candidate_predictions,
        summary=summarize_predictions(candidate_predictions),
    )
    comparison = compare_backtest_runs(baseline, candidate, bootstrap_samples=20)
    bias = comparison.deltas.loc[
        (comparison.deltas["cut_type"] == "overall")
        & (comparison.deltas["metric"] == "score_bias")
    ].iloc[0]
    assert bias["raw_delta"] == -1
    assert bias["improvement"] == 1
    assert bias["status"] == "improvement"


def test_comparison_artifact_restores_predictions_and_lock_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.model_patterns.expected_points.backtesting.run_expected_points_at_cutoff",
        _fake_cutoff_run,
    )
    frame = _historical_frame()
    frame["trace_payload"] = [{"provider": "test"} for _ in range(len(frame))]
    run = run_walk_forward(
        frame, TinyRecipe("1.0"), BacktestSpec(profile="quick", cache_root=tmp_path / "cache")
    )
    comparison = compare_backtest_runs(run, run, bootstrap_samples=0)
    artifact_path = write_comparison_artifacts(comparison, output_dir=tmp_path / "comparison")
    loaded = load_backtest_comparison(artifact_path)

    assert loaded.baseline_predictions.iloc[0]["trace_payload"] == {"provider": "test"}
    assert not loaded.lock_changes.empty
    assert set(loaded.lock_changes["spread_lock_change"]) <= {"common", "neither"}


def test_bootstrap_sufficient_statistics_match_pooled_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.model_patterns.expected_points.backtesting.run_expected_points_at_cutoff",
        _fake_cutoff_run,
    )
    run = run_walk_forward(
        _historical_frame(), TinyRecipe("1.0"), BacktestSpec(profile="quick", cache_root=tmp_path)
    )
    predictions = run.predictions.assign(
        __period=lambda value: value["season"].astype(str) + "_" + value["week"].astype(str)
    )
    periods = predictions["__period"].drop_duplicates().tolist()
    components = _period_metric_components(predictions, periods)
    selected = np.arange(len(periods), dtype=int).reshape(1, -1)
    pooled = prediction_metrics(predictions)

    for metric in _BOOTSTRAP_METRICS:
        if metric in pooled:
            value = _bootstrap_metric_values(components[metric], selected)[0]
            assert value == pytest.approx(pooled[metric]), metric


def test_source_bundle_compatibility_ignores_recipe_specific_feature_columns():
    baseline = _historical_frame().assign(old_recipe_feature=1.0)
    candidate = _historical_frame().assign(new_recipe_feature=2.0)
    assert source_bundle_fingerprint(baseline) == source_bundle_fingerprint(candidate)


def test_backtest_frame_round_trips_mixed_unused_object_columns(tmp_path):
    frame = _historical_frame().iloc[:2].copy()
    frame["raw_mixed_value"] = [1, "scheduled"]
    frame["raw_payload"] = [{"provider": "a"}, ["provider", "b"]]
    frame["provider_quotes"] = [(('a', -2.5), ('b', -3.0)), (('a', 45.5),)]
    path = save_backtest_frame(frame, ExpectedPointsLeague.NFL, root=tmp_path)
    loaded = load_backtest_frame(path)
    assert loaded["raw_mixed_value"].tolist() == ["1", "scheduled"]
    assert loaded["raw_payload"].tolist() == [{"provider": "a"}, ["provider", "b"]]
    assert loaded.loc[0, "provider_quotes"] == [["a", -2.5], ["b", -3.0]]


def test_lock_ranking_resets_for_each_week_and_pushes_stay_out_of_denominator():
    frame = pd.DataFrame({
        "season": [2024] * 4,
        "week": [1, 1, 2, 2],
        "game_id": ["1", "2", "3", "4"],
        "home_score": [21, 20, 24, 17],
        "away_score": [20, 20, 21, 17],
        "home_score_pred": [22, 21, 23, 18],
        "away_score_pred": [19, 20, 20, 17],
        "spread_pred": [-3, -1, -3, -1],
        "spread_line": [-2, 0, -2, 0],
        "spread_play": ["H"] * 4,
        "spread_win": [1, np.nan, 1, 0],
        "spread_win_prob": [70, 60, 69, 61],
        "total_pred": [41, 41, 43, 35],
        "total_line": [40, 40, 40, 35],
        "total_play": ["over", "over", "over", None],
        "total_win": [1, 0, 1, np.nan],
        "total_win_prob": [70, 60, 69, 61],
    })
    thresholds = PlayThresholds(
        max_spreads_plays=1,
        max_total_plays=1,
        min_spread_win_prob=50,
        min_total_win_prob=50,
    )
    evaluated = determine_plays_by_period(frame, thresholds=thresholds)
    assert evaluated.groupby("week")["spread_lock"].sum().tolist() == [1, 1]
    metrics = prediction_metrics(evaluated)
    assert metrics["spread_wins"] == 2
    assert metrics["spread_losses"] == 1
    assert metrics["spread_pushes"] == 1
    assert metrics["spread_win_pct"] == pytest.approx(100 * 2 / 3)


def test_unsupported_markets_are_excluded_from_all_pick_metrics():
    frame = pd.DataFrame({
        "home_score": [21, 21], "away_score": [17, 17],
        "home_score_pred": [20, 20], "away_score_pred": [18, 18],
        "spread_win": [1, 0], "total_win": [1, 0],
        "spread_lock": [0, 0], "total_lock": [0, 0],
        "spread_win_prob": [60, 60], "total_win_prob": [60, 60],
        "spread_market_supported": [True, False],
        "total_market_supported": [True, False],
        "season": [2024, 2024], "week": [1, 1],
    })
    metrics = prediction_metrics(frame)
    assert metrics["spread_eligible"] == 1
    assert metrics["spread_win_pct"] == 100


def test_zero_lock_record_is_undefined_instead_of_zero_percent():
    frame = pd.DataFrame({
        "home_score": [21], "away_score": [17],
        "home_score_pred": [20], "away_score_pred": [18],
        "spread_win": [1], "total_win": [0],
        "spread_lock": [0], "total_lock": [0],
        "spread_win_prob": [60], "total_win_prob": [60],
        "season": [2024], "week": [1],
    })
    metrics = prediction_metrics(frame)
    assert metrics["spread_lock_win_pct"] is None
    assert metrics["total_lock_win_pct"] is None
    assert metrics["spread_lock_mean_win_prob"] is None
    assert metrics["total_lock_calibration_gap"] is None


def test_market_benchmark_and_confidence_diagnostics_are_reported():
    frame = pd.DataFrame({
        "home_score": [21, 17], "away_score": [17, 24],
        "home_score_pred": [20, 20], "away_score_pred": [18, 21],
        "implied_points_home": [21, 18], "implied_points_away": [17, 23],
        "spread_line": [-4, 5], "total_line": [38, 41],
        "spread_win": [1, 0], "total_win": [1, 0],
        "spread_lock": [1, 1], "total_lock": [1, 1],
        "spread_win_prob": [80, 70], "total_win_prob": [75, 65],
        "season": [2024, 2024], "week": [1, 2],
    })
    metrics = prediction_metrics(frame)
    assert metrics["market_score_mae"] < metrics["score_mae"]
    assert metrics["score_mae_advantage_vs_market"] < 0
    assert metrics["spread_auc"] == 1
    assert metrics["spread_lock_mean_win_prob"] == pytest.approx(75)
    assert metrics["spread_lock_calibration_gap"] == pytest.approx(25)
    intervals = bootstrap_metric_intervals(frame, samples=20, seed=4)
    assert "spread_lock_win_pct" in intervals


def test_single_run_artifact_includes_report_and_timing(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(
        "src.model_patterns.expected_points.backtesting.run_expected_points_at_cutoff",
        _fake_cutoff_run,
    )
    caplog.set_level("INFO")
    run = run_walk_forward(
        _historical_frame(),
        TinyRecipe("1.0"),
        BacktestSpec(profile="quick", cache_root=tmp_path / "cache"),
    )
    artifact = write_run_artifacts(run, tmp_path / "run")
    loaded = load_backtest_run(artifact)
    assert loaded.elapsed_seconds >= 0
    assert len(loaded.cutoff_timings) == 2
    assert "cutoff 1/2" in caplog.text
    report = (artifact / "report.md").read_text(encoding="utf-8")
    assert "## Records" in report
    assert "95% season-week interval" in report


def test_notebook_inspection_exposes_exact_home_and_away_score_features():
    frame = _historical_frame().iloc[[0]].copy()
    config = TinyRecipe("1.0").build_config(frame, season=2022, week=1)
    inspected = inspect_game(frame, config, game_id="2022_1_0")
    assert inspected.score_features["predicted_side"].tolist() == ["home", "away"]
    assert inspected.score_features.loc[1, "metric_home"] == frame.iloc[0]["metric_away"]
    history = inspect_team_history(_historical_frame(), team="H0", before_game_id="2024_2_0")
    assert "2024_2_0" not in set(history["game_id"])


def test_league_recipes_select_only_registered_features_in_stable_order():
    nfl_metrics = [
        f"{base}_{venue}" for base in (
            "ewma_dynamic_window_rushing_offense", "ewma_dynamic_window_passing_offense",
            "ewma_dynamic_window_rushing_defense", "ewma_dynamic_window_passing_defense",
            "ewma_success_rate_rushing_offense", "ewma_success_rate_passing_offense",
            "ewma_success_rate_rushing_defense", "ewma_success_rate_passing_defense",
        ) for venue in ("home", "away")
    ]
    cfb_metrics = [f"{side}_explosiveness_ewma_dynamic_window_{venue}"
                   for venue in ("home", "away") for side in ("offense", "defense")]
    frame = pd.DataFrame(columns=[*nfl_metrics, *cfb_metrics, "future_outcome_ewma_dynamic"])
    nfl = NFLExpectedPointsRecipe().build_config(frame, season=2026, week=1)
    assert [c for c in nfl.features if c in nfl_metrics] == nfl_metrics
    cfb = CFBExpectedPointsRecipe(efficiency_metrics=("explosiveness",)).build_config(frame, season=2026, week=1)
    assert cfb.features == cfb_metrics
    assert "future_outcome_ewma_dynamic" not in nfl.features + cfb.features
    assert cfb.confidence_scoring == "neg_log_loss"
    assert cfb.betting_transform is not None
    assert cfb.spread_class_features[-1] == "spread_diff"
    assert cfb.total_class_features[-1] == "total_diff"
    for recipe, column in ((NFLExpectedPointsRecipe(), nfl_metrics[0]),
                           (CFBExpectedPointsRecipe(efficiency_metrics=("explosiveness",)), cfb_metrics[0])):
        with pytest.raises(ValueError, match="features are missing"):
            recipe.build_config(frame.drop(columns=column), season=2026, week=1)


def test_run_remains_loadable_when_report_rendering_fails(tmp_path, monkeypatch):
    from src.model_patterns.expected_points import backtest_artifacts

    monkeypatch.setattr(
        "src.model_patterns.expected_points.backtesting.run_expected_points_at_cutoff",
        _fake_cutoff_run,
    )
    run = run_walk_forward(_historical_frame(), TinyRecipe("working-tree"),
                           BacktestSpec(profile="quick", cache_root=tmp_path / "cache"))
    def fail_report(_run):
        raise RuntimeError("report rendering failed")
    monkeypatch.setattr(backtest_artifacts, "render_run_report", fail_report)
    with pytest.raises(RuntimeError, match="report rendering failed"):
        write_run_artifacts(run, tmp_path / "saved")
    restored = load_backtest_run(tmp_path / "saved")
    assert restored.predictions.game_id.tolist() == run.predictions.game_id.tolist()
    assert restored.source_fingerprint == run.source_fingerprint
