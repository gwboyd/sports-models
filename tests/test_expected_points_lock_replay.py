from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.model_patterns.expected_points.backtest_artifacts import write_json_atomic, write_parquet_atomic
from src.model_patterns.expected_points.lock_features import lock_feature_set
from src.model_patterns.expected_points.lock_models import LockVariantSpec, fit_probability_head
from src.model_patterns.expected_points.confidence import fit_configured_classifiers
from src.model_patterns.expected_points.betting import determine_plays
from src.model_patterns.expected_points.lock_policy import (
    LockPolicy,
    add_outcome_probabilities,
    lock_profit_metrics,
    select_weekly_locks,
)
from src.model_patterns.expected_points.lock_replay import (
    LockReplaySource,
    LockReplaySpec,
    build_lock_ledger,
    load_lock_replay_source,
    registered_lock_variants,
    replay_lock_variant,
)
from src.model_patterns.expected_points.types import (
    ExpectedPointsConfig,
    ExpectedPointsLeague,
    LockHeadConfig,
    PlayThresholds,
)


def _policy_frame(probabilities=(0.70, 0.69, 0.68, 0.67, 0.66, 0.65)):
    rows = []
    for index, probability in enumerate(probabilities):
        rows.append({
            "season": 2025,
            "week": 1,
            "game_id": f"g{index // 2}",
            "market": "spread" if index % 2 == 0 else "total",
            "kickoff": pd.Timestamp("2025-09-01", tz="UTC") + pd.Timedelta(hours=index),
            "edge": float(index + 1),
            "resolved_win_probability": probability,
            "expected_units": probability * (100 / 110) - (1 - probability),
            "market_supported": True,
            "locks_enabled": True,
            "outcome": "win" if index < 3 else "loss",
        })
    return pd.DataFrame(rows)


def test_combined_weekly_policy_caps_spread_and_total_at_five():
    selected = select_weekly_locks(_policy_frame(), policy=LockPolicy())
    assert selected["lock"].sum() == 5
    assert selected.loc[selected["lock"].eq(1), "market"].nunique() == 2
    assert selected.loc[selected["lock"].eq(0), "lock_reason"].tolist() == ["weekly_cap"]


def test_policy_allows_zero_locks_and_profit_uses_one_unit_risked():
    frame = _policy_frame((0.51, 0.52))
    selected = select_weekly_locks(frame, policy=LockPolicy())
    metrics = lock_profit_metrics(selected)
    assert metrics["locks"] == 0
    assert metrics["net_units"] == 0
    assert metrics["roi"] is None

    selected["lock"] = 1
    selected["outcome"] = ["win", "loss"]
    metrics = lock_profit_metrics(selected)
    assert metrics["net_units"] == pytest.approx(-1 / 11)
    assert metrics["roi"] == pytest.approx(-1 / 22)


def test_stronger_probability_can_exceed_five_without_forcing_a_quota():
    frame = _policy_frame((.59, .58, .57, .56, .555, .551, .54, .53))
    policy = LockPolicy(minimum_resolved_win_probability=.525, max_locks_per_week=3,
                        extra_lock_min_probability=.55)
    selected = select_weekly_locks(frame, policy=policy)
    assert selected["lock"].sum() == 6
    assert selected.loc[selected["lock"].eq(1), "resolved_win_probability"].min() >= .55
    assert select_weekly_locks(_policy_frame((.51,)), policy=policy)["lock"].sum() == 0


def test_preserved_locks_count_against_normal_capacity_but_allow_strong_extras():
    frame = _policy_frame((.59, .58, .57, .56, .54))
    preserved = frame.iloc[:3].assign(lock=1)
    result = select_weekly_locks(frame, policy=LockPolicy(
        minimum_resolved_win_probability=.525, max_locks_per_week=3,
        extra_lock_min_probability=.55,
    ), preserved=preserved)
    assert result["lock"].sum() == 4
    assert result.loc[0:2, "lock_reason"].eq("preserved").all()


def test_extra_probability_overrides_positive_market_caps_but_not_disabled_markets():
    result = select_weekly_locks(_policy_frame((.60,) * 8), policy=LockPolicy(
        max_locks_per_week=3, max_spreads_per_week=2, max_totals_per_week=1,
        extra_lock_min_probability=.58,
    ))
    assert result["lock"].sum() == 8
    disabled = select_weekly_locks(_policy_frame((.60,) * 8), policy=LockPolicy(
        max_locks_per_week=3, max_totals_per_week=0, extra_lock_min_probability=.58,
    ))
    assert disabled.loc[disabled["market"].eq("total"), "lock"].sum() == 0


def test_invalid_probabilities_fail_closed_instead_of_becoming_certain_wins():
    frame = _policy_frame((.6, .6))
    frame = add_outcome_probabilities(frame, np.array([np.inf, 1.5]), np.array([0., 0.]), policy=LockPolicy())
    assert select_weekly_locks(frame, policy=LockPolicy())["lock"].sum() == 0


def test_cadence_metrics_include_empty_weeks_and_initial_loss_drawdown():
    frame = _policy_frame((.6, .6, .51))
    frame["week"] = [1, 1, 2]
    frame["outcome"] = "loss"
    result = lock_profit_metrics(select_weekly_locks(frame, policy=LockPolicy()))
    assert result["evaluated_weeks"] == 2
    assert result["zero_lock_weeks"] == 1
    assert result["two_plus_week_fraction"] == .5
    assert result["max_drawdown"] == 2


def test_symmetric_residual_probabilities_are_monotone_shrunken_and_include_push_errors():
    history = pd.DataFrame({"score_residual": [-1., 2., 4., 8.], "win": [0., 1., np.nan, 1.]})
    head = fit_probability_head(LockVariantSpec("sym", "symmetric_residual", parameters={"trust": .2}),
                                history, numeric_features=("edge",))
    prediction = head.predict(pd.DataFrame({"edge": [0., 2., 5., 100.]}))
    np.testing.assert_allclose(prediction, [.5, .525, .575, .6])


def test_outcome_probabilities_sum_to_one_and_require_positive_ev():
    frame = _policy_frame((0.56,)).drop(columns=["resolved_win_probability", "expected_units"])
    result = add_outcome_probabilities(
        frame, np.array([0.56]), np.array([0.04]), policy=LockPolicy()
    )
    assert result.loc[0, ["p_win", "p_push", "p_loss"]].sum() == pytest.approx(1.0)
    assert result.loc[0, "expected_units"] > 0


def test_feature_registries_are_explicit_and_exclude_outcomes():
    for league in ExpectedPointsLeague:
        for group in ("core", "context", "market"):
            features = lock_feature_set(league, group, market="spread")
            assert "home_score" not in features.all
            assert "spread_win" not in features.all
            assert "recorded_probability" not in features.all
            assert all("postgame" not in column for column in features.all)


def _source_frame():
    rows = []
    for season in (2023, 2024):
        for week in range(1, 9):
            for game in range(8):
                home = 20 + ((week + game) % 10)
                away = 17 + ((week * 2 + game) % 10)
                spread_pred = float(away - home + (-1 if game % 2 else 1))
                total_pred = float(home + away + (-2 if game % 2 else 2))
                spread_line = spread_pred + (2.5 if game % 3 else -2.5)
                total_line = total_pred + (3.5 if game % 3 else -3.5)
                spread_play = "H" if spread_pred < spread_line else "A"
                true_spread = away - home
                correct_spread = "H" if true_spread < spread_line else "A"
                total_play = "under" if total_pred < total_line else "over"
                correct_total = "under" if home + away < total_line else "over"
                kickoff = pd.Timestamp(f"{season}-09-01", tz="America/New_York") + pd.Timedelta(days=week * 7, hours=game)
                rows.append({
                    "season": season, "week": week, "game_id": f"{season}-{week}-{game}",
                    "home_team": "H", "away_team": "A", "date_time": kickoff.strftime("%Y-%m-%d-%H:%M"),
                    "backtest_cutoff": (kickoff.tz_convert("UTC") - pd.Timedelta(hours=1)).isoformat(),
                    "home_score": home, "away_score": away,
                    "home_score_pred": (total_pred - spread_pred) / 2,
                    "away_score_pred": (total_pred + spread_pred) / 2,
                    "spread_pred": spread_pred, "total_pred": total_pred,
                    "spread_line": spread_line, "total_line": total_line,
                    "spread_play": spread_play, "total_play": total_play,
                    "spread_diff": abs(spread_pred - spread_line), "total_diff": abs(total_pred - total_line),
                    "spread_win_prob": 60.0, "total_win_prob": 60.0,
                    "spread_lock": 1, "total_lock": 1,
                    "spread_win": float(spread_play == correct_spread),
                    "total_win": float(total_play == correct_total),
                })
    return pd.DataFrame(rows)


def test_replay_is_chronological_and_keeps_frozen_score_outputs():
    source_frame = _source_frame()
    ledger = build_lock_ledger(source_frame, ExpectedPointsLeague.NFL)
    source = LockReplaySource(
        path=pd.NA,
        run=SimpleNamespace(league=ExpectedPointsLeague.NFL),
        predictions_sha256="sha",
        prediction_fingerprint="fingerprint",
        ledger=ledger,
    )
    replay = LockReplaySpec(
        minimum_training_decisions=20,
        minimum_class_decisions=5,
        bootstrap_samples=20,
    )
    result = replay_lock_variant(
        source,
        LockVariantSpec("edge", "empirical_edge", parameters={"strength": 25}),
        replay,
        market="spread",
        seasons=(2024,),
    )
    assert (result["trained_through"] < result["cutoff"]).all()
    expected = ledger.loc[ledger["market"].eq("spread") & ledger["season"].eq(2024)].set_index("game_id")
    actual = result.set_index("game_id")
    pd.testing.assert_series_equal(actual["prediction"].sort_index(), expected["prediction"].sort_index())


def test_registered_tournament_includes_release_policy_candidates():
    variants = {variant.name: variant for variant in registered_lock_variants(ExpectedPointsLeague.NFL)}
    assert variants["edge_threshold_6_top1"].parameters["max_locks_per_week"] == 1
    assert variants["edge_threshold_7_top4"].parameters["max_locks_per_week"] == 4


def test_weekly_edge_rank_excludes_unsupported_markets():
    frame = _source_frame().head(2).copy()
    frame["spread_market_supported"] = [False, True]
    frame["spread_diff"] = [20.0, 10.0]
    ledger = build_lock_ledger(frame, ExpectedPointsLeague.NFL)
    spread = ledger.loc[ledger["market"].eq("spread")].reset_index(drop=True)
    assert pd.isna(spread.loc[0, "weekly_edge_rank"])
    assert spread.loc[1, "weekly_edge_rank"] == 1


def test_unsupported_markets_cannot_be_locks():
    frame = _policy_frame((0.9,))
    frame["market_supported"] = False
    selected = select_weekly_locks(frame, policy=LockPolicy())
    assert selected.loc[0, "lock"] == 0
    assert selected.loc[0, "lock_reason"] == "market_unsupported"


def test_policy_minimum_edge_is_a_hard_gate():
    frame = _policy_frame((0.70, 0.69))
    selected = select_weekly_locks(frame, policy=LockPolicy(minimum_edge=2.0))
    assert selected.loc[0, "lock"] == 0
    assert selected.loc[1, "lock"] == 1


def _write_source_artifact(tmp_path, *, version="version:2.0", league="nfl"):
    frame = pd.concat([
        _source_frame(),
        _source_frame().assign(
            season=2025,
            game_id=lambda value: value["game_id"].str.replace("2024", "2025", regex=False),
            date_time=lambda value: value["date_time"].str.replace("2024", "2025", regex=False),
            backtest_cutoff=lambda value: value["backtest_cutoff"].str.replace("2024", "2025", regex=False),
        ).loc[lambda value: value["game_id"].str.startswith("2025")],
    ], ignore_index=True)
    frame["recipe_name"] = f"{league}_expected_points"
    frame["recipe_version"] = version
    frame["recipe_fingerprint"] = "recipe-fingerprint"
    write_parquet_atomic(tmp_path / "predictions.parquet", frame)
    write_json_atomic(tmp_path / "summary.json", [])
    write_json_atomic(tmp_path / "manifest.json", {
        "artifact_type": "expected_points_backtest_run",
        "league": league,
        "recipe_name": f"{league}_expected_points",
        "recipe_version": version,
        "recipe_fingerprint": "recipe-fingerprint",
        "profile": "standard",
        "seasons": [2023, 2024, 2025],
        "source_fingerprint": "source-fingerprint",
    })
    return frame


def test_source_preflight_rejects_wrong_version_before_replay(tmp_path):
    _write_source_artifact(tmp_path)
    with pytest.raises(ValueError, match="source version"):
        load_lock_replay_source(
            tmp_path, league="nfl", expected_version="version:3.0"
        )


def test_source_preflight_detects_settlement_corruption(tmp_path):
    frame = _write_source_artifact(tmp_path)
    frame.loc[0, "spread_win"] = 1.0 - frame.loc[0, "spread_win"]
    write_parquet_atomic(tmp_path / "predictions.parquet", frame)
    with pytest.raises(ValueError, match="settlements"):
        load_lock_replay_source(
            tmp_path, league="nfl", expected_version="version:2.0"
        )


def test_saved_calibration_history_round_trip_and_corruption_detection(tmp_path, monkeypatch):
    from dataclasses import replace
    from src.model_patterns.expected_points import backtest_artifacts as artifacts

    _write_source_artifact(tmp_path)
    run = artifacts.load_backtest_run(tmp_path)
    training = _source_frame().head(4).assign(lock_training_cutoff="2025-09-01T00:00:00Z")
    training["trace"] = [[{"feature": "edge", "value": 2.0}]] * len(training)
    monkeypatch.setattr(artifacts, "render_run_report", lambda run: "test")
    destination = tmp_path / "saved"
    artifacts.write_run_artifacts(replace(run, lock_training=training), destination)
    loaded = artifacts.load_backtest_run(destination)
    pd.testing.assert_frame_equal(loaded.lock_training, training)
    artifacts.write_parquet_atomic(destination / "lock_training.parquet", training.assign(home_score=999))
    with pytest.raises(ValueError, match="checksum"):
        artifacts.load_backtest_run(destination)


def test_score_holdout_replay_requires_saved_history_and_rejects_future_rows():
    from dataclasses import replace

    frame = _source_frame()
    ledger = build_lock_ledger(frame, ExpectedPointsLeague.NFL)
    ledger = ledger.loc[ledger["season"].eq(2024) & ledger["week"].eq(1)]
    source = LockReplaySource(pd.NA, SimpleNamespace(league=ExpectedPointsLeague.NFL), "sha", "fp", ledger)
    spec = LockReplaySpec(training_history="score-holdout", minimum_training_decisions=20, minimum_class_decisions=5)
    variant = LockVariantSpec("symmetric", "symmetric_residual", parameters={"trust": .2})
    with pytest.raises(ValueError, match="no saved"):
        replay_lock_variant(source, variant, spec, market="spread", seasons=(2024,))
    training = frame.loc[frame["season"].eq(2023)].copy()
    training.loc[training.index % 2 == 0, "spread_win"] = 0
    saved = pd.concat([training.assign(lock_training_cutoff=cutoff.isoformat()) for cutoff in ledger["cutoff"].unique()])
    source = replace(source, run=SimpleNamespace(league=ExpectedPointsLeague.NFL, lock_training=saved))
    result = replay_lock_variant(source, variant, spec, market="spread", seasons=(2024,))
    assert result["locks_enabled"].all()
    assert (result["trained_through"] < result["cutoff"]).all()
    source.run.lock_training = saved.assign(date_time="2099-09-01-12:00")
    with pytest.raises(ValueError, match="future"):
        replay_lock_variant(source, variant, spec, market="spread", seasons=(2024,))


def test_configured_probability_heads_use_shared_factory_and_disable_unproven_market():
    history = pd.concat([
        _source_frame().assign(game_id=lambda value, suffix=suffix: value["game_id"] + suffix)
        for suffix in ("-a", "-b", "-c")
    ], ignore_index=True)
    history.loc[history.index % 2 == 0, ["spread_win", "total_win"]] = 0.0
    config = ExpectedPointsConfig(
        current_year=2025,
        current_week=1,
        targets=["home_score", "away_score"],
        features=[],
        input_features=[],
        spread_class_features=[],
        total_class_features=[],
        cat_features=[],
        league=ExpectedPointsLeague.CFB,
        spread_lock_head=LockHeadConfig(
            family="spline_logit", parameters={"C": 0.3}, locks_enabled=True,
        ),
        total_lock_head=LockHeadConfig(family="base_rate", locks_enabled=False),
    )
    spread, total = fit_configured_classifiers(history, config)
    probability, push = spread.predict_outcomes(history.iloc[:3])
    assert np.isfinite(probability).all()
    assert ((probability > 0) & (probability < 1)).all()
    assert ((push >= 0) & (push <= 1)).all()
    assert spread.locks_enabled is True
    assert total.locks_enabled is False


def test_edge_threshold_head_estimates_only_two_shrunken_probability_buckets():
    history = pd.DataFrame({
        "edge": [2.0, 3.0, 6.0, 7.0, 8.0, 9.0],
        "weekly_edge_rank": [1, 2, 1, 2, 3, 4],
        "win": [0, 1, 1, 1, 0, 1],
    })
    head = fit_probability_head(
        LockVariantSpec(
            "selective",
            "edge_threshold",
            parameters={"minimum_edge": 6.0, "maximum_rank": 2, "strength": 2.0},
        ),
        history,
        numeric_features=("edge",),
    )
    probabilities = head.predict(pd.DataFrame({
        "edge": [5.99, 6.0, 20.0, 20.0],
        "weekly_edge_rank": [1, 2, 2, 3],
    }))
    assert probabilities[0] < probabilities[1]
    assert probabilities[1] == pytest.approx(probabilities[2])
    assert probabilities[3] == pytest.approx(probabilities[0])
    assert ((probabilities > 0) & (probabilities < 1)).all()


def test_wide_production_policy_uses_one_combined_cap():
    rows = []
    for index in range(4):
        rows.append({
            "season": 2025, "week": 1, "game_id": str(index),
            "date_time": f"2025-09-0{index + 1}-13:00",
            "spread_pred": -5.0 - index, "spread_line": -2.5,
            "total_pred": 50.0 + index, "total_line": 45.5,
            "spread_play": "H", "total_play": "over",
            "spread_win_prob": 70.0 - index, "total_win_prob": 69.5 - index,
            "spread_locks_enabled": True, "total_locks_enabled": True,
        })
    output = determine_plays(
        pd.DataFrame(rows),
        PlayThresholds(
            min_spread_diff=0.0,
            min_total_diff=0.0,
            max_combined_plays=5,
        ),
    )
    assert output["spread_lock"].sum() + output["total_lock"].sum() == 5


def test_wide_production_policy_enforces_each_market_cap_within_combined_cap():
    rows = []
    for index in range(5):
        rows.append({
            "season": 2025, "week": 1, "game_id": str(index),
            "date_time": f"2025-09-0{index + 1}-13:00",
            "spread_pred": -10.0 - index, "spread_line": -2.5,
            "total_pred": 55.0 + index, "total_line": 45.5,
            "spread_play": "H", "total_play": "over",
            "spread_win_prob": 70.0 - index, "total_win_prob": 69.5 - index,
            "spread_locks_enabled": True, "total_locks_enabled": True,
        })
    output = determine_plays(
        pd.DataFrame(rows),
        PlayThresholds(
            max_spreads_plays=1,
            max_total_plays=2,
            min_spread_diff=0.0,
            min_total_diff=0.0,
            max_combined_plays=5,
        ),
    )
    assert output["spread_lock"].sum() == 1
    assert output["total_lock"].sum() == 2
