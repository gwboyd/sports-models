from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.model_patterns.expected_points.backtest_artifacts import write_json_atomic, write_parquet_atomic
from src.model_patterns.expected_points.lock_features import lock_feature_set
from src.model_patterns.expected_points.lock_models import LockVariantSpec
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
    replay_lock_variant,
)
from src.model_patterns.expected_points.types import ExpectedPointsLeague


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


def test_unsupported_markets_cannot_be_locks():
    frame = _policy_frame((0.9,))
    frame["market_supported"] = False
    selected = select_weekly_locks(frame, policy=LockPolicy())
    assert selected.loc[0, "lock"] == 0
    assert selected.loc[0, "lock_reason"] == "market_unsupported"


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
