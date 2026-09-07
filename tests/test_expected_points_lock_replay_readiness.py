"""Regression coverage for optional inputs and reusable replay CLI controls."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import backtest_expected_points as cli
from src.model_patterns.expected_points.backtest_artifacts import write_json_atomic
from src.model_patterns.expected_points.lock_features import lock_feature_set
from src.model_patterns.expected_points.lock_models import fit_probability_head
from src.model_patterns.expected_points.lock_policy import LockPolicy
from src.model_patterns.expected_points.lock_replay import (
    LockReplaySpec,
    build_lock_market_frame,
    registered_lock_variants,
    screen_lock_cadence,
)
from src.model_patterns.expected_points.types import ExpectedPointsLeague


@pytest.mark.parametrize("league", list(ExpectedPointsLeague))
@pytest.mark.parametrize("name", [
    "spline_context_c01", "spline_context_c03", "spline_market_c01",
    "lightgbm_context", "lightgbm_market",
])
def test_optional_categories_support_missing_columns_and_pandas_na(league, name):
    count = 120
    frame = pd.DataFrame({
        "season": 2024, "week": np.arange(count) // 10 + 1,
        "game_id": [f"game-{index:03d}" for index in range(count)],
        "date_time": "2024-09-01-13:00", "home_team": "H", "away_team": "A",
        "home_score": 24, "away_score": 20, "spread_line": -2.5,
        "total_line": 45.5, "spread_pred": -4., "total_pred": 44.,
        "spread_play": "H", "spread_diff": 1.5,
        "spread_win": np.arange(count) % 2,
        "roof": pd.Series([pd.NA, "outdoors", "dome"] * 40, dtype="string"),
        "home_conference": pd.Series([pd.NA, "SEC", "Big Ten"] * 40, dtype="string"),
    })
    # Weekday, classifications, and other optional categories are wholly absent.
    history = build_lock_market_frame(frame, league, "spread", outcomes_known=True)
    variant = next(item for item in registered_lock_variants(league) if item.name == name)
    features = lock_feature_set(league, variant.feature_group, market="spread")
    head = fit_probability_head(
        variant, history, numeric_features=features.numeric,
        categorical_features=features.categorical,
    )
    probabilities = head.predict(history.iloc[:3])
    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0) & (probabilities < 1)).all()


def test_cadence_screen_skips_extra_floors_below_requested_minimum(tmp_path, monkeypatch):
    import src.model_patterns.expected_points.lock_replay as replay_module

    def forecast(_source, variant, _spec, *, market, seasons):
        return pd.DataFrame({
            "season": [2024, 2025], "week": [1, 1], "game_id": ["a", "b"],
            "market": market, "kickoff": pd.to_datetime(["2024-09-01", "2025-09-01"], utc=True),
            "edge": 10., "resolved_win_probability": .59, "expected_units": .12,
            "market_supported": True, "locks_enabled": True,
            "outcome": ["win", "loss"], "win": [1, 0],
        })

    monkeypatch.setattr(replay_module, "replay_lock_variant", forecast)
    source = SimpleNamespace(run=SimpleNamespace(league=ExpectedPointsLeague.NFL))
    spec = LockReplaySpec(
        bootstrap_samples=2,
        policy=LockPolicy(minimum_resolved_win_probability=.58),
    )
    variants = registered_lock_variants(ExpectedPointsLeague.NFL)[:1]
    result = screen_lock_cadence(source, spec, variants, tmp_path)
    assert len(result) == 10  # Five allocations, two compatible normal-cap policies.
    assert all(policy["extra_lock_min_probability"] is None for policy in result["policy"])


def test_source_discovery_includes_standalone_runs_and_skips_incomplete_bundles(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "_resolve_reference", lambda *_args: ("release-sha", "version:2.1"))
    root = tmp_path / ".backtests/expected_points"

    def save_manifest(path, created_at, *, complete=True, league="nfl"):
        write_json_atomic(path / "manifest.json", {
            "artifact_type": "expected_points_backtest_run", "league": league,
            "recipe_version": "version:2.1", "profile": "standard",
            "git_identity": "release-sha", "created_at": created_at,
        })
        if complete:
            (path / "predictions.parquet").touch()
            write_json_atomic(path / "summary.json", [])

    old_comparison = root / "comparisons/nfl/older/candidate"
    latest_run = root / "runs/nfl/newer"
    save_manifest(old_comparison, "2026-09-01")
    save_manifest(latest_run, "2026-09-02")
    save_manifest(root / "comparisons/nfl/incomplete/candidate", "2026-09-03", complete=False)
    save_manifest(root / "runs/nfl/wrong-league", "2026-09-04", league="cfb")
    assert cli._discover_lock_source("deployed", "nfl", None) == (latest_run, "version:2.1")
