"""Opponent features admit only results available before their snapshot.

The target team C has no overlapping game. A result from opponent A's game can
still change C's opponent-adjusted history through the shared rating snapshot.
"""

import numpy as np
import pandas as pd
import pytest

from src.sports.football.transforms import (
    OpponentMetricSpec,
    build_opponent_adjusted_team_metrics,
)


def _target_features(*, snapshot: str, elapsed_hours: float, unavailable_value: float):
    earlier = pd.Timestamp("2025-09-07T16:00Z")
    schedule = pd.DataFrame([
        dict(game_id="history", season=2025, week=1, home_team="C", away_team="A",
             start_date=earlier - pd.Timedelta(days=7)),
        dict(game_id="opponent_game", season=2025, week=2, home_team="A", away_team="B",
             start_date=earlier),
        dict(game_id="target", season=2025, week=3, home_team="C", away_team="D",
             start_date=earlier + pd.Timedelta(hours=elapsed_hours)),
    ])
    observations = pd.DataFrame([
        dict(game_id="history", team="C", metric="m", value=1.0),
        dict(game_id="history", team="A", metric="m", value=2.0),
        dict(game_id="opponent_game", team="A", metric="m", value=0.0),
        dict(game_id="opponent_game", team="B", metric="m", value=unavailable_value),
    ])
    output = build_opponent_adjusted_team_metrics(
        schedule,
        observations,
        [OpponentMetricSpec("m", "off", "def", "static")],
        config={"adjustment_strength": 0.5, "rating_snapshot": snapshot},
    )
    return output.loc[output.game_id.eq("target") & output.team.eq("C"), ["off_ewma", "def_ewma"]].to_numpy()


@pytest.mark.parametrize("snapshot", ["week", "kickoff"])
@pytest.mark.parametrize("elapsed_hours", [1.0, 8.0])
def test_unavailable_opponent_result_cannot_change_target(snapshot, elapsed_hours):
    before = _target_features(snapshot=snapshot, elapsed_hours=elapsed_hours, unavailable_value=0.0)
    changed = _target_features(snapshot=snapshot, elapsed_hours=elapsed_hours, unavailable_value=100.0)
    np.testing.assert_allclose(before, changed, rtol=0, atol=1e-12)


@pytest.mark.parametrize("snapshot", ["week", "kickoff"])
def test_available_opponent_result_can_change_target(snapshot):
    before = _target_features(snapshot=snapshot, elapsed_hours=8.001, unavailable_value=0.0)
    changed = _target_features(snapshot=snapshot, elapsed_hours=8.001, unavailable_value=100.0)
    assert not np.allclose(before, changed)
