import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.sports.football.cfb.expected_points.features import build_weekly_pregame_elo


def _schedule():
    return pd.DataFrame({
        "game_id": [1, 2, 3, 4], "season": 2025, "week": [1, 1, 2, 2],
        "start_date": pd.to_datetime(["2025-09-01T13:00Z", "2025-09-08T13:00Z", "2025-09-15T13:00Z", "2025-09-15T16:00Z"]),
        "home_team": ["A", "A", "A", "B"], "away_team": ["B", "C", "D", "E"],
        "home_classification": "fbs", "away_classification": "fbs",
        "home_points": [21., 0., np.nan, np.nan], "away_points": [7., 10., np.nan, np.nan],
    })


def test_elo_freezes_each_week_and_updates_only_later_week_forecasts():
    frame = _schedule()
    original = frame.copy(deep=True)
    result = build_weekly_pregame_elo(frame)
    assert result.home_pregame_elo.tolist() == [1500., 1500., 1499., 1488.]
    assert result.home_postgame_elo.iloc[:2].tolist() == [1512., 1499.]
    assert result.home_postgame_elo.iloc[2:].isna().all()
    changed = frame.copy()
    changed.loc[0, ["home_points", "away_points"]] = [0., 100.]
    alternate = build_weekly_pregame_elo(changed)
    columns = ["home_pregame_elo", "away_pregame_elo"]
    assert_frame_equal(result.loc[:1, columns], alternate.loc[:1, columns])
    assert result.loc[2, "home_pregame_elo"] != alternate.loc[2, "home_pregame_elo"]
    assert_frame_equal(frame, original)


def test_elo_excludes_recent_unavailable_and_incomplete_results():
    frame = _schedule().iloc[:3].copy()
    frame["week"] = [1, 2, 3]
    frame["start_date"] = pd.to_datetime(["2025-09-01T00:00Z", "2025-09-01T08:00Z", "2025-09-01T08:01Z"])
    frame.loc[1, "away_points"] = np.nan
    result = build_weekly_pregame_elo(frame)
    assert result.home_pregame_elo.tolist() == [1500., 1500., 1512.]
    assert pd.isna(result.loc[1, "home_postgame_elo"])


def test_elo_uses_kickoff_availability_for_postponed_week_labels():
    frame = _schedule()
    frame.loc[1, "start_date"] = pd.Timestamp("2025-09-20T13:00Z")
    result = build_weekly_pregame_elo(frame)
    # The postponed week-one loss has not happened at week two's first kickoff.
    assert result.loc[2, "home_pregame_elo"] == 1512.
    # It still shares the original week-one feature snapshot.
    assert result.loc[1, "home_pregame_elo"] == 1500.


def test_elo_preserves_row_order_and_initial_fcs_policy_without_future_updates():
    frame = _schedule()
    frame.loc[3, "away_classification"] = "fcs"
    shuffled = frame.iloc[[3, 1, 0, 2]].copy()
    shuffled.index = [5, 5, 2, 2]
    result = build_weekly_pregame_elo(shuffled)
    assert result.index.tolist() == [5, 5, 2, 2]
    assert result.game_id.tolist() == [4, 2, 1, 3]
    assert result.iloc[0].away_pregame_elo == 1200.
    prefix = build_weekly_pregame_elo(frame.iloc[:2])
    full = build_weekly_pregame_elo(frame)
    assert_frame_equal(prefix[["home_pregame_elo", "away_pregame_elo"]], full.loc[:1, ["home_pregame_elo", "away_pregame_elo"]])


def test_elo_rejects_duplicate_games():
    frame = _schedule()
    with pytest.raises(ValueError):
        build_weekly_pregame_elo(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
