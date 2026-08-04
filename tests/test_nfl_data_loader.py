from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from src.sports.football.nfl.expected_points import data_loader


def _frame(columns: tuple[str, ...], values: dict[str, object] | None = None) -> pl.DataFrame:
    values = values or {}
    return pl.DataFrame({column: [values.get(column, 1)] for column in columns})


def _install_complete_loaders(monkeypatch):
    pbp_frame = _frame(
        data_loader.PBP_REQUIRED_COLUMNS,
        {
            "home_team": "OAK",
            "away_team": "SD",
            "posteam": "STL",
            "defteam": "OAK",
            "side_of_field": "SD",
            "epa": 0.25,
            "qb_epa": 0.5,
        },
    )
    player_frame = _frame(
        data_loader.PLAYER_STATS_REQUIRED_COLUMNS,
        {"team": "OAK", "player_name": "Player", "position_group": "QB"},
    )
    schedule_frame = _frame(
        data_loader.SCHEDULE_REQUIRED_COLUMNS,
        {"home_team": "OAK", "away_team": "STL"},
    )
    team_frame = _frame(data_loader.TEAM_REQUIRED_COLUMNS)
    calls = {"pbp": [], "players": [], "schedules": [], "config": []}

    monkeypatch.setattr(
        data_loader.nfl, "load_pbp", lambda season: calls["pbp"].append(season) or pbp_frame
    )
    monkeypatch.setattr(
        data_loader.nfl,
        "load_player_stats",
        lambda season, summary_level: calls["players"].append((season, summary_level)) or player_frame,
    )
    monkeypatch.setattr(
        data_loader.nfl,
        "load_schedules",
        lambda seasons: calls["schedules"].append(seasons) or schedule_frame,
    )
    monkeypatch.setattr(data_loader.nfl, "load_teams", lambda: team_frame)
    monkeypatch.setattr(data_loader, "update_config", lambda **kwargs: calls["config"].append(kwargs))
    return calls, pbp_frame, player_frame, schedule_frame, team_frame


def test_loader_selects_seasons_converts_to_pandas_and_preserves_compatibility(monkeypatch):
    calls, *_frames = _install_complete_loaders(monkeypatch)

    inputs = data_loader.load_expected_points_inputs(2010, 2011, 2)

    assert calls["config"] == [{"cache_mode": "off", "verbose": False}]
    assert calls["pbp"] == [2009, 2010, 2011]
    assert calls["players"] == [(2009, "week"), (2010, "week"), (2011, "week")]
    assert calls["schedules"] == [[2010, 2011]]
    assert isinstance(inputs.pbp, pd.DataFrame)
    assert list(inputs.pbp.columns) == list(data_loader.PBP_REQUIRED_COLUMNS)
    assert inputs.pbp["epa"].dtype == "float32"
    assert inputs.pbp["qb_epa"].dtype == "float32"
    assert inputs.pbp.loc[0, "home_team"] == "LV"
    assert inputs.pbp.loc[0, "away_team"] == "LAC"
    assert inputs.pbp.loc[0, "posteam"] == "LA"
    assert inputs.pbp.loc[0, "defteam"] == "LV"
    assert inputs.pbp.loc[0, "side_of_field"] == "SD"
    assert inputs.schedules.loc[0, "home_team"] == "LV"
    assert inputs.schedules.loc[0, "away_team"] == "LA"
    assert "recent_team" in inputs.player_stats
    assert "interceptions" in inputs.player_stats
    assert "team" not in inputs.player_stats
    assert "passing_interceptions" not in inputs.player_stats


def test_loader_reports_dataset_specific_missing_columns(monkeypatch):
    calls, pbp_frame, _player_frame, _schedule_frame, _team_frame = _install_complete_loaders(monkeypatch)
    incomplete = pbp_frame.drop("qb_epa")
    monkeypatch.setattr(data_loader.nfl, "load_pbp", lambda _season: incomplete)

    with pytest.raises(ValueError, match="PBP data is missing required columns: qb_epa"):
        data_loader.load_expected_points_inputs(2010, 2010, 2)

    assert calls["config"] == [{"cache_mode": "off", "verbose": False}]


def test_week_one_skips_only_an_unavailable_current_season(monkeypatch):
    calls, pbp_frame, player_frame, _schedule_frame, _team_frame = _install_complete_loaders(monkeypatch)

    def load_pbp(season):
        calls["pbp"].append(season)
        if season == 2011:
            raise ValueError("2011 is not current")
        return pbp_frame

    def load_players(season, summary_level):
        calls["players"].append((season, summary_level))
        if season == 2011:
            raise ConnectionError("404 Client Error")
        return player_frame

    monkeypatch.setattr(data_loader.nfl, "load_pbp", load_pbp)
    monkeypatch.setattr(data_loader.nfl, "load_player_stats", load_players)

    inputs = data_loader.load_expected_points_inputs(2010, 2011, 1)

    assert len(inputs.pbp) == 2
    assert len(inputs.player_stats) == 2


@pytest.mark.parametrize("current_week", [2, 3])
def test_current_season_failure_after_week_one_is_propagated(monkeypatch, current_week):
    calls, pbp_frame, _player_frame, _schedule_frame, _team_frame = _install_complete_loaders(monkeypatch)

    def load_pbp(season):
        calls["pbp"].append(season)
        if season == 2011:
            raise ValueError("2011 is not current")
        return pbp_frame

    monkeypatch.setattr(data_loader.nfl, "load_pbp", load_pbp)

    with pytest.raises(ValueError, match="2011 is not current"):
        data_loader.load_expected_points_inputs(2010, 2011, current_week)


def test_historical_season_failure_is_propagated_during_week_one(monkeypatch):
    calls, _pbp_frame, _player_frame, _schedule_frame, _team_frame = _install_complete_loaders(monkeypatch)

    def load_pbp(season):
        calls["pbp"].append(season)
        if season == 2010:
            raise ValueError("historical release unavailable")
        return _frame(data_loader.PBP_REQUIRED_COLUMNS)

    monkeypatch.setattr(data_loader.nfl, "load_pbp", load_pbp)

    with pytest.raises(ValueError, match="historical release unavailable"):
        data_loader.load_expected_points_inputs(2010, 2011, 1)
