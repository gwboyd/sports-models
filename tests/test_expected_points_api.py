from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.sports.football import expected_points_api


def pick_row():
    return {
        "season": 2026,
        "week": "1",
        "home_team": "A",
        "away_team": "B",
        "home_conference": "SEC",
        "away_conference": "Big Ten",
        "home_score_pred": 27.0,
        "away_score_pred": 20.0,
        "spread_pred": -7.0,
        "spread_line": -3.5,
        "spread_play": "A",
        "spread_win_prob": 61.0,
        "spread_lock": 1,
        "total_pred": 47.0,
        "total_line": 44.5,
        "total_play": "over",
        "total_win_prob": 58.0,
        "total_lock": 1,
        "game_id": "1",
        "year_week": "2026_1",
        "date_time": "2026-09-01-17:00",
        "write_time": "2026-08-01 00:00:00",
    }


def make_client(update_runner):
    routers = expected_points_api.build_expected_points_routers(ExpectedPointsLeague.CFB, update_runner)
    app = FastAPI()
    app.include_router(routers.picks)
    app.include_router(routers.results)
    app.include_router(routers.update)
    return TestClient(app)


def test_cfb_picks_and_update_routes(monkeypatch):
    monkeypatch.setattr(expected_points_api, "get_expected_points_picks", lambda *_args, **_kwargs: [pick_row()])
    runner_result = {
        "write_time": "2026-08-01 00:00:00",
        "week": 1,
        "season": 2026,
        "environment": "TEST",
        "client_name": "pytest",
        "runtime": 1.0,
        "pick_changes": 1,
        "pick_changes_games": ["1"],
        "play_changes": 1,
        "play_changes_games": ["1"],
        "updates_skipped": 0,
        "picks_num": 1,
        "database_updated": True,
    }
    client = make_client(lambda _request, _client: runner_result)

    picks_response = client.get("/cfb-picks")
    update_response = client.post(
        "/cfb-update-picks",
        headers={"client-name": "pytest"},
        json={"season": 2026, "week": 1},
    )

    assert picks_response.status_code == 200
    assert picks_response.json()[0]["game_id"] == "1"
    assert picks_response.json()[0]["home_conference"] == "SEC"
    assert picks_response.json()[0]["away_conference"] == "Big Ten"
    assert update_response.status_code == 200
    assert update_response.json()["data"]["database_updated"] is True


def test_cfb_picks_returns_404_for_empty_database(monkeypatch):
    monkeypatch.setattr(expected_points_api, "get_expected_points_picks", lambda *_args, **_kwargs: [])
    client = make_client(lambda *_args: {})
    assert client.get("/cfb-picks").status_code == 404
