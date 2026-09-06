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


def make_client():
    routers = expected_points_api.build_expected_points_routers(ExpectedPointsLeague.CFB)
    app = FastAPI()
    app.include_router(routers.picks)
    app.include_router(routers.results)
    app.include_router(routers.update)
    app.include_router(expected_points_api.jobs)
    return TestClient(app)


def test_cfb_picks_and_update_routes(monkeypatch):
    monkeypatch.setattr(expected_points_api, "get_expected_points_picks", lambda *_a, **_k: [pick_row()])
    observed = []
    def submit(*args):
        observed.append(args)
        return {"status": "scheduled", "league": "cfb", "run_key": "api:cfb:1",
                "season": 2026, "week": 1, "status_url": "/model-update-jobs/api:cfb:1"}
    monkeypatch.setattr(expected_points_api, "submit_manual_update", submit)
    client = make_client()
    picks = client.get("/cfb-picks")
    assert picks.json()[0]["home_conference"] == "SEC"
    result = client.post("/cfb-update-picks", headers={"client-name": "will", "Idempotency-Key": "one"})
    assert result.status_code == 202
    assert result.headers["Location"] == result.json()["status_url"]
    assert result.headers["Idempotency-Key"] == "one"
    assert observed == [(ExpectedPointsLeague.CFB, "will", "one")]


def test_cfb_picks_returns_404_for_empty_database(monkeypatch):
    monkeypatch.setattr(expected_points_api, "get_expected_points_picks", lambda *_args, **_kwargs: [])
    client = make_client()
    assert client.get("/cfb-picks").status_code == 404


def test_update_rejects_explicit_week_and_write_flag(monkeypatch):
    monkeypatch.setattr(expected_points_api, "submit_manual_update", lambda *_a: (_ for _ in ()).throw(AssertionError()))
    client = make_client()
    for body in [{"season": 2026, "week": 1}, {"allow_non_aws_write": True}]:
        result = client.post("/cfb-update-picks", headers={"client-name": "will"}, json=body)
        assert result.status_code == 422


def test_failed_submission_returns_generated_key_for_safe_retry(monkeypatch):
    observed = []
    def fail(*args):
        observed.append(args)
        raise RuntimeError("private database error")
    monkeypatch.setattr(expected_points_api, "submit_manual_update", fail)
    result = make_client().post("/cfb-update-picks", headers={"client-name": "will"})
    assert result.status_code == 503
    assert result.headers["Idempotency-Key"] == observed[0][2]
    assert "private database error" not in result.text


def test_local_http_update_returns_403(monkeypatch):
    from src.sports.football import manual_updates
    monkeypatch.setattr(manual_updates, "is_aws_lambda_runtime", lambda: False)
    assert make_client().post("/cfb-update-picks", headers={"client-name": "will"}).status_code == 403


def test_status_read_and_completed_replay(monkeypatch):
    job = {"status": "completed", "league": "cfb", "run_key": "api:cfb:1", "update_id": 42}
    monkeypatch.setattr(expected_points_api, "get_manual_update_status", lambda *_a: job)
    monkeypatch.setattr(expected_points_api, "submit_manual_update", lambda *_a: job)
    client = make_client()
    assert client.get("/model-update-jobs/api:cfb:1").json()["update_id"] == 42
    assert client.post("/cfb-update-picks", headers={"client-name": "will"}).status_code == 200
    monkeypatch.setattr(expected_points_api, "get_manual_update_status", lambda *_a: None)
    assert client.get("/model-update-jobs/missing").status_code == 404


def test_update_and_status_routes_require_admin(monkeypatch):
    import main
    monkeypatch.delenv("LOCALHOST", raising=False)
    monkeypatch.setattr(main, "API_KEYS", {main.hash_key("reader"): ["read"], main.hash_key("admin"): ["admin"]})
    monkeypatch.setattr(expected_points_api, "get_manual_update_status", lambda *_a: {"status": "running", "league": "nfl"})
    monkeypatch.setattr(expected_points_api, "submit_manual_update", lambda *_a: {"status": "scheduled", "league": "nfl"})
    client = TestClient(main.app)
    for url, method in [("/nfl-update-picks", "post"), ("/model-update-jobs/api:nfl:1", "get")]:
        request = getattr(client, method)
        assert request(url, headers={"client-name": "will"}).status_code == 401
        assert request(url, headers={"Authorization": "reader", "client-name": "will"}).status_code == 403
        assert request(url, headers={"Authorization": "admin", "client-name": "will"}).status_code in (200, 202)


def test_status_serializes_legacy_update_without_source_sha(monkeypatch):
    job = {
        "status": "completed", "league": "nfl", "run_key": "aws-scheduler:nfl:legacy",
        "update_id": 42,
        "data": {
            "id": 42, "model_version": "1.0", "source_git_sha": None,
            "write_time": "2026-09-01T12:00:00Z", "runtime": 120,
            "picks_num": 16, "pick_changes": 1, "play_changes": 2, "updates_skipped": 0,
        },
    }
    monkeypatch.setattr(expected_points_api, "get_manual_update_status", lambda *_a: job)
    response = make_client().get("/model-update-jobs/aws-scheduler:nfl:legacy")
    assert response.status_code == 200
    assert response.json()["data"]["model_version"] == "1.0"
    assert "source_git_sha" not in response.json()["data"]


def test_interactive_notebook_client_is_rejected_before_submission(monkeypatch):
    monkeypatch.setattr(expected_points_api, 'submit_manual_update', lambda *_a: (_ for _ in ()).throw(AssertionError()))
    result = make_client().post('/cfb-update-picks', headers={'client-name': 'notebook'})
    assert result.status_code == 422
    assert 'reserved' in result.json()['detail']
