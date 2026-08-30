import pytest

import main
from src.sports.football import scheduled_updates


def scheduled_event(**overrides):
    event = {
        "job": "expected_points_update",
        "league": "nfl",
        "season": 2026,
        "week": 1,
        "run_key": "aws-scheduler:nfl:2026:1:opening",
    }
    event.update(overrides)
    return event


def test_scheduled_update_runs_explicit_plan_without_http(monkeypatch):
    observed = {}
    monkeypatch.setattr(scheduled_updates, "is_aws_lambda_runtime", lambda: False)

    def runner(request, client_name, *, run_key):
        observed.update(
            season=request.season,
            week=request.week,
            allow_non_aws_write=request.allow_non_aws_write,
            client_name=client_name,
            run_key=run_key,
        )
        return {"database_updated": False}

    monkeypatch.setattr(scheduled_updates, "_update_runner", lambda _league: runner)

    result = scheduled_updates.run_scheduled_expected_points_update(scheduled_event())

    assert result["status"] == "success"
    assert observed == {
        "season": 2026,
        "week": 1,
        "allow_non_aws_write": False,
        "client_name": "aws-scheduler",
        "run_key": "aws-scheduler:nfl:2026:1:opening",
    }


def test_scheduled_update_skips_completed_plan(monkeypatch):
    from src.utils.db import sports_models_db

    monkeypatch.setattr(scheduled_updates, "is_aws_lambda_runtime", lambda: True)
    monkeypatch.setattr(sports_models_db, "claim_scheduled_model_update", lambda *_a: "completed")
    monkeypatch.setattr(
        scheduled_updates,
        "_update_runner",
        lambda _league: pytest.fail("completed scheduled run should not execute"),
    )

    result = scheduled_updates.run_scheduled_expected_points_update(scheduled_event())

    assert result["status"] == "skipped"
    assert result["reason"] == "scheduled run is completed"


def test_aws_scheduled_update_claims_and_verifies_atomic_link(monkeypatch):
    from src.utils.db import sports_models_db

    monkeypatch.setattr(scheduled_updates, "is_aws_lambda_runtime", lambda: True)
    monkeypatch.setattr(sports_models_db, "claim_scheduled_model_update", lambda *_a: "claimed")
    monkeypatch.setattr(sports_models_db, "is_scheduled_model_update_completed", lambda *_a: True)
    monkeypatch.setattr(
        scheduled_updates,
        "_update_runner",
        lambda _league: lambda _request, _client, **_kwargs: {"database_updated": True},
    )

    result = scheduled_updates.run_scheduled_expected_points_update(
        scheduled_event(league="cfb", run_key="aws-scheduler:cfb:2026:1:opening")
    )

    assert result["status"] == "success"


def test_aws_scheduled_update_marks_failed_run_for_retry(monkeypatch):
    from src.utils.db import sports_models_db

    statuses = []
    monkeypatch.setattr(scheduled_updates, "is_aws_lambda_runtime", lambda: True)
    monkeypatch.setattr(sports_models_db, "claim_scheduled_model_update", lambda *_a: "claimed")
    monkeypatch.setattr(sports_models_db, "is_scheduled_model_update_completed", lambda *_a: False)
    monkeypatch.setattr(
        sports_models_db,
        "set_scheduled_model_update_status",
        lambda run_key, status, **kwargs: statuses.append((run_key, status, kwargs)),
    )

    def fail(_request, _client, **_kwargs):
        raise RuntimeError("notebook failed")

    monkeypatch.setattr(scheduled_updates, "_update_runner", lambda _league: fail)

    with pytest.raises(RuntimeError, match="notebook failed"):
        scheduled_updates.run_scheduled_expected_points_update(scheduled_event())

    assert statuses == [
        (
            "aws-scheduler:nfl:2026:1:opening",
            "failed",
            {"error": "notebook failed"},
        )
    ]


def test_aws_scheduled_update_recovers_post_commit_failure(monkeypatch):
    from src.utils.db import sports_models_db

    monkeypatch.setattr(scheduled_updates, "is_aws_lambda_runtime", lambda: True)
    monkeypatch.setattr(sports_models_db, "claim_scheduled_model_update", lambda *_a: "claimed")
    monkeypatch.setattr(sports_models_db, "is_scheduled_model_update_completed", lambda *_a: True)
    monkeypatch.setattr(
        sports_models_db,
        "set_scheduled_model_update_status",
        lambda *_a, **_k: pytest.fail("a committed run must not be marked failed"),
    )

    def fail_after_write(_request, _client, **_kwargs):
        raise RuntimeError("result file was unavailable after commit")

    monkeypatch.setattr(scheduled_updates, "_update_runner", lambda _league: fail_after_write)

    result = scheduled_updates.run_scheduled_expected_points_update(scheduled_event())

    assert result["status"] == "success"
    assert result["recovered"] is True


def test_main_handler_dispatches_scheduled_event(monkeypatch):
    event = scheduled_event()
    monkeypatch.setattr(
        main,
        "run_scheduled_expected_points_update",
        lambda value: {"event": value},
    )

    assert main.handler(event, {}) == {"event": event}


def test_main_handler_keeps_http_events_on_mangum(monkeypatch):
    event = {"requestContext": {"http": {"method": "GET"}}}
    monkeypatch.setattr(main, "api_handler", lambda value, context: (value, context))

    assert main.handler(event, {"request_id": "test"}) == (event, {"request_id": "test"})


@pytest.mark.parametrize(
    "event, message",
    [
        ({"job": "expected_points_update"}, "league='nfl' or league='cfb'"),
        (scheduled_event(season=None), "season must be an integer"),
        (scheduled_event(run_key=None), "run_key must start"),
    ],
)
def test_scheduled_update_rejects_invalid_events(event, message):
    with pytest.raises(ValueError, match=message):
        scheduled_updates.run_scheduled_expected_points_update(event)
