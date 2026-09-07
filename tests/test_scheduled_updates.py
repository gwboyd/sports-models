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


@pytest.mark.parametrize('league', ['nfl', 'cfb'])
def test_manual_event_uses_registered_client_and_same_versioned_runner(monkeypatch, league):
    from src.utils.db import sports_models_db as db
    module = scheduled_updates.nfl_update_picks if league == 'nfl' else scheduled_updates.cfb_update_picks
    run_key = f'api:{league}:abc'
    plan = {'trigger_source': 'api', 'league': league, 'season': 2026, 'week': 1, 'client_name': 'will'}
    monkeypatch.setattr(scheduled_updates, 'is_aws_lambda_runtime', lambda: True)
    monkeypatch.setattr(db, 'get_model_update_job', lambda *_a: plan)
    monkeypatch.setattr(db, 'claim_scheduled_model_update', lambda *_a: 'claimed')
    monkeypatch.setattr(db, 'get_expected_points_picks', lambda *_a, **_k: [{'season': 2026, 'week': '1'}])
    monkeypatch.setattr(db, 'is_scheduled_model_update_completed', lambda *_a: True)
    monkeypatch.setenv(f'{league.upper()}_EXPECTED_POINTS_VERSION', '2.3')
    monkeypatch.setenv('SOURCE_GIT_SHA', 'executing-deployment-sha')
    observed = {}
    def execute(path, **kwargs):
        observed.update(kwargs)
        return {'database_updated': True}
    monkeypatch.setattr(module, 'execute_expected_points_notebook', execute)
    result = scheduled_updates.run_scheduled_expected_points_update(scheduled_event(
        league=league, run_key=run_key, client_name='spoofed-payload-client', model_version='9.9',
    ))
    assert result['status'] == 'success'
    assert observed == {
        'season': 2026, 'week': 1, 'client_name': 'will', 'allow_non_aws_write': False,
        'model_version': '2.3', 'source_git_sha': 'executing-deployment-sha', 'run_key': run_key,
    }


def test_obsolete_manual_job_is_cancelled_without_notebook_or_pick_writes(monkeypatch):
    from src.utils.db import sports_models_db as db
    run_key = 'api:nfl:abc'
    cancelled = []
    monkeypatch.setattr(scheduled_updates, 'is_aws_lambda_runtime', lambda: True)
    monkeypatch.setattr(db, 'get_model_update_job', lambda *_a: {
        'trigger_source': 'api', 'league': 'nfl', 'season': 2026, 'week': 1, 'client_name': 'will',
    })
    monkeypatch.setattr(db, 'claim_scheduled_model_update', lambda *_a: 'claimed')
    monkeypatch.setattr(db, 'get_expected_points_picks', lambda *_a, **_k: [{'season': 2026, 'week': '2'}])
    monkeypatch.setattr(db, 'cancel_obsolete_manual_update', cancelled.append)
    monkeypatch.setattr(scheduled_updates.nfl_update_picks, 'main', lambda *_a, **_k: pytest.fail('obsolete run trained'))
    result = scheduled_updates.run_scheduled_expected_points_update(scheduled_event(run_key=run_key))
    assert result['status'] == 'cancelled'
    assert cancelled == [run_key]


def test_manual_payload_cannot_change_registered_week(monkeypatch):
    from src.utils.db import sports_models_db as db
    monkeypatch.setattr(scheduled_updates, 'is_aws_lambda_runtime', lambda: True)
    monkeypatch.setattr(db, 'get_model_update_job', lambda *_a: {
        'trigger_source': 'api', 'league': 'nfl', 'season': 2026, 'week': 2, 'client_name': 'will',
    })
    monkeypatch.setattr(db, 'claim_scheduled_model_update', lambda *_a: pytest.fail('invalid event claimed'))
    with pytest.raises(ValueError, match='registered plan'):
        scheduled_updates.run_scheduled_expected_points_update(scheduled_event(run_key='api:nfl:abc'))


def test_sam_routes_http_to_api_and_keeps_trainer_serialized():
    from pathlib import Path
    import yaml
    template = yaml.load((Path(__file__).resolve().parents[1] / 'template.yaml').read_text(), Loader=yaml.BaseLoader)
    resources = template['Resources']
    api = resources['ApiLambdaFunction']['Properties']
    trainer = resources['TrainingLambdaFunction']['Properties']
    paths = {event['Properties']['Path'] for event in api['Events'].values()}
    assert {'/nfl-update-picks', '/cfb-update-picks', '/model-update-jobs/{run_key}'} <= paths
    assert 'Events' not in trainer
    assert trainer['ReservedConcurrentExecutions'] == '1'
    assert api['Environment']['Variables']['TRAINING_FUNCTION_ARN'] == 'TrainingLambdaFunction.Arn'
    assert 'NFL_EXPECTED_POINTS_VERSION' not in api['Environment']['Variables']
    permissions = api['Policies'][0]['Statement']
    assert permissions[0]['Action'] == ['scheduler:CreateSchedule', 'scheduler:GetSchedule']
    assert 'manual-' in permissions[0]['Resource']
    assert permissions[1]['Condition']['StringEquals']['iam:PassedToService'] == 'scheduler.amazonaws.com'


def test_active_claim_is_not_acknowledged_as_a_successful_retry(monkeypatch):
    from src.utils.db import sports_models_db as db
    monkeypatch.setattr(scheduled_updates, 'is_aws_lambda_runtime', lambda: True)
    monkeypatch.setattr(db, 'claim_scheduled_model_update', lambda *_a: 'running')
    monkeypatch.setattr(scheduled_updates, '_update_runner', lambda *_a: pytest.fail('active claim ran twice'))
    monkeypatch.setattr(db, 'set_scheduled_model_update_status', lambda *_a, **_k: pytest.fail('active claim was overwritten'))
    with pytest.raises(RuntimeError, match='active claim'):
        scheduled_updates.run_scheduled_expected_points_update(scheduled_event())
