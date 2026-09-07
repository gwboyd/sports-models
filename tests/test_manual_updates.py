from datetime import datetime, timedelta, timezone
import json

from botocore.exceptions import ClientError
import pytest

from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.sports.football import manual_updates as manual
from src.sports.football.schedule_coordinator import ScheduledGame


NOW = datetime(2026, 9, 6, 16, 0, 15, tzinfo=timezone.utc)


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


class Scheduler:
    def __init__(self):
        self.rows = {}
        self.creates = 0
        self.fail_after_create = False
        self.conflict_on_create = False

    def get_schedule(self, Name, GroupName):
        if Name not in self.rows:
            raise ClientError({'Error': {'Code': 'ResourceNotFoundException'}}, 'GetSchedule')
        return self.rows[Name]

    def create_schedule(self, Name, GroupName, **kwargs):
        self.creates += 1
        self.rows[Name] = kwargs
        if self.conflict_on_create:
            raise ClientError({'Error': {'Code': 'ConflictException'}}, 'CreateSchedule')
        if self.fail_after_create:
            raise RuntimeError('response lost')


@pytest.fixture
def setup(monkeypatch):
    rows = {}
    scheduler = Scheduler()
    monkeypatch.setattr(manual, 'datetime', Clock)
    monkeypatch.setattr(manual, 'is_aws_lambda_runtime', lambda: True)
    monkeypatch.setattr(manual, 'scheduler_configuration', lambda: {
        'training_function_arn': 'training', 'scheduler_target_role_arn': 'role',
        'schedule_group_name': 'group', 'scheduler_dlq_arn': 'dlq',
    })
    monkeypatch.setattr(manual.boto3, 'client', lambda *_a, **_k: scheduler)
    monkeypatch.setattr(manual, 'get_model_update_job', lambda key: dict(rows[key]) if key in rows else None)

    def insert(record):
        rows.setdefault(record['run_key'], dict(record, trigger_source='api', status='planned',
                                                attempt_count=0, created_at=NOW))
        return dict(rows[record['run_key']])

    monkeypatch.setattr(manual, 'insert_manual_model_update', insert)
    monkeypatch.setattr(manual, 'confirm_manual_model_update_schedule', lambda key: rows[key].update(status='scheduled') if rows[key]['status'] == 'planned' else None)
    monkeypatch.setattr(manual, 'get_expected_points_picks', lambda *_a, **_k: [])
    monkeypatch.setattr(manual, 'load_schedule', lambda league, **_k: [
        ScheduledGame(league, 2026, 1, NOW + timedelta(hours=1), False)
    ])
    return rows, scheduler


def test_submit_current_week_then_replay_without_loading_feeds(setup, monkeypatch):
    rows, scheduler = setup
    first = manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will', 'request-1')
    assert first['status'] == 'scheduled'
    assert (first['season'], first['week'], first['client_name']) == (2026, 1, 'will')
    assert timedelta(minutes=1) <= first['scheduled_for'] - NOW < timedelta(minutes=2)
    config = next(iter(scheduler.rows.values()))
    assert config['ActionAfterCompletion'] == 'DELETE'
    assert config['FlexibleTimeWindow'] == {'Mode': 'OFF'}
    assert json.loads(config['Target']['Input']) == {
        'job': 'expected_points_update', 'league': 'nfl', 'season': 2026, 'week': 1,
        'run_key': first['run_key'],
    }
    monkeypatch.setattr(manual, 'load_schedule', lambda *_a, **_k: pytest.fail('replay loaded feed'))
    # Client metadata on the accepted plan is immutable too.
    second = manual.submit_manual_update(ExpectedPointsLeague.NFL, 'different-client', 'request-1')
    assert second == first
    assert scheduler.creates == 1
    assert len(rows) == 1


def test_optional_keys_create_fresh_jobs_and_keys_are_scoped_by_league(setup):
    rows, _ = setup
    manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will')
    manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will')
    manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will', 'same')
    manual.submit_manual_update(ExpectedPointsLeague.CFB, 'will', 'same')
    assert len(rows) == 4


def test_replay_after_uncertain_aws_create_confirms_existing_schedule(setup):
    rows, scheduler = setup
    scheduler.fail_after_create = True
    with pytest.raises(RuntimeError, match='response lost'):
        manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will', 'same')
    assert next(iter(rows.values()))['status'] == 'planned'
    original_time = next(iter(rows.values()))['scheduled_for']
    result = manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will', 'same')
    assert result['status'] == 'scheduled'
    assert result['scheduled_for'] == original_time
    assert scheduler.creates == 1


def test_concurrent_aws_create_conflict_accepts_only_matching_schedule(setup):
    _, scheduler = setup
    scheduler.conflict_on_create = True
    assert manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will', 'same')['status'] == 'scheduled'


def test_completed_job_with_deleted_schedule_is_never_recreated(setup, monkeypatch):
    rows, scheduler = setup
    job = manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will', 'same')
    rows[job['run_key']].update(status='completed', update_id=42)
    monkeypatch.setattr(manual, 'get_model_update_job_result', lambda league, id: {
        'id': id, 'model_version': '2.1', 'source_git_sha': 'executed-sha',
    })
    scheduler.rows.clear()
    result = manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will', 'same')
    assert result['data']['model_version'] == '2.1'
    assert scheduler.creates == 1
    assert result['outcome_unconfirmed'] is False


def test_planned_job_past_delivery_is_reported_uncertain_without_recreation(setup):
    rows, scheduler = setup
    job = manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will', 'same')
    rows[job['run_key']].update(status='planned', scheduled_for=NOW - timedelta(minutes=1))
    scheduler.rows.clear()
    result = manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will', 'same')
    assert result['outcome_unconfirmed'] is True
    assert scheduler.creates == 1


def test_replay_does_not_rewrite_mismatched_aws_schedule(setup):
    rows, scheduler = setup
    job = manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will', 'same')
    rows[job['run_key']]['status'] = 'planned'
    next(iter(scheduler.rows.values()))['Target']['Arn'] = 'unexpected-function'
    with pytest.raises(RuntimeError, match='immutable plan'):
        manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will', 'same')


@pytest.mark.parametrize('completed, hours', [(True, -1), (False, -1)])
def test_no_eligible_upcoming_games_do_not_create_a_job(setup, monkeypatch, completed, hours):
    rows, scheduler = setup
    monkeypatch.setattr(manual, 'load_schedule', lambda league, **_k: [
        ScheduledGame(league, 2026, 1, NOW + timedelta(hours=hours), completed)
    ])
    monkeypatch.setattr(manual, 'get_expected_points_picks', lambda *_a, **_k: [{'season': 2026, 'week': 1}])
    assert manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will')['status'] == 'not_scheduled'
    assert not rows and not scheduler.rows


def test_feed_failure_does_not_guess_week_or_insert_job(setup, monkeypatch):
    rows, _ = setup
    def fail(*_a, **_k):
        raise RuntimeError('feed unavailable')
    monkeypatch.setattr(manual, 'load_schedule', fail)
    with pytest.raises(RuntimeError, match='feed unavailable'):
        manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will')
    assert not rows


def test_local_and_sam_requests_cannot_even_read_or_create_plans(setup, monkeypatch):
    monkeypatch.setattr(manual, 'is_aws_lambda_runtime', lambda: False)
    monkeypatch.setattr(manual, 'get_model_update_job', lambda *_a: pytest.fail('local accessed job DB'))
    with pytest.raises(PermissionError, match='deployed AWS API'):
        manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will')


@pytest.mark.parametrize('state', ['scheduled', 'running', 'failed'])
def test_stale_status_is_read_only_and_does_not_claim_terminal_failure(setup, state):
    row = {'run_key': 'api:nfl:1', 'league': 'nfl', 'status': state,
           'scheduled_for': NOW - timedelta(hours=4)}
    result = manual.job_status(row)
    assert result['status'] == state
    assert result['outcome_unconfirmed'] is True
    assert 'outcome_unconfirmed' not in row


def test_acceptance_record_does_not_overwrite_completed_or_failed_worker(setup, monkeypatch):
    rows, _ = setup
    def delivered_during_create(client, *, plan, **kwargs):
        rows[plan.run_key]['status'] = 'failed'
        rows[plan.run_key]['last_error'] = 'training failed immediately'
    monkeypatch.setattr(manual, 'ensure_update_schedule', delivered_during_create)
    result = manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will', 'same')
    assert result['status'] == 'failed'
    assert result['last_error'] == 'training failed immediately'


@pytest.mark.parametrize('minute, expected', [(50, 'not_scheduled'), (61, 'scheduled')])
def test_manual_selection_honors_actual_five_am_rollover(setup, monkeypatch, minute, expected):
    compare_time = datetime(2026, 9, 7, 8, tzinfo=timezone.utc) + timedelta(minutes=minute)
    class RolloverClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return compare_time
    monkeypatch.setattr(manual, 'datetime', RolloverClock)
    monkeypatch.setattr(manual, 'load_schedule', lambda league, **_k: [
        ScheduledGame(league, 2026, 1, datetime(2026, 9, 7, 0, tzinfo=timezone.utc), True),
        ScheduledGame(league, 2026, 2, datetime(2026, 9, 13, 17, tzinfo=timezone.utc), False),
    ])
    monkeypatch.setattr(manual, 'get_expected_points_picks', lambda *_a, **_k: [{'season': 2026, 'week': 1}])
    result = manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will', 'rollover')
    assert result['status'] == expected
    if expected == 'scheduled':
        assert result['week'] == 2


def test_manual_schedule_must_delete_itself_after_delivery(setup):
    rows, scheduler = setup
    job = manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will', 'same')
    rows[job['run_key']]['status'] = 'planned'
    next(iter(scheduler.rows.values()))['ActionAfterCompletion'] = 'NONE'
    with pytest.raises(RuntimeError, match='immutable plan'):
        manual.submit_manual_update(ExpectedPointsLeague.NFL, 'will', 'same')
