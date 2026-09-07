from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.utils.db import sports_models_db


class FakeCursor:
    def __init__(self):
        self.calls = []
        self.return_time = datetime(2026, 8, 1, tzinfo=timezone.utc)
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=None):
        self.calls.append(("execute", query, params))

    def executemany(self, query, params):
        self.calls.append(("executemany", query, params))

    def fetchone(self):
        return {"id": 42, "write_time": self.return_time}


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def update_record(cursor, *, client_name="pytest"):
    return {
        "year_week": "2026_1",
        "write_time": cursor.return_time,
        "week": 1,
        "season": 2026,
        "environment": "TEST",
        "client_name": client_name,
        "runtime": 1.0,
        "model_version": "1.0",
        "pick_changes": 0,
        "pick_changes_games": [],
        "play_changes": 0,
        "play_changes_games": [],
        "updates_skipped": 0,
        "picks_num": 0,
        "difference_df": [],
        "picks_df": [],
    }


def test_atomic_cfb_run_uses_only_cfb_tables(monkeypatch):
    cursor = FakeCursor()

    @contextmanager
    def fake_connection():
        yield FakeConnection(cursor)

    monkeypatch.setattr(sports_models_db, "get_connection", fake_connection)
    update = update_record(cursor)
    pick_record = {column: 1 for column in sports_models_db.PICK_COLUMNS}
    pick_record.update(
        {
            "week": "1",
            "year_week": "2026_1",
            "game_id": "1",
            "model_version": "1.0",
            "home_conference": "SEC",
            "away_conference": "Big Ten",
        }
    )

    result = sports_models_db.write_expected_points_run(
        ExpectedPointsLeague.CFB,
        [pick_record],
        update,
        write_authorized=True,
    )

    assert result == cursor.return_time
    sql = "\n".join(call[1] for call in cursor.calls)
    assert "cfb_expected_points_pick_updates" in sql
    assert "cfb_expected_points_picks" in sql
    assert "home_conference" in sql
    assert "away_conference" in sql
    assert "nfl_expected_points" not in sql


def test_non_aws_database_write_requires_explicit_authorization(monkeypatch):
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    with pytest.raises(PermissionError, match="explicit authorization"):
        sports_models_db.write_expected_points_run(
            ExpectedPointsLeague.NFL,
            [],
            {},
        )


def test_scheduled_run_links_update_id_and_completion_in_same_transaction(monkeypatch):
    cursor = FakeCursor()
    monkeypatch.setenv(
        "EXPECTED_POINTS_RUN_KEY",
        "aws-scheduler:nfl:2026:1:daily:2026-09-01",
    )

    @contextmanager
    def fake_connection():
        yield FakeConnection(cursor)

    monkeypatch.setattr(sports_models_db, "get_connection", fake_connection)
    sports_models_db.write_expected_points_run(
        ExpectedPointsLeague.NFL,
        [],
        update_record(cursor, client_name="aws-scheduler"),
        write_authorized=True,
    )

    sql = "\n".join(call[1] for call in cursor.calls)
    assert "set update_id = %s" in sql
    assert "status = 'completed'" in sql
    completion_call = next(call for call in cursor.calls if "set update_id = %s" in call[1])
    assert completion_call[2] == (
        42,
        "aws-scheduler:nfl:2026:1:daily:2026-09-01",
        "nfl",
        "nfl_expected_points",
        2026,
        1,
    )


def test_manual_run_does_not_touch_schedule_table(monkeypatch):
    cursor = FakeCursor()
    monkeypatch.delenv("EXPECTED_POINTS_RUN_KEY", raising=False)

    @contextmanager
    def fake_connection():
        yield FakeConnection(cursor)

    monkeypatch.setattr(sports_models_db, "get_connection", fake_connection)
    sports_models_db.write_expected_points_run(
        ExpectedPointsLeague.NFL,
        [],
        update_record(cursor, client_name="notebook"),
        write_authorized=True,
    )

    assert not any("set update_id = %s" in call[1] for call in cursor.calls)
    assert not any("set first_pick_at" in call[1] for call in cursor.calls)


def test_real_aws_run_marks_release_live(monkeypatch):
    cursor = FakeCursor()
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "sports-models-training")
    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)

    @contextmanager
    def fake_connection():
        yield FakeConnection(cursor)

    monkeypatch.setattr(sports_models_db, "get_connection", fake_connection)
    sports_models_db.write_expected_points_run(
        ExpectedPointsLeague.NFL,
        [],
        update_record(cursor, client_name="aws-scheduler"),
    )

    live_call = next(call for call in cursor.calls if "set first_pick_at" in call[1])
    assert live_call[2] == (cursor.return_time, "nfl_expected_points", "1.0")


def test_invalid_league_cannot_be_used_as_identifier():
    try:
        sports_models_db.get_expected_points_picks("cfb; drop table x")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid league identifier was accepted")


def test_recipe_source_uses_immutable_registry_sha_for_requested_version(monkeypatch):
    cursor = FakeCursor()
    release = {
        "version": "1.0",
        "source_git_sha": "release-sha",
        "deployed_at": cursor.return_time,
        "first_pick_at": cursor.return_time,
    }
    cursor.fetchone = lambda: release

    @contextmanager
    def fake_connection():
        yield FakeConnection(cursor)

    monkeypatch.setattr(sports_models_db, "get_connection", fake_connection)
    resolved = sports_models_db.get_expected_points_recipe_source(
        ExpectedPointsLeague.NFL,
        version="1.0",
    )

    assert resolved["source_git_sha"] == "release-sha"
    assert cursor.calls[0][2] == ("nfl_expected_points", "1.0")
    assert "first_pick_at is not null" in cursor.calls[0][1]
    assert len(cursor.calls) == 1


def test_recipe_source_returns_none_without_a_live_registry_release(monkeypatch):
    cursor = FakeCursor()
    cursor.fetchone = lambda: None

    @contextmanager
    def fake_connection():
        yield FakeConnection(cursor)

    monkeypatch.setattr(sports_models_db, "get_connection", fake_connection)
    resolved = sports_models_db.get_expected_points_recipe_source(ExpectedPointsLeague.CFB)

    assert resolved is None
    assert cursor.calls[0][2] == ("cfb_expected_points",)
    assert len(cursor.calls) == 1


def test_bootstrap_release_source_is_initialized_only_when_missing(monkeypatch):
    cursor = FakeCursor()
    cursor.fetchone = lambda: {"source_git_sha": "first-protocol-sha"}

    @contextmanager
    def fake_connection():
        yield FakeConnection(cursor)

    monkeypatch.setattr(sports_models_db, "get_connection", fake_connection)
    resolved = sports_models_db.initialize_model_release_source(
        "nfl_expected_points",
        "1.0",
        source_git_sha="first-protocol-sha",
    )

    assert resolved == "first-protocol-sha"
    assert "source_git_sha is null" in cursor.calls[0][1]
    assert cursor.calls[0][2] == (
        "first-protocol-sha",
        "nfl_expected_points",
        "1.0",
    )
    assert cursor.calls[1][2] == ("nfl_expected_points", "1.0")


def test_bootstrap_release_source_never_replaces_an_existing_sha(monkeypatch):
    cursor = FakeCursor()
    cursor.fetchone = lambda: {"source_git_sha": "original-sha"}

    @contextmanager
    def fake_connection():
        yield FakeConnection(cursor)

    monkeypatch.setattr(sports_models_db, "get_connection", fake_connection)
    resolved = sports_models_db.initialize_model_release_source(
        "cfb_expected_points",
        "1.0",
        source_git_sha="later-deployment-sha",
    )

    assert resolved == "original-sha"
    assert "source_git_sha is null" in cursor.calls[0][1]


def test_scheduled_update_claim_is_atomic_and_leased(monkeypatch):
    cursor = FakeCursor()
    cursor.fetchone = lambda: {"status": "running"}

    @contextmanager
    def fake_connection():
        yield FakeConnection(cursor)

    monkeypatch.setattr(sports_models_db, "get_connection", fake_connection)

    result = sports_models_db.claim_scheduled_model_update(
        ExpectedPointsLeague.NFL,
        "aws-scheduler:nfl:2026:1:opening",
        2026,
        1,
        lease=timedelta(minutes=16),
    )

    assert result == "claimed"
    _, sql, params = cursor.calls[0]
    assert "status = 'running'" in sql
    assert "attempt_count = attempt_count + 1" in sql
    assert "claimed_at < now() - %s" in sql
    assert params == (
        "aws-scheduler:nfl:2026:1:opening",
        "nfl",
        "nfl_expected_points",
        2026,
        1,
        timedelta(minutes=16),
    )


def test_scheduled_update_claim_reports_existing_terminal_status(monkeypatch):
    cursor = FakeCursor()
    responses = iter([None, {"status": "completed"}])
    cursor.fetchone = lambda: next(responses)

    @contextmanager
    def fake_connection():
        yield FakeConnection(cursor)

    monkeypatch.setattr(sports_models_db, "get_connection", fake_connection)

    assert sports_models_db.claim_scheduled_model_update(
        ExpectedPointsLeague.CFB,
        "aws-scheduler:cfb:2026:1:opening",
        2026,
        1,
    ) == "completed"
    assert len(cursor.calls) == 2


def test_manual_api_run_links_versioned_update_in_same_transaction(monkeypatch):
    cursor = FakeCursor()
    transactions = []
    monkeypatch.setenv('AWS_LAMBDA_FUNCTION_NAME', 'training')
    monkeypatch.delenv('AWS_SAM_LOCAL', raising=False)
    monkeypatch.setenv('EXPECTED_POINTS_RUN_KEY', 'api:nfl:abc')

    @contextmanager
    def connection():
        transactions.append('begin')
        try:
            yield FakeConnection(cursor)
            transactions.append('commit')
        except Exception:
            transactions.append('rollback')
            raise

    monkeypatch.setattr(sports_models_db, 'get_connection', connection)
    record = update_record(cursor, client_name='will')
    record.update(model_version='2.3', source_git_sha='executing-sha')
    sports_models_db.write_expected_points_run(ExpectedPointsLeague.NFL, [], record)
    completion = next(call for call in cursor.calls if 'set update_id = %s' in call[1])
    assert completion[2][:3] == (42, 'api:nfl:abc', 'nfl')
    inserted = next(call for call in cursor.calls if 'insert into' in call[1] and 'pick_updates' in call[1])
    assert inserted[2]['model_version'] == '2.3'
    assert inserted[2]['source_git_sha'] == 'executing-sha'
    assert inserted[2]['client_name'] == 'will'
    assert transactions == ['begin', 'commit']
    cursor.rowcount = 0
    with pytest.raises(RuntimeError, match='completion'):
        sports_models_db.write_expected_points_run(ExpectedPointsLeague.NFL, [], record)
    assert transactions[-2:] == ['begin', 'rollback']


def test_calendar_query_explicitly_excludes_manual_jobs(monkeypatch):
    cursor = FakeCursor()
    cursor.fetchall = lambda: []
    @contextmanager
    def connection():
        yield FakeConnection(cursor)
    monkeypatch.setattr(sports_models_db, 'get_connection', connection)
    sports_models_db.get_scheduled_model_updates(ExpectedPointsLeague.NFL, pending_only=True)
    assert "trigger_source = 'scheduler'" in cursor.calls[0][1]


def test_manual_job_insert_never_rewrites_an_existing_plan(monkeypatch):
    cursor = FakeCursor()
    cursor.fetchone = lambda: {'run_key': 'api:nfl:key', 'season': 2026, 'week': 1}
    @contextmanager
    def connection():
        yield FakeConnection(cursor)
    monkeypatch.setattr(sports_models_db, 'get_connection', connection)
    job = sports_models_db.insert_manual_model_update({'run_key': 'api:nfl:key'})
    assert job['week'] == 1
    assert 'on conflict (run_key) do nothing' in cursor.calls[0][1]


@pytest.mark.parametrize('columns, ready', [(['trigger_source', 'client_name'], True), (['client_name'], False), ([], False)])
def test_manual_schema_preflight_checks_existing_database(monkeypatch, columns, ready):
    cursor = FakeCursor()
    cursor.fetchall = lambda: [{'column_name': name} for name in columns]
    @contextmanager
    def connection():
        yield FakeConnection(cursor)
    monkeypatch.setattr(sports_models_db, 'get_connection', connection)
    if ready:
        sports_models_db.verify_manual_update_schema()
    else:
        with pytest.raises(RuntimeError, match='setup SQL before deployment'):
            sports_models_db.verify_manual_update_schema()
    assert 'information_schema.columns' in cursor.calls[0][1]
