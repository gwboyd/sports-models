from dataclasses import replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from botocore.exceptions import ClientError
import pytest

from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.sports.football import schedule_coordinator


UTC = ZoneInfo("UTC")
EASTERN = ZoneInfo("America/New_York")


def game(
    league,
    kickoff,
    *,
    season=2026,
    week=1,
    completed=False,
):
    return schedule_coordinator.ScheduledGame(
        league=league,
        season=season,
        week=week,
        kickoff=kickoff,
        completed=completed,
    )


def pick(season=2026, week=1, game_id="game-1"):
    return {"season": season, "week": str(week), "game_id": game_id}


def test_initial_week_switches_from_weekly_offseason_to_active_horizon():
    league = ExpectedPointsLeague.CFB
    now = datetime(2026, 9, 1, 12, tzinfo=UTC)
    upcoming = [game(league, datetime(2026, 9, 4, 12, tzinfo=UTC), week=2)]

    selected = schedule_coordinator.select_plannable_week(
        league, games=upcoming, latest_picks=[], now=now
    )
    offseason = schedule_coordinator.select_plannable_week(
        league,
        games=[game(league, datetime(2026, 9, 10, 12, tzinfo=UTC))],
        latest_picks=[],
        now=now,
    )

    assert selected is not None
    assert selected.games[0].week == 2
    assert selected.cadence == "active"
    assert offseason is not None
    assert offseason.cadence == "offseason"


def test_existing_picks_do_not_open_new_season_before_active_horizon():
    league = ExpectedPointsLeague.NFL
    now = datetime(2026, 8, 29, 23, tzinfo=EASTERN)
    week_one = [
        game(league, datetime(2026, 9, 9, 20, 20, tzinfo=EASTERN)),
        game(league, datetime(2026, 9, 13, 13, 0, tzinfo=EASTERN)),
    ]

    selected = schedule_coordinator.select_plannable_week(
        league,
        games=week_one,
        latest_picks=[pick()],
        now=now,
    )

    assert selected is not None
    assert selected.cadence == "offseason"


def test_game_day_plans_daily_five_and_exactly_one_hour_before_each_window():
    league = ExpectedPointsLeague.CFB
    games = (
        game(league, datetime(2026, 9, 5, 12, 10, tzinfo=EASTERN)),
        game(league, datetime(2026, 9, 5, 15, 25, tzinfo=EASTERN)),
        game(league, datetime(2026, 9, 5, 19, 10, tzinfo=EASTERN)),
    )

    plans = schedule_coordinator.build_update_plans(
        schedule_coordinator.PlannableWeek(games),
        now=datetime(2026, 9, 4, 12, tzinfo=EASTERN),
    )
    by_window = {plan.window_key: plan for plan in plans}

    assert set(by_window) == {"daily", "early", "afternoon", "night"}
    assert by_window["daily"].scheduled_for.astimezone(EASTERN) == datetime(
        2026, 9, 5, 5, 0, tzinfo=EASTERN
    )
    assert by_window["early"].scheduled_for.astimezone(EASTERN) == datetime(
        2026, 9, 5, 11, 10, tzinfo=EASTERN
    )
    assert by_window["afternoon"].scheduled_for.astimezone(EASTERN) == datetime(
        2026, 9, 5, 14, 25, tzinfo=EASTERN
    )
    assert by_window["night"].scheduled_for.astimezone(EASTERN) == datetime(
        2026, 9, 5, 18, 10, tzinfo=EASTERN
    )
    assert by_window["night"].record()["model_key"] == "cfb_expected_points"


def test_monday_doubleheader_has_one_night_window_before_first_game():
    league = ExpectedPointsLeague.NFL
    first = datetime(2026, 9, 14, 19, 10, tzinfo=EASTERN)
    second = datetime(2026, 9, 14, 22, 15, tzinfo=EASTERN)
    plans = schedule_coordinator.build_update_plans(
        schedule_coordinator.PlannableWeek(
            (
                game(league, first),
                game(league, second),
            )
        ),
        now=datetime(2026, 9, 14, 12, tzinfo=EASTERN),
    )

    assert len(plans) == 1
    assert plans[0].window_key == "night"
    assert plans[0].scheduled_for.astimezone(EASTERN) == datetime(
        2026, 9, 14, 18, 10, tzinfo=EASTERN
    )


def test_daily_five_runs_every_day_until_the_last_game_date():
    league = ExpectedPointsLeague.NFL
    games = (
        game(league, datetime(2026, 9, 10, 20, 20, tzinfo=EASTERN)),
        game(league, datetime(2026, 9, 14, 20, 20, tzinfo=EASTERN)),
    )
    plans = schedule_coordinator.build_update_plans(
        schedule_coordinator.PlannableWeek(games),
        now=datetime(2026, 9, 8, 4, 45, tzinfo=EASTERN),
    )
    daily = [plan for plan in plans if plan.window_key == "daily"]

    assert [plan.game_date.isoformat() for plan in daily] == [
        "2026-09-08",
        "2026-09-09",
        "2026-09-10",
        "2026-09-11",
        "2026-09-12",
        "2026-09-13",
        "2026-09-14",
    ]
    assert all(plan.scheduled_for.astimezone(EASTERN).hour == 5 for plan in daily)


def test_offseason_with_future_games_plans_one_weekly_training_run():
    league = ExpectedPointsLeague.CFB
    now = datetime(2026, 3, 1, 12, tzinfo=EASTERN)
    future = game(league, datetime(2026, 8, 29, 12, tzinfo=EASTERN))

    plans = schedule_coordinator.build_update_plans(
        schedule_coordinator.PlannableWeek((future,), cadence="offseason"),
        now=now,
    )

    assert len(plans) == 1
    assert plans[0].window_key == "offseason"
    assert plans[0].scheduled_for.astimezone(EASTERN) == datetime(
        2026, 3, 2, 5, tzinfo=EASTERN
    )


def test_late_season_nfl_weekend_can_have_six_game_windows():
    league = ExpectedPointsLeague.NFL
    games = []
    for day in (19, 20):
        games.extend(
            [
                game(league, datetime(2026, 12, day, 12, tzinfo=EASTERN)),
                game(league, datetime(2026, 12, day, 16, tzinfo=EASTERN)),
                game(league, datetime(2026, 12, day, 20, tzinfo=EASTERN)),
            ]
        )

    plans = schedule_coordinator.build_update_plans(
        schedule_coordinator.PlannableWeek(tuple(games)),
        now=datetime(2026, 12, 18, 12, tzinfo=EASTERN),
    )
    windows = [plan for plan in plans if plan.window_key != "daily"]

    assert len(windows) == 6
    assert {(plan.game_date.day, plan.window_key) for plan in windows} == {
        (19, "early"),
        (19, "afternoon"),
        (19, "night"),
        (20, "early"),
        (20, "afternoon"),
        (20, "night"),
    }


def test_week_rollover_waits_for_completion_and_five_am():
    league = ExpectedPointsLeague.CFB
    current = game(
        league,
        datetime(2026, 9, 5, 22, tzinfo=EASTERN),
        completed=True,
    )
    following = game(
        league,
        datetime(2026, 9, 12, 12, tzinfo=EASTERN),
        week=2,
    )

    before = schedule_coordinator.select_plannable_week(
        league,
        games=[current, following],
        latest_picks=[pick()],
        now=datetime(2026, 9, 6, 4, 59, tzinfo=EASTERN),
    )
    after = schedule_coordinator.select_plannable_week(
        league,
        games=[current, following],
        latest_picks=[pick()],
        now=datetime(2026, 9, 6, 5, 0, tzinfo=EASTERN),
    )

    assert before is None
    assert after is not None
    assert after.games[0].week == 2


def test_unresolved_week_cannot_plan_the_following_week():
    league = ExpectedPointsLeague.NFL
    current = game(league, datetime(2026, 9, 10, 20, 20, tzinfo=EASTERN))
    following = game(
        league,
        datetime(2026, 9, 17, 20, 20, tzinfo=EASTERN),
        week=2,
    )

    selected = schedule_coordinator.select_plannable_week(
        league,
        games=[current, following],
        latest_picks=[pick()],
        now=datetime(2026, 9, 8, 6, tzinfo=EASTERN),
    )

    assert selected is not None
    assert {item.week for item in selected.games} == {1}


def test_pre_five_planner_uses_five_am_for_rollover_eligibility():
    before = datetime(2026, 9, 8, 4, 45, tzinfo=EASTERN)

    assert schedule_coordinator._planning_eligibility_time(before) == datetime(
        2026, 9, 8, 5, 0, tzinfo=EASTERN
    )


def test_new_season_stays_closed_until_active_horizon():
    league = ExpectedPointsLeague.NFL
    completed = game(
        league,
        datetime(2027, 2, 7, 18, 30, tzinfo=EASTERN),
        season=2026,
        week=22,
        completed=True,
    )
    next_season = game(
        league,
        datetime(2027, 9, 9, 20, 20, tzinfo=EASTERN),
        season=2027,
        week=1,
    )

    early = schedule_coordinator.select_plannable_week(
        league,
        games=[completed, next_season],
        latest_picks=[pick(season=2026, week=22)],
        now=datetime(2027, 2, 8, 5, tzinfo=EASTERN),
    )
    active = schedule_coordinator.select_plannable_week(
        league,
        games=[completed, next_season],
        latest_picks=[pick(season=2026, week=22)],
        now=datetime(2027, 9, 6, 5, tzinfo=EASTERN),
    )

    assert early is not None
    assert early.cadence == "offseason"
    assert active is not None
    assert active.games[0].season == 2027
    assert active.cadence == "active"


class FakeScheduler:
    def __init__(self):
        self.schedules = {}
        self.created = []
        self.updated = []
        self.deleted = []

    def get_schedule(self, Name, GroupName):
        try:
            return self.schedules[(GroupName, Name)]
        except KeyError as exc:
            raise ClientError(
                {"Error": {"Code": "ResourceNotFoundException", "Message": "missing"}},
                "GetSchedule",
            ) from exc

    def create_schedule(self, Name, GroupName, **kwargs):
        self.created.append((Name, GroupName, kwargs))
        self.schedules[(GroupName, Name)] = kwargs

    def update_schedule(self, Name, GroupName, **kwargs):
        self.updated.append((Name, GroupName, kwargs))
        self.schedules[(GroupName, Name)] = kwargs

    def delete_schedule(self, Name, GroupName):
        self.deleted.append((Name, GroupName))
        self.schedules.pop((GroupName, Name), None)


def scheduler_args():
    return {
        "training_function_arn": "arn:aws:lambda:us-east-1:123:function:training",
        "scheduler_target_role_arn": "arn:aws:iam::123:role/scheduler",
        "schedule_group_name": "training-updates",
        "scheduler_dlq_arn": "arn:aws:sqs:us-east-1:123:dlq",
    }


def test_reconciliation_creates_once_then_leaves_same_schedule_unchanged(monkeypatch):
    scheduler = FakeScheduler()
    statuses = []
    plan = schedule_coordinator._make_plan(
        ExpectedPointsLeague.NFL,
        2026,
        1,
        window_key="night",
        game_date=datetime(2026, 9, 6).date(),
        scheduled_for=datetime(2026, 9, 6, 18, 10, tzinfo=EASTERN),
        kickoff_at=datetime(2026, 9, 6, 19, 10, tzinfo=EASTERN),
        reason="night update",
    )
    monkeypatch.setattr(schedule_coordinator, "get_scheduled_model_updates", lambda *_a, **_k: [])
    monkeypatch.setattr(
        schedule_coordinator,
        "upsert_scheduled_model_update",
        lambda record: {"run_key": record["run_key"], "status": "planned"},
    )
    monkeypatch.setattr(
        schedule_coordinator,
        "set_scheduled_model_update_status",
        lambda run_key, status, **_k: statuses.append((run_key, status)),
    )

    first = schedule_coordinator.reconcile_update_schedules(
        scheduler,
        league=ExpectedPointsLeague.NFL,
        desired_plans=[plan],
        now=datetime(2026, 9, 1, tzinfo=UTC),
        **scheduler_args(),
    )
    second = schedule_coordinator.reconcile_update_schedules(
        scheduler,
        league=ExpectedPointsLeague.NFL,
        desired_plans=[plan],
        now=datetime(2026, 9, 1, tzinfo=UTC),
        **scheduler_args(),
    )

    assert first["scheduled"] == 1
    assert second["unchanged"] == 1
    assert len(scheduler.created) == 1
    assert scheduler.updated == []
    assert scheduler.created[0][2]["ScheduleExpression"] == "at(2026-09-06T22:10:00)"
    assert scheduler.created[0][2]["FlexibleTimeWindow"] == {"Mode": "OFF"}


def test_reconciliation_updates_same_named_schedule_when_kickoff_moves(monkeypatch):
    scheduler = FakeScheduler()
    plan = schedule_coordinator._make_plan(
        ExpectedPointsLeague.CFB,
        2026,
        1,
        window_key="night",
        game_date=datetime(2026, 9, 5).date(),
        scheduled_for=datetime(2026, 9, 5, 18, 10, tzinfo=EASTERN),
        kickoff_at=datetime(2026, 9, 5, 19, 10, tzinfo=EASTERN),
        reason="night update",
    )
    monkeypatch.setattr(schedule_coordinator, "get_scheduled_model_updates", lambda *_a, **_k: [])
    monkeypatch.setattr(
        schedule_coordinator,
        "upsert_scheduled_model_update",
        lambda record: {"run_key": record["run_key"], "status": "planned"},
    )
    monkeypatch.setattr(
        schedule_coordinator,
        "set_scheduled_model_update_status",
        lambda *_a, **_k: None,
    )
    schedule_coordinator.reconcile_update_schedules(
        scheduler,
        league=ExpectedPointsLeague.CFB,
        desired_plans=[plan],
        now=datetime(2026, 9, 1, tzinfo=UTC),
        **scheduler_args(),
    )

    moved = replace(
        plan,
        scheduled_for=plan.scheduled_for + timedelta(minutes=15),
        kickoff_at=plan.kickoff_at + timedelta(minutes=15),
    )
    schedule_coordinator.reconcile_update_schedules(
        scheduler,
        league=ExpectedPointsLeague.CFB,
        desired_plans=[moved],
        now=datetime(2026, 9, 1, tzinfo=UTC),
        **scheduler_args(),
    )

    assert len(scheduler.created) == 1
    assert len(scheduler.updated) == 1
    assert scheduler.updated[0][0] == scheduler.created[0][0]
    assert scheduler.updated[0][2]["ScheduleExpression"] == "at(2026-09-05T22:25:00)"


def test_obsolete_future_schedule_is_deleted_and_cancelled(monkeypatch):
    scheduler = FakeScheduler()
    row = {
        "run_key": "aws-scheduler:nfl:2026:1:night",
        "status": "scheduled",
        "scheduled_for": datetime(2026, 9, 6, 22, tzinfo=UTC),
        "aws_schedule_name": "sports-models-nfl-2026-w01-night",
    }
    statuses = []
    monkeypatch.setattr(
        schedule_coordinator,
        "get_scheduled_model_updates",
        lambda *_a, **_k: [row],
    )
    monkeypatch.setattr(
        schedule_coordinator,
        "set_scheduled_model_update_status",
        lambda run_key, status, **_k: statuses.append((run_key, status)),
    )

    schedule_coordinator.reconcile_update_schedules(
        scheduler,
        league=ExpectedPointsLeague.NFL,
        desired_plans=[],
        now=datetime(2026, 9, 1, tzinfo=UTC),
        **scheduler_args(),
    )

    assert scheduler.deleted == [(row["aws_schedule_name"], "training-updates")]
    assert statuses == [(row["run_key"], "cancelled")]


def test_coordinator_backoff_is_weekly_offseason_and_six_hours_active():
    league = ExpectedPointsLeague.NFL
    now = datetime(2026, 3, 1, 5, tzinfo=UTC)

    assert schedule_coordinator.next_coordinator_check(
        league, games=[], now=now
    ) == now + timedelta(days=7)
    assert schedule_coordinator.next_coordinator_check(
        league,
        games=[game(league, now + timedelta(days=1))],
        now=now,
    ) == now + timedelta(hours=6)


@pytest.mark.parametrize('fail', [False, True])
def test_nfl_schedule_fetch_is_fresh_bounded_and_restores_config(monkeypatch, fail):
    import nflreadpy as nfl
    from nflreadpy.config import get_config
    from types import SimpleNamespace

    before = {name: getattr(get_config(), name) for name in ('cache_mode', 'timeout', 'verbose')}
    def load(seasons):
        assert seasons == [2025, 2026]
        assert get_config().cache_mode == 'off'
        assert get_config().timeout == 5
        if fail:
            raise RuntimeError('feed unavailable')
        return SimpleNamespace(to_dicts=lambda: [{
            'season': 2026, 'week': 1, 'gameday': '2026-09-06', 'gametime': '13:00',
            'home_score': None, 'away_score': None,
        }])
    monkeypatch.setattr(nfl, 'load_schedules', load)
    if fail:
        with pytest.raises(RuntimeError, match='feed unavailable'):
            schedule_coordinator.load_schedule(ExpectedPointsLeague.NFL, now=datetime(2026, 9, 6, tzinfo=UTC), timeout=5)
    else:
        games = schedule_coordinator.load_schedule(ExpectedPointsLeague.NFL, now=datetime(2026, 9, 6, tzinfo=UTC), timeout=5)
        assert games[0].week == 1
    assert {name: getattr(get_config(), name) for name in before} == before
    assert not schedule_coordinator._NFL_SCHEDULE_LOCK.locked()
