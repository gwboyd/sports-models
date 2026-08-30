"""Plan and reconcile one-time expected-points training schedules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import hashlib
import json
import logging
import os
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import boto3
from botocore.exceptions import ClientError
import nflreadpy as nfl

from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.model_patterns.expected_points.versioning import ModelKey
from src.sports.football.cfb.expected_points.cfbd_client import CFBDClient
from src.utils.db.sports_models_db import (
    get_expected_points_picks,
    get_schedule_coordinator_state,
    get_scheduled_model_updates,
    set_scheduled_model_update_status,
    update_schedule_coordinator_state,
    upsert_scheduled_model_update,
)


LOGGER = logging.getLogger(__name__)
COORDINATOR_JOB = "coordinate_expected_points_updates"
TRAINING_JOB = "expected_points_update"
EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
ACTIVE_HORIZON = timedelta(days=4)
PREGAME_LEAD = timedelta(hours=1)
ROLLOVER_TIME = time(hour=5)
DAILY_UPDATE_TIME = time(hour=5)
PLANNER_LEAD = timedelta(minutes=15)
PENDING_STATUSES = {"planned", "scheduled", "failed"}
DELIVERY_GRACE = timedelta(minutes=30)
STALE_CLAIM_AGE = timedelta(minutes=30)


@dataclass(frozen=True)
class ScheduledGame:
    league: ExpectedPointsLeague
    season: int
    week: int
    kickoff: datetime
    completed: bool


@dataclass(frozen=True)
class UpdateWindow:
    scheduled_for: datetime
    kickoff_at: datetime
    label: str
    game_date: date


@dataclass(frozen=True)
class ScheduledUpdatePlan:
    league: ExpectedPointsLeague
    season: int
    week: int
    window_key: str
    game_date: date | None
    scheduled_for: datetime
    kickoff_at: datetime | None
    run_key: str
    aws_schedule_name: str
    reason: str

    def payload(self) -> dict[str, Any]:
        return {
            "job": TRAINING_JOB,
            "league": self.league.value,
            "season": self.season,
            "week": self.week,
            "run_key": self.run_key,
        }

    def record(self) -> dict[str, Any]:
        return {
            "run_key": self.run_key,
            "model_key": (
                ModelKey.NFL_EXPECTED_POINTS.value
                if self.league is ExpectedPointsLeague.NFL
                else ModelKey.CFB_EXPECTED_POINTS.value
            ),
            "league": self.league.value,
            "season": self.season,
            "week": self.week,
            "window_key": self.window_key,
            "game_date": self.game_date,
            "scheduled_for": self.scheduled_for.astimezone(UTC),
            "kickoff_at": self.kickoff_at.astimezone(UTC) if self.kickoff_at else None,
            "aws_schedule_name": self.aws_schedule_name,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PlannableWeek:
    games: tuple[ScheduledGame, ...]
    cadence: str = "active"


def handler(event: Mapping[str, Any], context: Any) -> dict[str, Any]:
    """Reconcile future Scheduler resources; never invoke training directly."""

    if event.get("job") != COORDINATOR_JOB:
        raise ValueError(f"Unsupported coordinator job: {event.get('job')!r}")
    configuration = _scheduler_configuration()
    scheduler_client = boto3.client("scheduler")
    now = datetime.now(UTC)
    outcomes: list[dict[str, Any]] = []
    failures: list[str] = []

    for league in ExpectedPointsLeague:
        try:
            state = get_schedule_coordinator_state(league)
            if state is not None and _state_datetime(state.get("next_check_at")) > now:
                outcomes.append(
                    {
                        "league": league.value,
                        "status": "sleeping",
                        "next_check_at": _state_datetime(state["next_check_at"]).isoformat(),
                    }
                )
                continue

            games = load_schedule(league, now=now)
            picks = get_expected_points_picks(league, latest=True)
            week_plan = select_plannable_week(
                league,
                games=games,
                latest_picks=picks,
                now=_planning_eligibility_time(now),
            )
            desired = build_update_plans(week_plan, now=now) if week_plan else []
            result = reconcile_update_schedules(
                scheduler_client,
                league=league,
                desired_plans=desired,
                now=now,
                **configuration,
            )
            outcomes.append(
                {
                    "league": league.value,
                    "status": "planned" if desired else "idle",
                    "scheduled": result["scheduled"],
                    "unchanged": result["unchanged"],
                    "cancelled": result["cancelled"],
                }
            )
            _save_next_check(league, games=games, now=now)
        except Exception as exc:
            LOGGER.exception("%s schedule planning failed", league.value.upper())
            failures.append(f"{league.value}: {exc}")

    if failures:
        raise RuntimeError("; ".join(failures))
    return {"status": "success", "outcomes": outcomes}


def _scheduler_configuration() -> dict[str, str]:
    names = (
        "TRAINING_FUNCTION_ARN",
        "SCHEDULER_TARGET_ROLE_ARN",
        "SCHEDULE_GROUP_NAME",
        "SCHEDULER_DLQ_ARN",
    )
    values = {name.lower(): os.getenv(name) for name in names}
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing coordinator configuration: {', '.join(missing)}")
    return {name: str(value) for name, value in values.items()}


def select_plannable_week(
    league: ExpectedPointsLeague,
    *,
    games: Sequence[ScheduledGame],
    latest_picks: Sequence[Mapping[str, Any]],
    now: datetime,
) -> PlannableWeek | None:
    """Select the one week allowed to publish without crossing unresolved picks."""

    compare_time = _aware(now)
    relevant = sorted(
        (game for game in games if game.league is league),
        key=lambda game: game.kickoff.astimezone(UTC),
    )
    if not relevant:
        return None

    latest_week = _latest_pick_week(latest_picks)
    if latest_week is not None:
        active_games = _games_for_week(relevant, *latest_week)
        if active_games and not _week_is_complete(active_games):
            first_kickoff = min(
                game.kickoff.astimezone(UTC) for game in active_games
            )
            cadence = (
                "offseason"
                if first_kickoff - compare_time.astimezone(UTC) > ACTIVE_HORIZON
                else "active"
            )
            return PlannableWeek(tuple(active_games), cadence=cadence)
        if active_games:
            if not _rollover_is_due(active_games, compare_time):
                return None
            next_week = _next_same_season_week(relevant, latest_week, compare_time)
            if next_week is not None:
                return PlannableWeek(tuple(next_week))

            next_season = _next_future_week(relevant, compare_time)
            if next_season is None:
                return None
            first_kickoff = min(game.kickoff.astimezone(UTC) for game in next_season)
            if first_kickoff - compare_time.astimezone(UTC) > ACTIVE_HORIZON:
                return PlannableWeek(tuple(next_season), cadence="offseason")
            return PlannableWeek(tuple(next_season), cadence="active")

    next_week = _next_future_week(relevant, compare_time)
    if next_week is None:
        return None
    first_kickoff = min(game.kickoff.astimezone(UTC) for game in next_week)
    if first_kickoff - compare_time.astimezone(UTC) > ACTIVE_HORIZON:
        return PlannableWeek(tuple(next_week), cadence="offseason")
    return PlannableWeek(tuple(next_week), cadence="active")


def build_update_plans(
    week_plan: PlannableWeek,
    *,
    now: datetime,
) -> list[ScheduledUpdatePlan]:
    """Build every still-future one-time update for the selected week."""

    games = list(week_plan.games)
    if not games:
        return []
    league = games[0].league
    season = games[0].season
    week = games[0].week
    compare_time = _aware(now).astimezone(UTC)
    plans: list[ScheduledUpdatePlan] = []

    future_games = [
        game
        for game in games
        if not game.completed and game.kickoff.astimezone(UTC) > compare_time
    ]
    if not future_games:
        return []

    if week_plan.cadence == "offseason":
        local_now = compare_time.astimezone(EASTERN)
        schedule_date = local_now.date()
        scheduled_for = datetime.combine(schedule_date, DAILY_UPDATE_TIME, tzinfo=EASTERN)
        if scheduled_for.astimezone(UTC) <= compare_time:
            schedule_date += timedelta(days=1)
            scheduled_for = datetime.combine(schedule_date, DAILY_UPDATE_TIME, tzinfo=EASTERN)
        return [
            _make_plan(
                league,
                season,
                week,
                window_key="offseason",
                game_date=schedule_date,
                scheduled_for=scheduled_for,
                kickoff_at=min(game.kickoff for game in future_games),
                reason="weekly offseason update with a future scheduled slate",
            )
        ]

    local_now = compare_time.astimezone(EASTERN)
    last_game_date = max(game.kickoff.astimezone(EASTERN).date() for game in future_games)
    daily_date = local_now.date()
    while daily_date <= last_game_date:
        scheduled_for = datetime.combine(daily_date, DAILY_UPDATE_TIME, tzinfo=EASTERN)
        if scheduled_for.astimezone(UTC) > compare_time:
            first_future_kickoff = min(game.kickoff for game in future_games)
            plans.append(
                _make_plan(
                    league,
                    season,
                    week,
                    window_key="daily",
                    game_date=daily_date,
                    scheduled_for=scheduled_for,
                    kickoff_at=first_future_kickoff,
                    reason=f"daily in-season update for {daily_date.isoformat()}",
                )
            )
        daily_date += timedelta(days=1)

    for window in _update_windows(games):
        scheduled_for = window.scheduled_for.astimezone(UTC)
        if scheduled_for <= compare_time:
            continue
        plans.append(
            _make_plan(
                league,
                season,
                week,
                window_key=window.label,
                game_date=window.game_date,
                scheduled_for=scheduled_for,
                kickoff_at=window.kickoff_at,
                reason=f"{window.label} update before {window.game_date.isoformat()} games",
            )
        )
    return sorted(plans, key=lambda plan: plan.scheduled_for)


def _make_plan(
    league: ExpectedPointsLeague,
    season: int,
    week: int,
    *,
    window_key: str,
    game_date: date | None,
    scheduled_for: datetime,
    kickoff_at: datetime | None,
    reason: str,
) -> ScheduledUpdatePlan:
    return ScheduledUpdatePlan(
        league=league,
        season=season,
        week=week,
        window_key=window_key,
        game_date=game_date,
        scheduled_for=_aware(scheduled_for),
        kickoff_at=_aware(kickoff_at) if kickoff_at else None,
        run_key=_run_key(league, season, week, window_key, game_date),
        aws_schedule_name=_schedule_name(league, season, week, window_key, game_date),
        reason=reason,
    )


def reconcile_update_schedules(
    scheduler_client: Any,
    *,
    league: ExpectedPointsLeague,
    desired_plans: Sequence[ScheduledUpdatePlan],
    now: datetime,
    training_function_arn: str,
    scheduler_target_role_arn: str,
    schedule_group_name: str,
    scheduler_dlq_arn: str,
) -> dict[str, int]:
    """Upsert desired schedules and remove obsolete, still-pending schedules."""

    existing_rows = get_scheduled_model_updates(league, pending_only=True)
    existing_by_key = {str(row["run_key"]): row for row in existing_rows}
    desired_by_key = {plan.run_key: plan for plan in desired_plans}
    counts = {"scheduled": 0, "unchanged": 0, "cancelled": 0}

    for plan in desired_plans:
        existing = existing_by_key.get(plan.run_key)
        if (
            existing
            and existing.get("status") == "running"
            and _state_datetime(existing.get("claimed_at")) + STALE_CLAIM_AGE
            <= _aware(now).astimezone(UTC)
        ):
            set_scheduled_model_update_status(
                plan.run_key,
                "failed",
                error="Planner recovered a stale training claim",
            )
            existing = {**existing, "status": "failed"}
        row = upsert_scheduled_model_update(plan.record())
        if row.get("status") in {"completed", "running"}:
            counts["unchanged"] += 1
            continue
        changed = _ensure_aws_schedule(
            scheduler_client,
            plan=plan,
            training_function_arn=training_function_arn,
            scheduler_target_role_arn=scheduler_target_role_arn,
            schedule_group_name=schedule_group_name,
            scheduler_dlq_arn=scheduler_dlq_arn,
        )
        set_scheduled_model_update_status(plan.run_key, "scheduled")
        counts["scheduled" if changed else "unchanged"] += 1

    compare_time = _aware(now).astimezone(UTC)
    for run_key, row in existing_by_key.items():
        if run_key in desired_by_key:
            continue
        status = str(row.get("status"))
        if status == "running":
            claimed_at = _state_datetime(row.get("claimed_at"))
            if claimed_at + STALE_CLAIM_AGE <= compare_time:
                set_scheduled_model_update_status(
                    run_key,
                    "missed",
                    error="Training claim expired without a completed run",
                )
                counts["cancelled"] += 1
            continue
        if status not in PENDING_STATUSES:
            continue
        scheduled_for = _state_datetime(row.get("scheduled_for"))
        if scheduled_for <= compare_time < scheduled_for + DELIVERY_GRACE:
            continue
        _delete_aws_schedule(
            scheduler_client,
            schedule_name=str(row["aws_schedule_name"]),
            schedule_group_name=schedule_group_name,
        )
        replacement_status = (
            "missed"
            if scheduled_for <= compare_time
            else "cancelled"
        )
        set_scheduled_model_update_status(run_key, replacement_status)
        counts["cancelled"] += 1
    return counts


def _ensure_aws_schedule(
    scheduler_client: Any,
    *,
    plan: ScheduledUpdatePlan,
    training_function_arn: str,
    scheduler_target_role_arn: str,
    schedule_group_name: str,
    scheduler_dlq_arn: str,
) -> bool:
    schedule_expression = f"at({plan.scheduled_for.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%S')})"
    target = {
        "Arn": training_function_arn,
        "RoleArn": scheduler_target_role_arn,
        "Input": json.dumps(plan.payload(), separators=(",", ":")),
        "RetryPolicy": {
            "MaximumEventAgeInSeconds": 3600,
            "MaximumRetryAttempts": 2,
        },
        "DeadLetterConfig": {"Arn": scheduler_dlq_arn},
    }
    common = {
        "Description": plan.reason[:512],
        "ScheduleExpression": schedule_expression,
        "ScheduleExpressionTimezone": "UTC",
        "FlexibleTimeWindow": {"Mode": "OFF"},
        "Target": target,
        "State": "ENABLED",
        "ActionAfterCompletion": "DELETE",
        "ClientToken": _client_token(plan.run_key, schedule_expression, target),
    }
    try:
        current = scheduler_client.get_schedule(
            Name=plan.aws_schedule_name,
            GroupName=schedule_group_name,
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise
        scheduler_client.create_schedule(
            Name=plan.aws_schedule_name,
            GroupName=schedule_group_name,
            **common,
        )
        return True

    if _aws_schedule_matches(current, schedule_expression, target):
        return False
    scheduler_client.update_schedule(
        Name=plan.aws_schedule_name,
        GroupName=schedule_group_name,
        **common,
    )
    return True


def _aws_schedule_matches(
    current: Mapping[str, Any],
    schedule_expression: str,
    target: Mapping[str, Any],
) -> bool:
    current_target = current.get("Target") or {}
    return (
        current.get("ScheduleExpression") == schedule_expression
        and current.get("ScheduleExpressionTimezone") == "UTC"
        and current.get("State") == "ENABLED"
        and current.get("FlexibleTimeWindow") == {"Mode": "OFF"}
        and current_target.get("Arn") == target["Arn"]
        and current_target.get("RoleArn") == target["RoleArn"]
        and current_target.get("Input") == target["Input"]
        and current_target.get("RetryPolicy") == target["RetryPolicy"]
        and current_target.get("DeadLetterConfig") == target["DeadLetterConfig"]
    )


def _delete_aws_schedule(
    scheduler_client: Any,
    *,
    schedule_name: str,
    schedule_group_name: str,
) -> None:
    try:
        scheduler_client.delete_schedule(Name=schedule_name, GroupName=schedule_group_name)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise


def _update_windows(games: Sequence[ScheduledGame]) -> list[UpdateWindow]:
    by_date: dict[date, list[ScheduledGame]] = {}
    for game in games:
        local_date = game.kickoff.astimezone(EASTERN).date()
        by_date.setdefault(local_date, []).append(game)

    windows: list[UpdateWindow] = []
    for game_date, date_games in by_date.items():
        ordered = sorted(date_games, key=lambda game: game.kickoff.astimezone(UTC))
        buckets = {
            "early": [game for game in ordered if game.kickoff.astimezone(EASTERN).hour < 14],
            "afternoon": [
                game for game in ordered if 14 <= game.kickoff.astimezone(EASTERN).hour < 19
            ],
            "night": [game for game in ordered if game.kickoff.astimezone(EASTERN).hour >= 19],
        }
        for label, bucket_games in buckets.items():
            if not bucket_games:
                continue
            kickoff = min(game.kickoff.astimezone(EASTERN) for game in bucket_games)
            windows.append(UpdateWindow(kickoff - PREGAME_LEAD, kickoff, label, game_date))
    return windows


def _save_next_check(
    league: ExpectedPointsLeague,
    *,
    games: Sequence[ScheduledGame],
    now: datetime,
) -> None:
    future_kickoffs = [
        game.kickoff.astimezone(UTC)
        for game in games
        if game.league is league
        and not game.completed
        and game.kickoff.astimezone(UTC) > now.astimezone(UTC)
    ]
    update_schedule_coordinator_state(
        league,
        next_check_at=next_coordinator_check(
            league,
            games=games,
            now=now,
        ),
        next_game_at=min(future_kickoffs) if future_kickoffs else None,
        checked_at=now,
    )


def next_coordinator_check(
    league: ExpectedPointsLeague,
    *,
    games: Sequence[ScheduledGame],
    now: datetime,
) -> datetime:
    """Use twice-daily checks in season and weekly checks with no future games."""

    compare_time = _aware(now).astimezone(UTC)
    active = compare_time + timedelta(hours=6)
    weekly = compare_time + timedelta(days=7)
    future = sorted(
        game.kickoff.astimezone(UTC)
        for game in games
        if game.league is league
        and not game.completed
        and game.kickoff.astimezone(UTC) > compare_time
    )
    if not future:
        return weekly
    horizon_entry = future[0] - ACTIVE_HORIZON
    if horizon_entry > compare_time:
        return min(weekly, horizon_entry)
    return active


def load_schedule(
    league: ExpectedPointsLeague,
    *,
    now: datetime,
) -> list[ScheduledGame]:
    if league is ExpectedPointsLeague.NFL:
        return _load_nfl_schedule(now)
    return _load_cfb_schedule(now)


def _load_nfl_schedule(now: datetime) -> list[ScheduledGame]:
    rows = nfl.load_schedules([now.year - 1, now.year]).to_dicts()
    games: list[ScheduledGame] = []
    for row in rows:
        try:
            kickoff = _nfl_kickoff(row.get("gameday"), row.get("gametime"))
            home_score = _optional_int(row.get("home_score"))
            away_score = _optional_int(row.get("away_score"))
            games.append(
                ScheduledGame(
                    league=ExpectedPointsLeague.NFL,
                    season=int(row["season"]),
                    week=int(row["week"]),
                    kickoff=kickoff,
                    completed=home_score is not None and away_score is not None,
                )
            )
        except (KeyError, TypeError, ValueError):
            LOGGER.warning("Skipping invalid NFL coordinator schedule row: %r", row)
    return games


def _load_cfb_schedule(now: datetime) -> list[ScheduledGame]:
    api_key = os.getenv("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD_API_KEY is required by the schedule coordinator")
    years = [now.year - 1, now.year]
    game_rows: list[dict[str, Any]] = []
    with CFBDClient(api_key) as client:
        for year in years:
            game_rows.extend(client.get_games(year=year, season_type="regular"))

    games: list[ScheduledGame] = []
    for row in game_rows:
        if not _cfb_game_is_relevant(row):
            continue
        try:
            home_score = _optional_int(row.get("homePoints", row.get("home_points")))
            away_score = _optional_int(row.get("awayPoints", row.get("away_points")))
            games.append(
                ScheduledGame(
                    league=ExpectedPointsLeague.CFB,
                    season=int(row["season"]),
                    week=int(row["week"]),
                    kickoff=_parse_iso_datetime(row.get("startDate", row.get("start_date"))),
                    completed=(
                        bool(row.get("completed"))
                        and home_score is not None
                        and away_score is not None
                    ),
                )
            )
        except (KeyError, TypeError, ValueError):
            LOGGER.warning("Skipping invalid CFB coordinator schedule row: %r", row)
    return games


def _cfb_game_is_relevant(row: Mapping[str, Any]) -> bool:
    classifications = {
        str(value).lower()
        for value in (
            row.get("homeClassification", row.get("home_classification")),
            row.get("awayClassification", row.get("away_classification")),
        )
        if value is not None
    }
    return not classifications or "fbs" in classifications


def _latest_pick_week(picks: Sequence[Mapping[str, Any]]) -> tuple[int, int] | None:
    if not picks:
        return None
    return max((int(pick["season"]), int(pick["week"])) for pick in picks)


def _games_for_week(
    games: Sequence[ScheduledGame], season: int, week: int
) -> list[ScheduledGame]:
    return [game for game in games if game.season == season and game.week == week]


def _week_is_complete(games: Sequence[ScheduledGame]) -> bool:
    return bool(games) and all(game.completed for game in games)


def _rollover_is_due(games: Sequence[ScheduledGame], now: datetime) -> bool:
    return _aware(now).astimezone(EASTERN) >= _rollover_at(games)


def _rollover_at(games: Sequence[ScheduledGame]) -> datetime:
    last_date = max(game.kickoff.astimezone(EASTERN).date() for game in games)
    return datetime.combine(last_date + timedelta(days=1), ROLLOVER_TIME, tzinfo=EASTERN)


def _next_same_season_week(
    games: Sequence[ScheduledGame],
    current_week: tuple[int, int],
    now: datetime,
) -> list[ScheduledGame] | None:
    season, week = current_week
    candidates = [
        game
        for game in games
        if game.season == season
        and game.week > week
        and not game.completed
        and game.kickoff.astimezone(UTC) > _aware(now).astimezone(UTC)
    ]
    return _earliest_week(candidates)


def _next_future_week(
    games: Sequence[ScheduledGame], now: datetime
) -> list[ScheduledGame] | None:
    candidates = [
        game
        for game in games
        if not game.completed and game.kickoff.astimezone(UTC) > _aware(now).astimezone(UTC)
    ]
    return _earliest_week(candidates)


def _earliest_week(games: Sequence[ScheduledGame]) -> list[ScheduledGame] | None:
    if not games:
        return None
    first = min(games, key=lambda game: game.kickoff.astimezone(UTC))
    return _games_for_week(games, first.season, first.week)


def _run_key(
    league: ExpectedPointsLeague,
    season: int,
    week: int,
    label: str,
    game_date: date | None = None,
) -> str:
    parts = ["aws-scheduler", league.value, str(season), str(week)]
    if game_date is not None:
        parts.append(game_date.isoformat())
    parts.append(label)
    return ":".join(parts)


def _schedule_name(
    league: ExpectedPointsLeague,
    season: int,
    week: int,
    label: str,
    game_date: date | None,
) -> str:
    date_part = f"-{game_date.isoformat()}" if game_date else ""
    return f"sports-models-{league.value}-{season}-w{week:02d}{date_part}-{label}"


def _client_token(
    run_key: str,
    schedule_expression: str,
    target: Mapping[str, Any],
) -> str:
    content = json.dumps(
        [run_key, schedule_expression, target],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _planning_eligibility_time(value: datetime) -> datetime:
    """Let the 4:45 AM planner register a run gated for the 5:00 AM rollover."""

    local = _aware(value).astimezone(EASTERN)
    rollover = datetime.combine(local.date(), ROLLOVER_TIME, tzinfo=EASTERN)
    if rollover - PLANNER_LEAD <= local < rollover:
        return rollover
    return value


def _nfl_kickoff(gameday: Any, gametime: Any) -> datetime:
    if gameday is None or gametime is None:
        raise ValueError("NFL kickoff requires gameday and gametime")
    day = gameday if isinstance(gameday, date) else date.fromisoformat(str(gameday))
    kickoff_time = gametime if isinstance(gametime, time) else time.fromisoformat(str(gametime))
    return datetime.combine(day, kickoff_time).replace(tzinfo=EASTERN)


def _parse_iso_datetime(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Kickoff timestamp is missing")
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Kickoff timestamp must include a timezone")
    return parsed


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Coordinator time must be timezone-aware")
    return value


def _state_datetime(value: Any) -> datetime:
    if value is None:
        return datetime.min.replace(tzinfo=UTC)
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
