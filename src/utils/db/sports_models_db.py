from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any

from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.model_patterns.expected_points.versioning import (
    ModelKey,
    ReleaseDraft,
    parse_version,
)
from src.model_patterns.expected_points.write_policy import is_aws_lambda_runtime
from src.utils.postgres import (
    get_connection,
    get_schema,
    json_dumps,
    normalize_record,
    normalize_records,
)

SCHEMA = get_schema()

PICK_COLUMNS = (
    "season", "week", "year_week", "game_id", "home_team", "away_team",
    "home_score_pred", "away_score_pred", "spread_pred", "spread_line", "spread_play",
    "spread_win_prob", "spread_lock", "total_pred", "total_line", "total_play",
    "total_win_prob", "total_lock", "date_time", "model_version", "write_time",
)
CFB_PICK_METADATA_COLUMNS = ("home_conference", "away_conference")
RESULT_COLUMNS = (
    "season", "week", "year_week", "game_id", "home_team", "away_team", "home_score",
    "away_score", "home_score_pred", "away_score_pred", "spread_pred", "spread_line",
    "true_spread", "spread_play", "spread_win_prob", "spread_lock", "correct_spread_play",
    "spread_win", "total_pred", "total_line", "true_total", "total_play", "total_win_prob",
    "total_lock", "correct_total_play", "total_win", "date_time", "model_version",
)
CFB_RESULT_METADATA_COLUMNS = CFB_PICK_METADATA_COLUMNS
UPDATE_COLUMNS = (
    "year_week", "write_time", "week", "season", "environment", "client_name", "runtime",
    "model_version", "source_git_sha", "evaluation_metrics",
    "pick_changes", "pick_changes_games", "play_changes", "play_changes_games",
    "updates_skipped", "picks_num", "difference_df", "picks_df",
)

_TABLE_PREFIXES = {
    ExpectedPointsLeague.NFL: "nfl",
    ExpectedPointsLeague.CFB: "cfb",
}


def _coerce_league(league: ExpectedPointsLeague | str) -> ExpectedPointsLeague:
    return league if isinstance(league, ExpectedPointsLeague) else ExpectedPointsLeague(league)


def model_key(league: ExpectedPointsLeague | str) -> ModelKey:
    league = _coerce_league(league)
    return ModelKey.CFB_EXPECTED_POINTS if league is ExpectedPointsLeague.CFB else ModelKey.NFL_EXPECTED_POINTS


def _table(league: ExpectedPointsLeague | str, suffix: str) -> str:
    prefix = _TABLE_PREFIXES[_coerce_league(league)]
    if suffix not in {"picks", "latest_picks", "results", "pick_updates"}:
        raise ValueError(f"Unsupported expected-points table suffix: {suffix}")
    return f"{SCHEMA}.{prefix}_expected_points_{suffix}"


def _pick_columns(league: ExpectedPointsLeague | str) -> tuple[str, ...]:
    league = _coerce_league(league)
    if league is ExpectedPointsLeague.CFB:
        return PICK_COLUMNS[:-2] + CFB_PICK_METADATA_COLUMNS + PICK_COLUMNS[-2:]
    return PICK_COLUMNS


def _result_columns(league: ExpectedPointsLeague | str) -> tuple[str, ...]:
    league = _coerce_league(league)
    if league is ExpectedPointsLeague.CFB:
        return RESULT_COLUMNS + CFB_RESULT_METADATA_COLUMNS
    return RESULT_COLUMNS


_MODEL_RELEASE_SELECT = """
    model_key, version, major_version, minor_version, title,
    public_summary, changes_md, evaluation_md, internal_notes_md,
    source_git_sha, deployed_at, first_pick_at
"""


def get_model_release(model: ModelKey | str, version: str) -> dict[str, Any] | None:
    model_key_value = model.value if isinstance(model, ModelKey) else ModelKey(model).value
    canonical_version = str(parse_version(version))
    query = f"""
        select {_MODEL_RELEASE_SELECT}
        from {SCHEMA}.model_releases
        where model_key = %s and version = %s
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, (model_key_value, canonical_version))
        return cur.fetchone()


def get_latest_model_release(model: ModelKey | str) -> dict[str, Any] | None:
    model_key_value = model.value if isinstance(model, ModelKey) else ModelKey(model).value
    query = f"""
        select {_MODEL_RELEASE_SELECT}
        from {SCHEMA}.model_releases
        where model_key = %s
        order by major_version desc, minor_version desc
        limit 1
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, (model_key_value,))
        return cur.fetchone()


def get_expected_points_recipe_source(
    league: ExpectedPointsLeague | str,
    *,
    version: str | None = None,
) -> dict[str, Any] | None:
    """Resolve a live version to its immutable release-registry SHA."""
    release_key = model_key(league).value
    params: list[Any] = [release_key]
    version_filter = ""
    if version is not None:
        canonical = str(parse_version(version))
        version_filter = "and version = %s"
        params.append(canonical)
    release_query = f"""
        select version, source_git_sha, deployed_at, first_pick_at
        from {SCHEMA}.model_releases
        where model_key = %s
          and first_pick_at is not null
          {version_filter}
        order by major_version desc, minor_version desc
        limit 1
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(release_query, tuple(params))
        return cur.fetchone()


def initialize_model_release_source(
    model: ModelKey | str,
    version: str,
    *,
    source_git_sha: str,
) -> str:
    """Fill a bootstrap release's missing SHA once and return its canonical SHA."""
    model_key_value = model.value if isinstance(model, ModelKey) else ModelKey(model).value
    canonical_version = str(parse_version(version))
    canonical_sha = source_git_sha.strip()
    if not canonical_sha:
        raise ValueError("Model release source Git SHA cannot be empty")
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            update {SCHEMA}.model_releases
            set source_git_sha = %s
            where model_key = %s
              and version = %s
              and source_git_sha is null
            """,
            (canonical_sha, model_key_value, canonical_version),
        )
        cur.execute(
            f"""
            select source_git_sha
            from {SCHEMA}.model_releases
            where model_key = %s and version = %s
            """,
            (model_key_value, canonical_version),
        )
        release = cur.fetchone()
        if release is None:
            raise ValueError(
                f"Model release {model_key_value} {canonical_version} does not exist"
            )
        registered_sha = release.get("source_git_sha")
        if not registered_sha:
            raise ValueError(
                f"Model release {model_key_value} {canonical_version} has no source Git SHA"
            )
        return str(registered_sha)


def insert_model_releases(
    releases: Iterable[tuple[ModelKey | str, str, ReleaseDraft]],
    *,
    source_git_sha: str | None,
    deployed_at: datetime,
) -> None:
    """Register one or more immutable releases in a single transaction."""
    normalized = []
    for model, version, draft in releases:
        model_key_value = model.value if isinstance(model, ModelKey) else ModelKey(model).value
        parsed_version = parse_version(version)
        normalized.append((model_key_value, parsed_version, draft))
    if not normalized:
        return

    query = f"""
        insert into {SCHEMA}.model_releases (
            model_key, version, major_version, minor_version, title,
            public_summary, changes_md, evaluation_md, internal_notes_md,
            source_git_sha, deployed_at
        ) values (
            %(model_key)s, %(version)s, %(major_version)s, %(minor_version)s, %(title)s,
            %(public_summary)s, %(changes_md)s, %(evaluation_md)s, %(internal_notes_md)s,
            %(source_git_sha)s, %(deployed_at)s
        )
        on conflict (model_key, version) do nothing
    """

    records = []
    for model_key_value, parsed_version, draft in normalized:
        records.append({
            "model_key": model_key_value,
            "version": str(parsed_version),
            "major_version": parsed_version.major,
            "minor_version": parsed_version.minor,
            "title": draft.title,
            "public_summary": draft.public_summary,
            "changes_md": draft.changes_md,
            "evaluation_md": draft.evaluation_md,
            "internal_notes_md": draft.internal_notes_md,
            "source_git_sha": source_git_sha,
            "deployed_at": deployed_at,
        })

    with get_connection() as conn, conn.cursor() as cur:
        cur.executemany(query, records)
        for record in records:
            cur.execute(
                f"""
                select title, public_summary, changes_md, evaluation_md,
                       internal_notes_md, source_git_sha
                from {SCHEMA}.model_releases
                where model_key = %s and version = %s
                """,
                (record["model_key"], record["version"]),
            )
            existing = cur.fetchone()
            expected = tuple(record[column] for column in (
                "title", "public_summary", "changes_md", "evaluation_md",
                "internal_notes_md", "source_git_sha",
            ))
            actual = tuple(existing[column] for column in (
                "title", "public_summary", "changes_md", "evaluation_md",
                "internal_notes_md", "source_git_sha",
            )) if existing else None
            if actual != expected:
                raise ValueError(
                    f"Model release {record['model_key']} {record['version']} "
                    "already exists with different contents"
                )


def _parse_write_time(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00").replace(" ", "T"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def get_expected_points_picks(
    league: ExpectedPointsLeague | str,
    *,
    latest: bool = False,
) -> list[dict[str, Any]]:
    table = _table(league, "latest_picks" if latest else "picks")
    columns = _pick_columns(league)
    selected = ",\n            ".join(columns[:-1])
    query = f"""
        select
            {selected},
            to_char(write_time at time zone 'UTC', 'YYYY-MM-DD HH24:MI:SS') as write_time
        from {table}
        order by {"date_time asc, game_id asc" if latest else "write_time asc, date_time asc, game_id asc"}
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query)
        return list(cur.fetchall())


def get_expected_points_results(league: ExpectedPointsLeague | str) -> list[dict[str, Any]]:
    columns = _result_columns(league)
    query = f"""
        select {', '.join(columns)}
        from {_table(league, 'results')}
        order by season desc, cast(week as integer) desc, date_time asc, game_id asc
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query)
        return list(cur.fetchall())


def is_scheduled_model_update_completed(
    league: ExpectedPointsLeague | str,
    run_key: str,
) -> bool:
    """Return whether a plan atomically linked its persisted model update."""

    if not run_key.startswith(("aws-scheduler:", "api:")):
        raise ValueError("Scheduled run key must start with 'aws-scheduler:' or 'api:'")
    league_value = _coerce_league(league).value
    query = f"""
        select 1
        from {SCHEMA}.scheduled_model_updates
        where run_key = %s
          and league = %s
          and model_key = %s
          and status = 'completed'
          and update_id is not null
        limit 1
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, (run_key, league_value, model_key(league).value))
        return cur.fetchone() is not None


def get_schedule_coordinator_state(
    league: ExpectedPointsLeague | str,
) -> dict[str, Any] | None:
    league_value = _coerce_league(league).value
    query = f"""
        select league, next_check_at, last_checked_at, next_game_at
        from {SCHEMA}.schedule_coordinator_state
        where league = %s
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, (league_value,))
        return cur.fetchone()


def update_schedule_coordinator_state(
    league: ExpectedPointsLeague | str,
    *,
    next_check_at: datetime,
    next_game_at: datetime | None,
    checked_at: datetime,
) -> None:
    league_value = _coerce_league(league).value
    query = f"""
        insert into {SCHEMA}.schedule_coordinator_state (
            league, next_check_at, last_checked_at, next_game_at
        ) values (%s, %s, %s, %s)
        on conflict (league) do update set
            next_check_at = excluded.next_check_at,
            last_checked_at = excluded.last_checked_at,
            next_game_at = excluded.next_game_at
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, (league_value, next_check_at, checked_at, next_game_at))


def get_scheduled_model_updates(
    league: ExpectedPointsLeague | str | None = None,
    *,
    season: int | None = None,
    week: int | None = None,
    pending_only: bool = False,
) -> list[dict[str, Any]]:
    """Return automatic plans for calendar reconciliation, excluding manual jobs."""

    # Calendar reconciliation must never cancel a manually requested schedule.
    filters: list[str] = ["trigger_source = 'scheduler'"]
    params: list[Any] = []
    if league is not None:
        filters.append("league = %s")
        params.append(_coerce_league(league).value)
    if season is not None:
        filters.append("season = %s")
        params.append(season)
    if week is not None:
        filters.append("week = %s")
        params.append(week)
    if pending_only:
        filters.append("status in ('planned', 'scheduled', 'running', 'failed')")
    where = f"where {' and '.join(filters)}" if filters else ""
    query = f"""
        select run_key, model_key, league, season, week, window_key, game_date,
               scheduled_for, kickoff_at, aws_schedule_name, status, reason,
               attempt_count, claimed_at, completed_at, update_id, last_error,
               created_at, updated_at
        from {SCHEMA}.scheduled_model_updates
        {where}
        order by scheduled_for asc, run_key asc
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, tuple(params))
        return list(cur.fetchall())


def verify_manual_update_schema() -> None:
    """Fail deployment before changing AWS if the additive job columns are missing."""

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select column_name from information_schema.columns
            where table_schema = %s and table_name = 'scheduled_model_updates'
              and column_name in ('trigger_source', 'client_name')
            """,
            (SCHEMA,),
        )
        columns = {row["column_name"] for row in cur.fetchall()}
    missing = {"trigger_source", "client_name"} - columns
    if missing:
        raise RuntimeError(
            "Apply the additive scheduled_model_updates setup SQL before deployment; "
            f"missing columns: {', '.join(sorted(missing))}. "
            "See db/sql/001_create_sports_models_schema.sql."
        )


def get_model_update_job(run_key: str) -> dict[str, Any] | None:
    """Read a scheduled or manual job without inferring completion from AWS delivery."""

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"select * from {SCHEMA}.scheduled_model_updates where run_key = %s",
            (run_key,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def insert_manual_model_update(record: dict[str, Any]) -> dict[str, Any]:
    """First submission wins, including its chosen week, client, and scheduled time."""

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            insert into {SCHEMA}.scheduled_model_updates (
                run_key, model_key, league, season, week, window_key,
                scheduled_for, aws_schedule_name, reason, trigger_source, client_name
            ) values (
                %(run_key)s, %(model_key)s, %(league)s, %(season)s, %(week)s, 'manual',
                %(scheduled_for)s, %(aws_schedule_name)s, %(reason)s, 'api', %(client_name)s
            ) on conflict (run_key) do nothing
            """,
            record,
        )
        cur.execute(
            f"select * from {SCHEMA}.scheduled_model_updates where run_key = %s",
            (record["run_key"],),
        )
        return dict(cur.fetchone())


def get_model_update_job_result(league: ExpectedPointsLeague | str, update_id: int) -> dict[str, Any] | None:
    """Read a compact result; large pick snapshots stay out of status responses."""

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            select id, model_version, source_git_sha, write_time, runtime,
                   picks_num, pick_changes, play_changes, updates_skipped
            from {_table(league, 'pick_updates')} where id = %s
            """,
            (update_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def confirm_manual_model_update_schedule(run_key: str) -> None:
    """Record AWS acceptance without overwriting a faster training attempt's state."""

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            update {SCHEMA}.scheduled_model_updates
            set status = 'scheduled', updated_at = now()
            where run_key = %s and trigger_source = 'api' and status = 'planned'
            """,
            (run_key,),
        )


def cancel_obsolete_manual_update(run_key: str) -> None:
    """Cancel only the claimed manual job, never a completed transaction."""

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            update {SCHEMA}.scheduled_model_updates
            set status = 'cancelled',
                last_error = 'A newer model week has already been published', updated_at = now()
            where run_key = %s and trigger_source = 'api'
              and status = 'running' and update_id is null
            """,
            (run_key,),
        )
        if cur.rowcount != 1:
            raise RuntimeError(f"Manual run is not claimable for cancellation: {run_key}")


def upsert_scheduled_model_update(record: dict[str, Any]) -> dict[str, Any]:
    """Create a plan or refresh its future AWS schedule without reopening history."""

    query = f"""
        insert into {SCHEMA}.scheduled_model_updates (
            run_key, model_key, league, season, week, window_key, game_date,
            scheduled_for, kickoff_at, aws_schedule_name, reason
        ) values (
            %(run_key)s, %(model_key)s, %(league)s, %(season)s, %(week)s, %(window_key)s, %(game_date)s,
            %(scheduled_for)s, %(kickoff_at)s, %(aws_schedule_name)s, %(reason)s
        )
        on conflict (run_key) do update set
            scheduled_for = case
                when {SCHEMA}.scheduled_model_updates.status in ('running', 'completed')
                    then {SCHEMA}.scheduled_model_updates.scheduled_for
                else excluded.scheduled_for
            end,
            kickoff_at = case
                when {SCHEMA}.scheduled_model_updates.status in ('running', 'completed')
                    then {SCHEMA}.scheduled_model_updates.kickoff_at
                else excluded.kickoff_at
            end,
            aws_schedule_name = case
                when {SCHEMA}.scheduled_model_updates.status in ('running', 'completed')
                    then {SCHEMA}.scheduled_model_updates.aws_schedule_name
                else excluded.aws_schedule_name
            end,
            reason = case
                when {SCHEMA}.scheduled_model_updates.status in ('running', 'completed')
                    then {SCHEMA}.scheduled_model_updates.reason
                else excluded.reason
            end,
            status = case
                when {SCHEMA}.scheduled_model_updates.status in ('cancelled', 'missed')
                    then 'planned'
                else {SCHEMA}.scheduled_model_updates.status
            end,
            last_error = case
                when {SCHEMA}.scheduled_model_updates.status in ('cancelled', 'missed')
                    then null
                else {SCHEMA}.scheduled_model_updates.last_error
            end,
            updated_at = now()
        returning run_key, status, scheduled_for, aws_schedule_name
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, normalize_record(record))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"Scheduled plan upsert returned no row for {record['run_key']}")
        return row


def set_scheduled_model_update_status(
    run_key: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    allowed = {"planned", "scheduled", "running", "failed", "cancelled", "missed"}
    if status not in allowed:
        raise ValueError(f"Unsupported scheduled update status: {status}")
    query = f"""
        update {SCHEMA}.scheduled_model_updates
        set status = case
                when status = 'completed' then status
                when %s in ('scheduled', 'cancelled', 'missed')
                     and status = 'running' then status
                else %s
            end,
            last_error = case
                when status = 'completed' then last_error
                when %s in ('scheduled', 'cancelled', 'missed')
                     and status = 'running' then last_error
                else %s
            end,
            updated_at = now()
        where run_key = %s
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            query,
            (
                status,
                status,
                status,
                error[:2000] if error else None,
                run_key,
            ),
        )
        if cur.rowcount != 1:
            raise ValueError(f"Unknown scheduled update run key: {run_key}")


def claim_scheduled_model_update(
    league: ExpectedPointsLeague | str,
    run_key: str,
    season: int,
    week: int,
    *,
    lease: timedelta = timedelta(minutes=15),
) -> str:
    """Claim one delivery using a lease matching the trainer's 900-second limit.

    Keep this bound aligned with the serialized training Lambda's timeout. AWS's
    first error retry can arrive one minute after timeout, so a 16-minute lease
    measured from the later database claim could reject that retry unnecessarily.
    """

    league_value = _coerce_league(league).value
    query = f"""
        update {SCHEMA}.scheduled_model_updates
        set status = 'running',
            claimed_at = now(),
            attempt_count = attempt_count + 1,
            last_error = null,
            updated_at = now()
        where run_key = %s
          and league = %s
          and model_key = %s
          and season = %s
          and week = %s
          and (
              status in ('planned', 'scheduled', 'failed')
              or (status = 'running' and claimed_at < now() - %s)
          )
        returning status
    """
    with get_connection() as conn, conn.cursor() as cur:
        identity = (run_key, league_value, model_key(league).value, season, week)
        cur.execute(query, (*identity, lease))
        claimed = cur.fetchone()
        if claimed is not None:
            return "claimed"
        cur.execute(
            f"""
            select status
            from {SCHEMA}.scheduled_model_updates
            where run_key = %s
              and league = %s
              and model_key = %s
              and season = %s
              and week = %s
            """,
            identity,
        )
        existing = cur.fetchone()
        return str(existing["status"]) if existing else "missing"


def _upsert_picks(cur, league: ExpectedPointsLeague | str, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    pick_columns = _pick_columns(league)
    columns = ", ".join(pick_columns)
    values = ", ".join(f"%({column})s" for column in pick_columns)
    updates = ",\n            ".join(
        f"{column} = excluded.{column}"
        for column in pick_columns
        if column not in {"year_week", "game_id"}
    )
    query = f"""
        insert into {_table(league, 'picks')} ({columns})
        values ({values})
        on conflict (year_week, game_id) do update set
            {updates}
    """
    cur.executemany(query, records)


def _upsert_results(cur, league: ExpectedPointsLeague | str, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    result_columns = _result_columns(league)
    columns = ", ".join(result_columns)
    values = ", ".join(f"%({column})s" for column in result_columns)
    updates = ",\n            ".join(
        f"{column} = excluded.{column}"
        for column in result_columns
        if column not in {"year_week", "game_id"}
    )
    query = f"""
        insert into {_table(league, 'results')} ({columns})
        values ({values})
        on conflict (year_week, game_id) do update set
            {updates}
    """
    cur.executemany(query, records)


def _prepare_update_record(result: dict[str, Any], write_time: datetime) -> dict[str, Any]:
    record = normalize_record(result)
    return {
        "year_week": record["year_week"],
        "write_time": write_time,
        "week": str(record["week"]),
        "season": record["season"],
        "environment": record.get("environment") or os.getenv("ENVIRONMENT") or "UNKNOWN",
        "client_name": record["client_name"],
        "runtime": record["runtime"],
        "model_version": record.get("model_version"),
        "source_git_sha": record.get("source_git_sha"),
        "evaluation_metrics": json_dumps(record.get("evaluation_metrics") or {}),
        "pick_changes": record["pick_changes"],
        "pick_changes_games": json_dumps(record["pick_changes_games"]),
        "play_changes": record["play_changes"],
        "play_changes_games": json_dumps(record["play_changes_games"]),
        "updates_skipped": record["updates_skipped"],
        "picks_num": record["picks_num"],
        "difference_df": json_dumps(record["difference_df"]),
        "picks_df": json_dumps(record["picks_df"]),
    }


def _insert_pick_update(
    cur,
    league: ExpectedPointsLeague | str,
    record: dict[str, Any],
) -> tuple[int, datetime]:
    columns = ", ".join(UPDATE_COLUMNS)
    values = ", ".join(f"%({column})s" for column in UPDATE_COLUMNS)
    query = f"""
        insert into {_table(league, 'pick_updates')} ({columns})
        values ({values})
        returning id, write_time
    """
    cur.execute(query, record)
    row = cur.fetchone()
    return int(row["id"]), row["write_time"]


def _complete_scheduled_model_update(
    cur,
    league: ExpectedPointsLeague | str,
    run_key: str,
    update_id: int,
    season: int,
    week: int,
) -> None:
    """Link a scheduled plan to its update row in the same write transaction."""

    if not run_key.startswith(("aws-scheduler:", "api:")):
        raise ValueError("Scheduled run key must start with 'aws-scheduler:' or 'api:'")
    league_value = _coerce_league(league).value
    query = f"""
        update {SCHEMA}.scheduled_model_updates
        set update_id = %s,
            status = 'completed',
            completed_at = now(),
            last_error = null,
            updated_at = now()
        where run_key = %s
          and league = %s
          and model_key = %s
          and season = %s
          and week = %s
          and status = 'running'
          and update_id is null
    """
    cur.execute(
        query,
        (update_id, run_key, league_value, model_key(league).value, season, week),
    )
    if cur.rowcount != 1:
        raise RuntimeError(f"Scheduled plan is not claimable for completion: {run_key}")


def write_expected_points_run(
    league: ExpectedPointsLeague | str,
    picks: Iterable[dict[str, Any]],
    update: dict[str, Any],
    results: Iterable[dict[str, Any]] = (),
    *,
    write_authorized: bool = False,
) -> datetime:
    if not is_aws_lambda_runtime() and not write_authorized:
        raise PermissionError(
            "Non-AWS expected-points writes require explicit authorization"
        )
    write_time = _parse_write_time(update.get("write_time") or datetime.now(timezone.utc))
    pick_records = normalize_records(picks)
    expected_model_version = update.get("model_version")
    if not expected_model_version:
        raise ValueError("Expected-points writes require model_version")
    parsed_version = parse_version(str(expected_model_version))
    expected_model_version = str(parsed_version)
    for record in pick_records:
        record["write_time"] = write_time
        record["model_version"] = str(parse_version(str(record.get("model_version") or expected_model_version)))
    result_records = normalize_records(results)
    for record in result_records:
        record["model_version"] = str(parse_version(str(record.get("model_version") or expected_model_version)))
    update_record = _prepare_update_record(update, write_time)
    update_record["model_version"] = expected_model_version
    update_record["source_git_sha"] = update.get("source_git_sha") or "unknown"

    with get_connection() as conn, conn.cursor() as cur:
        release_key = model_key(league).value
        referenced_versions = {
            expected_model_version,
            *(record["model_version"] for record in pick_records),
            *(record["model_version"] for record in result_records),
        }
        for referenced_version in referenced_versions:
            cur.execute(
                f"""
                select version
                from {SCHEMA}.model_releases
                where model_key = %s and version = %s
                for share
                """,
                (release_key, referenced_version),
            )
            if cur.fetchone() is None:
                raise ValueError(f"No registered release for {release_key} {referenced_version}")
        update_id, persisted_time = _insert_pick_update(cur, league, update_record)
        _upsert_picks(cur, league, pick_records)
        _upsert_results(cur, league, result_records)
        # A release becomes live only after code running in the real deployed
        # Lambda writes picks. Explicitly authorized local writes are useful for
        # recovery/testing but must never publish a release implicitly.
        if is_aws_lambda_runtime():
            cur.execute(
                f"""
                update {SCHEMA}.model_releases
                set first_pick_at = coalesce(first_pick_at, %s)
                where model_key = %s and version = %s
                """,
                (write_time, release_key, expected_model_version),
            )
        scheduled_run_key = os.getenv("EXPECTED_POINTS_RUN_KEY")
        if scheduled_run_key:
            _complete_scheduled_model_update(
                cur,
                league,
                scheduled_run_key,
                update_id,
                int(update_record["season"]),
                int(update_record["week"]),
            )
        return persisted_time


def upsert_expected_points_picks(
    league: ExpectedPointsLeague | str,
    picks: Iterable[dict[str, Any]],
) -> None:
    records = normalize_records(picks)
    if not records:
        return
    with get_connection() as conn, conn.cursor() as cur:
        _upsert_picks(cur, league, records)


def upsert_expected_points_results(
    league: ExpectedPointsLeague | str,
    results: Iterable[dict[str, Any]],
) -> None:
    records = normalize_records(results)
    if not records:
        return
    with get_connection() as conn, conn.cursor() as cur:
        _upsert_results(cur, league, records)


def insert_expected_points_pick_update(
    league: ExpectedPointsLeague | str,
    result: dict[str, Any],
) -> datetime:
    write_time = _parse_write_time(result.get("write_time") or datetime.now(timezone.utc))
    record = _prepare_update_record(result, write_time)
    with get_connection() as conn, conn.cursor() as cur:
        return _insert_pick_update(cur, league, record)


def clear_expected_points_data(league: ExpectedPointsLeague | str) -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"truncate table {_table(league, 'picks')}")
        cur.execute(f"truncate table {_table(league, 'pick_updates')} restart identity")
        cur.execute(f"truncate table {_table(league, 'results')}")


def get_nba_first_basket_picks() -> list[dict[str, Any]]:
    query = f"""
        select
            to_char(pick_date, 'YYYY-MM-DD') as date,
            player_name, team, fb_model_prob, fb_model_odds, odds, sportsbook, units
        from {SCHEMA}.nba_first_basket_picks
        order by pick_date desc, player_name asc
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query)
        return list(cur.fetchall())


def replace_nba_first_basket_picks(picks: Iterable[dict[str, Any]]) -> int:
    records = normalize_records(picks)
    if not records:
        return 0
    unique_dates = sorted({record["date"] for record in records})
    delete_query = f"delete from {SCHEMA}.nba_first_basket_picks where pick_date = any(%s)"
    insert_query = f"""
        insert into {SCHEMA}.nba_first_basket_picks (
            pick_date, player_name, team, fb_model_prob, fb_model_odds, odds, sportsbook, units
        ) values (
            %(date)s, %(player_name)s, %(team)s, %(fb_model_prob)s, %(fb_model_odds)s,
            %(odds)s, %(sportsbook)s, %(units)s
        )
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(delete_query, (unique_dates,))
        cur.executemany(insert_query, records)
    return len(records)
