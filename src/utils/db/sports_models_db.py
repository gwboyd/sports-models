from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Iterable

from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.model_patterns.expected_points.versioning import ModelKey, ReleaseDraft, parse_version
from src.utils.postgres import get_connection, get_schema, json_dumps, normalize_record, normalize_records


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


def insert_model_release(
    model: ModelKey | str,
    version: str,
    draft: ReleaseDraft,
    *,
    source_git_sha: str | None,
    deployed_at: datetime,
) -> None:
    insert_model_releases(
        [(model, version, draft)],
        source_git_sha=source_git_sha,
        deployed_at=deployed_at,
    )


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
) -> datetime:
    columns = ", ".join(UPDATE_COLUMNS)
    values = ", ".join(f"%({column})s" for column in UPDATE_COLUMNS)
    query = f"""
        insert into {_table(league, 'pick_updates')} ({columns})
        values ({values})
        returning write_time
    """
    cur.execute(query, record)
    return cur.fetchone()["write_time"]


def write_expected_points_run(
    league: ExpectedPointsLeague | str,
    picks: Iterable[dict[str, Any]],
    update: dict[str, Any],
    results: Iterable[dict[str, Any]] = (),
) -> datetime:
    write_time = _parse_write_time(update.get("write_time") or datetime.now(timezone.utc))
    pick_records = normalize_records(picks)
    expected_model_version = update.get("model_version")
    if not expected_model_version and (update.get("environment") or "").upper() != "PROD":
        # Preserve the pre-versioning test/notebook contract outside production.
        expected_model_version = "1.0"
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
        persisted_time = _insert_pick_update(cur, league, update_record)
        _upsert_picks(cur, league, pick_records)
        _upsert_results(cur, league, result_records)
        cur.execute(
            f"""
            update {SCHEMA}.model_releases
            set first_pick_at = coalesce(first_pick_at, %s)
            where model_key = %s and version = %s
            """,
            (write_time, release_key, expected_model_version),
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
