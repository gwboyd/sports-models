from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Iterable

from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.utils.postgres import get_connection, get_schema, json_dumps, normalize_record, normalize_records


SCHEMA = get_schema()

PICK_COLUMNS = (
    "season", "week", "year_week", "game_id", "home_team", "away_team",
    "home_score_pred", "away_score_pred", "spread_pred", "spread_line", "spread_play",
    "spread_win_prob", "spread_lock", "total_pred", "total_line", "total_play",
    "total_win_prob", "total_lock", "date_time", "write_time",
)
CFB_PICK_METADATA_COLUMNS = ("home_conference", "away_conference")
RESULT_COLUMNS = (
    "season", "week", "year_week", "game_id", "home_team", "away_team", "home_score",
    "away_score", "home_score_pred", "away_score_pred", "spread_pred", "spread_line",
    "true_spread", "spread_play", "spread_win_prob", "spread_lock", "correct_spread_play",
    "spread_win", "total_pred", "total_line", "true_total", "total_play", "total_win_prob",
    "total_lock", "correct_total_play", "total_win", "date_time",
)
CFB_RESULT_METADATA_COLUMNS = CFB_PICK_METADATA_COLUMNS
UPDATE_COLUMNS = (
    "year_week", "write_time", "week", "season", "environment", "client_name", "runtime",
    "pick_changes", "pick_changes_games", "play_changes", "play_changes_games",
    "updates_skipped", "picks_num", "difference_df", "picks_df",
)

_TABLE_PREFIXES = {
    ExpectedPointsLeague.NFL: "nfl",
    ExpectedPointsLeague.CFB: "cfb",
}


def _coerce_league(league: ExpectedPointsLeague | str) -> ExpectedPointsLeague:
    return league if isinstance(league, ExpectedPointsLeague) else ExpectedPointsLeague(league)


def _table(league: ExpectedPointsLeague | str, suffix: str) -> str:
    prefix = _TABLE_PREFIXES[_coerce_league(league)]
    if suffix not in {"picks", "latest_picks", "results", "pick_updates"}:
        raise ValueError(f"Unsupported expected-points table suffix: {suffix}")
    return f"{SCHEMA}.{prefix}_expected_points_{suffix}"


def _pick_columns(league: ExpectedPointsLeague | str) -> tuple[str, ...]:
    league = _coerce_league(league)
    if league is ExpectedPointsLeague.CFB:
        return PICK_COLUMNS[:-1] + CFB_PICK_METADATA_COLUMNS + PICK_COLUMNS[-1:]
    return PICK_COLUMNS


def _result_columns(league: ExpectedPointsLeague | str) -> tuple[str, ...]:
    league = _coerce_league(league)
    if league is ExpectedPointsLeague.CFB:
        return RESULT_COLUMNS + CFB_RESULT_METADATA_COLUMNS
    return RESULT_COLUMNS


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
    for record in pick_records:
        record["write_time"] = write_time
    result_records = normalize_records(results)
    update_record = _prepare_update_record(update, write_time)

    with get_connection() as conn, conn.cursor() as cur:
        persisted_time = _insert_pick_update(cur, league, update_record)
        _upsert_picks(cur, league, pick_records)
        _upsert_results(cur, league, result_records)
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


# Compatibility wrappers for existing callers.
def get_latest_nfl_picks() -> list[dict[str, Any]]:
    return get_expected_points_picks(ExpectedPointsLeague.NFL, latest=True)


def get_nfl_picks() -> list[dict[str, Any]]:
    return get_expected_points_picks(ExpectedPointsLeague.NFL)


def get_nfl_results() -> list[dict[str, Any]]:
    return get_expected_points_results(ExpectedPointsLeague.NFL)


def upsert_nfl_picks(picks: Iterable[dict[str, Any]]) -> None:
    upsert_expected_points_picks(ExpectedPointsLeague.NFL, picks)


def upsert_nfl_results(results: Iterable[dict[str, Any]]) -> None:
    upsert_expected_points_results(ExpectedPointsLeague.NFL, results)


def insert_nfl_pick_update(result: dict[str, Any]) -> datetime:
    return insert_expected_points_pick_update(ExpectedPointsLeague.NFL, result)


def clear_nfl_picks() -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"truncate table {_table(ExpectedPointsLeague.NFL, 'picks')}")


def clear_nfl_pick_updates() -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"truncate table {_table(ExpectedPointsLeague.NFL, 'pick_updates')} restart identity")


def clear_nfl_results() -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"truncate table {_table(ExpectedPointsLeague.NFL, 'results')}")


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
