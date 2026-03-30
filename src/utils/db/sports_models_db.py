from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable
import os

from src.utils.postgres import get_connection, get_schema, json_dumps, normalize_record, normalize_records


SCHEMA = get_schema()


def _parse_write_time(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00").replace(" ", "T"))


def get_latest_nfl_picks() -> list[dict[str, Any]]:
    query = f"""
        select
            season,
            week,
            home_team,
            away_team,
            home_score_pred,
            away_score_pred,
            spread_pred,
            spread_line,
            spread_play,
            spread_win_prob,
            spread_lock,
            total_pred,
            total_line,
            total_play,
            total_win_prob,
            total_lock,
            game_id,
            year_week,
            date_time,
            to_char(write_time at time zone 'UTC', 'YYYY-MM-DD HH24:MI:SS') as write_time
        from {SCHEMA}.nfl_expected_points_latest_picks
        order by date_time asc, game_id asc
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query)
        return list(cur.fetchall())


def get_nfl_picks() -> list[dict[str, Any]]:
    query = f"""
        select
            season,
            week,
            home_team,
            away_team,
            home_score_pred,
            away_score_pred,
            spread_pred,
            spread_line,
            spread_play,
            spread_win_prob,
            spread_lock,
            total_pred,
            total_line,
            total_play,
            total_win_prob,
            total_lock,
            game_id,
            year_week,
            date_time,
            to_char(write_time at time zone 'UTC', 'YYYY-MM-DD HH24:MI:SS') as write_time
        from {SCHEMA}.nfl_expected_points_picks
        order by write_time asc, date_time asc, game_id asc
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query)
        return list(cur.fetchall())


def get_nfl_results() -> list[dict[str, Any]]:
    query = f"""
        select
            season,
            week,
            home_team,
            away_team,
            home_score,
            away_score,
            home_score_pred,
            away_score_pred,
            spread_pred,
            spread_line,
            true_spread,
            spread_play,
            spread_win_prob,
            spread_lock,
            correct_spread_play,
            spread_win,
            total_pred,
            total_line,
            true_total,
            total_play,
            total_win_prob,
            total_lock,
            correct_total_play,
            total_win,
            year_week,
            game_id,
            date_time
        from {SCHEMA}.nfl_expected_points_results
        order by season desc, week desc, date_time asc, game_id asc
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query)
        return list(cur.fetchall())


def upsert_nfl_picks(picks: Iterable[dict[str, Any]]) -> None:
    records = normalize_records(picks)
    query = f"""
        insert into {SCHEMA}.nfl_expected_points_picks (
            season, week, year_week, game_id, home_team, away_team, home_score_pred,
            away_score_pred, spread_pred, spread_line, spread_play, spread_win_prob,
            spread_lock, total_pred, total_line, total_play, total_win_prob, total_lock,
            date_time, write_time
        ) values (
            %(season)s, %(week)s, %(year_week)s, %(game_id)s, %(home_team)s, %(away_team)s,
            %(home_score_pred)s, %(away_score_pred)s, %(spread_pred)s, %(spread_line)s,
            %(spread_play)s, %(spread_win_prob)s, %(spread_lock)s, %(total_pred)s,
            %(total_line)s, %(total_play)s, %(total_win_prob)s, %(total_lock)s,
            %(date_time)s, %(write_time)s
        )
        on conflict (year_week, game_id) do update set
            season = excluded.season,
            week = excluded.week,
            home_team = excluded.home_team,
            away_team = excluded.away_team,
            home_score_pred = excluded.home_score_pred,
            away_score_pred = excluded.away_score_pred,
            spread_pred = excluded.spread_pred,
            spread_line = excluded.spread_line,
            spread_play = excluded.spread_play,
            spread_win_prob = excluded.spread_win_prob,
            spread_lock = excluded.spread_lock,
            total_pred = excluded.total_pred,
            total_line = excluded.total_line,
            total_play = excluded.total_play,
            total_win_prob = excluded.total_win_prob,
            total_lock = excluded.total_lock,
            date_time = excluded.date_time,
            write_time = excluded.write_time
    """
    if not records:
        return
    with get_connection() as conn, conn.cursor() as cur:
        cur.executemany(query, records)


def upsert_nfl_results(results: Iterable[dict[str, Any]]) -> None:
    records = normalize_records(results)
    query = f"""
        insert into {SCHEMA}.nfl_expected_points_results (
            season, week, year_week, game_id, home_team, away_team, home_score, away_score,
            home_score_pred, away_score_pred, spread_pred, spread_line, true_spread, spread_play,
            spread_win_prob, spread_lock, correct_spread_play, spread_win, total_pred,
            total_line, true_total, total_play, total_win_prob, total_lock,
            correct_total_play, total_win, date_time
        ) values (
            %(season)s, %(week)s, %(year_week)s, %(game_id)s, %(home_team)s, %(away_team)s,
            %(home_score)s, %(away_score)s, %(home_score_pred)s, %(away_score_pred)s,
            %(spread_pred)s, %(spread_line)s, %(true_spread)s, %(spread_play)s,
            %(spread_win_prob)s, %(spread_lock)s, %(correct_spread_play)s, %(spread_win)s,
            %(total_pred)s, %(total_line)s, %(true_total)s, %(total_play)s,
            %(total_win_prob)s, %(total_lock)s, %(correct_total_play)s, %(total_win)s,
            %(date_time)s
        )
        on conflict (year_week, game_id) do update set
            season = excluded.season,
            week = excluded.week,
            home_team = excluded.home_team,
            away_team = excluded.away_team,
            home_score = excluded.home_score,
            away_score = excluded.away_score,
            home_score_pred = excluded.home_score_pred,
            away_score_pred = excluded.away_score_pred,
            spread_pred = excluded.spread_pred,
            spread_line = excluded.spread_line,
            true_spread = excluded.true_spread,
            spread_play = excluded.spread_play,
            spread_win_prob = excluded.spread_win_prob,
            spread_lock = excluded.spread_lock,
            correct_spread_play = excluded.correct_spread_play,
            spread_win = excluded.spread_win,
            total_pred = excluded.total_pred,
            total_line = excluded.total_line,
            true_total = excluded.true_total,
            total_play = excluded.total_play,
            total_win_prob = excluded.total_win_prob,
            total_lock = excluded.total_lock,
            correct_total_play = excluded.correct_total_play,
            total_win = excluded.total_win,
            date_time = excluded.date_time
    """
    if not records:
        return
    with get_connection() as conn, conn.cursor() as cur:
        cur.executemany(query, records)


def insert_nfl_pick_update(result: dict[str, Any]) -> datetime:
    record = normalize_record(result)
    record["write_time"] = _parse_write_time(record["write_time"])
    query = f"""
        insert into {SCHEMA}.nfl_expected_points_pick_updates (
            year_week, write_time, week, season, environment, client_name, runtime,
            pick_changes, pick_changes_games, play_changes, play_changes_games,
            updates_skipped, picks_num, difference_df, picks_df
        ) values (
            %(year_week)s, %(write_time)s, %(week)s, %(season)s, %(environment)s,
            %(client_name)s, %(runtime)s, %(pick_changes)s, %(pick_changes_games)s,
            %(play_changes)s, %(play_changes_games)s, %(updates_skipped)s, %(picks_num)s,
            %(difference_df)s, %(picks_df)s
        )
        returning write_time
    """
    params = {
        "year_week": record["year_week"],
        "write_time": record["write_time"],
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
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()
        return row["write_time"]


def get_nba_first_basket_picks() -> list[dict[str, Any]]:
    query = f"""
        select
            to_char(pick_date, 'YYYY-MM-DD') as date,
            player_name,
            team,
            fb_model_prob,
            fb_model_odds,
            odds,
            sportsbook,
            units
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


def clear_nfl_picks() -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"truncate table {SCHEMA}.nfl_expected_points_picks")


def clear_nfl_pick_updates() -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"truncate table {SCHEMA}.nfl_expected_points_pick_updates restart identity")


def clear_nfl_results() -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"truncate table {SCHEMA}.nfl_expected_points_results")
