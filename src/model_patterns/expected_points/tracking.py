from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from .betting import calculate_wins, determine_plays
from .types import ExpectedPointsTrackingConfig


PICK_COLUMNS = [
    "season",
    "week",
    "year_week",
    "game_id",
    "home_team",
    "away_team",
    "home_score_pred",
    "away_score_pred",
    "spread_pred",
    "spread_line",
    "spread_play",
    "spread_win_prob",
    "spread_lock",
    "total_pred",
    "total_line",
    "total_play",
    "total_win_prob",
    "total_lock",
    "date_time",
]

RESULT_COLUMNS = [
    "season",
    "week",
    "year_week",
    "game_id",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_score_pred",
    "away_score_pred",
    "spread_pred",
    "spread_line",
    "true_spread",
    "spread_play",
    "spread_win_prob",
    "spread_lock",
    "correct_spread_play",
    "spread_win",
    "total_pred",
    "total_line",
    "true_total",
    "total_play",
    "total_win_prob",
    "total_lock",
    "correct_total_play",
    "total_win",
    "date_time",
]

PICK_COMPARISON_COLUMNS = ["spread_play", "total_play"]
PLAY_COMPARISON_COLUMNS = ["spread_lock", "total_lock"]


@dataclass
class TrackingRun:
    picks: pd.DataFrame
    differences: pd.DataFrame
    locked_game_ids: list[str]
    pick_changes_games: list[str]
    play_changes_games: list[str]
    pick_metadata_columns: tuple[str, ...] = ()

    @property
    def pick_changes(self) -> int:
        return len(self.pick_changes_games)

    @property
    def play_changes(self) -> int:
        return len(self.play_changes_games)

    @property
    def updates_skipped(self) -> int:
        return len(self.locked_game_ids)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    # Model joins may leave duplicate auxiliary feature labels. Tracking only
    # needs one canonical value per column and later narrows to PICK_COLUMNS.
    output = frame.loc[:, ~frame.columns.duplicated()].copy()
    if "game_id" in output.columns:
        output["game_id"] = output["game_id"].astype(str)
    if "week" in output.columns:
        output["week"] = output["week"].astype(str)
    return output


def validate_pick_frame(
    frame: pd.DataFrame,
    expected_game_ids: Iterable[str] | None = None,
    metadata_columns: Iterable[str] = (),
) -> pd.DataFrame:
    output = _normalize_keys(frame)
    metadata_columns = tuple(metadata_columns)
    _require_columns(output, PICK_COLUMNS + list(metadata_columns), "Pick frame")

    if output.empty:
        raise ValueError("Pick frame cannot be empty")
    if output.duplicated(subset=["year_week", "game_id"]).any():
        duplicates = output.loc[
            output.duplicated(subset=["year_week", "game_id"], keep=False), "game_id"
        ].tolist()
        raise ValueError(f"Pick frame contains duplicate game IDs: {duplicates}")

    null_columns = [column for column in PICK_COLUMNS if output[column].isna().any()]
    if null_columns:
        raise ValueError(f"Pick frame contains null required values: {', '.join(null_columns)}")

    parsed_kickoffs = pd.to_datetime(output["date_time"], format="%Y-%m-%d-%H:%M", utc=True, errors="coerce")
    if parsed_kickoffs.isna().any():
        invalid = output.loc[parsed_kickoffs.isna(), "game_id"].tolist()
        raise ValueError(f"Pick frame contains invalid date_time values for games: {invalid}")

    if expected_game_ids is not None:
        expected = {str(game_id) for game_id in expected_game_ids}
        actual = set(output["game_id"])
        if expected != actual:
            raise ValueError(
                "Prediction game IDs do not match eligible games "
                f"(missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)})"
            )
    return output


def get_current_period_picks(all_picks: pd.DataFrame, year_week: str) -> pd.DataFrame:
    if all_picks.empty:
        return all_picks.copy()
    return _normalize_keys(all_picks[all_picks["year_week"] == year_week].copy())


def get_locked_picks(
    existing_picks: pd.DataFrame,
    config: ExpectedPointsTrackingConfig,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if existing_picks.empty or not config.lock_started_games:
        return existing_picks.iloc[0:0].copy()

    compare_time = (now or pd.Timestamp.now(tz="UTC")) + pd.Timedelta(minutes=config.lock_window_minutes)
    kickoff_times = pd.to_datetime(
        existing_picks["date_time"], format="%Y-%m-%d-%H:%M", utc=True, errors="coerce"
    )
    return existing_picks[kickoff_times < compare_time].copy()


def _changed_game_ids(existing: pd.DataFrame, predicted: pd.DataFrame, columns: list[str]) -> list[str]:
    existing_by_id = existing.set_index("game_id") if not existing.empty else pd.DataFrame()
    predicted_by_id = predicted.set_index("game_id") if not predicted.empty else pd.DataFrame()
    game_ids = sorted(set(existing.get("game_id", [])) | set(predicted.get("game_id", [])))
    changed: list[str] = []
    for game_id in game_ids:
        if game_id not in existing_by_id.index or game_id not in predicted_by_id.index:
            changed.append(str(game_id))
            continue
        if any(existing_by_id.at[game_id, column] != predicted_by_id.at[game_id, column] for column in columns):
            changed.append(str(game_id))
    return changed


def summarize_pick_diffs(
    existing: pd.DataFrame,
    predicted: pd.DataFrame,
    metadata_columns: Iterable[str] = (),
) -> tuple[pd.DataFrame, list[str], list[str]]:
    existing = _normalize_keys(existing)
    predicted = _normalize_keys(predicted)
    metadata_columns = tuple(metadata_columns)
    pick_changes = _changed_game_ids(existing, predicted, PICK_COMPARISON_COLUMNS)
    play_changes = _changed_game_ids(existing, predicted, PLAY_COMPARISON_COLUMNS)
    changed_ids = set(pick_changes) | set(play_changes)
    snapshot_columns = PICK_COLUMNS + list(metadata_columns) + ["write_time"]

    frames: list[pd.DataFrame] = []
    if not existing.empty:
        columns = [column for column in snapshot_columns if column in existing.columns]
        frames.append(
            existing.loc[existing["game_id"].isin(changed_ids), columns].assign(source="database")
        )
    if not predicted.empty:
        columns = [column for column in snapshot_columns if column in predicted.columns]
        frames.append(
            predicted.loc[predicted["game_id"].isin(changed_ids), columns].assign(source="predictions")
        )
    if not frames:
        return pd.DataFrame(
            columns=PICK_COLUMNS + list(metadata_columns) + ["write_time", "source"]
        ), pick_changes, play_changes

    differences = pd.concat(frames, ignore_index=True, sort=False)
    return differences.sort_values(["date_time", "home_team", "source"]).reset_index(drop=True), pick_changes, play_changes


def prepare_tracking_run(
    predicted_picks: pd.DataFrame,
    all_existing_picks: pd.DataFrame,
    year_week: str,
    config: ExpectedPointsTrackingConfig,
    now: pd.Timestamp | None = None,
) -> TrackingRun:
    predicted_input = _normalize_keys(predicted_picks)
    expected_game_ids = predicted_input["game_id"].tolist()
    metadata_columns = config.pick_metadata_columns
    pick_columns = PICK_COLUMNS + list(metadata_columns)
    predicted = validate_pick_frame(
        predicted_input,
        expected_game_ids,
        metadata_columns,
    )[pick_columns].copy()
    existing = get_current_period_picks(all_existing_picks, year_week)
    locked = get_locked_picks(existing, config=config, now=now)

    final_picks = predicted.set_index("game_id")
    if not locked.empty:
        final_picks.update(locked.set_index("game_id"))
    final_picks = final_picks.reset_index()
    final_picks = determine_plays(
        final_picks,
        thresholds=config.play_thresholds,
        dont_update=locked["game_id"].tolist() if not locked.empty else [],
    )
    final_picks = validate_pick_frame(final_picks, expected_game_ids, metadata_columns)
    differences, pick_changes, play_changes = summarize_pick_diffs(
        existing,
        final_picks,
        metadata_columns,
    )
    return TrackingRun(
        picks=final_picks,
        differences=differences,
        locked_game_ids=locked["game_id"].tolist() if not locked.empty else [],
        pick_changes_games=pick_changes,
        play_changes_games=play_changes,
        pick_metadata_columns=metadata_columns,
    )


def build_pick_records(
    frame: pd.DataFrame,
    write_time: datetime,
    metadata_columns: Iterable[str] = (),
) -> list[dict[str, Any]]:
    metadata_columns = tuple(metadata_columns)
    output = validate_pick_frame(frame, metadata_columns=metadata_columns)
    records = output[PICK_COLUMNS + list(metadata_columns)].to_dict(orient="records")
    for record in records:
        record["write_time"] = write_time
    return records


def build_result_records(
    frame: pd.DataFrame,
    metadata_columns: Iterable[str] = (),
) -> list[dict[str, Any]]:
    output = _normalize_keys(frame)
    metadata_columns = tuple(metadata_columns)
    _require_columns(output, RESULT_COLUMNS + list(metadata_columns), "Result frame")
    records = output[RESULT_COLUMNS + list(metadata_columns)].to_dict(orient="records")
    for record in records:
        record["home_score"] = int(record["home_score"])
        record["away_score"] = int(record["away_score"])
    return records


def serialize_pick_frame(
    frame: pd.DataFrame,
    metadata_columns: Iterable[str] = (),
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    columns = [
        column
        for column in PICK_COLUMNS + list(metadata_columns) + ["write_time", "source"]
        if column in frame.columns
    ]
    output = _normalize_keys(frame[columns])
    records = output.to_dict(orient="records")
    return [
        {key: None if pd.isna(value) else value for key, value in record.items()}
        for record in records
    ]


def build_update_record(
    run: TrackingRun,
    *,
    year_week: str,
    week: int,
    season: int,
    environment: str,
    client_name: str,
    runtime: float,
    write_time: datetime | None = None,
) -> dict[str, Any]:
    return {
        "year_week": year_week,
        "write_time": write_time or datetime.now(timezone.utc),
        "week": week,
        "season": season,
        "environment": environment,
        "client_name": client_name,
        "runtime": runtime,
        "pick_changes": run.pick_changes,
        "pick_changes_games": run.pick_changes_games,
        "play_changes": run.play_changes,
        "play_changes_games": run.play_changes_games,
        "updates_skipped": run.updates_skipped,
        "picks_num": len(run.picks),
        "difference_df": serialize_pick_frame(run.differences, run.pick_metadata_columns),
        "picks_df": serialize_pick_frame(run.picks, run.pick_metadata_columns),
    }


def grade_completed_picks(
    existing_picks: pd.DataFrame,
    existing_results: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    if existing_picks.empty or scores.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    picks = _normalize_keys(existing_picks)
    results = _normalize_keys(existing_results)
    score_frame = _normalize_keys(scores[["game_id", "home_score", "away_score"]])
    completed_ids = set(score_frame.dropna(subset=["home_score", "away_score"])["game_id"])
    result_ids = set(results["game_id"]) if not results.empty else set()
    pending = picks[picks["game_id"].isin(completed_ids - result_ids)].copy()
    if pending.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    pending = pending.merge(score_frame, on="game_id", how="inner")
    return calculate_wins(pending)
