"""Shared kickoff-time contract for NFL and CFB workflows."""

from __future__ import annotations

import pandas as pd


EASTERN_TIME_ZONE = "America/New_York"
KICKOFF_FORMAT = "%Y-%m-%d-%H:%M"


def parse_eastern_kickoffs(values: pd.Series) -> pd.Series:
    """Interpret API-facing kickoff strings as New York wall-clock time."""
    parsed = pd.to_datetime(values, format=KICKOFF_FORMAT, errors="coerce")
    try:
        return parsed.dt.tz_localize(
            EASTERN_TIME_ZONE,
            ambiguous="raise",
            nonexistent="raise",
        ).dt.tz_convert("UTC")
    except (TypeError, ValueError) as exc:
        raise ValueError("Kickoff values must be valid America/New_York wall times") from exc


def utc_kickoffs_to_eastern_strings(values: pd.Series) -> pd.Series:
    """Convert CFBD UTC timestamps to the shared Eastern wall-time contract."""
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return parsed.dt.tz_convert(EASTERN_TIME_ZONE).dt.strftime(KICKOFF_FORMAT)


def nfl_kickoffs_to_eastern_strings(gameday: pd.Series, gametime: pd.Series) -> pd.Series:
    """Validate and format nflverse date/time fields, which are already Eastern."""
    combined = gameday.astype(str) + "-" + gametime.astype(str)
    parsed = parse_eastern_kickoffs(combined)
    return parsed.dt.tz_convert(EASTERN_TIME_ZONE).dt.strftime(KICKOFF_FORMAT)


def exclude_started_games(
    frame: pd.DataFrame,
    *,
    kickoff_col: str = "date_time",
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return games whose kickoff is strictly after the comparison instant."""
    if kickoff_col not in frame.columns:
        raise ValueError(f"Prediction frame is missing kickoff column: {kickoff_col}")
    kickoffs = parse_eastern_kickoffs(frame[kickoff_col])
    if kickoffs.isna().any():
        raise ValueError("Prediction frame contains invalid or null kickoff values")

    compare_time = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if compare_time.tzinfo is None:
        raise ValueError("Prediction comparison time must be timezone-aware")
    compare_time = compare_time.tz_convert("UTC")
    return frame.loc[kickoffs > compare_time].copy()


__all__ = [
    "EASTERN_TIME_ZONE",
    "KICKOFF_FORMAT",
    "exclude_started_games",
    "nfl_kickoffs_to_eastern_strings",
    "parse_eastern_kickoffs",
    "utc_kickoffs_to_eastern_strings",
]
