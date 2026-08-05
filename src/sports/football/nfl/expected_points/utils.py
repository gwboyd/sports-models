"""nflverse adapters for the NFL expected-points workflow."""

from __future__ import annotations

import pandas as pd

from src.sports.football.kickoff import (
    EASTERN_TIME_ZONE,
    KICKOFF_FORMAT,
    parse_eastern_kickoffs,
)


def nflverse_kickoffs_to_eastern_strings(
    gameday: pd.Series,
    gametime: pd.Series,
) -> pd.Series:
    """Validate nflverse date/time fields, which are Eastern wall times."""
    combined = gameday.astype(str) + "-" + gametime.astype(str)
    parsed = parse_eastern_kickoffs(combined)
    return parsed.dt.tz_convert(EASTERN_TIME_ZONE).dt.strftime(KICKOFF_FORMAT)


__all__ = ["nflverse_kickoffs_to_eastern_strings"]
