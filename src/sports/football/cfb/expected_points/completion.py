"""Honor CFBD final status while retaining unfinished games on the schedule."""

from __future__ import annotations

import pandas as pd


def cfb_completed_game_mask(frame: pd.DataFrame) -> pd.Series:
    """Use explicit final status when supplied; retain legacy status-free inputs.

    Historical CSV rows use ``status="completed"`` and API rows use a boolean
    ``completed`` flag. The CSV status can fill missing API flags after their
    frames are combined, but cannot override an explicit false flag.
    """
    if "completed" not in frame and "status" not in frame:
        return pd.Series(True, index=frame.index, dtype=bool)
    status = (
        frame["status"].astype("string").str.strip().str.lower().eq("completed")
        if "status" in frame else pd.Series(False, index=frame.index, dtype=bool)
    )
    if "completed" in frame:
        status = frame["completed"].eq(True).where(frame["completed"].notna(), status)
    return status.fillna(False).astype(bool)


def mask_incomplete_cfb_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep scheduled rows, but prevent partial scores from becoming final labels."""
    output = frame.copy()
    if "completed" not in frame and "status" not in frame:
        return output
    completed = cfb_completed_game_mask(frame)
    output["completed"] = completed
    for column in ("home_points", "away_points", "home_score", "away_score"):
        if column in output:
            output[column] = output[column].where(completed)
    return output
