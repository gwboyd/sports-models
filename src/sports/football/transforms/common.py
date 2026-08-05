"""League-neutral feature transforms shared by football models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.sports.data_validation import enforce_contract, validate_frame


def dynamic_period_ewma(
    frame: pd.DataFrame,
    shifted_col: str,
    *,
    period_col: str = "week",
    minimum_span: int = 10,
) -> pd.Series:
    """Return an EWMA whose span grows with the in-season period number."""
    values = np.zeros(len(frame))
    for position, (_, row) in enumerate(frame.iterrows()):
        history = frame[shifted_col].iloc[: position + 1]
        span = max(minimum_span, int(row[period_col]))
        values[position] = history.ewm(min_periods=1, span=span).mean().iloc[-1]
    return pd.Series(values, index=frame.index)


def build_lagged_team_metrics(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    group_col: str = "team",
    game_col: str = "game_id",
    period_col: str = "week",
    time_col: str = "start_date",
) -> pd.DataFrame:
    """Build leakage-safe team features in actual event order.

    Feed ordering must never decide which event is considered previous. The
    caller remains responsible for adapting league-specific feed columns to this
    common team/event contract.
    """
    required = (group_col, game_col, period_col, time_col, *columns)
    validate_frame(
        df,
        label="football team metrics",
        required_columns=required,
        non_null_columns=(group_col, game_col, period_col, time_col),
        unique_keys=((game_col, group_col),),
        # Missing chronology inputs cannot be made safe by warning mode.
        strict=True,
    )

    output = df.copy()
    output[time_col] = pd.to_datetime(output[time_col], utc=True, errors="coerce")
    enforce_contract(
        ["football team metrics contains invalid event timestamps"]
        if output[time_col].isna().any()
        else [],
        strict=True,
    )
    output = output.sort_values(
        [group_col, time_col, game_col], kind="stable"
    ).reset_index(drop=True)

    previous_event = output.groupby(group_col, sort=False)[time_col].shift()
    invalid_history = previous_event.notna() & (previous_event >= output[time_col])
    enforce_contract(
        [
            "football team metrics cannot establish a strictly earlier prior event "
            f"for {int(invalid_history.sum())} rows"
        ]
        if invalid_history.any()
        else [],
        strict=True,
    )

    for col in columns:
        col_shifted = f"{col}_shifted"
        col_ewma = f"{col}_ewma"
        col_ewma_dynamic_window = f"{col}_ewma_dynamic_window"

        output[col_shifted] = output.groupby(group_col, sort=False)[col].shift()
        output[col_ewma] = output.groupby(group_col, sort=False)[col_shifted].transform(
            lambda x: x.ewm(min_periods=1, span=10).mean()
        )
        dynamic_values = pd.Series(index=output.index, dtype=float)
        for _, group in output.groupby(group_col, sort=False):
            dynamic_values.loc[group.index] = dynamic_period_ewma(
                group,
                col_shifted,
                period_col=period_col,
            )
        output[col_ewma_dynamic_window] = dynamic_values

    return output


__all__ = ["build_lagged_team_metrics", "dynamic_period_ewma"]
