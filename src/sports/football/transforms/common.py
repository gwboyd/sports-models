import numpy as np
import pandas as pd

from src.sports.data_validation import enforce_contract, validate_frame


def dynamic_window_ewma(x: pd.DataFrame, col_shifted: str) -> pd.Series:
    values = np.zeros(len(x))
    for i, (_, row) in enumerate(x.iterrows()):
        epa_col = x[col_shifted][: i + 1]
        span = row["week"] if row["week"] > 10 else 10
        values[i] = epa_col.ewm(min_periods=1, span=span).mean().iloc[-1]
    return pd.Series(values, index=x.index)


def get_averaged_game_stats(
    df: pd.DataFrame,
    columns,
    *,
    time_col: str = "start_date",
) -> pd.DataFrame:
    """Build lagged team features in actual kickoff order.

    Sorting is part of the feature contract: API and CSV concatenation order is
    not chronological and must never decide which game is considered previous.
    """
    required = ("team", "game_id", "week", time_col, *columns)
    validate_frame(
        df,
        label="CFB advanced game stats",
        required_columns=required,
        non_null_columns=("team", "game_id", "week", time_col),
        unique_keys=(("game_id", "team"),),
        # Missing chronology inputs cannot be made safe by warning mode.
        strict=True,
    )

    output = df.copy()
    output[time_col] = pd.to_datetime(output[time_col], utc=True, errors="coerce")
    enforce_contract(
        ["CFB advanced game stats contains invalid kickoff timestamps"]
        if output[time_col].isna().any()
        else [],
        strict=True,
    )
    output = output.sort_values(["team", time_col, "game_id"], kind="stable").reset_index(drop=True)

    previous_kickoff = output.groupby("team", sort=False)[time_col].shift()
    invalid_history = previous_kickoff.notna() & (previous_kickoff >= output[time_col])
    enforce_contract(
        [
            "CFB advanced game stats cannot establish a strictly earlier prior game "
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

        output[col_shifted] = output.groupby("team", sort=False)[col].shift()
        output[col_ewma] = output.groupby("team", sort=False)[col_shifted].transform(
            lambda x: x.ewm(min_periods=1, span=10).mean()
        )
        output[col_ewma_dynamic_window] = output.groupby("team", sort=False).apply(
            lambda x: dynamic_window_ewma(x, col_shifted)
        ).reset_index(level=0, drop=True)

    return output


def get_implied_totals(row):
    home_points = (row["total_line"] / 2) + (row["spread_line"] / 2)
    away_points = (row["total_line"] / 2) - (row["spread_line"] / 2)
    return pd.Series([home_points, away_points])
