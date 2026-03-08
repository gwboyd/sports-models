import numpy as np
import pandas as pd


def dynamic_window_ewma(x: pd.DataFrame, col_shifted: str) -> pd.Series:
    values = np.zeros(len(x))
    for i, (_, row) in enumerate(x.iterrows()):
        epa_col = x[col_shifted][: i + 1]
        span = row["week"] if row["week"] > 10 else 10
        values[i] = epa_col.ewm(min_periods=1, span=span).mean().iloc[-1]
    return pd.Series(values, index=x.index)


def get_averaged_game_stats(df: pd.DataFrame, columns) -> pd.DataFrame:
    output = df.copy()
    for col in columns:
        col_shifted = f"{col}_shifted"
        col_ewma = f"{col}_ewma"
        col_ewma_dynamic_window = f"{col}_ewma_dynamic_window"

        output[col_shifted] = output.groupby("team")[col].shift()
        output[col_ewma] = output.groupby("team")[col_shifted].transform(
            lambda x: x.ewm(min_periods=1, span=10).mean()
        )
        output[col_ewma_dynamic_window] = output.groupby("team").apply(
            lambda x: dynamic_window_ewma(x, col_shifted)
        ).reset_index(level=0, drop=True)

    return output


def get_implied_totals(row):
    home_points = (row["total_line"] / 2) + (row["spread_line"] / 2)
    away_points = (row["total_line"] / 2) - (row["spread_line"] / 2)
    return pd.Series([home_points, away_points])
