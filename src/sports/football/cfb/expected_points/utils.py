import pandas as pd

from src.sports.football.kickoff import EASTERN_TIME_ZONE, KICKOFF_FORMAT


def cfbd_kickoffs_to_eastern_strings(values: pd.Series) -> pd.Series:
    """Convert CFBD's UTC timestamps to the football Eastern-time contract."""
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return parsed.dt.tz_convert(EASTERN_TIME_ZONE).dt.strftime(KICKOFF_FORMAT)


def prepare_cfb_expected_points_df(df):
    output = df.copy()

    rename_map = {
        "home_points": "home_score",
        "away_points": "away_score",
        "spread": "spread_line",
        "over_under": "total_line",
        "home_moneyline": "moneyline_home",
        "away_moneyline": "moneyline_away",
    }
    output = output.rename(columns={k: v for k, v in rename_map.items() if k in output.columns})

    output["game_id"] = output["game_id"].astype(str)
    output["year_week"] = output["season"].astype(str) + "_" + output["week"].astype(str)
    output["date_time"] = cfbd_kickoffs_to_eastern_strings(output["start_date"])

    return output


__all__ = ["cfbd_kickoffs_to_eastern_strings", "prepare_cfb_expected_points_df"]
