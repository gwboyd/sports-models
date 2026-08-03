import pandas as pd


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
    kickoff = pd.to_datetime(output["start_date"], utc=True, errors="coerce")
    output["date_time"] = kickoff.dt.strftime("%Y-%m-%d-%H:%M")

    return output


__all__ = ["prepare_cfb_expected_points_df"]
