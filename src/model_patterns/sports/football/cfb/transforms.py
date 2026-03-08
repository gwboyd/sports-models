import numpy as np

from src.model_patterns.sports.football.transforms.common import (
    dynamic_window_ewma,
    get_averaged_game_stats,
)


def prepare_cfb_expected_points_df(df):
    output = df.copy()

    rename_map = {
        "home_points": "home_score",
        "away_points": "away_score",
        "spread": "spread_line",
        "over_under": "total_line",
        "home_moneyline": "moneyline_home",
        "away_moneyline": "moneyline_away",
        "conference_game": "div_game",
    }
    output = output.rename(columns={k: v for k, v in rename_map.items() if k in output.columns})

    defaults = {
        "spread_odds_home": np.nan,
        "spread_odds_away": np.nan,
        "over_odds": np.nan,
        "rest_home": np.nan,
        "rest_away": np.nan,
        "pred_team": "undefined",
    }

    for key, value in defaults.items():
        if key not in output.columns:
            output[key] = value

    if "div_game" in output.columns:
        output["div_game"] = output["div_game"].fillna(False).astype(bool)

    return output


__all__ = ["dynamic_window_ewma", "get_averaged_game_stats", "prepare_cfb_expected_points_df"]
