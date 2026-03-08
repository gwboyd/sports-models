import pandas as pd

from .types import PlayThresholds


def scores_to_bets(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    output["spread_pred"] = output["away_score_pred"] - output["home_score_pred"]
    output["total_pred"] = output["home_score_pred"] + output["away_score_pred"]
    output["spread_play"] = output.apply(
        lambda row: row["home_team"] if row["spread_pred"] < row["spread_line"] else row["away_team"], axis=1
    )
    output["total_play"] = output.apply(
        lambda row: "under" if row["total_pred"] < row["total_line"] else ("over" if row["total_pred"] > row["total_line"] else None),
        axis=1,
    )
    output["spread_diff"] = (output["spread_line"] - output["spread_pred"]).abs()
    output["total_diff"] = (output["total_line"] - output["total_pred"]).abs()
    return output


def calculate_wins(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["true_spread"] = output["away_score"] - output["home_score"]
    output["correct_spread_play"] = output.apply(
        lambda row: row["home_team"]
        if row["true_spread"] < row["spread_line"]
        else (row["away_team"] if row["true_spread"] > row["spread_line"] else None),
        axis=1,
    )
    output["spread_win"] = output.apply(
        lambda row: None if pd.isnull(row["correct_spread_play"]) else (1 if row["spread_play"] == row["correct_spread_play"] else 0),
        axis=1,
    )
    output["true_total"] = output["away_score"] + output["home_score"]
    output["correct_total_play"] = output.apply(
        lambda row: "under"
        if row["true_total"] < row["total_line"]
        else ("over" if row["true_total"] > row["total_line"] else None),
        axis=1,
    )
    output["total_win"] = output.apply(
        lambda row: None if pd.isnull(row["correct_total_play"]) else (1 if row["total_play"] == row["correct_total_play"] else 0),
        axis=1,
    )
    return output


def determine_plays(df: pd.DataFrame, thresholds: PlayThresholds, dont_update=None) -> pd.DataFrame:
    dont_update = dont_update or []
    output = df.copy()
    output["is_top_n_spread"] = output["spread_win_prob"].rank(method="first", ascending=False) <= thresholds.max_spreads_plays
    output["is_top_n_total"] = output["total_win_prob"].rank(method="first", ascending=False) <= thresholds.max_total_plays

    output["new_spread_lock"] = (
        output["is_top_n_spread"]
        & ((output["spread_pred"] - output["spread_line"]).abs() >= thresholds.min_spread_diff)
        & (output["spread_win_prob"] > thresholds.min_spread_win_prob)
    ).astype(int)

    output["new_total_lock"] = (
        output["is_top_n_total"]
        & ((output["total_pred"] - output["total_line"]).abs() >= thresholds.min_total_diff)
        & (output["total_win_prob"] > thresholds.min_total_win_prob)
    ).astype(int)

    if "spread_lock" in output.columns:
        output["spread_lock"] = output.apply(
            lambda row: row["spread_lock"] if row.get("game_id") in dont_update else row["new_spread_lock"], axis=1
        )
    else:
        output["spread_lock"] = output["new_spread_lock"]

    if "total_lock" in output.columns:
        output["total_lock"] = output.apply(
            lambda row: row["total_lock"] if row.get("game_id") in dont_update else row["new_total_lock"], axis=1
        )
    else:
        output["total_lock"] = output["new_total_lock"]

    output.drop(["is_top_n_spread", "is_top_n_total"], axis=1, inplace=True)
    return output


def win_probability(df: pd.DataFrame, classifier, features):
    return classifier.predict_proba(df[features])[:, 1] * 100
