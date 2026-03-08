from typing import Dict, List

import pandas as pd


def summarize_eval_results(results: pd.DataFrame) -> Dict[str, float]:
    return {
        "total_win_pct": float(100.0 * results["total_win"].mean()),
        "spread_win_pct": float(100.0 * results["spread_win"].mean()),
    }


def get_feature_importance_df(score_model, features: List[str], estimator_step: str = "lgbmregressor") -> pd.DataFrame:
    if not hasattr(score_model, "named_steps"):
        return pd.DataFrame(columns=["feature_name", "feature_importance"])
    estimator = score_model.named_steps.get(estimator_step)
    if estimator is None or not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=["feature_name", "feature_importance"])

    importances = estimator.feature_importances_
    return pd.DataFrame(
        zip(features, importances), columns=["feature_name", "feature_importance"]
    ).sort_values(by="feature_importance", ascending=True)


def print_plays(df: pd.DataFrame) -> None:
    print("Spread plays:")
    spread_plays = df[df["spread_lock"] == 1].sort_values(by="spread_win_prob", ascending=False)
    for _, row in spread_plays.iterrows():
        mult = -1 if row["spread_play"] == row["away_team"] else 1
        pref = "+" if row["spread_line"] * mult > 0 else ""
        pref2 = "+" if row["spread_pred"] * mult > 0 else ""
        print(
            f"{row['home_team']}/{row['away_team']}: {row['spread_play']} {pref}{row['spread_line']*mult} "
            f"(model {row['spread_play']} {pref2}{(row['spread_pred']*mult):.2f}, {row['spread_win_prob']:.2f}% win probability)"
        )

    print("\nTotal plays:")
    total_plays = df[df["total_lock"] == 1].sort_values(by="total_win_prob", ascending=False)
    for _, row in total_plays.iterrows():
        print(
            f"{row['home_team']}/{row['away_team']}: {row['total_play']} {row['total_line']} "
            f"(model {row['total_pred']:.2f}, {row['total_win_prob']:.2f}% win probability)"
        )
