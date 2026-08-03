from typing import Dict, List

import pandas as pd


def summarize_eval_results(results: pd.DataFrame) -> Dict[str, float]:
    return {
        "total_win_pct": float(100.0 * results["total_win"].mean()),
        "spread_win_pct": float(100.0 * results["spread_win"].mean()),
    }


def get_result_stats(df, verbose: bool = False):
    results = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    spread_wins = results["spread_win"] == 1
    spread_losses = results["spread_win"] == 0
    spread_pushes = results["spread_win"].isnull()
    total_wins = results["total_win"] == 1
    total_losses = results["total_win"] == 0
    total_pushes = results["total_win"].isnull()

    spread_games = int(spread_wins.sum() + spread_losses.sum())
    total_games = int(total_wins.sum() + total_losses.sum())
    spread_lock_df = results[results["spread_lock"] == 1]
    total_lock_df = results[results["total_lock"] == 1]
    spread_lock_wins = spread_lock_df["spread_win"] == 1
    spread_lock_losses = spread_lock_df["spread_win"] == 0
    spread_lock_pushes = spread_lock_df["spread_win"].isnull()
    total_lock_wins = total_lock_df["total_win"] == 1
    total_lock_losses = total_lock_df["total_win"] == 0
    total_lock_pushes = total_lock_df["total_win"].isnull()
    spread_lock_games = int(spread_lock_wins.sum() + spread_lock_losses.sum())
    total_lock_games = int(total_lock_wins.sum() + total_lock_losses.sum())

    data = {
        "predicted_games": len(results),
        "spread_wins": int(spread_wins.sum()),
        "spread_losses": int(spread_losses.sum()),
        "spread_pushes": int(spread_pushes.sum()),
        "spread_win_pct": 100 * spread_wins.sum() / spread_games if spread_games else 0,
        "spread_lock_predictions": int(len(spread_lock_df)),
        "spread_lock_wins": int(spread_lock_wins.sum()),
        "spread_lock_losses": int(spread_lock_losses.sum()),
        "spread_lock_pushes": int(spread_lock_pushes.sum()),
        "spread_lock_win_pct": 100 * spread_lock_wins.sum() / spread_lock_games if spread_lock_games else 0,
        "total_wins": int(total_wins.sum()),
        "total_losses": int(total_losses.sum()),
        "total_pushes": int(total_pushes.sum()),
        "total_win_pct": 100 * total_wins.sum() / total_games if total_games else 0,
        "total_lock_predictions": int(len(total_lock_df)),
        "total_lock_wins": int(total_lock_wins.sum()),
        "total_lock_losses": int(total_lock_losses.sum()),
        "total_lock_pushes": int(total_lock_pushes.sum()),
        "total_lock_win_pct": 100 * total_lock_wins.sum() / total_lock_games if total_lock_games else 0,
    }
    if verbose:
        print(f"Spread Win Percentage: {data['spread_win_pct']:.2f}%")
        print(f"Total Win Percentage: {data['total_win_pct']:.2f}%")
    return data


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
