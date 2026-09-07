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
    if thresholds.max_combined_plays is not None:
        return _determine_combined_plays(df, thresholds, dont_update=dont_update)
    dont_update = dont_update or []
    output = df.copy()
    spread_supported = output.get(
        "spread_market_supported",
        pd.Series(True, index=output.index),
    ).fillna(False).astype(bool)
    total_supported = output.get(
        "total_market_supported",
        pd.Series(True, index=output.index),
    ).fillna(False).astype(bool)
    output["is_top_n_spread"] = (
        output["spread_win_prob"].where(spread_supported).rank(method="first", ascending=False)
        <= thresholds.max_spreads_plays
    )
    output["is_top_n_total"] = (
        output["total_win_prob"].where(total_supported).rank(method="first", ascending=False)
        <= thresholds.max_total_plays
    )

    output["new_spread_lock"] = (
        output["is_top_n_spread"]
        & spread_supported
        & ((output["spread_pred"] - output["spread_line"]).abs() >= thresholds.min_spread_diff)
        & (output["spread_win_prob"] > thresholds.min_spread_win_prob)
    ).astype(int)

    output["new_total_lock"] = (
        output["is_top_n_total"]
        & total_supported
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


def _determine_combined_plays(
    df: pd.DataFrame,
    thresholds: PlayThresholds,
    *,
    dont_update=None,
) -> pd.DataFrame:
    """Adapt production picks to the same weekly selector used by replay."""
    from .lock_policy import LockPolicy, add_outcome_probabilities, select_weekly_locks
    from src.sports.football.kickoff import parse_eastern_kickoffs

    output = df.copy()
    frozen_ids = {str(value) for value in (dont_update or [])}
    policy = LockPolicy(
        american_odds=thresholds.american_odds,
        minimum_resolved_win_probability=0.5,
        max_locks_per_week=thresholds.max_combined_plays,
        max_spreads_per_week=thresholds.max_spreads_plays,
        max_totals_per_week=thresholds.max_total_plays,
        extra_lock_min_probability=thresholds.extra_lock_min_probability,
    )
    decisions = []
    for market in ("spread", "total"):
        part = pd.DataFrame(index=output.index)
        part["row_position"] = range(len(output))
        part["season"] = output.get("season", 0)
        part["week"] = output.get("week", 0)
        part["game_id"] = output.get("game_id", pd.Series(output.index, index=output.index)).astype(str)
        part["market"] = market
        part["kickoff"] = (
            parse_eastern_kickoffs(output["date_time"])
            if "date_time" in output else pd.Timestamp("1970-01-01", tz="UTC")
        )
        part["edge"] = (output[f"{market}_pred"] - output[f"{market}_line"]).abs()
        probability = pd.to_numeric(output[f"{market}_win_prob"], errors="coerce") / 100.0
        push = pd.to_numeric(
            output.get(f"{market}_push_prob", pd.Series(0.0, index=output.index)),
            errors="coerce",
        ).fillna(0.0)
        part["market_supported"] = output.get(
            f"{market}_market_supported", pd.Series(True, index=output.index)
        ).fillna(False).astype(bool)
        part["locks_enabled"] = output.get(
            f"{market}_locks_enabled", pd.Series(True, index=output.index)
        ).fillna(False).astype(bool)
        part["lock"] = pd.to_numeric(
            output.get(f"{market}_lock", pd.Series(0, index=output.index)), errors="coerce"
        ).fillna(0).astype(int)
        part["policy_eligible"] = (
            probability.ge(getattr(thresholds, f"min_{market}_win_prob") / 100.0)
            & probability.le(1.0)
            & part["edge"].ge(getattr(thresholds, f"min_{market}_diff"))
            & output.get(f"{market}_play", pd.Series(True, index=output.index)).notna()
            & ~part["game_id"].isin(frozen_ids)
        )
        part = add_outcome_probabilities(part, probability.to_numpy(), push.to_numpy(), policy=policy)
        decisions.append(part)
    combined = pd.concat(decisions, ignore_index=True)
    preserved = combined.loc[combined["game_id"].isin(frozen_ids)]
    selected = select_weekly_locks(combined, policy=policy, preserved=preserved)
    for market in ("spread", "total"):
        part = selected.loc[selected["market"].eq(market)].sort_values("row_position")
        frozen = output.get("game_id", pd.Series(output.index, index=output.index)).astype(str).isin(frozen_ids)
        new_locks = part["lock"].to_numpy()
        output[f"new_{market}_lock"] = pd.Series(new_locks, index=output.index).where(~frozen, 0)
        existing = output.get(f"{market}_lock", pd.Series(0, index=output.index))
        output[f"{market}_lock"] = pd.Series(new_locks, index=output.index).where(~frozen, existing).fillna(0).astype(int)
        output[f"{market}_expected_units"] = part["expected_units"].to_numpy()
        output[f"{market}_lock_rank"] = pd.array(part["lock_rank"], dtype="Int64")
    return output


def win_probability(df: pd.DataFrame, classifier, features):
    return classifier.predict_proba(df[features])[:, 1] * 100
