"""Probability, expected-value, and combined weekly Lock policy utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LockPolicy:
    american_odds: int = -110
    minimum_resolved_win_probability: float = 0.55
    max_locks_per_week: int = 5

    def __post_init__(self) -> None:
        if self.american_odds == 0:
            raise ValueError("American odds cannot be zero")
        if not 0.5 <= self.minimum_resolved_win_probability <= 1.0:
            raise ValueError("Minimum resolved win probability must be between .5 and 1")
        if self.max_locks_per_week < 0:
            raise ValueError("Weekly Lock cap cannot be negative")


def win_profit(american_odds: int) -> float:
    return 100.0 / abs(american_odds) if american_odds < 0 else american_odds / 100.0


def estimate_push_probability(
    history: pd.DataFrame,
    target: pd.DataFrame,
    *,
    prior_probability: float = 0.03,
    prior_strength: float = 50.0,
) -> np.ndarray:
    """Estimate pushes without leaking current outcomes.

    Half-point markets cannot push. Integer spreads use stable key-number groups;
    integer totals pool together because individual total values are too sparse.
    """
    output = np.zeros(len(target), dtype=float)
    line = pd.to_numeric(target["line"], errors="coerce").to_numpy(float)
    integer = np.isfinite(line) & np.isclose(line, np.round(line))
    if not integer.any() or history.empty:
        return output
    historic = history.copy()
    historic_line = pd.to_numeric(historic["line"], errors="coerce")
    historic = historic.loc[historic_line.notna() & np.isclose(historic_line, np.round(historic_line))]
    if historic.empty:
        output[integer] = prior_probability
        return output
    historic_push = historic["outcome"].eq("push").astype(float)
    global_rate = float((historic_push.sum() + prior_probability * prior_strength) / (len(historic) + prior_strength))
    for position in np.flatnonzero(integer):
        market = str(target.iloc[position]["market"])
        selected = historic.loc[historic["market"].eq(market)]
        if market == "spread":
            value = abs(float(line[position]))
            group = value if value in {3.0, 7.0, 10.0, 14.0} else -1.0
            selected_line = pd.to_numeric(selected["line"], errors="coerce").abs()
            selected = selected.loc[selected_line.eq(group) if group >= 0 else ~selected_line.isin([3, 7, 10, 14])]
        pushes = selected["outcome"].eq("push").sum()
        output[position] = float((pushes + global_rate * prior_strength) / (len(selected) + prior_strength))
    return np.clip(output, 0.0, 0.25)


def add_outcome_probabilities(
    decisions: pd.DataFrame,
    resolved_win_probability: np.ndarray,
    push_probability: np.ndarray,
    *,
    policy: LockPolicy,
) -> pd.DataFrame:
    output = decisions.copy()
    q = np.clip(np.asarray(resolved_win_probability, dtype=float), 0.0, 1.0)
    push = np.clip(np.asarray(push_probability, dtype=float), 0.0, 1.0)
    output["resolved_win_probability"] = q
    output["p_push"] = push
    output["p_win"] = (1.0 - push) * q
    output["p_loss"] = (1.0 - push) * (1.0 - q)
    output["expected_units"] = win_profit(policy.american_odds) * output["p_win"] - output["p_loss"]
    return output


def select_weekly_locks(
    decisions: pd.DataFrame,
    *,
    policy: LockPolicy,
    preserved: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply one deterministic spread+total cap to each league week."""
    output = decisions.copy()
    required = {
        "season", "week", "game_id", "market", "kickoff", "edge",
        "resolved_win_probability", "expected_units", "market_supported", "locks_enabled",
    }
    missing = sorted(required - set(output.columns))
    if missing:
        raise ValueError(f"Lock decisions are missing policy columns: {missing}")
    output["lock"] = 0
    output["lock_rank"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output["lock_reason"] = "below_probability_or_ev"
    output.loc[~output["market_supported"].fillna(False), "lock_reason"] = "market_unsupported"
    output.loc[~output["locks_enabled"].fillna(False), "lock_reason"] = "model_not_ready"
    eligible = (
        output["market_supported"].fillna(False)
        & output["locks_enabled"].fillna(False)
        & output["resolved_win_probability"].ge(policy.minimum_resolved_win_probability)
        & output["expected_units"].gt(0.0)
    )
    output.loc[eligible, "lock_reason"] = "eligible"
    preserved_keys: set[tuple[int, object, str, str]] = set()
    if preserved is not None and not preserved.empty:
        preserved_keys = {
            (int(row.season), row.week, str(row.game_id), str(row.market))
            for row in preserved.itertuples(index=False)
            if bool(row.lock)
        }
    output["_preserved"] = [
        (int(row.season), row.week, str(row.game_id), str(row.market)) in preserved_keys
        for row in output.itertuples(index=False)
    ]
    for (_season, _week), indices in output.groupby(["season", "week"], sort=False).groups.items():
        group = output.loc[indices]
        frozen = group.loc[group["_preserved"]]
        output.loc[frozen.index, ["lock", "lock_reason"]] = [1, "preserved"]
        capacity = policy.max_locks_per_week - len(frozen)
        if capacity <= 0:
            reason = "legacy_cap_overflow" if len(frozen) > policy.max_locks_per_week else "weekly_cap"
            output.loc[group.index.intersection(output.index[eligible]), "lock_reason"] = reason
            continue
        candidates = group.loc[eligible.loc[group.index] & ~group["_preserved"]].copy()
        candidates["_market_order"] = candidates["market"].map({"spread": 0, "total": 1}).fillna(2)
        candidates = candidates.sort_values(
            ["expected_units", "resolved_win_probability", "edge", "kickoff", "game_id", "_market_order"],
            ascending=[False, False, False, True, True, True],
            kind="stable",
        )
        ranked = candidates.index[:capacity]
        output.loc[candidates.index, "lock_rank"] = np.arange(1, len(candidates) + 1)
        output.loc[ranked, ["lock", "lock_reason"]] = [1, "selected"]
        output.loc[candidates.index[capacity:], "lock_reason"] = "weekly_cap"
    return output.drop(columns="_preserved")


def realized_units(outcome: pd.Series, *, american_odds: int = -110) -> pd.Series:
    return outcome.map({"win": win_profit(american_odds), "loss": -1.0, "push": 0.0}).astype(float)


def lock_profit_metrics(decisions: pd.DataFrame, *, american_odds: int = -110) -> dict[str, float | int | None]:
    selected = decisions.loc[decisions["lock"].eq(1)].copy()
    resolved = selected.loc[selected["outcome"].isin(["win", "loss", "push"])].copy()
    wins = int(resolved["outcome"].eq("win").sum())
    losses = int(resolved["outcome"].eq("loss").sum())
    pushes = int(resolved["outcome"].eq("push").sum())
    units = realized_units(resolved["outcome"], american_odds=american_odds) if len(resolved) else pd.Series(dtype=float)
    net = float(units.sum()) if len(units) else 0.0
    decisions_count = wins + losses + pushes
    non_push = wins + losses
    weekly = (
        resolved.assign(_units=units.to_numpy())
        .groupby(["season", "week"], sort=True)["_units"].sum()
        if len(resolved) else pd.Series(dtype=float)
    )
    cumulative = weekly.cumsum()
    drawdown = cumulative.cummax() - cumulative
    return {
        "locks": decisions_count,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": wins / non_push if non_push else None,
        "net_units": net,
        "roi": net / decisions_count if decisions_count else None,
        "max_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "mean_probability": float(resolved["resolved_win_probability"].mean()) if len(resolved) else None,
        "calibration_gap": (
            float(resolved.loc[resolved["outcome"].ne("push"), "resolved_win_probability"].mean()) - wins / non_push
            if non_push else None
        ),
    }


__all__ = [
    "LockPolicy",
    "add_outcome_probabilities",
    "estimate_push_probability",
    "lock_profit_metrics",
    "realized_units",
    "select_weekly_locks",
    "win_profit",
]
