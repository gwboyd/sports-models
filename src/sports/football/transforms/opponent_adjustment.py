"""Leakage-safe opponent adjustment for football team-game metrics.

The public entry point deliberately works with one offensive observation per
game.  A defense's observed value is the opposing offense's observation, which
prevents mirrored provider rows from being counted twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.sports.data_validation import enforce_contract, validate_frame

@dataclass(frozen=True)
class OpponentMetricSpec:
    """Names and smoothing policy for one offense-versus-defense metric."""

    key: str
    offense_output: str
    defense_output: str
    smoothing: Literal["static", "dynamic"]
    span: int = 10


@dataclass(frozen=True)
class OpponentAdjustmentConfig:
    """Versioned assumptions used while estimating opponent quality.

    ``season_carryover`` discounts observations from each prior season.  It is
    intentionally an exposed lever: it is not a claim that a fixed value is
    optimal for NFL or CFB.
    """

    ridge_alpha: float = 5.0
    season_carryover: float = 0.5
    fcs_policy: Literal["partial_pool", "pooled"] = "partial_pool"
    rating_snapshot: Literal["week", "kickoff"] = "week"
    protocol_version: str = "1"

    def __post_init__(self) -> None:
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be finite and positive")
        if not np.isfinite(self.season_carryover) or not 0 <= self.season_carryover <= 1:
            raise ValueError("season_carryover must be between zero and one")
        if self.rating_snapshot not in {"week", "kickoff"}:
            raise ValueError("rating_snapshot must be 'week' or 'kickoff'")


DEFAULT_OPPONENT_ADJUSTMENT = OpponentAdjustmentConfig()


def build_opponent_adjusted_team_metrics(
    schedule: pd.DataFrame,
    observations: pd.DataFrame,
    specs: Sequence[OpponentMetricSpec],
    *,
    config: OpponentAdjustmentConfig | Mapping[str, object] | None = DEFAULT_OPPONENT_ADJUSTMENT,
    strict: bool = True,
) -> pd.DataFrame:
    """Return team-game rows with adjusted values under existing feature names.

    ``observations`` has one row per offensive team/game/metric and contains
    ``game_id``, ``team``, ``metric``, and ``value``. By default, ratings are
    fit once at the opening kickoff of each game week, so no same-week outcome
    can enter any pregame feature. ``rating_snapshot='kickoff'`` is available
    for an exact, more expensive intraday simulation. Earlier games are then
    reassessed with the applicable snapshot before the original EWMA is
    calculated.
    """
    config = resolve_opponent_adjustment_config(config)
    _validate_inputs(schedule, observations, specs, strict=strict)
    targets = _team_timeline(schedule)
    observed = _attach_opponents(observations, schedule)
    output = targets.copy()
    output["start_date"] = pd.to_datetime(output["start_date"], utc=True, errors="coerce")
    specs_by_key = {spec.key: spec for spec in specs}

    for spec in specs:
        metric = observed.loc[observed["metric"] == spec.key].copy()
        adjusted = _adjust_metric_at_targets(targets, metric, spec, config)
        for column in adjusted.columns:
            if column not in {"game_id", "season", "week", "team"}:
                output[column] = adjusted[column].to_numpy()

    # A missing metric is legitimate for a scheduled game.  Chronology fields
    # are not; they are validated above and converted once here.
    return output.drop(columns=["start_date"], errors="ignore")


def resolve_opponent_adjustment_config(
    config: OpponentAdjustmentConfig | Mapping[str, object] | None,
) -> OpponentAdjustmentConfig:
    """Accept a notebook/Papermill mapping while keeping the core typed."""
    if config is None:
        return DEFAULT_OPPONENT_ADJUSTMENT
    if isinstance(config, OpponentAdjustmentConfig):
        return config
    if isinstance(config, Mapping):
        return OpponentAdjustmentConfig(**config)
    raise TypeError("opponent adjustment config must be a mapping or OpponentAdjustmentConfig")


def _validate_inputs(
    schedule: pd.DataFrame,
    observations: pd.DataFrame,
    specs: Sequence[OpponentMetricSpec],
    *,
    strict: bool,
) -> None:
    validate_frame(
        schedule,
        label="opponent adjustment schedule",
        required_columns=("game_id", "season", "week", "home_team", "away_team", "start_date"),
        non_null_columns=("game_id", "season", "week", "home_team", "away_team", "start_date"),
        unique_keys=(("game_id",),),
        strict=strict,
    )
    validate_frame(
        observations,
        label="opponent adjustment observations",
        required_columns=("game_id", "team", "metric", "value"),
        non_null_columns=("game_id", "team", "metric"),
        unique_keys=(("game_id", "team", "metric"),),
        strict=strict,
    )
    configured = {spec.key for spec in specs}
    enforce_contract(
        ["opponent adjustment needs at least one metric specification"] if not configured else [],
        strict=strict,
    )
    unknown = sorted(set(observations["metric"].dropna()) - configured)
    enforce_contract(
        [f"opponent adjustment has unregistered metrics: {', '.join(unknown)}"] if unknown else [],
        strict=strict,
    )


def _team_timeline(schedule: pd.DataFrame) -> pd.DataFrame:
    keys = ["game_id", "season", "week", "start_date"]
    extra = [column for column in ("home_classification", "away_classification", "neutral_site") if column in schedule]
    home = schedule[keys + extra + ["home_team", "away_team"]].copy()
    home = home.rename(columns={"home_team": "team", "away_team": "opponent", "home_classification": "group", "away_classification": "opponent_group"})
    away = schedule[keys + extra + ["away_team", "home_team"]].copy()
    away = away.rename(columns={"away_team": "team", "home_team": "opponent", "away_classification": "group", "home_classification": "opponent_group"})
    timeline = pd.concat([home, away], ignore_index=True, sort=False)
    timeline["group"] = timeline.get("group", pd.Series("nfl", index=timeline.index)).fillna("unknown").str.lower()
    timeline["opponent_group"] = timeline.get("opponent_group", pd.Series("nfl", index=timeline.index)).fillna("unknown").str.lower()
    neutral = timeline.get("neutral_site", pd.Series(False, index=timeline.index)).fillna(False).astype(bool)
    timeline["venue_sign"] = np.where(neutral, 0.0, np.where(timeline.index < len(home), 1.0, -1.0))
    return timeline.sort_values(["start_date", "game_id", "team"], kind="stable").reset_index(drop=True)


def _attach_opponents(observations: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    timeline = _team_timeline(schedule)
    columns = ["game_id", "team", "opponent", "season", "week", "start_date", "group", "opponent_group", "venue_sign"]
    # Schedule is authoritative for game chronology and opponent identity.  The
    # adapters may retain these convenience columns, so discard them before the
    # merge rather than creating ambiguous ``_x``/``_y`` columns.
    source = observations.drop(
        columns=[column for column in ("season", "week", "start_date") if column in observations],
    )
    output = source.merge(timeline[columns], on=["game_id", "team"], how="left", validate="many_to_one")
    output["start_date"] = pd.to_datetime(output["start_date"], utc=True, errors="coerce")
    return output.dropna(subset=["value", "start_date", "opponent"])


def _adjust_metric_at_targets(
    targets: pd.DataFrame,
    observations: pd.DataFrame,
    spec: OpponentMetricSpec,
    config: OpponentAdjustmentConfig,
) -> pd.DataFrame:
    output = targets[["game_id", "season", "week", "team"]].copy()
    values: dict[str, list[float]] = {name: [] for name in _metric_columns(spec)}
    cache: dict[pd.Timestamp, tuple[dict[str, float], dict[str, float]]] = {}
    observations = observations.sort_values(["start_date", "game_id"], kind="stable").reset_index(drop=True)
    week_openers = (
        targets.groupby(["season", "week"], sort=False)["start_date"].min().to_dict()
    )
    offense_histories = {
        team: rows for team, rows in observations.groupby("team", sort=False)
    }
    defense_histories = {
        team: rows for team, rows in observations.groupby("opponent", sort=False)
    }
    team_histories = {
        team: pd.concat(
            [offense_histories.get(team, observations.iloc[0:0]), defense_histories.get(team, observations.iloc[0:0])],
            ignore_index=True,
            sort=False,
        )
        for team in set(offense_histories) | set(defense_histories)
    }

    for target in targets.itertuples(index=False):
        cutoff = (
            pd.Timestamp(week_openers[(target.season, target.week)])
            if config.rating_snapshot == "week"
            else pd.Timestamp(target.start_date)
        )
        if cutoff not in cache:
            history = observations.loc[observations["start_date"] < cutoff]
            cache[cutoff] = _fit_snapshot(history, int(target.season), config)
        offense_ratings, defense_ratings = cache[cutoff]
        team_history = team_histories.get(target.team, observations.iloc[0:0])
        team_history = team_history.loc[team_history["start_date"] < cutoff]
        rendered = _render_target_history(
            team_history, target, offense_ratings, defense_ratings, spec
        )
        for name, value in rendered.items():
            values[name].append(value)

    for name, value in values.items():
        output[name] = value
    return output


def _fit_snapshot(
    history: pd.DataFrame,
    target_season: int,
    config: OpponentAdjustmentConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    if history.empty:
        return {}, {}
    frame = history.copy()
    if config.fcs_policy == "pooled":
        frame["offense_key"] = np.where(frame["group"] == "fcs", "fcs", frame["team"])
        frame["defense_key"] = np.where(frame["opponent_group"] == "fcs", "fcs", frame["opponent"])
    else:
        frame["offense_key"] = frame["team"]
        frame["defense_key"] = frame["opponent"]

    design = pd.concat([
        pd.get_dummies(frame["offense_key"], prefix="off", dtype=float),
        pd.get_dummies(frame["defense_key"], prefix="def", dtype=float),
        pd.get_dummies(frame["group"], prefix="off_group", dtype=float),
        pd.get_dummies(frame["opponent_group"], prefix="def_group", dtype=float),
        frame[["venue_sign"]].astype(float),
    ], axis=1)
    years_back = (target_season - pd.to_numeric(frame["season"], errors="coerce")).clip(lower=0)
    weights = np.power(config.season_carryover, years_back.to_numpy(dtype=float))
    # The design has only team, group, and venue columns. Dense Cholesky avoids
    # iterative-solver startup cost across the many weekly historical fits.
    model = Ridge(alpha=config.ridge_alpha, fit_intercept=True, solver="cholesky")
    model.fit(design, pd.to_numeric(frame["value"], errors="coerce"), sample_weight=weights)
    coefficients = dict(zip(design.columns, model.coef_))

    # Return ratings keyed by the real team name, even when FCS teams were
    # pooled into a shared entity.  The group effects are part of a team's
    # rating: otherwise partial pooling would estimate FBS/FCS separation but
    # discard it before applying the opponent correction.
    offense = {}
    for row in frame[["team", "group", "offense_key"]].drop_duplicates().itertuples(index=False):
        offense[row.team] = (
            coefficients.get(f"off_{row.offense_key}", 0.0)
            + coefficients.get(f"off_group_{row.group}", 0.0)
        )
    defense = {}
    for row in frame[["opponent", "opponent_group", "defense_key"]].drop_duplicates().itertuples(index=False):
        defense[row.opponent] = (
            coefficients.get(f"def_{row.defense_key}", 0.0)
            + coefficients.get(f"def_group_{row.opponent_group}", 0.0)
        )
    return offense, defense


def _render_target_history(
    history: pd.DataFrame,
    target: object,
    offense_ratings: dict[str, float],
    defense_ratings: dict[str, float],
    spec: OpponentMetricSpec,
) -> dict[str, float]:
    offense_rows = history.loc[history["team"] == target.team].copy()
    defense_rows = history.loc[history["opponent"] == target.team].copy()
    offense_rows["adjusted"] = offense_rows["value"] - offense_rows["opponent"].map(defense_ratings).fillna(0.0)
    defense_rows["adjusted"] = defense_rows["value"] - defense_rows["team"].map(offense_ratings).fillna(0.0)
    return {
        **_smooth(offense_rows, target, spec, spec.offense_output),
        **_smooth(defense_rows, target, spec, spec.defense_output),
    }


def _smooth(history: pd.DataFrame, target: object, spec: OpponentMetricSpec, output: str) -> dict[str, float]:
    raw = np.nan
    shifted = np.nan
    ewma = np.nan
    dynamic = np.nan
    if not history.empty:
        ordered = history.sort_values(["start_date", "game_id"], kind="stable")
        values = ordered["adjusted"].astype(float)
        raw = float(values.iloc[-1])
        shifted = raw
        ewma = float(values.ewm(min_periods=1, span=spec.span).mean().iloc[-1])
        if spec.smoothing == "dynamic":
            # ``dynamic_period_ewma`` uses the current row's week to choose
            # the final span. Only that final value is needed for a target,
            # so calculating the earlier intermediate points is redundant.
            span = max(spec.span, int(ordered["week"].iloc[-1]))
            dynamic = float(values.ewm(min_periods=1, span=span).mean().iloc[-1])
    result = {
        output: raw,
        f"{output}_shifted": shifted,
        f"{output}_ewma": ewma,
    }
    if spec.smoothing == "dynamic":
        result[f"{output}_ewma_dynamic_window"] = dynamic
    return result


def _metric_columns(spec: OpponentMetricSpec) -> tuple[str, ...]:
    names: list[str] = []
    for output in (spec.offense_output, spec.defense_output):
        names.extend((output, f"{output}_shifted", f"{output}_ewma"))
        if spec.smoothing == "dynamic":
            names.append(f"{output}_ewma_dynamic_window")
    return tuple(names)


__all__ = [
    "DEFAULT_OPPONENT_ADJUSTMENT",
    "OpponentAdjustmentConfig",
    "OpponentMetricSpec",
    "build_opponent_adjusted_team_metrics",
    "resolve_opponent_adjustment_config",
]
