"""Lightweight chronological replay for expected-points Lock probability heads.

The replay consumes a completed :class:`BacktestRun`.  Score predictions, play
directions, execution lines, and outcomes are immutable inputs; only the small
second-stage probability head is fitted.  No feed loading, notebook execution,
score-model fitting, database access, or operational writes belong here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.sports.football.kickoff import parse_eastern_kickoffs

from .backtest_artifacts import _jsonable, load_backtest_run
from .backtesting import BacktestRun, frame_fingerprint
from .lock_features import LockFeatureSet, lock_feature_set
from .lock_models import (
    CalibratedHead,
    LockVariantSpec,
    fit_calibrator,
    fit_probability_head,
    shrunken_base_rate,
)
from .lock_policy import (
    LockPolicy,
    add_outcome_probabilities,
    estimate_push_probability,
    lock_profit_metrics,
    select_weekly_locks,
)
from .types import ExpectedPointsLeague


LOCK_REPLAY_PROTOCOL_VERSION = "2"
RESULT_AVAILABILITY_LAG = pd.Timedelta(hours=8)

REQUIRED_SOURCE_COLUMNS = {
    "season", "week", "game_id", "date_time", "backtest_cutoff",
    "home_team", "away_team", "home_score", "away_score",
    "home_score_pred", "away_score_pred", "spread_pred", "total_pred",
    "spread_line", "total_line", "spread_play", "total_play",
    "spread_diff", "total_diff", "spread_win_prob", "total_win_prob",
    "spread_lock", "total_lock", "spread_win", "total_win",
    "recipe_name", "recipe_version", "recipe_fingerprint",
}


@dataclass(frozen=True)
class LockReplaySpec:
    warmup_seasons: tuple[int, ...] = (2023,)
    development_seasons: tuple[int, ...] = (2024,)
    confirmation_seasons: tuple[int, ...] = (2025,)
    minimum_training_decisions: int = 100
    minimum_class_decisions: int = 25
    bootstrap_samples: int = 2000
    random_seed: int = 31
    policy: LockPolicy = LockPolicy()
    minimum_two_plus_week_fraction: float = 0.8
    training_history: str = "weekly"

    def __post_init__(self) -> None:
        groups = [set(self.warmup_seasons), set(self.development_seasons), set(self.confirmation_seasons)]
        if any(left & right for index, left in enumerate(groups) for right in groups[index + 1:]):
            raise ValueError("Warmup, development, and confirmation seasons must be disjoint")
        if self.minimum_training_decisions < 1 or self.minimum_class_decisions < 1:
            raise ValueError("Lock replay training minimums must be positive")
        if self.bootstrap_samples < 1:
            raise ValueError("Bootstrap samples must be positive")
        if not 0 <= self.minimum_two_plus_week_fraction <= 1:
            raise ValueError("Weekly coverage target must be between zero and one")
        if self.training_history not in {"weekly", "score-holdout"}:
            raise ValueError("Unknown Lock training history")


@dataclass(frozen=True)
class LockReplaySource:
    path: Path
    run: BacktestRun
    predictions_sha256: str
    prediction_fingerprint: str
    ledger: pd.DataFrame


@dataclass(frozen=True)
class LockSelection:
    league: str
    source_fingerprint: str
    protocol_version: str
    spread_variant: str | None
    total_variant: str | None
    enabled_markets: tuple[str, ...]
    development_metrics: dict
    rejection_reasons: tuple[str, ...]
    selection_fingerprint: str


def _json_digest(value: object) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _american_probability(value: object) -> float:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(odds) or odds == 0:
        return np.nan
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def _outcome(win: object) -> str | None:
    if pd.isna(win):
        return "push"
    return "win" if int(win) == 1 else "loss"


def _safe_numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _safe_object(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype="object")
    return frame[name].astype("object")


def build_lock_market_frame(
    frame: pd.DataFrame,
    league: ExpectedPointsLeague,
    market: str,
    *,
    outcomes_known: bool,
) -> pd.DataFrame:
    """Build the identical explicit feature view used by replay and production."""
    output = pd.DataFrame(index=frame.index)
    for column in ("season", "week", "game_id", "home_team", "away_team"):
        output[column] = frame[column]
    output["market"] = market
    output["kickoff"] = parse_eastern_kickoffs(frame["date_time"])
    output["cutoff"] = (
        pd.to_datetime(frame["backtest_cutoff"], utc=True)
        if "backtest_cutoff" in frame else output["kickoff"]
    )
    output["result_available_at"] = output["kickoff"] + RESULT_AVAILABILITY_LAG
    output["line"] = _safe_numeric(frame, f"{market}_line")
    output["prediction"] = _safe_numeric(frame, f"{market}_pred")
    output["edge"] = _safe_numeric(frame, f"{market}_diff").abs()
    output["market_supported"] = (
        frame.get(f"{market}_market_supported", pd.Series(True, index=frame.index))
        .fillna(False).astype(bool)
    )
    output["weekly_edge_rank"] = (
        output.sort_values(["kickoff", "game_id"], kind="stable")["edge"].where(output["market_supported"])
        .groupby([output["season"], output["week"]], sort=False)
        .rank(method="first", ascending=False)
    )
    output["signed_edge"] = output["prediction"] - output["line"]
    output["play"] = frame[f"{market}_play"].astype(str)
    output["direction"] = np.where(
        market == "spread",
        np.where(output["play"].eq(frame["home_team"].astype(str)), "home", "away"),
        output["play"],
    )
    output["recorded_probability"] = _safe_numeric(frame, f"{market}_win_prob")
    output["recorded_lock"] = _safe_numeric(frame, f"{market}_lock").fillna(0).astype(int)
    output["win"] = _safe_numeric(frame, f"{market}_win")
    output["outcome"] = (
        output["win"].map(_outcome)
        if outcomes_known else pd.Series(None, index=frame.index, dtype="object")
    )
    if market == "spread" and outcomes_known:
        output["score_residual"] = (
            _safe_numeric(frame, "away_score") - _safe_numeric(frame, "home_score")
            - _safe_numeric(frame, "spread_pred")
        )
    elif outcomes_known:
        output["score_residual"] = (
            _safe_numeric(frame, "away_score") + _safe_numeric(frame, "home_score")
            - _safe_numeric(frame, "total_pred")
        )
    else:
        output["score_residual"] = np.nan
    output["spread_line"] = _safe_numeric(frame, "spread_line")
    output["total_line"] = _safe_numeric(frame, "total_line")
    output["predicted_margin"] = -_safe_numeric(frame, "spread_pred")
    output["predicted_total"] = _safe_numeric(frame, "total_pred")
    output["line_is_integer"] = np.isclose(output["line"], np.round(output["line"])).astype(float)
    output["spread_key_3_distance"] = (output["spread_line"].abs() - 3.0).abs()
    output["spread_key_7_distance"] = (output["spread_line"].abs() - 7.0).abs()
    output["weekday"] = _safe_object(frame, "weekday")
    output["implied_points_home"] = _safe_numeric(frame, "implied_points_home")
    output["implied_points_away"] = _safe_numeric(frame, "implied_points_away")

    home_pick = output["direction"].eq("home")
    if league is ExpectedPointsLeague.NFL:
        output["roof"] = _safe_object(frame, "roof")
        output["division_game"] = _safe_numeric(frame, "div_game")
        rest_advantage = _safe_numeric(frame, "rest_home") - _safe_numeric(frame, "rest_away")
        qbr_advantage = _safe_numeric(frame, "ewma_qbr_home") - _safe_numeric(frame, "ewma_qbr_away")
        output["picked_rest_advantage"] = np.where(home_pick, rest_advantage, -rest_advantage)
        output["picked_qbr_advantage"] = np.where(home_pick, qbr_advantage, -qbr_advantage)
        for column in lock_feature_set(league, "context").numeric:
            if column not in output:
                output[column] = _safe_numeric(frame, column)
        home_odds = _safe_numeric(frame, "spread_odds_home")
        away_odds = _safe_numeric(frame, "spread_odds_away")
        output["selected_spread_odds"] = np.where(home_pick, home_odds, away_odds) if market == "spread" else np.nan
        output["opposing_spread_odds"] = np.where(home_pick, away_odds, home_odds) if market == "spread" else np.nan
        output["moneyline_home"] = _safe_numeric(frame, "moneyline_home")
        output["moneyline_away"] = _safe_numeric(frame, "moneyline_away")
        output["over_odds_for_selected_play"] = np.where(
            (market == "total") & output["direction"].eq("over"), _safe_numeric(frame, "over_odds"), np.nan
        )
    else:
        output["neutral_site"] = _safe_numeric(frame, "neutral_site").fillna(_safe_numeric(frame, "neutral_site_x"))
        output["conference_game"] = _safe_numeric(frame, "conference_game")
        elo_advantage = _safe_numeric(frame, "home_pregame_elo") - _safe_numeric(frame, "away_pregame_elo")
        output["picked_elo_advantage"] = np.where(home_pick, elo_advantage, -elo_advantage)
        output["elo_gap"] = elo_advantage.abs()
        home_conference = _safe_object(frame, "home_conference")
        away_conference = _safe_object(frame, "away_conference")
        output["picked_conference"] = np.where(home_pick, home_conference, away_conference)
        output["opponent_conference"] = np.where(home_pick, away_conference, home_conference)
        output["home_classification"] = _safe_object(frame, "home_classification")
        output["away_classification"] = _safe_object(frame, "away_classification")
        for column in lock_feature_set(league, "context").numeric:
            if column not in output:
                output[column] = _safe_numeric(frame, column)
        prefix = market
        output["reference_line"] = _safe_numeric(frame, f"{prefix}_reference_line")
        output["open_line"] = _safe_numeric(frame, f"{prefix}_open_line")
        output["line_move"] = _safe_numeric(frame, f"{prefix}_line_move")
        output["line_move_abs"] = _safe_numeric(frame, f"{prefix}_line_move_abs")
        output["quote_count"] = _safe_numeric(frame, f"{prefix}_quote_count")
        output["quote_range"] = _safe_numeric(frame, f"{prefix}_quote_range")
        output["shopping_points"] = _safe_numeric(frame, f"{prefix}_shopping_points")
        output["move_toward_pick"] = _safe_numeric(frame, f"{prefix}_move_toward_pick")
        output["consensus_home_win_prob"] = _safe_numeric(frame, "consensus_home_win_prob")
        output["provider"] = _safe_object(frame, f"{prefix}_provider")
        output["selection_reason"] = _safe_object(frame, f"{prefix}_selection_reason")
    return output.reset_index(drop=True)


def build_lock_ledger(frame: pd.DataFrame, league: ExpectedPointsLeague) -> pd.DataFrame:
    output = pd.concat(
        [
            build_lock_market_frame(frame, league, market, outcomes_known=True)
            for market in ("spread", "total")
        ],
        ignore_index=True,
    )
    keys = ["season", "week", "game_id", "market"]
    if output.duplicated(keys).any():
        raise ValueError("Lock source contains duplicate game-market keys")
    return output.sort_values(["cutoff", "kickoff", "game_id", "market"], kind="stable").reset_index(drop=True)


def load_lock_replay_source(
    path: str | Path,
    *,
    league: ExpectedPointsLeague | str,
    expected_version: str,
    expected_profile: str = "standard",
    expected_seasons: Sequence[int] | None = None,
) -> LockReplaySource:
    source_path = Path(path).resolve()
    run = load_backtest_run(source_path)
    parsed_league = league if isinstance(league, ExpectedPointsLeague) else ExpectedPointsLeague(league)
    if run.league is not parsed_league:
        raise ValueError(f"Lock source league is {run.league.value}, expected {parsed_league.value}")
    if run.recipe_version != expected_version:
        raise ValueError(f"Lock source version is {run.recipe_version}, expected {expected_version}")
    if run.profile != expected_profile:
        raise ValueError(f"Lock source profile is {run.profile}, expected {expected_profile}")
    if expected_seasons is not None and tuple(run.seasons) != tuple(int(value) for value in expected_seasons):
        raise ValueError(f"Lock source seasons are {run.seasons}, expected {tuple(expected_seasons)}")
    frame = run.predictions
    missing = sorted(REQUIRED_SOURCE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Lock source is missing required columns: {missing}")
    if frame.duplicated(["season", "week", "game_id"]).any():
        raise ValueError("Lock source contains duplicate game keys")
    for column, expected in (
        ("recipe_name", run.recipe_name),
        ("recipe_version", run.recipe_version),
        ("recipe_fingerprint", run.recipe_fingerprint),
    ):
        if set(frame[column].dropna().astype(str)) != {str(expected)}:
            raise ValueError(f"Lock source has mixed or inconsistent {column}")
    cutoff = pd.to_datetime(frame["backtest_cutoff"], utc=True, errors="coerce")
    kickoff = parse_eastern_kickoffs(frame["date_time"])
    if cutoff.isna().any() or kickoff.isna().any() or not (kickoff > cutoff).all():
        raise ValueError("Every lock source kickoff must be strictly after its timezone-aware cutoff")
    if not np.allclose(frame["spread_pred"], frame["away_score_pred"] - frame["home_score_pred"], equal_nan=False):
        raise ValueError("Lock source spread predictions do not match frozen score predictions")
    if not np.allclose(frame["total_pred"], frame["away_score_pred"] + frame["home_score_pred"], equal_nan=False):
        raise ValueError("Lock source total predictions do not match frozen score predictions")
    expected_spread = frame["away_score"] - frame["home_score"]
    push = np.isclose(expected_spread, frame["spread_line"])
    correct_spread = np.where(expected_spread < frame["spread_line"], frame["home_team"], frame["away_team"])
    recomputed_spread = np.where(push, np.nan, (frame["spread_play"].astype(str) == correct_spread).astype(float))
    expected_total = frame["away_score"] + frame["home_score"]
    total_push = np.isclose(expected_total, frame["total_line"])
    correct_total = np.where(expected_total < frame["total_line"], "under", "over")
    recomputed_total = np.where(total_push, np.nan, (frame["total_play"].astype(str) == correct_total).astype(float))
    for market, recomputed in (("spread", recomputed_spread), ("total", recomputed_total)):
        actual = pd.to_numeric(frame[f"{market}_win"], errors="coerce").to_numpy(float)
        if not np.array_equal(np.isnan(actual), np.isnan(recomputed)) or not np.allclose(
            actual[~np.isnan(actual)], recomputed[~np.isnan(recomputed)]
        ):
            raise ValueError(f"Lock source {market} settlements do not match scores and execution lines")
    predictions_path = source_path / "predictions.parquet"
    return LockReplaySource(
        path=source_path,
        run=run,
        predictions_sha256=hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        prediction_fingerprint=frame_fingerprint(frame),
        ledger=build_lock_ledger(frame, parsed_league),
    )


def registered_lock_variants(league: ExpectedPointsLeague) -> tuple[LockVariantSpec, ...]:
    """Predeclared tournament; market-timing variants are evidence-only."""
    return (
        LockVariantSpec("base_rate", "base_rate"),
        LockVariantSpec("recorded_raw", "recorded_raw"),
        LockVariantSpec("recorded_platt", "recorded_platt", calibrator="platt"),
        LockVariantSpec("recorded_beta", "recorded_platt", calibrator="beta"),
        LockVariantSpec("empirical_edge_25", "empirical_edge", parameters={"strength": 25.0}),
        LockVariantSpec("empirical_edge_50", "empirical_edge", parameters={"strength": 50.0}),
        *(LockVariantSpec(
            f"symmetric_residual_{trust:g}", "symmetric_residual", parameters={"trust": trust},
        ) for trust in (0.2, 0.35, 0.5)),
        LockVariantSpec(
            "edge_threshold_6_top1",
            "edge_threshold",
            parameters={
                "minimum_edge": 6.0,
                "maximum_rank": 1,
                "strength": 10.0,
                "max_locks_per_week": 1,
            },
        ),
        LockVariantSpec(
            "edge_threshold_7_top4",
            "edge_threshold",
            parameters={
                "minimum_edge": 7.0,
                "maximum_rank": 4,
                "strength": 10.0,
                "max_locks_per_week": 4,
            },
        ),
        LockVariantSpec("residual_distribution_10", "residual_distribution", parameters={"strength": 10.0}),
        LockVariantSpec("residual_distribution_25", "residual_distribution", parameters={"strength": 25.0}),
        LockVariantSpec("spline_core_c03", "spline_logit", parameters={"C": 0.3}),
        LockVariantSpec("spline_context_c01", "spline_logit", feature_group="context", parameters={"C": 0.1}),
        LockVariantSpec("spline_context_c03", "spline_logit", feature_group="context", parameters={"C": 0.3}),
        LockVariantSpec(
            "spline_market_c01", "spline_logit", feature_group="market", parameters={"C": 0.1},
            promotion_eligible=False,
        ),
        LockVariantSpec(
            "lightgbm_context", "lightgbm", feature_group="context",
            parameters={"n_estimators": 50, "max_depth": 2, "learning_rate": 0.03, "min_child_samples": 40},
        ),
        LockVariantSpec(
            "lightgbm_market", "lightgbm", feature_group="market",
            parameters={"n_estimators": 50, "max_depth": 2, "learning_rate": 0.03, "min_child_samples": 40},
            promotion_eligible=False,
        ),
    )


def _fit_for_period(
    spec: LockVariantSpec,
    history: pd.DataFrame,
    features: LockFeatureSet,
    replay: LockReplaySpec,
):
    resolved = history.loc[history["outcome"].isin(["win", "loss"])].copy()
    ready = (
        len(resolved) >= replay.minimum_training_decisions
        and resolved["win"].nunique() == 2
        and resolved["win"].value_counts().min() >= replay.minimum_class_decisions
    )
    if not ready and spec.family not in {"recorded_raw"}:
        return fit_probability_head(
            LockVariantSpec("fallback", "base_rate"), resolved,
            numeric_features=(), label_column="win",
        ), False
    head = fit_probability_head(
        spec,
        history,
        numeric_features=features.numeric,
        categorical_features=features.categorical,
        label_column="win",
    )
    if spec.calibrator != "identity" and spec.family not in {"recorded_platt"}:
        periods = resolved[["cutoff"]].drop_duplicates().sort_values("cutoff")
        split = max(1, int(len(periods) * 0.8))
        calibration_periods = set(periods.iloc[split:]["cutoff"])
        calibration = resolved.loc[resolved["cutoff"].isin(calibration_periods)]
        fitting = resolved.loc[~resolved["cutoff"].isin(calibration_periods)]
        if len(calibration) >= 40 and fitting["win"].nunique() == 2:
            calibration_head = fit_probability_head(
                spec, fitting, numeric_features=features.numeric,
                categorical_features=features.categorical, label_column="win",
            )
            head = CalibratedHead(head, fit_calibrator(spec.calibrator, calibration_head.predict(calibration), calibration["win"]))
    return head, ready or spec.family == "recorded_raw"


def replay_lock_variant(
    source: LockReplaySource,
    variant: LockVariantSpec,
    replay: LockReplaySpec,
    *,
    market: str,
    seasons: Iterable[int],
) -> pd.DataFrame:
    if market not in {"spread", "total"}:
        raise ValueError(f"Unsupported market: {market}")
    selected_seasons = {int(value) for value in seasons}
    ledger = source.ledger.loc[source.ledger["market"].eq(market)].copy()
    target = ledger.loc[ledger["season"].astype(int).isin(selected_seasons)].copy()
    features = lock_feature_set(source.run.league, variant.feature_group, market=market)
    decisions = []
    saved_training = getattr(source.run, "lock_training", None)
    if replay.training_history == "score-holdout" and saved_training is None:
        raise ValueError("This source has no saved per-cutoff score holdout; use weekly history or create a new standard run")
    for cutoff, slate in target.groupby("cutoff", sort=True):
        if replay.training_history == "score-holdout":
            rows = saved_training.loc[pd.to_datetime(saved_training["lock_training_cutoff"], utc=True).eq(cutoff)]
            if rows.empty:
                raise ValueError(f"Missing saved score holdout at {cutoff}")
            history = build_lock_market_frame(rows, source.run.league, market, outcomes_known=True)
            if not history["kickoff"].lt(cutoff).all():
                raise ValueError("Saved score holdout contains a target/future game")
            if variant.family == "recorded_platt" and history["recorded_probability"].isna().any():
                raise ValueError("Recorded-probability calibration is unavailable in this saved score holdout")
        else:
            history = ledger.loc[ledger["result_available_at"].lt(cutoff)].copy()
        head, ready = _fit_for_period(variant, history, features, replay)
        q = head.predict(slate)
        push = estimate_push_probability(history, slate)
        predicted = add_outcome_probabilities(slate, q, push, policy=replay.policy)
        predicted["variant"] = variant.name
        predicted["training_history"] = replay.training_history
        predicted["feature_group"] = variant.feature_group
        predicted["family"] = variant.family
        predicted["calibrator"] = variant.calibrator
        predicted["trained_rows"] = len(history.loc[history["outcome"].isin(["win", "loss"])])
        predicted["trained_through"] = history["result_available_at"].max() if len(history) else pd.NaT
        predicted["locks_enabled"] = bool(ready and variant.promotion_eligible)
        decisions.append(predicted)
    if not decisions:
        return pd.DataFrame()
    result = pd.concat(decisions, ignore_index=True)
    variant_policy = replace(
        replay.policy,
        minimum_edge=float(variant.parameters.get("minimum_edge", replay.policy.minimum_edge)),
        max_locks_per_week=int(
            variant.parameters.get("max_locks_per_week", replay.policy.max_locks_per_week)
        ),
    )
    result["policy_eligible"] = result["edge"].ge(variant_policy.minimum_edge)
    selected = select_weekly_locks(result, policy=variant_policy)
    # Preserve the frozen variant's caps when markets are combined later.
    selected["policy_eligible"] &= selected["lock"].eq(1)
    return selected


def probability_metrics(decisions: pd.DataFrame) -> dict[str, float | int | None]:
    resolved = decisions.loc[decisions["outcome"].isin(["win", "loss"])].copy()
    if not len(resolved):
        return {"decisions": 0, "brier": None, "log_loss": None, "auc": None}
    labels = resolved["win"].astype(int)
    probability = resolved["resolved_win_probability"].clip(1e-6, 1 - 1e-6)
    return {
        "decisions": len(resolved),
        "brier": float(brier_score_loss(labels, probability)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "auc": float(roc_auc_score(labels, probability)) if labels.nunique() == 2 else None,
    }


def screen_lock_cadence(
    source: LockReplaySource,
    replay: LockReplaySpec,
    variants: Sequence[LockVariantSpec],
    output: Path,
) -> pd.DataFrame:
    """Exploratory volume/profit screen; refit heads once, reuse forecasts for policies.

    All evaluation seasons are exposed. This is not untouched confirmation or
    automatic production promotion. Each row references saved bet-level evidence.
    """
    import logging
    from .backtest_artifacts import write_json_atomic, write_parquet_atomic

    seasons = (*replay.development_seasons, *replay.confirmation_seasons)
    # Policy experiments never change fitted probabilities.
    forecast_spec = replace(replay, policy=replace(
        replay.policy, minimum_resolved_win_probability=0.5,
        max_locks_per_week=1_000_000, extra_lock_min_probability=None,
    ))
    records = []
    for variant in variants:
        logging.info("Cadence probability head: %s", variant.name)
        forecasts = {}
        for market in ("spread", "total"):
            forecast = replay_lock_variant(source, variant, forecast_spec, market=market, seasons=seasons)
            forecasts[market] = forecast
            write_parquet_atomic(output / "forecasts" / variant.name / f"{market}.parquet", forecast)
        for mode, spread_cap, total_cap in (
            ("spread", None, 0), ("total", 0, None), ("both", None, None),
            ("spread2_total1", 2, 1), ("spread1_total2", 1, 2),
        ):
            frame = pd.concat(forecasts.values(), ignore_index=True)
            for normal_cap, extra in ((2, None), (3, None), (3, 0.55), (3, 0.575)):
                policy = replace(
                    replay.policy, max_locks_per_week=normal_cap,
                    extra_lock_min_probability=extra,
                    max_spreads_per_week=spread_cap, max_totals_per_week=total_cap,
                )
                decisions = select_weekly_locks(frame, policy=policy)
                name = f"{variant.name}-{mode}-top{normal_cap}-extra{extra}"
                destination = output / "policies" / name
                write_parquet_atomic(destination / "decisions.parquet", decisions)
                metrics = summarize_lock_variant(decisions, replay)
                per_season = {
                    str(season): summarize_lock_variant(group, replay)
                    for season, group in decisions.groupby("season")
                }
                # Existing weekly source uses NFL weeks 1–18; CFB ordinary slates
                # 1–14, with conference titles/Army-Navy/bowls disclosed separately.
                regular = decisions.loc[pd.to_numeric(decisions["week"]).between(
                    1, 18 if source.run.league is ExpectedPointsLeague.NFL else 14
                )]
                cadence = lock_profit_metrics(regular, american_odds=policy.american_odds)
                record = {
                    "name": name, "variant": asdict(variant), "mode": mode,
                    "policy": asdict(policy), "metrics": metrics,
                    "per_season": per_season, "regular_season": cadence,
                    "coverage_target_met": (
                        cadence["two_plus_week_fraction"] is not None
                        and cadence["two_plus_week_fraction"] >= replay.minimum_two_plus_week_fraction
                        and cadence["zero_lock_weeks"] <= .1 * cadence["evaluated_weeks"]
                    ),
                    "decisions_path": str(destination / "decisions.parquet"),
                }
                write_json_atomic(destination / "metrics.json", record)
                records.append(record)
    write_json_atomic(output / "screen.json", {
        "warning": "Retrospective research: all evaluation seasons exposed; intervals are not selection-adjusted.",
        "policies": records,
    })
    return pd.DataFrame(records)


def profit_bootstrap_interval(
    decisions: pd.DataFrame,
    *,
    samples: int,
    seed: int,
    american_odds: int,
) -> tuple[float, float]:
    selected = decisions.loc[decisions["lock"].eq(1) & decisions["outcome"].notna()].copy()
    periods = list(decisions[["season", "week"]].drop_duplicates().itertuples(index=False, name=None))
    if len(periods) < 2:
        return (float("nan"), float("nan"))
    unit_map = {"win": 100.0 / abs(american_odds) if american_odds < 0 else american_odds / 100.0,
                "loss": -1.0, "push": 0.0}
    values = np.asarray([
        selected.loc[(selected["season"] == season) & (selected["week"] == week), "outcome"].map(unit_map).sum()
        for season, week in periods
    ])
    rng = np.random.default_rng(seed)
    # Resample full season-week blocks, including zero-bet weeks, within season.
    draws = np.zeros(samples)
    for season in sorted({season for season, _ in periods}):
        season_values = values[[index for index, period in enumerate(periods) if period[0] == season]]
        draws += season_values[rng.integers(0, len(season_values), size=(samples, len(season_values)))].sum(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def summarize_lock_variant(decisions: pd.DataFrame, replay: LockReplaySpec) -> dict:
    profit = lock_profit_metrics(decisions, american_odds=replay.policy.american_odds)
    interval = profit_bootstrap_interval(
        decisions, samples=replay.bootstrap_samples, seed=replay.random_seed,
        american_odds=replay.policy.american_odds,
    )
    return {**probability_metrics(decisions), **profit, "net_units_interval": interval}


def choose_lock_selection(
    source: LockReplaySource,
    replay: LockReplaySpec,
    decisions: dict[tuple[str, str], pd.DataFrame],
    variants: Sequence[LockVariantSpec],
) -> LockSelection:
    summaries = {
        f"{market}:{variant.name}": summarize_lock_variant(decisions[(market, variant.name)], replay)
        for market in ("spread", "total") for variant in variants
    }
    chosen: dict[str, str | None] = {}
    enabled_markets: list[str] = []
    reasons: list[str] = []
    for market in ("spread", "total"):
        baseline = summaries[f"{market}:base_rate"]
        candidates = []
        for variant in variants:
            metric = summaries[f"{market}:{variant.name}"]
            brier_ok = (
                metric["brier"] is not None and baseline["brier"] is not None
                and metric["brier"] <= baseline["brier"] * 0.999
            )
            log_ok = metric["log_loss"] is not None and metric["log_loss"] <= baseline["log_loss"]
            # The bootstrap lower bound already accounts for volume. Twenty-five
            # decisions is a hard floor; requiring fifty as well rejected a
            # genuinely selective, positive-lower-bound strategy by construction.
            profit_ok = metric["locks"] >= 25 and np.isfinite(metric["net_units_interval"][0]) and metric["net_units_interval"][0] > 0
            # Overstating a Lock is dangerous; conservative under-confidence is
            # allowed and is still visible in the probability metrics.
            calibration_ok = metric["calibration_gap"] is None or metric["calibration_gap"] <= 0.05
            if variant.promotion_eligible and brier_ok and log_ok and profit_ok and calibration_ok:
                candidates.append((metric["net_units_interval"][0], -metric["log_loss"], variant.name))
        if candidates:
            chosen[market] = max(candidates)[2]
            enabled_markets.append(market)
        else:
            honest = [
                (summaries[f"{market}:{variant.name}"]["log_loss"], variant.name)
                for variant in variants
                if variant.promotion_eligible
                and summaries[f"{market}:{variant.name}"]["log_loss"] is not None
            ]
            chosen[market] = min(honest)[1] if honest else "base_rate"
            reasons.append(f"{market}: no candidate passed probability, overconfidence, volume, and profit gates")

    enabled = tuple(enabled_markets)
    payload = {
        "league": source.run.league.value,
        "source_fingerprint": source.prediction_fingerprint,
        "protocol_version": LOCK_REPLAY_PROTOCOL_VERSION,
        "spread_variant": chosen["spread"],
        "total_variant": chosen["total"],
        "enabled_markets": enabled,
        "development_metrics": summaries,
        "rejection_reasons": reasons,
    }
    return LockSelection(**payload, selection_fingerprint=_json_digest(payload))


def selection_as_dict(selection: LockSelection) -> dict:
    return asdict(selection)


def assess_lock_promotion(
    selection: LockSelection,
    replay: LockReplaySpec,
    *,
    development: dict[tuple[str, str], pd.DataFrame],
    confirmation: dict[str, pd.DataFrame],
    confirmation_baseline: dict[str, pd.DataFrame],
) -> dict:
    """Apply untouched-season and combined-evidence gates to a frozen selection."""
    markets = {}
    promoted = []
    for market, selected_name in (
        ("spread", selection.spread_variant), ("total", selection.total_variant)
    ):
        candidate_metrics = summarize_lock_variant(confirmation[market], replay)
        baseline_metrics = probability_metrics(confirmation_baseline[market])
        combined_metrics = summarize_lock_variant(
            pd.concat([development[(market, selected_name)], confirmation[market]], ignore_index=True),
            replay,
        )
        checks = {
            "development_selected": market in selection.enabled_markets,
            "confirmation_volume": candidate_metrics["locks"] >= 15,
            "confirmation_profit": candidate_metrics["net_units"] > 0,
            "confirmation_brier": candidate_metrics["brier"] <= baseline_metrics["brier"],
            "confirmation_log_loss": candidate_metrics["log_loss"] <= baseline_metrics["log_loss"],
            "combined_profit_lower_bound": (
                np.isfinite(combined_metrics["net_units_interval"][0])
                and combined_metrics["net_units_interval"][0] > 0
            ),
        }
        passed = all(checks.values())
        if passed:
            promoted.append(market)
        markets[market] = {
            "variant": selected_name,
            "promoted": passed,
            "checks": checks,
            "confirmation": candidate_metrics,
            "confirmation_base_rate": baseline_metrics,
            "development_and_confirmation": combined_metrics,
        }
    return {"promoted_markets": promoted, "markets": markets}


__all__ = [
    "LOCK_REPLAY_PROTOCOL_VERSION",
    "LockReplaySource",
    "LockReplaySpec",
    "LockSelection",
    "build_lock_ledger",
    "build_lock_market_frame",
    "assess_lock_promotion",
    "choose_lock_selection",
    "load_lock_replay_source",
    "probability_metrics",
    "profit_bootstrap_interval",
    "registered_lock_variants",
    "replay_lock_variant",
    "selection_as_dict",
    "summarize_lock_variant",
]
