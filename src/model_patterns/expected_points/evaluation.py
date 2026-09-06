"""Shared, chronology-safe evaluation for expected-points models."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from .betting import determine_plays
from .chronology import chronological_train_test_split
from .confidence import fit_classifiers
from .types import ExpectedPointsConfig, PlayThresholds

LOGGER = logging.getLogger(__name__)


def _finite_metric(value: float) -> float:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def score_prediction_metrics(results: pd.DataFrame) -> dict[str, float]:
    """Measure score, margin, and total error on score-model OOS rows."""
    required = {
        "home_score", "away_score", "home_score_pred", "away_score_pred",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Score evaluation is missing columns: {', '.join(missing)}")
    evaluated = results.dropna(subset=list(required)).copy()
    if evaluated.empty:
        raise ValueError("Score evaluation requires at least one complete prediction")

    actual_team = np.concatenate([evaluated["home_score"], evaluated["away_score"]])
    predicted_team = np.concatenate(
        [evaluated["home_score_pred"], evaluated["away_score_pred"]]
    )
    actual_margin = evaluated["away_score"] - evaluated["home_score"]
    predicted_margin = evaluated["away_score_pred"] - evaluated["home_score_pred"]
    actual_total = evaluated["away_score"] + evaluated["home_score"]
    predicted_total = evaluated["away_score_pred"] + evaluated["home_score_pred"]

    def errors(actual, predicted, prefix: str) -> dict[str, float]:
        residual = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
        return {
            f"{prefix}_mae": _finite_metric(mean_absolute_error(actual, predicted)),
            f"{prefix}_rmse": _finite_metric(mean_squared_error(actual, predicted) ** 0.5),
            f"{prefix}_bias": _finite_metric(residual.mean()),
        }

    return {
        **errors(actual_team, predicted_team, "score"),
        **errors(evaluated["home_score"], evaluated["home_score_pred"], "home_score"),
        **errors(evaluated["away_score"], evaluated["away_score_pred"], "away_score"),
        **errors(actual_margin, predicted_margin, "margin"),
        **errors(actual_total, predicted_total, "total_points"),
        "score_eval_rows": float(len(evaluated)),
    }


def market_score_benchmarks(results: pd.DataFrame) -> dict[str, float]:
    """Compare point predictions with the market-implied score baseline."""
    required = {
        "home_score", "away_score", "home_score_pred", "away_score_pred",
        "implied_points_home", "implied_points_away",
    }
    if not required.issubset(results.columns):
        return {}
    evaluated = results.dropna(subset=list(required)).copy()
    if evaluated.empty:
        return {}

    actual_home = pd.to_numeric(evaluated["home_score"], errors="coerce")
    actual_away = pd.to_numeric(evaluated["away_score"], errors="coerce")
    predicted_home = pd.to_numeric(evaluated["home_score_pred"], errors="coerce")
    predicted_away = pd.to_numeric(evaluated["away_score_pred"], errors="coerce")
    market_home = pd.to_numeric(evaluated["implied_points_home"], errors="coerce")
    market_away = pd.to_numeric(evaluated["implied_points_away"], errors="coerce")

    model_score_error = np.concatenate([
        predicted_home - actual_home,
        predicted_away - actual_away,
    ])
    market_score_error = np.concatenate([
        market_home - actual_home,
        market_away - actual_away,
    ])
    metrics = {
        "market_score_mae": _finite_metric(np.abs(market_score_error).mean()),
        "score_mae_advantage_vs_market": _finite_metric(
            np.abs(market_score_error).mean() - np.abs(model_score_error).mean()
        ),
    }

    actual_margin = actual_away - actual_home
    predicted_margin = predicted_away - predicted_home
    spread_column = (
        "spread_reference_line"
        if "spread_reference_line" in evaluated
        else "spread_line"
    )
    if spread_column in evaluated:
        market_margin = pd.to_numeric(evaluated[spread_column], errors="coerce")
        valid = market_margin.notna() & actual_margin.notna() & predicted_margin.notna()
        if valid.any():
            market_mae = np.abs(market_margin[valid] - actual_margin[valid]).mean()
            model_mae = np.abs(predicted_margin[valid] - actual_margin[valid]).mean()
            metrics["market_margin_mae"] = _finite_metric(market_mae)
            metrics["margin_mae_advantage_vs_market"] = _finite_metric(
                market_mae - model_mae
            )

    actual_total = actual_home + actual_away
    predicted_total = predicted_home + predicted_away
    total_column = (
        "total_reference_line"
        if "total_reference_line" in evaluated
        else "total_line"
    )
    if total_column in evaluated:
        market_total = pd.to_numeric(evaluated[total_column], errors="coerce")
        valid = market_total.notna() & actual_total.notna() & predicted_total.notna()
        if valid.any():
            market_mae = np.abs(market_total[valid] - actual_total[valid]).mean()
            model_mae = np.abs(predicted_total[valid] - actual_total[valid]).mean()
            metrics["market_total_points_mae"] = _finite_metric(market_mae)
            metrics["total_points_mae_advantage_vs_market"] = _finite_metric(
                market_mae - model_mae
            )
    return metrics


def determine_plays_by_period(
    frame: pd.DataFrame,
    *,
    thresholds: PlayThresholds,
    period_columns: Iterable[str] = ("season", "week"),
) -> pd.DataFrame:
    """Apply top-N lock rules independently to each historical slate."""
    columns = tuple(period_columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Lock evaluation is missing period columns: {', '.join(missing)}")
    if frame.empty:
        return frame.copy()
    groups = []
    for _, group in frame.groupby(list(columns), sort=False, dropna=False):
        groups.append(determine_plays(group, thresholds=thresholds))
    return pd.concat(groups).sort_index()


def _market_metrics(frame: pd.DataFrame, market: str) -> dict[str, float | None]:
    win_col = f"{market}_win"
    lock_col = f"{market}_lock"
    probability_col = f"{market}_win_prob"
    supported_col = f"{market}_market_supported"
    if win_col not in frame:
        return {}

    supported = (
        frame[supported_col].fillna(False).astype(bool)
        if supported_col in frame
        else pd.Series(True, index=frame.index)
    )
    eligible = frame.loc[supported]
    wins = pd.to_numeric(eligible[win_col], errors="coerce")
    decisions = wins.dropna()
    lock_values = eligible.get(lock_col, pd.Series(0, index=eligible.index))
    locks = eligible.loc[pd.to_numeric(lock_values, errors="coerce").fillna(0) == 1]
    lock_decisions = pd.to_numeric(locks.get(win_col), errors="coerce").dropna()
    metrics: dict[str, float | None] = {
        f"{market}_wins": float((decisions == 1).sum()),
        f"{market}_losses": float((decisions == 0).sum()),
        f"{market}_pushes": float(wins.isna().sum()),
        f"{market}_win_pct": (
            _finite_metric(decisions.mean() * 100) if len(decisions) else None
        ),
        f"{market}_lock_wins": float((lock_decisions == 1).sum()),
        f"{market}_lock_losses": float((lock_decisions == 0).sum()),
        f"{market}_lock_pushes": float(pd.to_numeric(locks.get(win_col), errors="coerce").isna().sum()),
        f"{market}_lock_win_pct": (
            _finite_metric(lock_decisions.mean() * 100)
            if len(lock_decisions)
            else None
        ),
        f"{market}_locks": float(len(locks)),
        f"{market}_eligible": float(len(eligible)),
        f"{market}_lock_coverage_pct": (
            _finite_metric(len(locks) / len(eligible) * 100) if len(eligible) else 0.0
        ),
    }
    if probability_col in frame and len(decisions):
        evaluated = eligible.loc[wins.notna(), [win_col, probability_col]].dropna()
        if not evaluated.empty:
            probabilities = pd.to_numeric(
                evaluated[probability_col], errors="coerce"
            ).clip(0.0, 100.0) / 100.0
            outcomes = pd.to_numeric(evaluated[win_col], errors="coerce").astype(int)
            valid = probabilities.notna()
            probabilities = probabilities.loc[valid].clip(1e-9, 1 - 1e-9)
            outcomes = outcomes.loc[valid]
            if not outcomes.empty:
                metrics[f"{market}_brier"] = float(
                    brier_score_loss(outcomes, probabilities)
                )
                metrics[f"{market}_log_loss"] = float(
                    log_loss(outcomes, probabilities, labels=[0, 1])
                )
                metrics[f"{market}_calibration_error"] = expected_calibration_error(
                    outcomes,
                    probabilities,
                )
                metrics[f"{market}_auc"] = (
                    float(roc_auc_score(outcomes, probabilities))
                    if outcomes.nunique() > 1
                    else None
                )

    lock_probabilities = locks[[win_col, probability_col]].dropna() if probability_col in locks else pd.DataFrame()
    if not lock_probabilities.empty:
        observed_lock_rate = pd.to_numeric(
            lock_probabilities[win_col], errors="coerce"
        ).mean() * 100.0
        mean_lock_probability = pd.to_numeric(
            lock_probabilities[probability_col], errors="coerce"
        ).mean()
        metrics[f"{market}_lock_mean_win_prob"] = _finite_metric(mean_lock_probability)
        metrics[f"{market}_lock_calibration_gap"] = _finite_metric(
            mean_lock_probability - observed_lock_rate
        )
    else:
        metrics[f"{market}_lock_mean_win_prob"] = None
        metrics[f"{market}_lock_calibration_gap"] = None
    return metrics


def expected_calibration_error(
    outcomes: pd.Series,
    probabilities: pd.Series,
    *,
    bins: int = 10,
) -> float:
    evaluated = pd.DataFrame({"outcome": outcomes, "probability": probabilities}).dropna()
    if evaluated.empty:
        return 0.0
    evaluated["bucket"] = pd.cut(
        evaluated["probability"],
        bins=np.linspace(0, 1, bins + 1),
        include_lowest=True,
        duplicates="drop",
    )
    weighted = 0.0
    for _, group in evaluated.groupby("bucket", observed=True):
        weighted += len(group) * abs(group["probability"].mean() - group["outcome"].mean())
    return float(weighted / len(evaluated))


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    metrics = score_prediction_metrics(frame)
    metrics.update(market_score_benchmarks(frame))
    metrics.update(_market_metrics(frame, "spread"))
    metrics.update(_market_metrics(frame, "total"))
    weeks = frame[[column for column in ("season", "week") if column in frame]].drop_duplicates()
    week_count = len(weeks)
    metrics["evaluated_weeks"] = float(week_count)
    for market in ("spread", "total"):
        metrics[f"{market}_locks_per_week"] = (
            metrics.get(f"{market}_locks", 0.0) / week_count if week_count else 0.0
        )
    return metrics


def _positive_probability(classifier, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ValueError("Confidence classifier did not learn the win class")
    return classifier.predict_proba(frame[features])[:, classes.index(1)] * 100.0


def confidence_health_evaluation(
    results: pd.DataFrame,
    config: ExpectedPointsConfig,
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """Evaluate confidence and lock behavior on an untouched chronological tail."""
    development, validation = chronological_train_test_split(
        results,
        time_col=config.time_col,
        test_size=config.confidence_validation_size,
    )
    spread_clf, total_clf = fit_classifiers(
        development,
        config.spread_class_features,
        config.total_class_features,
        config.spread_class_cat_features,
        config.total_class_cat_features,
        config.confidence_param_grid,
        time_col=config.time_col,
        validation_size=config.confidence_validation_size,
        n_jobs=config.confidence_n_jobs,
        scoring=config.confidence_scoring,
    )
    evaluated = validation.copy()
    evaluated["spread_win_prob"] = _positive_probability(
        spread_clf,
        evaluated,
        config.spread_class_features,
    )
    evaluated["total_win_prob"] = _positive_probability(
        total_clf,
        evaluated,
        config.total_class_features,
    )
    evaluated = determine_plays_by_period(evaluated, thresholds=config.play_thresholds)
    metrics = {f"health_{key}": value for key, value in prediction_metrics(evaluated).items()}
    metrics["health_available"] = 1.0
    return evaluated, metrics


def safe_confidence_health_evaluation(
    results: pd.DataFrame,
    config: ExpectedPointsConfig,
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """Keep a sparse early-season health slice from breaking production training."""
    try:
        return confidence_health_evaluation(results, config)
    except ValueError as exc:
        LOGGER.warning("Confidence health evaluation unavailable: %s", exc)
        return pd.DataFrame(), {"health_available": 0.0}


__all__ = [
    "confidence_health_evaluation",
    "determine_plays_by_period",
    "expected_calibration_error",
    "market_score_benchmarks",
    "prediction_metrics",
    "safe_confidence_health_evaluation",
    "score_prediction_metrics",
]
