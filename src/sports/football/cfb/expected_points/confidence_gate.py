"""Probability-quality metrics for optional CFB confidence feature groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.model_patterns.expected_points.chronology import chronological_train_test_split
from src.model_patterns.expected_points.confidence import fit_classifiers


@dataclass(frozen=True)
class ProbabilityEvaluation:
    brier_score: float
    log_loss: float
    brier_by_season: dict[int, float]
    calibration: tuple[dict[str, float | int], ...]
    expected_calibration_error: float


def evaluate_probabilities(
    frame: pd.DataFrame,
    *,
    target_col: str,
    probability_col: str,
    season_col: str = "season",
    calibration_bins: int = 10,
) -> ProbabilityEvaluation:
    """Summarize genuinely out-of-time binary probabilities on a 0-1 scale."""
    required = {target_col, probability_col, season_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Probability frame is missing required columns: {', '.join(missing)}")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least 2")

    evaluated = frame[[target_col, probability_col, season_col]].dropna().copy()
    if evaluated.empty:
        raise ValueError("Probability evaluation requires at least one complete row")
    evaluated[target_col] = pd.to_numeric(evaluated[target_col], errors="raise").astype(int)
    evaluated[probability_col] = pd.to_numeric(
        evaluated[probability_col], errors="raise"
    ).clip(1e-9, 1 - 1e-9)
    if not set(evaluated[target_col].unique()).issubset({0, 1}):
        raise ValueError("Probability targets must be binary")

    brier_by_season = {
        int(season): float(brier_score_loss(group[target_col], group[probability_col]))
        for season, group in evaluated.groupby(season_col)
    }
    bin_edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    evaluated["__probability_bin"] = pd.cut(
        evaluated[probability_col],
        bins=bin_edges,
        include_lowest=True,
        duplicates="drop",
    )
    calibration = tuple(
        {
            "bin": str(bucket),
            "count": int(len(group)),
            "mean_probability": float(group[probability_col].mean()),
            "win_rate": float(group[target_col].mean()),
        }
        for bucket, group in evaluated.groupby("__probability_bin", observed=True)
    )
    expected_calibration_error = sum(
        bucket["count"]
        * abs(float(bucket["mean_probability"]) - float(bucket["win_rate"]))
        for bucket in calibration
    ) / len(evaluated)
    return ProbabilityEvaluation(
        brier_score=float(brier_score_loss(evaluated[target_col], evaluated[probability_col])),
        log_loss=float(log_loss(evaluated[target_col], evaluated[probability_col], labels=[0, 1])),
        brier_by_season=brier_by_season,
        calibration=calibration,
        expected_calibration_error=float(expected_calibration_error),
    )


def passes_optional_feature_gate(
    baseline: ProbabilityEvaluation,
    candidate: ProbabilityEvaluation,
    *,
    minimum_brier_improvement: float = 0.01,
    maximum_season_brier_degradation: float = 0.02,
) -> bool:
    """Apply the documented production gate for optional betting features."""
    if not 0 <= minimum_brier_improvement < 1:
        raise ValueError("minimum_brier_improvement must be in [0, 1)")
    if maximum_season_brier_degradation < 0:
        raise ValueError("maximum_season_brier_degradation cannot be negative")
    if set(baseline.brier_by_season) != set(candidate.brier_by_season):
        raise ValueError("Baseline and candidate evaluations must cover identical seasons")

    pooled_improved = candidate.brier_score <= baseline.brier_score * (
        1 - minimum_brier_improvement
    )
    log_loss_not_worse = candidate.log_loss <= baseline.log_loss
    seasons_acceptable = all(
        candidate.brier_by_season[season]
        <= baseline_score * (1 + maximum_season_brier_degradation)
        for season, baseline_score in baseline.brier_by_season.items()
    )
    return pooled_improved and log_loss_not_worse and seasons_acceptable


def _ordered_union(*column_groups: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(column for group in column_groups for column in group))


def _evaluate_classifier(
    frame: pd.DataFrame,
    *,
    classifier: Any,
    features: Sequence[str],
    target_col: str,
    season_col: str,
) -> tuple[ProbabilityEvaluation, int]:
    evaluated = frame.loc[frame[target_col].notna()].copy()
    if evaluated.empty:
        raise ValueError(f"No non-push {target_col} rows remain for probability evaluation")
    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ValueError(f"Classifier for {target_col} did not learn the win class")
    evaluated["__win_probability"] = classifier.predict_proba(
        evaluated[list(features)]
    )[:, classes.index(1)]
    return (
        evaluate_probabilities(
            evaluated,
            target_col=target_col,
            probability_col="__win_probability",
            season_col=season_col,
        ),
        len(evaluated),
    )


def backtest_optional_confidence_feature_groups(
    results: pd.DataFrame,
    *,
    baseline_spread_features: Sequence[str],
    baseline_total_features: Sequence[str],
    spread_cat_features: Sequence[str],
    total_cat_features: Sequence[str],
    feature_groups: Mapping[str, Mapping[str, Sequence[str]]],
    param_grid: Mapping[str, Sequence[Any]],
    time_col: str = "date_time",
    season_col: str = "season",
    test_size: float = 0.2,
    inner_validation_size: float = 0.2,
    n_jobs: int = 1,
) -> dict[str, dict[str, Any]]:
    """Backtest optional groups on a later slice of outer-OOS score predictions.

    The supplied results must already be predictions from the score model's
    chronological outer holdout. Classifiers tune only on the earlier portion
    of those rows and are evaluated on the untouched later portion.
    """
    if set(feature_groups) != {"spread", "total"}:
        raise ValueError("Feature groups must define spread and total markets")
    spread_group_names = set(feature_groups["spread"])
    total_group_names = set(feature_groups["total"])
    if spread_group_names != total_group_names:
        raise ValueError("Spread and total optional feature groups must have matching names")

    development, validation = chronological_train_test_split(
        results,
        time_col=time_col,
        test_size=test_size,
    )
    baseline_spread_features = _ordered_union(baseline_spread_features)
    baseline_total_features = _ordered_union(baseline_total_features)
    baseline_spread_clf, baseline_total_clf = fit_classifiers(
        development,
        baseline_spread_features,
        baseline_total_features,
        list(spread_cat_features),
        list(total_cat_features),
        dict(param_grid),
        time_col=time_col,
        validation_size=inner_validation_size,
        n_jobs=n_jobs,
        scoring="neg_log_loss",
    )
    baseline_spread, spread_rows = _evaluate_classifier(
        validation,
        classifier=baseline_spread_clf,
        features=baseline_spread_features,
        target_col="spread_win",
        season_col=season_col,
    )
    baseline_total, total_rows = _evaluate_classifier(
        validation,
        classifier=baseline_total_clf,
        features=baseline_total_features,
        target_col="total_win",
        season_col=season_col,
    )

    report: dict[str, dict[str, Any]] = {}
    for group_name in sorted(spread_group_names):
        candidate_spread_features = _ordered_union(
            baseline_spread_features,
            feature_groups["spread"][group_name],
        )
        candidate_total_features = _ordered_union(
            baseline_total_features,
            feature_groups["total"][group_name],
        )
        spread_clf, total_clf = fit_classifiers(
            development,
            candidate_spread_features,
            candidate_total_features,
            list(spread_cat_features),
            list(total_cat_features),
            dict(param_grid),
            time_col=time_col,
            validation_size=inner_validation_size,
            n_jobs=n_jobs,
            scoring="neg_log_loss",
        )
        candidate_spread, candidate_spread_rows = _evaluate_classifier(
            validation,
            classifier=spread_clf,
            features=candidate_spread_features,
            target_col="spread_win",
            season_col=season_col,
        )
        candidate_total, candidate_total_rows = _evaluate_classifier(
            validation,
            classifier=total_clf,
            features=candidate_total_features,
            target_col="total_win",
            season_col=season_col,
        )
        if candidate_spread_rows != spread_rows or candidate_total_rows != total_rows:
            raise ValueError("Baseline and candidate evaluations must use identical rows")
        report[group_name] = {
            "spread": {
                "passes": passes_optional_feature_gate(
                    baseline_spread,
                    candidate_spread,
                ),
                "rows": spread_rows,
                "baseline": asdict(baseline_spread),
                "candidate": asdict(candidate_spread),
            },
            "total": {
                "passes": passes_optional_feature_gate(
                    baseline_total,
                    candidate_total,
                ),
                "rows": total_rows,
                "baseline": asdict(baseline_total),
                "candidate": asdict(candidate_total),
            },
        }
    return report


__all__ = [
    "ProbabilityEvaluation",
    "backtest_optional_confidence_feature_groups",
    "evaluate_probabilities",
    "passes_optional_feature_gate",
]
