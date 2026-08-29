import numpy as np
import pandas as pd
import pytest

from src.sports.football.cfb.expected_points import confidence_gate
from src.sports.football.cfb.expected_points.confidence_gate import (
    backtest_optional_confidence_feature_groups,
    evaluate_probabilities,
    passes_optional_feature_gate,
)


def probabilities(values):
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2025, 2025],
            "won": [1, 0, 1, 0],
            "probability": values,
        }
    )


def test_probability_evaluation_reports_seasons_and_calibration():
    evaluation = evaluate_probabilities(
        probabilities([0.8, 0.2, 0.7, 0.3]),
        target_col="won",
        probability_col="probability",
    )

    assert evaluation.brier_score == pytest.approx(0.065)
    assert evaluation.expected_calibration_error == pytest.approx(0.25)
    assert set(evaluation.brier_by_season) == {2024, 2025}
    assert sum(bucket["count"] for bucket in evaluation.calibration) == 4


def test_optional_feature_gate_requires_every_documented_condition():
    baseline = evaluate_probabilities(
        probabilities([0.7, 0.3, 0.7, 0.3]),
        target_col="won",
        probability_col="probability",
    )
    candidate = evaluate_probabilities(
        probabilities([0.75, 0.25, 0.72, 0.28]),
        target_col="won",
        probability_col="probability",
    )
    worse_season = evaluate_probabilities(
        probabilities([0.9, 0.1, 0.6, 0.4]),
        target_col="won",
        probability_col="probability",
    )

    assert passes_optional_feature_gate(baseline, candidate) is True
    assert passes_optional_feature_gate(baseline, worse_season) is False


def test_feature_group_backtest_uses_a_later_untouched_probability_slice(monkeypatch):
    class FeatureProbabilityClassifier:
        classes_ = np.array([0, 1])

        def __init__(self, features):
            self.features = features

        def predict_proba(self, frame):
            optional = [feature for feature in self.features if feature.startswith("optional_")]
            probability_col = optional[0] if optional else "baseline_probability"
            win_probability = frame[probability_col].to_numpy()
            return np.column_stack([1 - win_probability, win_probability])

    def fake_fit_classifiers(
        results,
        spread_features,
        total_features,
        spread_cat_features,
        total_cat_features,
        param_grid,
        **kwargs,
    ):
        return (
            FeatureProbabilityClassifier(spread_features),
            FeatureProbabilityClassifier(total_features),
        )

    monkeypatch.setattr(confidence_gate, "fit_classifiers", fake_fit_classifiers)
    dates = pd.date_range("2024-09-01", periods=40, freq="7D")
    wins = np.tile([0, 1], 20)
    frame = pd.DataFrame(
        {
            "date_time": dates.strftime("%Y-%m-%d-%H:%M"),
            "season": dates.year,
            "spread_win": wins,
            "total_win": wins,
            "baseline_probability": np.where(wins == 1, 0.65, 0.35),
            "optional_spread": np.where(wins == 1, 0.85, 0.15),
            "optional_total": np.where(wins == 1, 0.85, 0.15),
        }
    )

    report = backtest_optional_confidence_feature_groups(
        frame,
        baseline_spread_features=["baseline_probability"],
        baseline_total_features=["baseline_probability"],
        spread_cat_features=[],
        total_cat_features=[],
        feature_groups={
            "spread": {"signal": ("optional_spread",)},
            "total": {"signal": ("optional_total",)},
        },
        param_grid={},
        n_jobs=1,
    )

    assert report["signal"]["spread"]["passes"] is True
    assert report["signal"]["total"]["passes"] is True
    assert report["signal"]["spread"]["rows"] < len(frame)
    assert (
        report["signal"]["spread"]["candidate"]["brier_score"]
        < report["signal"]["spread"]["baseline"]["brier_score"]
    )
