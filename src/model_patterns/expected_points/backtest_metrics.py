"""Metrics, uncertainty, and comparisons for expected-points backtests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .backtesting import BacktestComparison, BacktestRun
from .evaluation import prediction_metrics

_METRIC_DIRECTIONS = {
    "score_mae": "lower",
    "score_rmse": "lower",
    "score_bias": "closer_to_zero",
    "home_score_mae": "lower",
    "home_score_rmse": "lower",
    "home_score_bias": "closer_to_zero",
    "away_score_mae": "lower",
    "away_score_rmse": "lower",
    "away_score_bias": "closer_to_zero",
    "margin_mae": "lower",
    "margin_rmse": "lower",
    "margin_bias": "closer_to_zero",
    "total_points_mae": "lower",
    "total_points_rmse": "lower",
    "total_points_bias": "closer_to_zero",
    "spread_win_pct": "higher",
    "total_win_pct": "higher",
    "spread_lock_win_pct": "higher",
    "total_lock_win_pct": "higher",
    "spread_brier": "lower",
    "total_brier": "lower",
    "spread_log_loss": "lower",
    "total_log_loss": "lower",
    "spread_calibration_error": "lower",
    "total_calibration_error": "lower",
    "score_mae_advantage_vs_market": "higher",
    "margin_mae_advantage_vs_market": "higher",
    "total_points_mae_advantage_vs_market": "higher",
    "spread_lock_calibration_gap": "closer_to_zero",
    "total_lock_calibration_gap": "closer_to_zero",
}

_COUNT_METRICS = {"score_eval_rows", "evaluated_weeks"}
for _market in ("spread", "total"):
    _COUNT_METRICS.update({
        f"{_market}_wins",
        f"{_market}_losses",
        f"{_market}_pushes",
        f"{_market}_lock_wins",
        f"{_market}_lock_losses",
        f"{_market}_lock_pushes",
        f"{_market}_locks",
        f"{_market}_eligible",
    })

_RATIO_METRICS = {"spread_win_pct", "total_win_pct"}
for _market in ("spread", "total"):
    _RATIO_METRICS.update({
        f"{_market}_lock_win_pct",
        f"{_market}_lock_coverage_pct",
        f"{_market}_locks_per_week",
        f"{_market}_brier",
        f"{_market}_log_loss",
        f"{_market}_lock_mean_win_prob",
        f"{_market}_lock_calibration_gap",
    })

_ERROR_METRICS = {
    f"{prefix}_{suffix}"
    for prefix in ("score", "home_score", "away_score", "margin", "total_points")
    for suffix in ("mae", "rmse", "bias")
}
_MARKET_ERROR_METRICS = {
    "market_score_mae",
    "market_margin_mae",
    "market_total_points_mae",
    "score_mae_advantage_vs_market",
    "margin_mae_advantage_vs_market",
    "total_points_mae_advantage_vs_market",
}
_CALIBRATION_METRICS = {"spread_calibration_error", "total_calibration_error"}
_BOOTSTRAP_METRICS = tuple(sorted(
    _ERROR_METRICS
    | _MARKET_ERROR_METRICS
    | _COUNT_METRICS
    | _RATIO_METRICS
    | _CALIBRATION_METRICS
))

def add_season_phase(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["season_phase"] = ""
    labels = ("early", "middle", "late")
    for season, group in output.groupby("season", sort=False):
        weeks = (
            group.groupby("week")["date_time"].min().sort_values(kind="stable").index.tolist()
        )
        for label, phase_weeks in zip(labels, np.array_split(np.asarray(weeks, dtype=object), 3)):
            output.loc[
                (output["season"] == season) & output["week"].isin(phase_weeks.tolist()),
                "season_phase",
            ] = label
    return output


def summarize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append(cut_type: str, cut_value: str, group: pd.DataFrame) -> None:
        for metric, value in prediction_metrics(group).items():
            rows.append({
                "cut_type": cut_type,
                "cut_value": cut_value,
                "metric": metric,
                "value": value,
                "games": len(group),
            })

    append("overall", "all", frame)
    latest_season = int(pd.to_numeric(frame["season"]).max())
    append(
        "latest_season",
        str(latest_season),
        frame.loc[pd.to_numeric(frame["season"]) == latest_season],
    )
    for season, group in frame.groupby("season", sort=True):
        append("season", str(int(season)), group)
    for phase, group in frame.groupby("season_phase", sort=False):
        append("season_phase", str(phase), group)
    for (season, week), group in frame.groupby(["season", "week"], sort=True):
        append("week", f"{int(season)}_{week}", group)
    return pd.DataFrame(rows)


def _filter_cut(frame: pd.DataFrame, cut_type: str, cut_value: str) -> pd.DataFrame:
    if cut_type == "overall":
        return frame
    if cut_type == "season":
        return frame.loc[frame["season"].astype(int) == int(cut_value)]
    if cut_type == "latest_season":
        return frame.loc[frame["season"].astype(int) == int(cut_value)]
    if cut_type == "season_phase":
        return frame.loc[frame["season_phase"] == cut_value]
    season, week = cut_value.split("_", 1)
    return frame.loc[(frame["season"].astype(int) == int(season)) & (frame["week"].astype(str) == week)]


def _bootstrap_intervals(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    metrics: tuple[str, ...],
    *,
    samples: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    empty = {metric: (float("nan"), float("nan")) for metric in metrics}
    if samples <= 0:
        return empty
    baseline = baseline.assign(__period=baseline["season"].astype(str) + "_" + baseline["week"].astype(str))
    candidate = candidate.assign(__period=candidate["season"].astype(str) + "_" + candidate["week"].astype(str))
    periods = sorted(set(baseline["__period"]) & set(candidate["__period"]))
    if len(periods) < 2:
        return empty
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(periods), size=(samples, len(periods)))
    baseline_components = _period_metric_components(baseline, periods)
    candidate_components = _period_metric_components(candidate, periods)
    output = {}
    for metric in metrics:
        baseline_values = _bootstrap_metric_values(baseline_components[metric], selected)
        candidate_values = _bootstrap_metric_values(candidate_components[metric], selected)
        valid = np.isfinite(baseline_values) & np.isfinite(candidate_values)
        if not valid.any():
            output[metric] = (float("nan"), float("nan"))
            continue
        baseline_values = baseline_values[valid]
        candidate_values = candidate_values[valid]
        improvement = _metric_improvement(
            baseline_values,
            candidate_values,
            _METRIC_DIRECTIONS.get(metric),
        )
        output[metric] = tuple(
            float(value) for value in np.quantile(improvement, [0.025, 0.975])
        )
    return output


def bootstrap_metric_intervals(
    frame: pd.DataFrame,
    metrics: tuple[str, ...] | None = None,
    *,
    samples: int = 2000,
    seed: int = 31,
) -> dict[str, tuple[float, float]]:
    """Return absolute season-week block-bootstrap intervals for one run."""
    selected_metrics = metrics or _BOOTSTRAP_METRICS
    empty = {
        metric: (float("nan"), float("nan")) for metric in selected_metrics
    }
    if samples <= 0:
        return empty
    prepared = frame.assign(
        __period=frame["season"].astype(str) + "_" + frame["week"].astype(str)
    )
    periods = sorted(prepared["__period"].dropna().unique().tolist())
    if len(periods) < 2:
        return empty
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(periods), size=(samples, len(periods)))
    components = _period_metric_components(prepared, periods)
    output = {}
    for metric in selected_metrics:
        values = _bootstrap_metric_values(components[metric], selected)
        values = values[np.isfinite(values)]
        output[metric] = (
            tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))
            if len(values)
            else (float("nan"), float("nan"))
        )
    return output


def _period_metric_components(
    frame: pd.DataFrame,
    periods: list[str],
) -> dict[str, tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None]]:
    """Return sufficient statistics for a fast season-week block bootstrap."""
    size = len(periods)
    components: dict[str, tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None]] = {}
    for metric in _ERROR_METRICS:
        kind = "rmse" if metric.endswith("_rmse") else "mean"
        components[metric] = (kind, np.zeros(size), np.zeros(size), None)
    for metric in _MARKET_ERROR_METRICS:
        components[metric] = ("mean", np.zeros(size), np.zeros(size), None)
    for metric in _COUNT_METRICS:
        components[metric] = ("sum", np.zeros(size), None, None)
    for metric in _RATIO_METRICS:
        components[metric] = ("mean", np.zeros(size), np.zeros(size), None)
    for metric in _CALIBRATION_METRICS:
        components[metric] = (
            "calibration",
            np.zeros((size, 10)),
            np.zeros((size, 10)),
            np.zeros((size, 10)),
        )

    def set_mean(metric: str, index: int, values: np.ndarray, *, squared: bool = False) -> None:
        finite = np.isfinite(values)
        selected_values = values[finite]
        numerator = components[metric][1]
        denominator = components[metric][2]
        numerator[index] = np.square(selected_values).sum() if squared else selected_values.sum()
        assert denominator is not None
        denominator[index] = finite.sum()

    def set_sum(metric: str, index: int, value: float) -> None:
        components[metric][1][index] = value

    for index, period in enumerate(periods):
        group = frame.loc[frame["__period"] == period]
        home_actual = pd.to_numeric(group["home_score"], errors="coerce").to_numpy(float)
        away_actual = pd.to_numeric(group["away_score"], errors="coerce").to_numpy(float)
        home_pred = pd.to_numeric(group["home_score_pred"], errors="coerce").to_numpy(float)
        away_pred = pd.to_numeric(group["away_score_pred"], errors="coerce").to_numpy(float)
        residuals = {
            "home_score": home_pred - home_actual,
            "away_score": away_pred - away_actual,
            "margin": (away_pred - home_pred) - (away_actual - home_actual),
            "total_points": (away_pred + home_pred) - (away_actual + home_actual),
        }
        residuals["score"] = np.concatenate([
            residuals["home_score"], residuals["away_score"]
        ])
        for prefix, values in residuals.items():
            set_mean(f"{prefix}_mae", index, np.abs(values))
            set_mean(f"{prefix}_rmse", index, values, squared=True)
            set_mean(f"{prefix}_bias", index, values)
        if {"implied_points_home", "implied_points_away"}.issubset(group.columns):
            market_home = pd.to_numeric(
                group["implied_points_home"], errors="coerce"
            ).to_numpy(float)
            market_away = pd.to_numeric(
                group["implied_points_away"], errors="coerce"
            ).to_numpy(float)
            market_score_residual = np.concatenate([
                market_home - home_actual,
                market_away - away_actual,
            ])
            set_mean("market_score_mae", index, np.abs(market_score_residual))
            set_mean(
                "score_mae_advantage_vs_market",
                index,
                np.abs(market_score_residual) - np.abs(residuals["score"]),
            )

            spread_column = (
                "spread_reference_line"
                if "spread_reference_line" in group
                else "spread_line"
            )
            if spread_column in group:
                market_margin = pd.to_numeric(
                    group[spread_column], errors="coerce"
                ).to_numpy(float)
                market_margin_residual = market_margin - (away_actual - home_actual)
                set_mean("market_margin_mae", index, np.abs(market_margin_residual))
                set_mean(
                    "margin_mae_advantage_vs_market",
                    index,
                    np.abs(market_margin_residual) - np.abs(residuals["margin"]),
                )

            total_column = (
                "total_reference_line"
                if "total_reference_line" in group
                else "total_line"
            )
            if total_column in group:
                market_total = pd.to_numeric(
                    group[total_column], errors="coerce"
                ).to_numpy(float)
                market_total_residual = market_total - (away_actual + home_actual)
                set_mean(
                    "market_total_points_mae",
                    index,
                    np.abs(market_total_residual),
                )
                set_mean(
                    "total_points_mae_advantage_vs_market",
                    index,
                    np.abs(market_total_residual) - np.abs(residuals["total_points"]),
                )
        complete_scores = (
            np.isfinite(home_actual)
            & np.isfinite(away_actual)
            & np.isfinite(home_pred)
            & np.isfinite(away_pred)
        )
        set_sum("score_eval_rows", index, float(complete_scores.sum()))
        set_sum("evaluated_weeks", index, 1.0)

        for market in ("spread", "total"):
            supported_col = f"{market}_market_supported"
            supported = (
                group[supported_col].fillna(False).astype(bool).to_numpy()
                if supported_col in group else np.ones(len(group), dtype=bool)
            )
            outcomes = pd.to_numeric(group[f"{market}_win"], errors="coerce").to_numpy(float)
            decision = supported & np.isfinite(outcomes)
            locks = pd.to_numeric(
                group.get(f"{market}_lock", pd.Series(0, index=group.index)), errors="coerce"
            ).fillna(0).to_numpy() == 1
            locks &= supported
            lock_decision = decision & locks

            set_sum(f"{market}_wins", index, float((outcomes[decision] == 1).sum()))
            set_sum(f"{market}_losses", index, float((outcomes[decision] == 0).sum()))
            set_sum(f"{market}_pushes", index, float((supported & ~np.isfinite(outcomes)).sum()))
            set_sum(f"{market}_lock_wins", index, float((outcomes[lock_decision] == 1).sum()))
            set_sum(f"{market}_lock_losses", index, float((outcomes[lock_decision] == 0).sum()))
            set_sum(
                f"{market}_lock_pushes",
                index,
                float((locks & ~np.isfinite(outcomes)).sum()),
            )
            set_sum(f"{market}_locks", index, float(locks.sum()))
            set_sum(f"{market}_eligible", index, float(supported.sum()))

            for metric, numerator_value, denominator_value in (
                (f"{market}_win_pct", outcomes[decision].sum() * 100.0, decision.sum()),
                (
                    f"{market}_lock_win_pct",
                    outcomes[lock_decision].sum() * 100.0,
                    lock_decision.sum(),
                ),
                (f"{market}_lock_coverage_pct", locks.sum() * 100.0, supported.sum()),
                (f"{market}_locks_per_week", locks.sum(), 1.0),
            ):
                components[metric][1][index] = numerator_value
                assert components[metric][2] is not None
                components[metric][2][index] = denominator_value

            probabilities = pd.to_numeric(
                group[f"{market}_win_prob"], errors="coerce"
            ).to_numpy(float) / 100.0
            probability_rows = decision & np.isfinite(probabilities)
            clipped = np.clip(probabilities[probability_rows], 1e-9, 1 - 1e-9)
            observed = outcomes[probability_rows]
            lock_probability_rows = lock_decision & np.isfinite(probabilities)
            lock_probabilities = probabilities[lock_probability_rows] * 100.0
            lock_outcomes = outcomes[lock_probability_rows] * 100.0
            for metric, numerator_value in (
                (f"{market}_lock_mean_win_prob", lock_probabilities.sum()),
                (
                    f"{market}_lock_calibration_gap",
                    (lock_probabilities - lock_outcomes).sum(),
                ),
            ):
                components[metric][1][index] = numerator_value
                assert components[metric][2] is not None
                components[metric][2][index] = len(lock_probabilities)
            brier_metric = f"{market}_brier"
            log_metric = f"{market}_log_loss"
            components[brier_metric][1][index] = np.square(clipped - observed).sum()
            assert components[brier_metric][2] is not None
            components[brier_metric][2][index] = len(observed)
            components[log_metric][1][index] = (
                -(observed * np.log(clipped) + (1 - observed) * np.log(1 - clipped)).sum()
            )
            assert components[log_metric][2] is not None
            components[log_metric][2][index] = len(observed)

            calibration = components[f"{market}_calibration_error"]
            bin_count, probability_sum, outcome_sum = calibration[1:]
            assert probability_sum is not None and outcome_sum is not None
            bin_indexes = np.searchsorted(
                np.linspace(0.0, 1.0, 11)[1:], clipped, side="left"
            )
            for bin_index in range(10):
                in_bin = bin_indexes == bin_index
                bin_count[index, bin_index] = in_bin.sum()
                probability_sum[index, bin_index] = clipped[in_bin].sum()
                outcome_sum[index, bin_index] = observed[in_bin].sum()
    return components


def _bootstrap_metric_values(
    component: tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None],
    selected: np.ndarray,
) -> np.ndarray:
    kind, numerator, denominator, extra = component
    if kind == "sum":
        return numerator[selected].sum(axis=1)
    if kind in {"mean", "rmse"}:
        assert denominator is not None
        totals = numerator[selected].sum(axis=1)
        counts = denominator[selected].sum(axis=1)
        values = np.divide(
            totals,
            counts,
            out=np.full(len(selected), np.nan),
            where=counts > 0,
        )
        return np.sqrt(values) if kind == "rmse" else values
    if kind == "calibration":
        assert denominator is not None and extra is not None
        counts = numerator[selected].sum(axis=1)
        probability_sums = denominator[selected].sum(axis=1)
        outcome_sums = extra[selected].sum(axis=1)
        total_counts = counts.sum(axis=1)
        probability_means = np.divide(
            probability_sums,
            counts,
            out=np.zeros_like(probability_sums),
            where=counts > 0,
        )
        outcome_means = np.divide(
            outcome_sums,
            counts,
            out=np.zeros_like(outcome_sums),
            where=counts > 0,
        )
        weighted_error = (counts * np.abs(probability_means - outcome_means)).sum(axis=1)
        return np.divide(
            weighted_error,
            total_counts,
            out=np.full(len(selected), np.nan),
            where=total_counts > 0,
        )
    raise ValueError(f"Unsupported bootstrap component kind: {kind}")


def _metric_improvement(
    baseline: np.ndarray | float,
    candidate: np.ndarray | float,
    direction: str | None,
) -> np.ndarray | float:
    if direction == "lower":
        return np.asarray(baseline) - np.asarray(candidate)
    if direction == "closer_to_zero":
        return np.abs(np.asarray(baseline)) - np.abs(np.asarray(candidate))
    return np.asarray(candidate) - np.asarray(baseline)


def compare_backtest_runs(
    baseline: BacktestRun,
    candidate: BacktestRun,
    *,
    bootstrap_samples: int = 2000,
    random_seed: int = 31,
) -> BacktestComparison:
    if baseline.league is not candidate.league:
        raise ValueError("Backtest comparison requires the same league")
    if baseline.source_fingerprint != candidate.source_fingerprint:
        raise ValueError("Backtest comparison requires compatible source bundles and evaluation universes")
    if baseline.profile != candidate.profile or baseline.seasons != candidate.seasons:
        raise ValueError("Backtest comparison requires the same profile and evaluated seasons")
    keys = ["season", "week", "game_id"]
    baseline_common = _canonicalize_evaluation_keys(baseline.predictions, keys)
    candidate_common = _canonicalize_evaluation_keys(candidate.predictions, keys)
    baseline_keys = pd.MultiIndex.from_frame(baseline_common[keys].sort_values(keys))
    candidate_keys = pd.MultiIndex.from_frame(candidate_common[keys].sort_values(keys))
    if not baseline_keys.equals(candidate_keys):
        raise ValueError("Backtest comparison requires identical evaluated game universes")
    if baseline_common.empty:
        raise ValueError("Baseline and candidate have no evaluated games")

    baseline_summary = summarize_predictions(baseline_common).assign(recipe="baseline")
    candidate_summary = summarize_predictions(candidate_common).assign(recipe="candidate")
    summary = pd.concat([baseline_summary, candidate_summary], ignore_index=True)
    merged = baseline_summary.merge(
        candidate_summary,
        on=["cut_type", "cut_value", "metric"],
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    interval_cache: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    delta_rows = []
    for row in merged.itertuples(index=False):
        direction = _METRIC_DIRECTIONS.get(row.metric)
        # Identically unavailable metrics (for example a one-game week's AUC
        # or a slate with no locks) are the same result, so their comparison
        # delta is zero even though both absolute values remain undefined.
        both_undefined = pd.isna(row.value_baseline) and pd.isna(row.value_candidate)
        if both_undefined:
            raw_delta = improvement = 0.0
        else:
            raw_delta = float(row.value_candidate - row.value_baseline)
            improvement = float(_metric_improvement(
                row.value_baseline,
                row.value_candidate,
                direction,
            ))
        lower = upper = float("nan")
        status = "informational"
        if row.metric in _BOOTSTRAP_METRICS:
            cut_key = (row.cut_type, row.cut_value)
            if cut_key not in interval_cache:
                baseline_cut = _filter_cut(baseline_common, row.cut_type, row.cut_value)
                candidate_cut = _filter_cut(candidate_common, row.cut_type, row.cut_value)
                interval_cache[cut_key] = _bootstrap_intervals(
                    baseline_cut,
                    candidate_cut,
                    tuple(_BOOTSTRAP_METRICS),
                    samples=bootstrap_samples,
                    seed=random_seed,
                )
            lower, upper = interval_cache[cut_key][row.metric]
            if direction is not None and not np.isnan(lower):
                status = "improvement" if lower > 0 else ("regression" if upper < 0 else "inconclusive")
            elif direction is not None:
                status = "inconclusive"
        delta_rows.append({
            "cut_type": row.cut_type,
            "cut_value": row.cut_value,
            "metric": row.metric,
            "baseline": row.value_baseline,
            "candidate": row.value_candidate,
            "raw_delta": raw_delta,
            "improvement": improvement,
            "improvement_ci_low": lower,
            "improvement_ci_high": upper,
            "status": status,
            "games": min(row.games_baseline, row.games_candidate),
        })

    trace_columns = [
        "spread_lock", "total_lock", "spread_play", "total_play",
        "spread_win_prob", "total_win_prob", "spread_diff", "total_diff",
    ]
    base_locks = baseline_common[keys + trace_columns]
    cand_locks = candidate_common[keys + trace_columns]
    lock_changes = base_locks.merge(cand_locks, on=keys, suffixes=("_baseline", "_candidate"))
    for market in ("spread", "total"):
        baseline_lock = lock_changes[f"{market}_lock_baseline"].fillna(0).astype(bool)
        candidate_lock = lock_changes[f"{market}_lock_candidate"].fillna(0).astype(bool)
        lock_changes[f"{market}_lock_change"] = np.select(
            [baseline_lock & candidate_lock, baseline_lock & ~candidate_lock, ~baseline_lock & candidate_lock],
            ["common", "baseline_only", "candidate_only"],
            default="neither",
        )
        lock_changes[f"{market}_direction_changed"] = (
            lock_changes[f"{market}_play_baseline"].fillna("")
            != lock_changes[f"{market}_play_candidate"].fillna("")
        )
        lock_changes[f"{market}_confidence_delta"] = (
            lock_changes[f"{market}_win_prob_candidate"]
            - lock_changes[f"{market}_win_prob_baseline"]
        )
        lock_changes[f"{market}_edge_delta"] = (
            lock_changes[f"{market}_diff_candidate"]
            - lock_changes[f"{market}_diff_baseline"]
        )
    lock_changes = lock_changes.loc[
        (lock_changes["spread_lock_change"] != "neither")
        | (lock_changes["total_lock_change"] != "neither")
        | lock_changes["spread_direction_changed"]
        | lock_changes["total_direction_changed"]
    ].reset_index(drop=True)
    return BacktestComparison(
        baseline=baseline,
        candidate=candidate,
        summary=summary,
        deltas=pd.DataFrame(delta_rows),
        lock_changes=lock_changes,
    )


def _canonicalize_evaluation_keys(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise ValueError(f"Backtest predictions are missing evaluation keys: {', '.join(missing)}")
    output = frame.copy()
    output["season"] = pd.to_numeric(output["season"], errors="raise").astype(int)
    output["week"] = output["week"].astype(str)
    output["game_id"] = output["game_id"].astype(str)
    if output.duplicated(keys).any():
        raise ValueError("Backtest predictions require unique season/week/game_id keys")
    return output



__all__ = [
    "add_season_phase",
    "bootstrap_metric_intervals",
    "compare_backtest_runs",
    "summarize_predictions",
]
