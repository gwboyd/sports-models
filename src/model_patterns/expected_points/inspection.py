"""Notebook-friendly inspection helpers for expected-points runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .pipeline import HomeAwayTransformer
from .types import ExpectedPointsConfig, ExpectedPointsRunResult


@dataclass(frozen=True)
class GameInspection:
    schedule: pd.DataFrame
    market: pd.DataFrame
    model_frame: pd.DataFrame
    score_features: pd.DataFrame
    spread_confidence_features: pd.DataFrame
    total_confidence_features: pd.DataFrame
    prediction: pd.DataFrame
    decision_thresholds: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "schedule": self.schedule,
            "market": self.market,
            "model_frame": self.model_frame,
            "score_features": self.score_features,
            "spread_confidence_features": self.spread_confidence_features,
            "total_confidence_features": self.total_confidence_features,
            "prediction": self.prediction,
            "decision_thresholds": self.decision_thresholds,
        }


def inspect_game(
    frame: pd.DataFrame,
    config: ExpectedPointsConfig,
    *,
    game_id: str,
    plays: pd.DataFrame | None = None,
    schedule: pd.DataFrame | None = None,
    market: pd.DataFrame | None = None,
) -> GameInspection:
    matches = frame.loc[frame["game_id"].astype(str) == str(game_id)].copy()
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one model row for game_id={game_id!r}; found {len(matches)}")
    raw = matches.iloc[[0]].copy()
    representations = []
    transformer = HomeAwayTransformer()
    for team in ("home", "away"):
        represented = raw.copy()
        represented["pred_team"] = team
        represented = transformer.transform(represented)
        selected = represented.reindex(columns=config.features).copy()
        selected.insert(0, "predicted_side", team)
        representations.append(selected)
    score_features = pd.concat(representations, ignore_index=True)

    prediction = pd.DataFrame()
    if plays is not None and not plays.empty:
        prediction = plays.loc[plays["game_id"].astype(str) == str(game_id)].copy()
    confidence_source = prediction if len(prediction) == 1 else raw
    spread = confidence_source.reindex(columns=config.spread_class_features).copy()
    total = confidence_source.reindex(columns=config.total_class_features).copy()
    if config.spread_lock_head is not None and config.total_lock_head is not None:
        from .lock_replay import build_lock_market_frame
        from .lock_features import lock_feature_set

        inspected = []
        for name, head in (("spread", config.spread_lock_head), ("total", config.total_lock_head)):
            if prediction.empty:
                inspected.append(pd.DataFrame([{"head": head.family, "status": "Prediction required to inspect Lock inputs"}]))
                continue
            # Tracking/persistence retains projections and execution lines but
            # drops transient difference columns. Reconstruct the exact input
            # on an inspection-only copy, leaving the caller's picks unchanged.
            lock_input = prediction.copy()
            lock_input[f"{name}_diff"] = (
                lock_input[f"{name}_pred"] - lock_input[f"{name}_line"]
            ).abs()
            features = build_lock_market_frame(lock_input, config.league, name, outcomes_known=False)
            columns = ("edge",) if head.family == "symmetric_residual" else (
                () if head.family == "base_rate" else lock_feature_set(config.league, head.feature_group, market=name).all
            )
            inputs = features.reindex(columns=columns).copy()
            inputs["head"] = head.family
            inputs["parameters"] = [dict(head.parameters)] * len(inputs)
            inputs["locks_configured"] = head.locks_enabled
            # A configured market may still be disabled by insufficient
            # training history. Missing runtime metadata is unknown, not ready.
            inputs["locks_enabled"] = (
                prediction[f"{name}_locks_enabled"].to_numpy()
                if f"{name}_locks_enabled" in prediction else pd.NA
            )
            inspected.append(inputs)
        spread, total = inspected
    schedule_row = _matching_game_row(schedule, game_id, raw)
    market_row = _matching_game_row(market, game_id, raw)
    thresholds = pd.DataFrame([asdict(config.play_thresholds)])
    return GameInspection(
        schedule_row, market_row, raw, score_features, spread, total, prediction, thresholds
    )


def _matching_game_row(
    frame: pd.DataFrame | None,
    game_id: str,
    model_row: pd.DataFrame,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    if "game_id" in frame:
        return frame.loc[frame["game_id"].astype(str) == str(game_id)].copy()
    if "id" in frame and "id" in model_row:
        identifier = str(model_row.iloc[0]["id"])
        return frame.loc[frame["id"].astype(str) == identifier].copy()
    return pd.DataFrame()


def inspect_team_history(
    frame: pd.DataFrame,
    *,
    team: str,
    before_game_id: str | None = None,
    time_col: str = "date_time",
) -> pd.DataFrame:
    history = frame.loc[
        (frame["home_team"] == team) | (frame["away_team"] == team)
    ].copy()
    if before_game_id is not None:
        target = frame.loc[frame["game_id"].astype(str) == str(before_game_id)]
        if len(target) != 1:
            raise ValueError(f"Expected one target game for {before_game_id!r}")
        cutoff = pd.to_datetime(target.iloc[0][time_col], errors="raise")
        parsed = pd.to_datetime(history[time_col], errors="coerce")
        history = history.loc[parsed < cutoff]
    return history.sort_values(time_col, kind="stable").reset_index(drop=True)


def health_metrics_frame(run: ExpectedPointsRunResult) -> pd.DataFrame:
    rows = []
    for key, value in sorted(run.metrics.items()):
        section = "health" if key.startswith("health_") else "score_holdout"
        metric = key.removeprefix("health_")
        rows.append({"section": section, "metric": metric, "value": value})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class LoadedBacktestComparison:
    path: Path
    summary: pd.DataFrame
    deltas: pd.DataFrame
    baseline_predictions: pd.DataFrame
    candidate_predictions: pd.DataFrame
    lock_changes: pd.DataFrame
    manifest: dict[str, Any]


def load_backtest_comparison(path: str | Path) -> LoadedBacktestComparison:
    import json

    from .backtest_artifacts import restore_json_columns

    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    summary = pd.read_json(root / "summary.json", orient="records")
    deltas_path = root / "deltas.json"
    deltas = pd.read_json(deltas_path, orient="records") if deltas_path.exists() else pd.DataFrame()
    return LoadedBacktestComparison(
        path=root,
        summary=summary,
        deltas=deltas,
        baseline_predictions=restore_json_columns(
            pd.read_parquet(root / "baseline_predictions.parquet"),
            manifest.get("baseline_json_columns", []),
        ),
        candidate_predictions=restore_json_columns(
            pd.read_parquet(root / "candidate_predictions.parquet"),
            manifest.get("candidate_json_columns", []),
        ),
        lock_changes=restore_json_columns(
            pd.read_parquet(root / "lock_changes.parquet"),
            manifest.get("lock_change_json_columns", []),
        ),
        manifest=manifest,
    )


__all__ = [
    "GameInspection",
    "LoadedBacktestComparison",
    "health_metrics_frame",
    "inspect_game",
    "inspect_team_history",
    "load_backtest_comparison",
]
