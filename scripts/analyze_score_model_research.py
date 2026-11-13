"""Summarize completed score-estimator experiments without choosing a winner."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_patterns.expected_points.backtest_artifacts import load_backtest_run


def correction_diagnostics(baseline, candidate):
    keys = ["season", "week", "game_id"]
    left = baseline.set_index(keys).sort_index()
    right = candidate.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("Different evaluation populations")
    rows = []
    for season in [None, *sorted(left.index.get_level_values("season").unique())]:
        a = left if season is None else left.xs(season, level="season")
        b = right if season is None else right.xs(season, level="season")
        actual = np.concatenate([a[f"{s}_score"].to_numpy(float) for s in ("home", "away")])
        control = np.concatenate([a[f"{s}_score_pred"].to_numpy(float) for s in ("home", "away")])
        challenger = np.concatenate([b[f"{s}_score_pred"].to_numpy(float) for s in ("home", "away")])
        delta = challenger - control
        error = actual - control
        rows.append({
            "season": "all" if season is None else int(season), "team_scores": len(actual),
            "mean_absolute_change": float(np.mean(np.abs(delta))),
            "correction_alignment": float(2 * np.mean(error * delta)),
            "correction_size_cost": float(np.mean(delta ** 2)),
            "mse_improvement": float(np.mean(error ** 2 - (actual - challenger) ** 2)),
            "baseline_bias_prediction_minus_actual": float(np.mean(control - actual)),
            "candidate_bias_prediction_minus_actual": float(np.mean(challenger - actual)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    out = args.study / "analysis"
    out.mkdir(exist_ok=True)
    metrics, corrections, tuning, buckets, ledger = [], [], [], [], []
    for directory in sorted(args.study.iterdir()):
        if not (directory / "request.json").exists():
            continue
        request = json.loads((directory / "request.json").read_text())
        status = json.loads((directory / "status.json").read_text())
        league = request["arguments"]["league"]
        variant = request["arguments"]["variant"]
        books = not request["arguments"]["no_sportsbook"]
        common = {"run": directory.name, "league": league, "variant": variant, "sportsbook": books}
        script = {"legacy-tree": "experiment_legacy_score_tree.py",
                  "residual": "experiment_residual_score_tree.py"}.get(variant, "experiment_score_models.py")
        command = [".venv/bin/python", f"scripts/{script}", "--league", league]
        if variant not in {"legacy-tree", "residual"}:
            command += ["--variant", variant]
        for argument in ["frame", "baseline", "output", "seasons"]:
            command += [f"--{argument}", str(request["arguments"][argument])]
        if not books:
            command.append("--no-sportsbook")
        entry = {**common, "status": status["status"], "output": str(directory),
                 "source_sha256": request["research_sha256"], "profile": "standard",
                 "seasons": request["arguments"]["seasons"],
                 "command": shlex.join(command),
                 "settings": json.dumps(request["config"]["score_param_grid"], sort_keys=True)}
        ledger.append(entry)
        if status["status"] != "completed":
            continue
        delta = pd.read_json(directory / "deltas.json")
        headline = delta.loc[(delta.cut_type == "overall") & (delta.metric == "score_mae")].iloc[0]
        entry.update(candidate_score_mae=headline.candidate, score_mae_improvement=headline.improvement,
                     improvement_ci_low=headline.improvement_ci_low,
                     improvement_ci_high=headline.improvement_ci_high)
        for row in delta.loc[delta.cut_type.isin(["overall", "season"])].to_dict("records"):
            metrics.append({**common, **row})
        base = load_backtest_run(request["arguments"]["baseline"])
        candidate = load_backtest_run(directory / "candidate")
        for row in correction_diagnostics(base.predictions, candidate.predictions):
            corrections.append({**common, **row})
        choices = json.loads((directory / "tuning.json").read_text())
        counts = Counter(json.dumps(entry["best_params"], sort_keys=True) for entry in choices)
        for parameters, count in counts.items():
            tuning.append({**common, "parameters": parameters, "cutoffs": count,
                           "fraction": count / len(choices)})
        # Fixed, interpretable implied-score bands, never chosen from outcomes.
        actual, predicted, implied = [], [], []
        for side in ("home", "away"):
            actual.extend(candidate.predictions[f"{side}_score"])
            predicted.extend(candidate.predictions[f"{side}_score_pred"])
            implied.extend(candidate.predictions[f"implied_points_{side}"])
        sample = pd.DataFrame({"actual": actual, "predicted": predicted, "implied": implied})
        sample["band"] = pd.cut(sample.implied, [-np.inf, 14, 21, 28, 35, np.inf], right=False)
        for band, group in sample.groupby("band", observed=True):
            buckets.append({**common, "implied_band": str(band), "team_scores": len(group),
                            "actual_mean": group.actual.mean(), "prediction_mean": group.predicted.mean(),
                            "market_mean": group.implied.mean(),
                            "bias": (group.predicted - group.actual).mean(),
                            "mae": (group.predicted - group.actual).abs().mean()})
    pd.DataFrame(ledger).to_csv(args.study / "runs.csv", index=False)
    for name, rows in [("metrics", metrics), ("corrections", corrections),
                       ("tuning", tuning), ("implied-score-bands", buckets)]:
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    frame = pd.DataFrame(metrics)
    if frame.empty:
        return
    selected = frame.loc[(frame.cut_type == "overall") & frame.metric.isin(
        ["score_mae", "score_rmse", "margin_mae", "total_points_mae"])]
    print(selected[["run", "metric", "baseline", "candidate", "improvement",
                    "improvement_ci_low", "improvement_ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
