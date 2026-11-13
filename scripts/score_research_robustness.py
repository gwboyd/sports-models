"""Paired year-sensitivity diagnostics, supplemental to standard backtest reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_patterns.expected_points.backtest_artifacts import load_backtest_run, write_json_atomic


def paired_year_sensitivity(baseline, candidate, *, samples=10000, seed=3118):
    keys = ["season", "week", "game_id"]
    left = baseline.set_index(keys).sort_index()
    right = candidate.set_index(keys).sort_index()
    if not left.index.equals(right.index) or left.index.has_duplicates:
        raise ValueError("Require identical unique paired games")
    sums = np.zeros(len(left))
    for side in ("home", "away"):
        actual = left[f"{side}_score"].to_numpy(float)
        np.testing.assert_array_equal(actual, right[f"{side}_score"].to_numpy(float))
        sums += np.abs(actual - left[f"{side}_score_pred"]) - np.abs(actual - right[f"{side}_score_pred"])
    games = pd.DataFrame({"sum": sums, "count": 2}, index=left.index).reset_index()
    weekly = games.groupby(["season", "week"])[["sum", "count"]].sum()
    yearly = weekly.groupby("season").sum()
    rng = np.random.default_rng(seed)
    blocks = [group.to_numpy(float) for _, group in weekly.groupby(level="season")]
    draws = np.zeros(samples)
    for index in range(samples):
        total = np.zeros(2)
        for season in rng.integers(0, len(blocks), len(blocks)):
            weeks = blocks[season]
            total += weeks[rng.integers(0, len(weeks), len(weeks))].sum(axis=0)
        draws[index] = total[0] / total[1]
    loo = {}
    for season in yearly.index:
        remain = yearly.drop(season).sum()
        if remain["count"]:
            loo[str(season)] = float(remain["sum"] / remain["count"])
    return {
        "positive_favors": "candidate", "metric": "team_score_mae_improvement",
        "games": len(left), "seasons": list(map(int, yearly.index)),
        "improvement": float(yearly["sum"].sum() / yearly["count"].sum()),
        "hierarchical_95_ci": np.quantile(draws, [.025, .975]).tolist(),
        "per_season_improvement": (yearly["sum"] / yearly["count"]).to_dict(),
        "leave_one_season_out_improvement": loo, "bootstrap_samples": samples, "seed": seed,
        "limitation": "Already researched seasons; this is not untouched confirmation or correction for model selection.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--legacy-comparison", action="append", default=[], metavar="LEAGUE=PATH",
                        help="Optional prior ridge/tree comparison to include and pool with a matching earlier run")
    args = parser.parse_args()
    results = {}
    for directory in sorted(args.study.iterdir()):
        if not (directory / "request.json").exists():
            continue
        status = json.loads((directory / "status.json").read_text())
        if status["status"] != "completed":
            continue
        request = json.loads((directory / "request.json").read_text())
        if request["arguments"]["variant"] == "ridge":
            continue
        base = load_backtest_run(request["arguments"]["baseline"])
        candidate = load_backtest_run(directory / "candidate")
        results[directory.name] = paired_year_sensitivity(base.predictions, candidate.predictions)
    for reference in args.legacy_comparison:
        league, separator, path = reference.partition("=")
        if not separator or league not in {"nfl", "cfb"}:
            parser.error("--legacy-comparison requires nfl=PATH or cfb=PATH")
        root = Path(path)
        base = pd.read_parquet(root / "baseline_predictions.parquet")
        tree = pd.read_parquet(root / "candidate_predictions.parquet")
        results[f"{league}-legacy-tree-recent"] = paired_year_sensitivity(base, tree)
        older = args.study / f"{league}-legacy-tree-earlier"
        if (older / "status.json").exists() and json.loads((older / "status.json").read_text())["status"] == "completed":
            req = json.loads((older / "request.json").read_text())
            older_base = load_backtest_run(req["arguments"]["baseline"]).predictions
            older_tree = load_backtest_run(older / "candidate").predictions
            results[f"{league}-legacy-tree-pooled"] = paired_year_sensitivity(
                pd.concat([older_base, base]), pd.concat([older_tree, tree]))
    write_json_atomic(args.study / "analysis/year-sensitivity.json", results)
    for name, result in results.items():
        print(name, round(result["improvement"], 4), result["hierarchical_95_ci"], flush=True)


if __name__ == "__main__":
    main()
