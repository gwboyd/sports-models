"""Retest the original direct LightGBM challenger on explicit frozen populations."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiment_score_models import ResearchRecipe, research_factory
from src.model_patterns.expected_points import trainer
from src.model_patterns.expected_points.backtest_artifacts import (
    load_backtest_frame, load_backtest_run, write_comparison_artifacts,
    write_json_atomic, write_run_artifacts,
)
from src.model_patterns.expected_points.backtesting import (
    BacktestSpec, compare_backtest_runs, recipe_code_identity, run_walk_forward,
)
from src.model_patterns.expected_points.recipes import get_expected_points_recipe


@dataclass(frozen=True)
class LegacyTreeRecipe(ResearchRecipe):
    def build_config(self, frame, **kwargs):
        config = self.base.build_config(frame, **kwargs)
        config.score_n_jobs = 1
        config.confidence_n_jobs = 1
        config.score_estimator = "research-tree"
        config.score_param_grid = {
            "lgbmregressor__n_estimators": [200, 400],
            "lgbmregressor__max_depth": [5],
            "lgbmregressor__learning_rate": [.03, .06],
            "lgbmregressor__num_leaves": [7, 15],
            "lgbmregressor__min_child_samples": [50, 100],
            "lgbmregressor__reg_lambda": [10.],
        }
        return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", required=True, choices=["nfl", "cfb"])
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seasons", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), Path(__file__).with_name("experiment_score_models.py")]
    sources = {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
    for file in files:
        (args.output / file.name).write_bytes(file.read_bytes())
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    recipe = LegacyTreeRecipe(get_expected_points_recipe(args.league), "legacy-tree", digest)
    identity = recipe_code_identity(recipe.league)
    frame = load_backtest_frame(args.frame)
    seasons = tuple(map(int, args.seasons.split(",")))
    write_json_atomic(args.output / "request.json", {
        "arguments": {**vars(args), "variant": "legacy-tree", "no_sportsbook": False},
        "recipe": recipe, "source_identity": identity, "research_sha256": digest,
        "source_files": sources, "frozen_inputs": True, "use_cache": False,
        "config": recipe.build_config(frame, season=seasons[0], week=1),
    })
    original_search = trainer.fit_score_model
    tuning = []

    def capture_search(*positional, **keywords):
        search = original_search(*positional, **keywords)
        tuning.append({"cutoff_index": len(tuning) + 1, "best_params": search.best_params_,
                       "best_validation_mse": -search.best_score_})
        write_json_atomic(args.output / "tuning.json", tuning)
        return search

    write_json_atomic(args.output / "status.json", {"status": "running"})
    try:
        with patch.object(trainer, "make_score_pipeline", research_factory), \
             patch.object(trainer, "fit_score_model", capture_search):
            result = run_walk_forward(frame, recipe, BacktestSpec(
                profile="standard", seasons=seasons, use_cache=False))
        assert recipe_code_identity(recipe.league) == identity
        assert all(hashlib.sha256(Path(file).read_bytes()).hexdigest() == sha for file, sha in sources.items())
        write_run_artifacts(result, args.output / "candidate")
        write_comparison_artifacts(compare_backtest_runs(load_backtest_run(args.baseline), result),
                                   output_dir=args.output)
        write_json_atomic(args.output / "status.json", {"status": "completed"})
    except Exception as exc:
        write_json_atomic(args.output / "status.json", {"status": "failed", "error": str(exc)})
        raise
    print(f"Completed {args.output}", flush=True)


if __name__ == "__main__":
    main()
