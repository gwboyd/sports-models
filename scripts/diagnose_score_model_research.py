"""Training-only fit diagnostics for the first cutoff of completed research runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiment_score_models import ResearchRecipe, research_factory
from scripts.experiment_legacy_score_tree import LegacyTreeRecipe
from scripts.experiment_residual_score_tree import ResidualRecipe, residual_factory
from src.model_patterns.expected_points.backtest_artifacts import load_backtest_frame, write_json_atomic
from src.model_patterns.expected_points.chronology import chronological_train_test_split
from src.model_patterns.expected_points.pipeline import HomeAwayTransformer
from src.model_patterns.expected_points.recipes import get_expected_points_recipe
from src.model_patterns.expected_points.trainer import (
    _build_train_df_at_cutoff, _duplicate_score_training_rows, _split,
)
from src.sports.football.kickoff import parse_eastern_kickoffs
from dataclasses import replace


def diagnose(directory):
    request = json.loads((directory / "request.json").read_text())
    args = request["arguments"]
    base = replace(get_expected_points_recipe(args["league"]),
                   include_sportsbook_features=not args["no_sportsbook"])
    recipe_class = {"legacy-tree": LegacyTreeRecipe, "residual": ResidualRecipe}.get(args["variant"], ResearchRecipe)
    recipe = recipe_class(base, args["variant"], request["research_sha256"])
    frame = load_backtest_frame(args["frame"])
    first_year = int(args["seasons"].split(",")[0])
    week = int(frame.loc[frame.season == first_year, "week"].min())
    target = frame.loc[(frame.season == first_year) & (frame.week == week)]
    cutoff = parse_eastern_kickoffs(target.date_time).min() - pd.Timedelta(nanoseconds=1)
    config = recipe.build_config(frame, season=first_year, week=week, prediction_now=cutoff)
    history = _build_train_df_at_cutoff(frame, config, cutoff=cutoff)
    outer_train, _ = _split(history, config)
    inner_train, validation = chronological_train_test_split(
        outer_train, time_col=config.time_col, test_size=config.inner_validation_size)
    parameters = json.loads((directory / "tuning.json").read_text())[0]["best_params"]
    factory = residual_factory if args["variant"] == "residual" else research_factory
    model = factory(config.features, config.cat_features,
                              score_estimator=config.score_estimator,
                              team_features=config.score_team_features).set_params(**parameters)
    X, y = _duplicate_score_training_rows(inner_train[config.input_features], inner_train[config.targets])
    vx, vy = _duplicate_score_training_rows(validation[config.input_features], validation[config.targets])
    model.fit(X, y)
    result = {"run": directory.name, "cutoff": cutoff.isoformat(), "parameters": parameters,
              "inner_train_games": len(inner_train), "inner_validation_games": len(validation),
              "inner_train_mse": float(np.mean((model.predict(X) - y) ** 2)),
              "inner_validation_mse": float(np.mean((model.predict(vx) - vy) ** 2)),
              "interpretation": "Training-only diagnostic; validation selected these parameters, so not unbiased evidence."}
    if args["variant"] in ["tree", "compact", "legacy-tree"]:
        tree = model.named_steps["lgbmregressor"]
        gain = tree.booster_.feature_importance(importance_type="gain")
        result["gain_fraction"] = dict(zip(tree.feature_name_, (gain / max(gain.sum(), 1)).tolist()))
    if args["variant"] == "ridge":
        prep = model.named_steps["columntransformer"]
        result["coefficients_on_transformed_features"] = dict(zip(
            prep.get_feature_names_out().tolist(), model.named_steps["ridge"].coef_.tolist()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    rows = []
    for directory in sorted(args.study.iterdir()):
        status = directory / "status.json"
        if not status.exists() or json.loads(status.read_text())["status"] != "completed":
            continue
        rows.append(diagnose(directory))
    write_json_atomic(args.study / "analysis/training-diagnostics.json", rows)
    print(pd.DataFrame(rows)[["run", "inner_train_games", "inner_train_mse", "inner_validation_mse"]].to_string(index=False))


if __name__ == "__main__":
    main()
