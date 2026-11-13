"""Frozen-input estimator research using the unmodified weekly backtest fitter.

The factory substitution is scoped to this read-only command. Production recipes,
chronological splits, confidence/Lock fitting and backtest caches are not changed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_patterns.expected_points import trainer
from src.model_patterns.expected_points.backtest_artifacts import (
    load_backtest_frame, load_backtest_run, write_comparison_artifacts,
    write_json_atomic, write_run_artifacts,
)
from src.model_patterns.expected_points.backtesting import (
    BacktestSpec, compare_backtest_runs, recipe_code_identity, run_walk_forward,
)
from src.model_patterns.expected_points.pipeline import (
    HomeAwayTransformer, ScorePipeline, make_score_pipeline,
)
from src.model_patterns.expected_points.recipes import get_expected_points_recipe


class MatchupTransformer(BaseEstimator, TransformerMixin):
    """Combine rotated own offense and opposing defense on their existing scales."""

    def __init__(self, pairs=(), elo=False):
        self.pairs = pairs
        self.elo = elo

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        for output, offense, defense in self.pairs:
            # Missing input remains missing; do not average a single observed side.
            result[output] = (result[offense] + result[defense]) / 2.0
        if self.elo:
            result["matchup_elo"] = result["home_pregame_elo"] - result["away_pregame_elo"]
        return result


def matchup_spec(features):
    pairs = []
    efficiency = [name for name in features if "ewma" in name]
    for offense in efficiency:
        if "offense" in offense and offense.endswith("_home"):
            defense = offense.replace("offense", "defense")[:-5] + "_away"
            if defense not in features:
                raise ValueError(f"Missing matchup defense {defense}")
            pairs.append(("matchup_" + offense[:-5], offense, defense))
    removed = set(efficiency) | {
        "home_pregame_elo", "away_pregame_elo", "moneyline_home", "spread_line",
        "spread_odds_home", "total_line", "over_odds",
    }
    selected = [name for name in features if name not in removed]
    selected += [pair[0] for pair in pairs]
    elo = "home_pregame_elo" in features
    if elo:
        selected.append("matchup_elo")
    return selected, tuple(pairs), elo


def tree_pipeline(features, cat_features, *, compact=False):
    selected, pairs, elo = matchup_spec(features) if compact else (features, (), False)
    model = make_score_pipeline(selected, cat_features, score_estimator="lightgbm")
    if compact:
        model.steps.insert(1, ("matchup", MatchupTransformer(pairs=pairs, elo=elo)))
    model.set_params(lgbmregressor__n_jobs=1, lgbmregressor__verbosity=-1,
                     lgbmregressor__n_estimators=300,
                     lgbmregressor__learning_rate=.03)
    return model


class RidgeTreeBlend(RegressorMixin, BaseEstimator):
    """Both components fit only supplied training rows; outer GridSearch picks weight."""

    def __init__(self, features, cat_features, team_features, alpha=100.,
                 team_weight=1., tree_weight=0., compact=False):
        self.features = features
        self.cat_features = cat_features
        self.team_features = team_features
        self.alpha = alpha
        self.team_weight = team_weight
        self.tree_weight = tree_weight
        self.compact = compact

    def fit(self, X, y):
        if not 0 <= self.tree_weight <= 1:
            raise ValueError("tree_weight must be between zero and one")
        self.ridge_ = make_score_pipeline(
            self.features, self.cat_features, score_estimator="ridge",
            team_features=self.team_features,
        ).set_params(ridge__alpha=self.alpha,
                     columntransformer__transformer_weights={"team": self.team_weight})
        self.ridge_.fit(X, y)
        self.tree_ = None
        if self.tree_weight:
            self.tree_ = tree_pipeline(self.features, self.cat_features, compact=self.compact)
            self.tree_.set_params(lgbmregressor__max_depth=2, lgbmregressor__num_leaves=4,
                                  lgbmregressor__min_child_samples=200,
                                  lgbmregressor__reg_lambda=50.)
            self.tree_.fit(X, y)
        return self

    def predict(self, X):
        ridge = self.ridge_.predict(X)
        if self.tree_ is None:
            return ridge
        return (1 - self.tree_weight) * ridge + self.tree_weight * self.tree_.predict(X)


def research_factory(features, cat_features, *, score_estimator, team_features=None, **kwargs):
    if score_estimator in {"research-tree", "research-compact"}:
        return tree_pipeline(features, cat_features, compact=score_estimator == "research-compact")
    if score_estimator == "research-blend":
        return ScorePipeline([("blend", RidgeTreeBlend(features, cat_features, team_features))])
    return make_score_pipeline(features, cat_features, score_estimator=score_estimator,
                               team_features=team_features, **kwargs)


@dataclass(frozen=True)
class ResearchRecipe:
    base: object
    variant: str
    implementation_sha256: str
    version: str = "working-tree"
    protocol_version: str = "nonlinear-research-v1"

    @property
    def league(self):
        return self.base.league

    @property
    def name(self):
        return f"{self.base.name}-{self.variant}"

    def build_config(self, frame, **kwargs):
        config = self.base.build_config(frame, **kwargs)
        config.score_n_jobs = 1
        config.confidence_n_jobs = 1
        if self.variant == "ridge":
            return config
        config.score_estimator = "research-" + self.variant
        if self.variant == "blend":
            config.score_param_grid = {
                "blend__alpha": [10., 100., 1000., 10000.],
                "blend__team_weight": [0., .25, 1.],
                "blend__tree_weight": [0., .25, .5],
            }
        else:
            config.score_param_grid = [
                {"lgbmregressor__max_depth": [depth], "lgbmregressor__num_leaves": [leaves],
                 "lgbmregressor__min_child_samples": [minimum],
                 "lgbmregressor__reg_lambda": [penalty]}
                for depth, leaves in [(2, 4), (3, 8)]
                for minimum, penalty in [(100, 10.), (300, 100.)]
            ]
        return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", required=True, choices=["nfl", "cfb"])
    parser.add_argument("--variant", required=True, choices=["ridge", "tree", "compact", "blend"])
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seasons", default="2023,2024,2025")
    parser.add_argument("--no-sportsbook", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    (args.output / "source.py").write_bytes(source)
    base = get_expected_points_recipe(args.league)
    base = replace(base, include_sportsbook_features=not args.no_sportsbook)
    recipe = ResearchRecipe(base, args.variant, source_hash)
    identity = recipe_code_identity(recipe.league)
    frame = load_backtest_frame(args.frame)
    baseline = load_backtest_run(args.baseline)
    seasons = tuple(int(year) for year in args.seasons.split(","))
    spec = BacktestSpec(profile="standard", seasons=seasons, use_cache=False)
    metadata = json.loads(args.frame.with_suffix(".json").read_text())
    request = dict(arguments=vars(args), recipe=recipe, source_identity=identity,
                   frame_metadata=metadata, research_sha256=source_hash,
                   base_git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   config=recipe.build_config(frame, season=seasons[0], week=1),
                   frozen_inputs=True, use_cache=False)
    write_json_atomic(args.output / "request.json", request)
    diagnostics = []
    original_search = trainer.fit_score_model

    def capture_search(*positional, **keywords):
        search = original_search(*positional, **keywords)
        diagnostics.append({
            "cutoff_index": len(diagnostics) + 1,
            "best_params": search.best_params_, "best_validation_mse": -search.best_score_,
            "train_games": len(keywords["split_frame"]),
            "training_last_kickoff": str(keywords["split_frame"]["date_time"].max()),
            "grid": [{"params": parameters, "validation_mse": -float(score)}
                     for parameters, score in zip(search.cv_results_["params"],
                                                   search.cv_results_["mean_test_score"])],
        })
        write_json_atomic(args.output / "tuning.json", diagnostics)
        return search

    write_json_atomic(args.output / "status.json", {"status": "running"})
    try:
        with patch.object(trainer, "make_score_pipeline", research_factory), \
             patch.object(trainer, "fit_score_model", capture_search):
            run = run_walk_forward(frame, recipe, spec)
        assert recipe_code_identity(recipe.league) == identity, "Production source changed during run"
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == source_hash
        write_run_artifacts(run, args.output / "candidate")
        comparison = compare_backtest_runs(baseline, run)
        write_comparison_artifacts(comparison, output_dir=args.output)
        write_json_atomic(args.output / "status.json", {"status": "completed"})
    except Exception as exc:
        write_json_atomic(args.output / "status.json", {"status": "failed", "error": str(exc)})
        raise
    print(f"Completed {args.output}", flush=True)


if __name__ == "__main__":
    main()
