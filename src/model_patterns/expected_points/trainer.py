from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from src.sports.football.kickoff import exclude_started_games, parse_eastern_kickoffs

from .betting import calculate_wins, determine_plays, scores_to_bets, win_probability
from .chronology import chronological_train_test_split, predefined_chronological_split
from .confidence import fit_classifiers
from .evaluation import safe_confidence_health_evaluation, score_prediction_metrics
from .pipeline import make_score_pipeline
from .types import ExpectedPointsConfig, ExpectedPointsRunResult


def _duplicate_score_training_rows(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X_home = X.copy()
    X_away = X.copy()
    X_home["pred_team"] = "home"
    X_away["pred_team"] = "away"
    full_X = pd.concat([X_home, X_away], ignore_index=True)
    full_y = pd.concat([y.iloc[:, 0], y.iloc[:, 1]], ignore_index=True)
    return full_X, full_y


def _duplicate_predefined_split(split: PredefinedSplit) -> PredefinedSplit:
    base_fold = np.asarray(split.test_fold, dtype=int)
    return PredefinedSplit(np.concatenate([base_fold, base_fold]))


def fit_score_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    score_pipeline,
    param_grid,
    *,
    split_frame: pd.DataFrame,
    time_col: str,
    validation_size: float,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Tune once with one train-before-validation split at the game level."""
    full_X, full_y = _duplicate_score_training_rows(X, y)
    game_split = predefined_chronological_split(
        split_frame,
        time_col=time_col,
        test_size=validation_size,
    )
    search = GridSearchCV(
        score_pipeline,
        param_grid,
        cv=_duplicate_predefined_split(game_split),
        n_jobs=n_jobs,
        scoring="neg_mean_squared_error",
    )
    search.fit(full_X, full_y)
    return search


def _fit_final_score_model(best_estimator, X: pd.DataFrame, y: pd.DataFrame):
    """Refit selected score parameters on every completed game without retuning."""
    full_X, full_y = _duplicate_score_training_rows(X, y)
    final_model = clone(best_estimator)
    final_model.fit(full_X, full_y)
    return final_model


def fit_eval(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    score_pipeline,
    score_param_grid,
    *,
    time_col: str,
    validation_size: float,
    score_n_jobs: int,
    betting_transform=scores_to_bets,
) -> tuple[pd.DataFrame, GridSearchCV]:
    score_search = fit_score_model(
        X_train,
        y_train,
        score_pipeline,
        score_param_grid,
        split_frame=df.loc[X_train.index],
        time_col=time_col,
        validation_size=validation_size,
        n_jobs=score_n_jobs,
    )
    score_model = score_search.best_estimator_
    away_scores = score_model.predict_scores(X_test, "away")
    home_scores = score_model.predict_scores(X_test, "home")

    results = df.loc[X_test.index].copy()
    results["home_score"] = y_test.iloc[:, 0]
    results["away_score"] = y_test.iloc[:, 1]
    results["away_score_pred"] = away_scores
    results["home_score_pred"] = home_scores
    results = betting_transform(results)
    results = calculate_wins(results)
    return results, score_search


def _build_train_df(df: pd.DataFrame, config: ExpectedPointsConfig) -> pd.DataFrame:
    # A score model predicts every configured target, so partial results cannot
    # be used as labels. This excludes scheduled, cancelled, and incomplete games.
    train_df = df.copy()
    for target in config.targets:
        train_df[target] = pd.to_numeric(train_df[target], errors="coerce")
    train_df = train_df.dropna(subset=config.targets, inplace=False)
    train_df = train_df[
        ~(
            (train_df[config.season_col] == config.current_year)
            & (train_df[config.week_col] >= config.current_week)
        )
    ]
    return train_df


def _build_train_df_at_cutoff(
    df: pd.DataFrame,
    config: ExpectedPointsConfig,
    *,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Build completed training history strictly before a simulation cutoff."""
    train_df = df.copy()
    for target in config.targets:
        train_df[target] = pd.to_numeric(train_df[target], errors="coerce")
    kickoffs = parse_eastern_kickoffs(train_df[config.time_col])
    parsed_cutoff = pd.Timestamp(cutoff)
    if parsed_cutoff.tzinfo is None:
        raise ValueError("Historical cutoff must be timezone-aware")
    return train_df.loc[
        train_df[list(config.targets)].notna().all(axis=1)
        & (kickoffs < parsed_cutoff.tz_convert("UTC"))
    ].copy()


def _split(
    frame: pd.DataFrame,
    config: ExpectedPointsConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.split_strategy != "chronological":
        raise ValueError(f"Unsupported split_strategy: {config.split_strategy}")
    return chronological_train_test_split(
        frame,
        time_col=config.time_col,
        test_size=config.test_size,
    )


def _fit_expected_points(
    train_df: pd.DataFrame,
    config: ExpectedPointsConfig,
):
    score_pipeline = make_score_pipeline(config.features, config.cat_features)
    betting_transform = config.betting_transform or scores_to_bets
    train_partition, test_partition = _split(train_df, config)
    X = train_df[config.input_features]
    y = train_df[config.targets]
    X_train = train_partition[config.input_features]
    X_test = test_partition[config.input_features]
    y_train = train_partition[config.targets]
    y_test = test_partition[config.targets]

    eval_results, score_search = fit_eval(
        train_df,
        X_train,
        X_test,
        y_train,
        y_test,
        score_pipeline,
        config.score_param_grid,
        time_col=config.time_col,
        validation_size=config.inner_validation_size,
        score_n_jobs=config.score_n_jobs,
        betting_transform=betting_transform,
    )

    health_results, health_metrics = safe_confidence_health_evaluation(
        eval_results,
        config,
    )

    spread_clf, total_clf = fit_classifiers(
        eval_results,
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

    score_model = _fit_final_score_model(score_search.best_estimator_, X, y)

    return (
        score_model,
        spread_clf,
        total_clf,
        eval_results,
        health_results,
        health_metrics,
    )


def _predict_period(
    prediction_frame: pd.DataFrame,
    config: ExpectedPointsConfig,
    *,
    score_model,
    spread_clf,
    total_clf,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    betting_transform = config.betting_transform or scores_to_bets
    scheduled_this_week = prediction_frame.copy()
    this_week = exclude_started_games(
        scheduled_this_week,
        kickoff_col=config.time_col,
        now=config.prediction_now,
    )
    if this_week.empty:
        raise ValueError("No unstarted games remain in the requested prediction period")

    away_prediction_features = config.away_prediction_features or config.input_features
    home_prediction_features = config.home_prediction_features or config.features

    this_week["away_score_pred"] = score_model.predict_scores(
        this_week[away_prediction_features], "away"
    )
    this_week["home_score_pred"] = score_model.predict_scores(
        this_week[home_prediction_features], "home"
    )

    plays = betting_transform(this_week)
    plays["spread_win_prob"] = win_probability(
        plays,
        classifier=spread_clf,
        features=config.spread_class_features,
    )
    plays["total_win_prob"] = win_probability(
        plays,
        classifier=total_clf,
        features=config.total_class_features,
    )
    plays = determine_plays(plays, thresholds=config.play_thresholds)
    return this_week, plays


def _build_metrics(
    *,
    train_df: pd.DataFrame,
    eval_results: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    this_week: pd.DataFrame,
    health_metrics: dict[str, float | None],
) -> dict[str, float | None]:
    metrics = {
        "eval_spread_win_pct": float(eval_results["spread_win"].mean() * 100),
        "eval_total_win_pct": float(eval_results["total_win"].mean() * 100),
        "train_rows": float(len(train_df)),
        "eval_rows": float(len(eval_results)),
        "this_week_rows": float(len(this_week)),
        "started_games_excluded": float(len(prediction_frame) - len(this_week)),
    }
    metrics.update(score_prediction_metrics(eval_results))
    metrics.update(health_metrics)
    return metrics


def run_expected_points(df: pd.DataFrame, config: ExpectedPointsConfig) -> ExpectedPointsRunResult:
    train_df = _build_train_df(df, config)
    (
        score_model,
        spread_clf,
        total_clf,
        eval_results,
        health_results,
        health_metrics,
    ) = _fit_expected_points(train_df, config)

    scheduled_this_week = df[
        (df[config.season_col] == config.current_year)
        & (df[config.week_col] == config.current_week)
    ].copy()
    this_week, plays = _predict_period(
        scheduled_this_week,
        config,
        score_model=score_model,
        spread_clf=spread_clf,
        total_clf=total_clf,
    )
    metrics = _build_metrics(
        train_df=train_df,
        eval_results=eval_results,
        prediction_frame=scheduled_this_week,
        this_week=this_week,
        health_metrics=health_metrics,
    )

    return ExpectedPointsRunResult(
        score_model=score_model,
        spread_clf=spread_clf,
        total_clf=total_clf,
        eval_results=eval_results,
        this_week=this_week,
        plays=plays,
        train_df=train_df,
        metrics=metrics,
        health_results=health_results,
    )


def run_expected_points_at_cutoff(
    df: pd.DataFrame,
    config: ExpectedPointsConfig,
    *,
    cutoff: pd.Timestamp,
    prediction_frame: pd.DataFrame | None = None,
) -> ExpectedPointsRunResult:
    """Retrain and predict one historical slate without using its outcomes."""
    train_df = _build_train_df_at_cutoff(df, config, cutoff=cutoff)
    (
        score_model,
        spread_clf,
        total_clf,
        eval_results,
        health_results,
        health_metrics,
    ) = _fit_expected_points(train_df, config)
    target = (
        prediction_frame.copy()
        if prediction_frame is not None
        else df.loc[
            (df[config.season_col] == config.current_year)
            & (df[config.week_col] == config.current_week)
        ].copy()
    )
    outcomes = target[["game_id", *config.targets]].copy()
    hidden = target.copy()
    hidden.loc[:, config.targets] = np.nan
    historical_config = ExpectedPointsConfig(**{
        **config.__dict__,
        "prediction_now": pd.Timestamp(cutoff),
    })
    this_week, plays = _predict_period(
        hidden,
        historical_config,
        score_model=score_model,
        spread_clf=spread_clf,
        total_clf=total_clf,
    )
    plays = plays.drop(columns=list(config.targets), errors="ignore").merge(
        outcomes,
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    metrics = _build_metrics(
        train_df=train_df,
        eval_results=eval_results,
        prediction_frame=target,
        this_week=this_week,
        health_metrics=health_metrics,
    )
    return ExpectedPointsRunResult(
        score_model=score_model,
        spread_clf=spread_clf,
        total_clf=total_clf,
        eval_results=eval_results,
        this_week=this_week,
        plays=plays,
        train_df=train_df,
        metrics=metrics,
        health_results=health_results,
    )
