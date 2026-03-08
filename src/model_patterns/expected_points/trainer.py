from typing import Tuple

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

from .betting import calculate_wins, determine_plays, scores_to_bets, win_probability
from .confidence import fit_classifiers
from .pipeline import make_score_pipeline
from .types import ExpectedPointsConfig, ExpectedPointsRunResult


def fit_score_model(X, y, score_pipeline, param_grid, cv=2, n_jobs=-1):
    X_home = X.copy()
    X_away = X.copy()
    y_home = y.iloc[:, 0]
    y_away = y.iloc[:, 1]

    X_home["pred_team"] = "home"
    X_away["pred_team"] = "away"

    full_X = pd.concat([X_home, X_away], ignore_index=True)
    full_y = pd.concat([y_home, y_away], ignore_index=True)

    search = GridSearchCV(score_pipeline, param_grid, cv=cv, n_jobs=n_jobs, scoring="neg_mean_squared_error")
    search.fit(full_X, full_y)
    return search.best_estimator_


def fit_eval(df, X_train, X_test, y_train, y_test, score_pipeline, score_param_grid, score_cv, score_n_jobs):
    score_model = fit_score_model(
        X_train, y_train, score_pipeline, score_param_grid, cv=score_cv, n_jobs=score_n_jobs
    )
    away_scores = score_model.predict_scores(X_test, "away")
    home_scores = score_model.predict_scores(X_test, "home")

    results = df.loc[X_test.index].copy()
    results["home_score"] = y_test.iloc[:, 0]
    results["away_score"] = y_test.iloc[:, 1]
    results["away_score_pred"] = away_scores
    results["home_score_pred"] = home_scores
    results = scores_to_bets(results)
    results = calculate_wins(results)
    return results


def _build_train_df(df, config: ExpectedPointsConfig):
    train_df = df.dropna(subset=[config.targets[0]], inplace=False)
    train_df = train_df[
        ~(
            (train_df[config.season_col] == config.current_year)
            & (train_df[config.week_col] >= config.current_week)
        )
    ]
    return train_df


def _split(X, y, config: ExpectedPointsConfig) -> Tuple:
    if config.split_strategy != "random":
        raise ValueError(f"Unsupported split_strategy: {config.split_strategy}")
    return train_test_split(X, y, test_size=config.test_size, random_state=config.random_state)


def run_expected_points(df, config: ExpectedPointsConfig) -> ExpectedPointsRunResult:
    score_pipeline = make_score_pipeline(config.features, config.cat_features)

    train_df = _build_train_df(df, config)
    X = train_df[config.input_features]
    y = train_df[config.targets]
    X_train, X_test, y_train, y_test = _split(X, y, config)

    eval_results = fit_eval(
        train_df,
        X_train,
        X_test,
        y_train,
        y_test,
        score_pipeline,
        config.score_param_grid,
        config.score_cv,
        config.score_n_jobs,
    )

    spread_clf, total_clf = fit_classifiers(
        eval_results,
        config.spread_class_features,
        config.total_class_features,
        config.spread_class_cat_features,
        config.total_class_cat_features,
        config.confidence_param_grid,
        cv=config.confidence_cv,
        n_jobs=config.confidence_n_jobs,
    )


    score_model = fit_score_model(
        X, y, score_pipeline, config.score_param_grid, cv=config.score_cv, n_jobs=config.score_n_jobs
    )

    this_week = df[(df[config.season_col] == config.current_year) & (df[config.week_col] == config.current_week)].copy()
    away_prediction_features = config.away_prediction_features or config.input_features
    home_prediction_features = config.home_prediction_features or config.features

    this_week["away_score_pred"] = score_model.predict_scores(this_week[away_prediction_features], "away")
    this_week["home_score_pred"] = score_model.predict_scores(this_week[home_prediction_features], "home")

    plays = scores_to_bets(this_week)
    plays["spread_win_prob"] = win_probability(plays, classifier=spread_clf, features=config.spread_class_features)
    plays["total_win_prob"] = win_probability(plays, classifier=total_clf, features=config.total_class_features)
    plays = determine_plays(plays, thresholds=config.play_thresholds)

    metrics = {
        "eval_spread_win_pct": float(eval_results["spread_win"].mean() * 100),
        "eval_total_win_pct": float(eval_results["total_win"].mean() * 100),
        "train_rows": float(len(train_df)),
        "this_week_rows": float(len(this_week)),
    }

    return ExpectedPointsRunResult(
        score_model=score_model,
        spread_clf=spread_clf,
        total_clf=total_clf,
        eval_results=eval_results,
        this_week=this_week,
        plays=plays,
        train_df=train_df,
        metrics=metrics,
    )
