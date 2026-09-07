from typing import List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class HomeAwayTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X_, y=None):
        X = X_.copy()
        if "pred_team" in X.columns:
            mask = X["pred_team"] == "away"
            if mask.any():
                home_cols = [col for col in X.columns if col.endswith("_home")]
                for home_col in home_cols:
                    base_col = home_col[:-5]
                    away_col = base_col + "_away"
                    if away_col in X.columns:
                        temp = X.loc[mask, home_col].copy()
                        X.loc[mask, home_col] = X.loc[mask, away_col]
                        X.loc[mask, away_col] = temp
                # Some feeds name team inputs with a home_/away_ prefix
                # (for example pregame Elo) rather than a suffix.
                for home_col in [col for col in X.columns if col.startswith("home_")]:
                    away_col = "away_" + home_col[5:]
                    if away_col in X.columns:
                        temp = X.loc[mask, home_col].copy()
                        X.loc[mask, home_col] = X.loc[mask, away_col]
                        X.loc[mask, away_col] = temp
                if "spread_line" in X.columns:
                    X.loc[mask, "spread_line"] = -1.0 * X.loc[mask, "spread_line"]
        return X


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class AmericanOddsProbabilityTransformer(BaseEstimator, TransformerMixin):
    """Put existing betting-odds inputs on a meaningful linear-model scale."""

    def fit(self, X, y=None):
        return self

    def transform(self, X_, y=None):
        X = X_.copy()
        for column in (
            "moneyline_home", "moneyline_away", "spread_odds_home",
            "spread_odds_away", "over_odds", "under_odds",
        ):
            if column not in X:
                continue
            odds = pd.to_numeric(X[column], errors="coerce")
            probability = pd.Series(np.nan, index=X.index, dtype=float)
            positive = odds.gt(0) & np.isfinite(odds)
            negative = odds.lt(0) & np.isfinite(odds)
            probability.loc[positive] = 100.0 / (100.0 + odds.loc[positive])
            probability.loc[negative] = -odds.loc[negative] / (100.0 - odds.loc[negative])
            X[column] = probability
        return X


class CatTypeChangeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names: List[str], cat_names: List[str]):
        self.feature_names = feature_names
        self.cat_names = cat_names

    def fit(self, X, y=None):
        return self

    def transform(self, X_, y=None):
        X = pd.DataFrame(X_)
        changes = dict(zip(X.columns, self.feature_names))
        X.rename(columns=changes, inplace=True)
        for col in X.columns:
            if col in self.cat_names:
                X[col] = X[col].astype("category")
            else:
                X[col] = X[col].astype("float")
        return X


class ScorePipeline(Pipeline):
    def __init__(self, steps, *, score_target: str = "points", memory=None, verbose: bool = False):
        self.score_target = score_target
        super().__init__(steps=steps, memory=memory, verbose=verbose)

    def _market_baseline(self, X: pd.DataFrame) -> np.ndarray:
        required = {"pred_team", "implied_points_home", "implied_points_away"}
        if not required.issubset(X.columns) or not X["pred_team"].isin(["home", "away"]).all():
            raise ValueError("Residual score prediction requires a team and both market-implied scores")
        home = pd.to_numeric(X["implied_points_home"], errors="coerce").to_numpy(float)
        away = pd.to_numeric(X["implied_points_away"], errors="coerce").to_numpy(float)
        if not (np.isfinite(home).all() and np.isfinite(away).all()):
            raise ValueError("Residual score prediction requires finite market-implied scores")
        return np.where(X["pred_team"].eq("away"), away, home)

    def fit(self, X: pd.DataFrame, y=None, **params):
        if self.score_target == "market_residual":
            y = np.asarray(y, dtype=float) - self._market_baseline(X)
        elif self.score_target != "points":
            raise ValueError(f"Unknown score target: {self.score_target}")
        return super().fit(X, y, **params)

    def predict(self, X: pd.DataFrame, **params) -> np.ndarray:
        baseline = self._market_baseline(X) if self.score_target == "market_residual" else None
        predictions = super().predict(X, **params)
        if baseline is not None:
            return np.maximum(0.0, np.asarray(predictions) + baseline)
        return predictions

    def predict_scores(self, X, team: str):
        X = X.copy()
        X["pred_team"] = team
        return self.predict(X)

    def score(self, X, y, sample_weight=None):
        # Pipeline.score delegates directly to the final estimator and would
        # compare residual predictions to point labels for a residual target.
        return r2_score(y, self.predict(X), sample_weight=sample_weight)


def make_score_pipeline(
    features: List[str], cat_features: List[str], *,
    score_target: str = "points", score_estimator: str = "lightgbm",
    team_features: List[str] | None = None,
):
    if score_estimator == "ridge":
        numeric = [column for column in features if column not in cat_features]
        categorical = [column for column in features if column in cat_features]
        team = list(team_features or [])
        if set(team) - set(numeric):
            raise ValueError("Ridge team-feature group must contain selected numeric inputs")
        context = [column for column in numeric if column not in team]
        transformers = [
            ("numeric", make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                StandardScaler(),
            ), context),
            ("categorical", make_pipeline(
                SimpleImputer(strategy="constant", fill_value="unknown", keep_empty_features=True),
                OneHotEncoder(handle_unknown="ignore"),
            ), categorical),
        ]
        if team:
            # Weight the entire standardized group, including its missingness
            # indicators, so ridge can shrink team history more than context.
            transformers.append(("team", make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                StandardScaler(),
            ), team))
        preprocessing = ColumnTransformer(transformers)
        pipeline_template = make_pipeline(
            HomeAwayTransformer(),
            AmericanOddsProbabilityTransformer(),
            ItemSelector(features),
            CatTypeChangeTransformer(feature_names=features, cat_names=cat_features),
            preprocessing,
            Ridge(solver="lsqr", tol=1e-10),
        )
        return ScorePipeline(steps=pipeline_template.steps, score_target=score_target)
    if score_estimator != "lightgbm":
        raise ValueError(f"Unknown score estimator: {score_estimator}")
    pipeline_template = make_pipeline(
        HomeAwayTransformer(),
        ItemSelector(features),
        CatTypeChangeTransformer(feature_names=features, cat_names=cat_features),
        LGBMRegressor(verbose=-1, n_jobs=-1, random_state=31),
    )
    return ScorePipeline(steps=pipeline_template.steps, score_target=score_target)
