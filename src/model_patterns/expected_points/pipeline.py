from typing import List

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline


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
    def predict_scores(self, X, team: str):
        X = X.copy()
        X["pred_team"] = team
        return super().predict(X)


def make_score_pipeline(features: List[str], cat_features: List[str]):
    pipeline_template = make_pipeline(
        HomeAwayTransformer(),
        ItemSelector(features),
        CatTypeChangeTransformer(feature_names=features, cat_names=cat_features),
        LGBMRegressor(verbose=-1, n_jobs=-1, random_state=31),
    )
    return ScorePipeline(steps=pipeline_template.steps)
