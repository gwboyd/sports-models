from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .chronology import predefined_chronological_split, sort_chronologically
from .types import ExpectedPointsConfig, ExpectedPointsLeague

if TYPE_CHECKING:
    from .lock_models import ProbabilityHead


class ConfiguredLockClassifier:
    """Compatibility adapter around the shared replay probability head."""

    classes_ = np.array([0, 1])

    def __init__(
        self,
        head: ProbabilityHead,
        history: pd.DataFrame,
        *,
        league: ExpectedPointsLeague,
        market: str,
        locks_enabled: bool,
    ) -> None:
        self.head = head
        self.history = history
        self.league = league
        self.market = market
        self.locks_enabled = bool(locks_enabled)

    def predict_outcomes(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        from .lock_policy import estimate_push_probability
        from .lock_replay import build_lock_market_frame

        features = build_lock_market_frame(
            frame, self.league, self.market, outcomes_known=False,
        )
        return self.head.predict(features), estimate_push_probability(self.history, features)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probability, _push = self.predict_outcomes(frame)
        return np.column_stack([1.0 - probability, probability])


def _build_classifier_pipeline(num_features, cat_features):
    transformers = []
    if num_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                num_features,
            )
        )
    if cat_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", LGBMClassifier(random_state=2, verbose=-1)),
        ]
    )


def fit_classifiers(
    results,
    spread_class_features,
    total_class_features,
    spread_class_cat_features,
    total_class_cat_features,
    param_grid,
    time_col="date_time",
    validation_size=0.2,
    n_jobs=-1,
    scoring=None,
):
    results = sort_chronologically(results, time_col=time_col)

    spread_rows = results.loc[results["spread_win"].notna()].copy()
    total_rows = results.loc[results["total_win"].notna()].copy()
    spread_X = spread_rows[spread_class_features]
    spread_y = spread_rows["spread_win"]
    total_X = total_rows[total_class_features]
    total_y = total_rows["total_win"]

    spread_cat = [c for c in spread_class_cat_features if c in spread_X.columns]
    total_cat = [c for c in total_class_cat_features if c in total_X.columns]
    spread_num = [c for c in spread_X.columns if c not in spread_cat]
    total_num = [c for c in total_X.columns if c not in total_cat]

    spread_pipe = _build_classifier_pipeline(spread_num, spread_cat)
    total_pipe = _build_classifier_pipeline(total_num, total_cat)

    # Grid now needs clf__ prefix
    clf_param_grid = {f"clf__{k}": v for k, v in param_grid.items()}

    spread_cv = predefined_chronological_split(
        spread_rows,
        time_col=time_col,
        test_size=validation_size,
    )
    total_cv = predefined_chronological_split(
        total_rows,
        time_col=time_col,
        test_size=validation_size,
    )

    spread_clf = GridSearchCV(
        spread_pipe,
        clf_param_grid,
        cv=spread_cv,
        n_jobs=n_jobs,
        scoring=scoring,
    )
    total_clf = GridSearchCV(
        total_pipe,
        clf_param_grid,
        cv=total_cv,
        n_jobs=n_jobs,
        scoring=scoring,
    )

    spread_clf.fit(spread_X, spread_y)
    total_clf.fit(total_X, total_y)

    return spread_clf, total_clf


def fit_configured_classifiers(
    results: pd.DataFrame,
    config: ExpectedPointsConfig,
) -> tuple[ConfiguredLockClassifier, ConfiguredLockClassifier] | tuple[GridSearchCV, GridSearchCV]:
    """Fit either the released lock heads or the legacy confidence classifiers."""
    configured = (config.spread_lock_head is not None, config.total_lock_head is not None)
    if any(configured) and not all(configured):
        raise ValueError("Configure both spread and total Lock heads, or neither for legacy confidence")
    if not any(configured):
        return fit_classifiers(
            results,
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
    if config.league is None:
        raise ValueError("Configured Lock heads require an explicit league")

    from .lock_features import lock_feature_set
    from .lock_models import LockVariantSpec
    from .lock_replay import build_lock_market_frame, _fit_for_period, LockReplaySpec

    fitted = []
    for market, head_config in (
        ("spread", config.spread_lock_head),
        ("total", config.total_lock_head),
    ):
        history = build_lock_market_frame(
            results, config.league, market, outcomes_known=True,
        )
        variant = LockVariantSpec(
            name=f"production_{market}",
            family=head_config.family,
            feature_group=head_config.feature_group,
            calibrator=head_config.calibrator,
            parameters=dict(head_config.parameters),
        )
        if variant.family in {"recorded_raw", "recorded_platt"}:
            raise ValueError("Recorded classifier probabilities are replay-only, not a production head")
        feature_set = lock_feature_set(
            config.league, head_config.feature_group, market=market,
        )
        head, ready = _fit_for_period(variant, history, feature_set, LockReplaySpec())
        fitted.append(ConfiguredLockClassifier(
            head,
            history,
            league=config.league,
            market=market,
            locks_enabled=head_config.locks_enabled and ready,
        ))
    return tuple(fitted)
