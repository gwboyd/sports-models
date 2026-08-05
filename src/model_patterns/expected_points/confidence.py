from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .chronology import predefined_chronological_split, sort_chronologically


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

    spread_clf = GridSearchCV(spread_pipe, clf_param_grid, cv=spread_cv, n_jobs=n_jobs)
    total_clf = GridSearchCV(total_pipe, clf_param_grid, cv=total_cv, n_jobs=n_jobs)

    spread_clf.fit(spread_X, spread_y)
    total_clf.fit(total_X, total_y)

    return spread_clf, total_clf
