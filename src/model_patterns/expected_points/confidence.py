from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV


def fit_classifiers(results, spread_class_features, total_class_features, param_grid, cv=5, n_jobs=-1):
    spread_X = results[spread_class_features]
    spread_y = results["spread_win"]
    total_X = results[total_class_features]
    total_y = results["total_win"]

    spread_X = spread_X[spread_y.notna()]
    spread_y = spread_y.dropna()
    total_X = total_X[total_y.notna()]
    total_y = total_y.dropna()

    spread_clf = GridSearchCV(LGBMClassifier(random_state=2, verbose=-1), param_grid, cv=cv, n_jobs=n_jobs)
    total_clf = GridSearchCV(LGBMClassifier(random_state=2, verbose=-1), param_grid, cv=cv, n_jobs=n_jobs)
    spread_clf.fit(spread_X, spread_y)
    total_clf.fit(total_X, total_y)
    return spread_clf, total_clf
