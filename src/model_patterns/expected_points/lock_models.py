"""Small, reproducible probability heads for expected-points betting decisions.

These estimators deliberately operate on frozen expected-points outputs and an
explicit feature matrix.  They never choose a side: the score model owns the
play direction and a lock head only estimates whether that play will win.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


PROBABILITY_EPSILON = 1e-6


class ProbabilityHead(Protocol):
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return resolved-bet win probabilities on the zero-to-one scale."""


@dataclass(frozen=True)
class LockVariantSpec:
    """A registered, JSON-serializable lock probability candidate."""

    name: str
    family: Literal[
        "base_rate",
        "recorded_raw",
        "recorded_platt",
        "empirical_edge",
        "edge_threshold",
        "residual_distribution",
        "symmetric_residual",
        "spline_logit",
        "lightgbm",
    ]
    feature_group: str = "core"
    calibrator: Literal["identity", "platt", "beta"] = "identity"
    parameters: dict[str, float | int | str] = field(default_factory=dict)
    promotion_eligible: bool = True


@dataclass
class ConstantHead:
    probability: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), float(self.probability), dtype=float)


@dataclass
class ColumnHead:
    column: str

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(
            pd.to_numeric(frame[self.column], errors="coerce").fillna(50.0).to_numpy(float) / 100.0,
            PROBABILITY_EPSILON,
            1.0 - PROBABILITY_EPSILON,
        )


@dataclass
class SklearnHead:
    estimator: object
    columns: tuple[str, ...]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(
            self.estimator.predict_proba(frame.loc[:, self.columns])[:, 1],
            PROBABILITY_EPSILON,
            1.0 - PROBABILITY_EPSILON,
        )


@dataclass
class EmpiricalEdgeHead:
    """Beta-shrunk historical win rates in fixed edge buckets."""

    edges: tuple[float, ...]
    probabilities: tuple[float, ...]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        edge = pd.to_numeric(frame["edge"], errors="coerce").fillna(0.0).to_numpy(float)
        positions = np.digitize(edge, np.asarray(self.edges), right=False)
        return np.asarray(self.probabilities, dtype=float)[positions]


@dataclass
class EdgeThresholdHead:
    """Two-bucket empirical head for a release-fixed minimum score edge.

    The threshold decides which predictions are eligible for the selective
    policy.  The probabilities themselves are estimated only from the
    chronological score holdout supplied by the caller and shrink toward the
    overall resolved-bet rate.
    """

    minimum_edge: float
    maximum_rank: int
    base_probability: float
    qualifying_probability: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        edge = pd.to_numeric(frame["edge"], errors="coerce").fillna(0.0).to_numpy(float)
        rank = pd.to_numeric(frame["weekly_edge_rank"], errors="coerce").fillna(np.inf).to_numpy(float)
        return np.where(
            (edge >= self.minimum_edge) & (rank <= self.maximum_rank),
            self.qualifying_probability,
            self.base_probability,
        )


@dataclass
class ResidualDistributionHead:
    """Convert the score model's historical errors into bet-win probabilities."""

    residuals: np.ndarray
    prior_strength: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        edge = pd.to_numeric(frame["edge"], errors="coerce").fillna(0.0).to_numpy(float)
        direction = frame["direction"].astype(str).to_numpy()
        left = np.isin(direction, ["home", "under"])
        # Home/under wins below +edge; away/over wins above -edge.
        successes = np.where(
            left,
            np.searchsorted(self.residuals, edge, side="left"),
            len(self.residuals) - np.searchsorted(self.residuals, -edge, side="right"),
        )
        probability = (successes + 0.5 * self.prior_strength) / (len(self.residuals) + self.prior_strength)
        return np.clip(probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


@dataclass
class SymmetricResidualHead:
    """A conservative score-error distribution translated into bet probabilities.

    Symmetrizing errors estimates P(win) = .5 + .5 P(|error| < edge).
    Mixing this with an even-chance forecast discounts the score model's edge.
    The mixture weight is a versioned research choice, not fitted to target outcomes.
    """

    absolute_errors: np.ndarray
    trust: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        edge = pd.to_numeric(frame["edge"], errors="coerce").fillna(0.0).clip(lower=0).to_numpy(float)
        inside = np.searchsorted(self.absolute_errors, edge, side="left")
        return 0.5 + 0.5 * self.trust * inside / len(self.absolute_errors)


@dataclass
class PlattCalibrator:
    estimator: LogisticRegression | None

    def transform(self, probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
        if self.estimator is None:
            return clipped
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return np.clip(
            self.estimator.predict_proba(logits)[:, 1],
            PROBABILITY_EPSILON,
            1.0 - PROBABILITY_EPSILON,
        )


@dataclass
class BetaCalibrator:
    parameters: tuple[float, float, float] | None

    def transform(self, probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
        if self.parameters is None:
            return clipped
        a, b, c = self.parameters
        score = a * np.log(clipped) - b * np.log1p(-clipped) + c
        return 1.0 / (1.0 + np.exp(-np.clip(score, -35.0, 35.0)))


@dataclass
class CalibratedHead:
    head: ProbabilityHead
    calibrator: PlattCalibrator | BetaCalibrator

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.calibrator.transform(self.head.predict(frame))


def shrunken_base_rate(labels: Sequence[int | float], *, strength: float = 50.0) -> float:
    values = np.asarray(labels, dtype=float)
    values = values[np.isfinite(values)]
    return float((values.sum() + 0.5 * strength) / (len(values) + strength))


def fit_calibrator(
    method: str,
    probabilities: Sequence[float],
    labels: Sequence[int | float],
) -> PlattCalibrator | BetaCalibrator:
    probability = np.clip(np.asarray(probabilities, dtype=float), PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    target = np.asarray(labels, dtype=int)
    valid = np.isfinite(probability)
    probability, target = probability[valid], target[valid]
    if len(target) < 40 or len(np.unique(target)) < 2 or method == "identity":
        return PlattCalibrator(None)
    if method == "platt":
        logits = np.log(probability / (1.0 - probability)).reshape(-1, 1)
        return PlattCalibrator(LogisticRegression(C=1.0, max_iter=1000).fit(logits, target))
    if method != "beta":
        raise ValueError(f"Unsupported calibrator: {method}")

    def objective(parameters: np.ndarray) -> float:
        a, b, c = parameters
        score = a * np.log(probability) - b * np.log1p(-probability) + c
        predicted = np.clip(1.0 / (1.0 + np.exp(-np.clip(score, -35, 35))), PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
        return float(-(target * np.log(predicted) + (1 - target) * np.log1p(-predicted)).mean())

    result = minimize(objective, x0=np.array([1.0, 1.0, 0.0]), bounds=((0.0, None), (0.0, None), (None, None)))
    return BetaCalibrator(tuple(float(value) for value in result.x) if result.success else None)


def _column_pipeline(
    numeric: Sequence[str],
    categorical: Sequence[str],
    *,
    spline_edge: bool,
) -> ColumnTransformer:
    transformers: list[tuple[str, object, list[str]]] = []
    numeric = list(numeric)
    if spline_edge and "edge" in numeric:
        numeric.remove("edge")
        transformers.append((
            "edge",
            Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("spline", SplineTransformer(n_knots=4, degree=2, include_bias=False)),
                ("scale", StandardScaler()),
            ]),
            ["edge"],
        ))
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            list(categorical),
        ))
    return ColumnTransformer(transformers, remainder="drop")


def fit_probability_head(
    spec: LockVariantSpec,
    train: pd.DataFrame,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] = (),
    label_column: str = "win",
) -> ProbabilityHead:
    """Fit one registered family; callers own chronology and fallback policy."""
    labels = pd.to_numeric(train[label_column], errors="coerce")
    resolved = train.loc[labels.notna()].copy()
    labels = labels.loc[labels.notna()].astype(int)
    if not len(resolved) or labels.nunique() < 2:
        return ConstantHead(shrunken_base_rate(labels))
    if spec.family == "base_rate":
        return ConstantHead(shrunken_base_rate(labels))
    if spec.family == "symmetric_residual":
        trust = float(spec.parameters.get("trust", 0.2))
        if not np.isfinite(trust) or not 0 <= trust <= 1:
            raise ValueError("Residual trust must be between zero and one")
        # Pushes also reveal the score model's error; exclude them only from
        # resolved-win classifiers, not from the score-error distribution.
        errors = pd.to_numeric(train["score_residual"], errors="coerce")
        errors = np.sort(errors.loc[np.isfinite(errors)].abs().to_numpy(float))
        return SymmetricResidualHead(errors, trust) if len(errors) else ConstantHead(0.5)
    if spec.family in {"recorded_raw", "recorded_platt"}:
        raw = ColumnHead("recorded_probability")
        if spec.family == "recorded_raw":
            return raw
        calibrator = fit_calibrator(spec.calibrator, raw.predict(resolved), labels)
        return CalibratedHead(raw, calibrator)
    if spec.family == "empirical_edge":
        boundaries = (1.0, 2.0, 3.0, 5.0, 7.0)
        positions = np.digitize(pd.to_numeric(resolved["edge"], errors="coerce").fillna(0.0), boundaries)
        global_rate = shrunken_base_rate(labels)
        probabilities = []
        strength = float(spec.parameters.get("strength", 50.0))
        for position in range(len(boundaries) + 1):
            selected = labels.to_numpy()[positions == position]
            probabilities.append(float((selected.sum() + global_rate * strength) / (len(selected) + strength)))
        return EmpiricalEdgeHead(boundaries, tuple(probabilities))
    if spec.family == "edge_threshold":
        minimum_edge = float(spec.parameters["minimum_edge"])
        maximum_rank = int(spec.parameters["maximum_rank"])
        strength = float(spec.parameters.get("strength", 10.0))
        global_rate = shrunken_base_rate(labels)
        qualifying_mask = (
            pd.to_numeric(resolved["edge"], errors="coerce").fillna(0.0).ge(minimum_edge)
            & pd.to_numeric(resolved["weekly_edge_rank"], errors="coerce").le(maximum_rank)
        )
        qualifying = labels.loc[qualifying_mask]
        qualifying_rate = float(
            (qualifying.sum() + global_rate * strength) / (len(qualifying) + strength)
        )
        return EdgeThresholdHead(minimum_edge, maximum_rank, global_rate, qualifying_rate)
    if spec.family == "residual_distribution":
        residuals = pd.to_numeric(resolved["score_residual"], errors="coerce").dropna().sort_values().to_numpy(float)
        if not len(residuals):
            return ConstantHead(shrunken_base_rate(labels))
        return ResidualDistributionHead(
            residuals,
            prior_strength=float(spec.parameters.get("strength", 25.0)),
        )

    columns = tuple(dict.fromkeys([*numeric_features, *categorical_features]))
    if spec.family == "spline_logit":
        estimator = Pipeline([
            ("features", _column_pipeline(numeric_features, categorical_features, spline_edge=True)),
            ("model", LogisticRegression(
                C=float(spec.parameters.get("C", 0.1)),
                max_iter=2000,
                random_state=2,
            )),
        ]).fit(resolved.loc[:, columns], labels)
        return SklearnHead(estimator, columns)
    if spec.family == "lightgbm":
        from lightgbm import LGBMClassifier

        estimator = Pipeline([
            ("features", _column_pipeline(numeric_features, categorical_features, spline_edge=False)),
            ("model", LGBMClassifier(
                n_estimators=int(spec.parameters.get("n_estimators", 50)),
                max_depth=int(spec.parameters.get("max_depth", 2)),
                learning_rate=float(spec.parameters.get("learning_rate", 0.03)),
                num_leaves=7,
                min_child_samples=int(spec.parameters.get("min_child_samples", 40)),
                reg_lambda=10.0,
                verbosity=-1,
                random_state=2,
                n_jobs=1,
            )),
        ]).fit(resolved.loc[:, columns], labels)
        return SklearnHead(estimator, columns)
    raise ValueError(f"Unsupported lock model family: {spec.family}")
