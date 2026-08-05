"""Reusable dataframe contracts for model data boundaries."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import pandas as pd


LOGGER = logging.getLogger(__name__)


class DataContractError(ValueError):
    """Raised when a model input violates a declared data contract."""


def enforce_contract(problems: Sequence[str], *, strict: bool) -> bool:
    """Raise or warn for collected contract problems.

    Returns ``True`` when no problems were found so callers can use the result
    in diagnostics without duplicating the strict/warn policy.
    """
    if not problems:
        return True
    message = "; ".join(problems)
    if strict:
        raise DataContractError(message)
    LOGGER.warning("Data contract warning: %s", message)
    return False


def validate_frame(
    frame: pd.DataFrame,
    *,
    label: str,
    required_columns: Iterable[str] = (),
    non_null_columns: Iterable[str] = (),
    unique_keys: Iterable[Sequence[str]] = (),
    expected_seasons: Iterable[int] | None = None,
    season_col: str = "season",
    allow_empty: bool = False,
    strict: bool = True,
) -> bool:
    """Validate only structural facts that are stable across seasons."""
    if frame.empty and allow_empty:
        return True
    problems: list[str] = []
    required = tuple(required_columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        problems.append(f"{label} is missing required columns: {', '.join(missing)}")

    if frame.empty and not allow_empty:
        problems.append(f"{label} is empty")

    available_non_null = [column for column in non_null_columns if column in frame.columns]
    null_columns = [column for column in available_non_null if frame[column].isna().any()]
    if null_columns:
        problems.append(f"{label} contains null values in: {', '.join(null_columns)}")

    for key in unique_keys:
        columns = tuple(key)
        if columns and all(column in frame.columns for column in columns):
            duplicate_count = int(frame.duplicated(subset=list(columns), keep=False).sum())
            if duplicate_count:
                problems.append(
                    f"{label} contains {duplicate_count} rows with duplicate key "
                    f"({', '.join(columns)})"
                )

    if expected_seasons is not None and season_col in frame.columns:
        expected = {int(season) for season in expected_seasons}
        actual = {
            int(season)
            for season in pd.to_numeric(frame[season_col], errors="coerce").dropna().unique()
        }
        if actual != expected:
            problems.append(
                f"{label} season coverage differs "
                f"(missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)})"
            )

    return enforce_contract(problems, strict=strict)


def validate_feature_coverage(
    frame: pd.DataFrame,
    *,
    label: str,
    feature_columns: Iterable[str],
    prediction_mask: pd.Series,
    strict: bool = True,
) -> bool:
    """Ensure eligible prediction rows are not silently all-imputed."""
    features = tuple(feature_columns)
    problems: list[str] = []
    if not features:
        problems.append(f"{label} has no monitored prediction features")
    missing = [column for column in features if column not in frame.columns]
    if missing:
        problems.append(f"{label} is missing prediction features: {', '.join(missing)}")
    prediction_rows = frame.loc[prediction_mask]
    if prediction_rows.empty:
        problems.append(f"{label} has no eligible prediction rows")
    available = [column for column in features if column in frame.columns]
    if available and not prediction_rows.empty:
        all_null_rows = int(prediction_rows[available].isna().all(axis=1).sum())
        if all_null_rows:
            problems.append(
                f"{label} has {all_null_rows} prediction rows with all monitored features null"
            )
    return enforce_contract(problems, strict=strict)


__all__ = [
    "DataContractError",
    "enforce_contract",
    "validate_feature_coverage",
    "validate_frame",
]
