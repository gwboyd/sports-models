"""Chronological split helpers that keep each game in one partition."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit

from src.sports.football.kickoff import parse_eastern_kickoffs


def sort_chronologically(frame: pd.DataFrame, *, time_col: str) -> pd.DataFrame:
    if time_col not in frame.columns:
        raise ValueError(f"Training frame is missing chronological column: {time_col}")
    output = frame.copy()
    output["__kickoff_utc"] = parse_eastern_kickoffs(output[time_col])
    if output["__kickoff_utc"].isna().any():
        raise ValueError(f"Training frame contains invalid values in {time_col}")
    tie_breakers = [column for column in ("season", "week", "game_id") if column in output.columns]
    output = output.sort_values(["__kickoff_utc", *tie_breakers], kind="stable")
    return output.drop(columns="__kickoff_utc")


def chronological_train_test_split(
    frame: pd.DataFrame,
    *,
    time_col: str,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("Chronological test_size must be between 0 and 1")
    ordered = sort_chronologically(frame, time_col=time_col)
    if len(ordered) < 2:
        raise ValueError("Chronological splitting requires at least two completed games")
    target_test_rows = max(1, math.ceil(len(ordered) * test_size))
    kickoffs = parse_eastern_kickoffs(ordered[time_col])
    unique_kickoffs = kickoffs.drop_duplicates().sort_values()
    if len(unique_kickoffs) < 2:
        raise ValueError("Chronological split requires at least two distinct kickoff times")

    candidates: list[tuple[int, int, pd.Timestamp]] = []
    for cutoff in unique_kickoffs.iloc[1:]:
        candidate_rows = int((kickoffs >= cutoff).sum())
        candidates.append(
            (
                abs(candidate_rows - target_test_rows),
                0 if candidate_rows >= target_test_rows else 1,
                cutoff,
            )
        )
    _, _, cutoff = min(candidates)
    test_mask = kickoffs >= cutoff
    train = ordered.loc[~test_mask].copy()
    test = ordered.loc[test_mask].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split leaves an empty partition")
    return train, test


def predefined_chronological_split(
    frame: pd.DataFrame,
    *,
    time_col: str,
    test_size: float,
) -> PredefinedSplit:
    """Create one train-before-validation split for an already ordered game frame."""
    if not frame.index.is_unique:
        raise ValueError("Chronological split requires unique dataframe indices")
    train, validation = chronological_train_test_split(
        frame,
        time_col=time_col,
        test_size=test_size,
    )
    validation_indices = set(validation.index)
    test_fold = np.array([0 if index in validation_indices else -1 for index in frame.index])
    if int((test_fold == -1).sum()) != len(train):
        raise ValueError("Chronological split requires unique dataframe indices")
    return PredefinedSplit(test_fold)


__all__ = [
    "chronological_train_test_split",
    "predefined_chronological_split",
    "sort_chronologically",
]
