"""Write authorization for expected-points notebook executions."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import subprocess
from typing import Callable, Mapping

from .types import ExpectedPointsLeague
from .versioning import ModelKey, parse_version


logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ExpectedPointsWritePlan:
    should_write: bool
    model_version: str | None
    source_git_sha: str | None
    is_aws_runtime: bool
    requires_interactive_confirmation: bool = False
    release_source_git_sha: str | None = None


def is_aws_lambda_runtime(environment: Mapping[str, str] | None = None) -> bool:
    """Identify a real Lambda runtime while excluding SAM's local emulator."""

    values = os.environ if environment is None else environment
    return bool(values.get("AWS_LAMBDA_FUNCTION_NAME")) and (
        values.get("AWS_SAM_LOCAL", "").lower() != "true"
    )


def _model_key(league: ExpectedPointsLeague) -> ModelKey:
    if league is ExpectedPointsLeague.CFB:
        return ModelKey.CFB_EXPECTED_POINTS
    return ModelKey.NFL_EXPECTED_POINTS


def _local_git_identity() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def resolve_expected_points_write_plan(
    league: ExpectedPointsLeague,
    *,
    client_name: str,
    allow_non_aws_write: bool,
) -> ExpectedPointsWritePlan:
    """Resolve the version before tracking and decide whether a write is eligible."""

    if is_aws_lambda_runtime():
        version = os.getenv("EXPECTED_POINTS_MODEL_VERSION")
        if not version:
            raise RuntimeError("AWS expected-points writes require a deployed model version")
        return ExpectedPointsWritePlan(
            should_write=True,
            model_version=str(parse_version(version)),
            source_git_sha=os.getenv("SOURCE_GIT_SHA") or "unknown",
            is_aws_runtime=True,
        )

    if not allow_non_aws_write:
        return ExpectedPointsWritePlan(
            should_write=False,
            model_version=None,
            source_git_sha=None,
            is_aws_runtime=False,
        )

    # Import lazily so read-only notebooks do not open a database connection
    # merely to decide that they will not persist.
    from src.utils.db.sports_models_db import get_latest_model_release

    release = get_latest_model_release(_model_key(league))
    if release is None:
        raise RuntimeError(f"No registered release exists for {league.value} expected points")
    version = str(parse_version(str(release["version"])))
    return ExpectedPointsWritePlan(
        should_write=True,
        model_version=version,
        source_git_sha=_local_git_identity(),
        is_aws_runtime=False,
        requires_interactive_confirmation=client_name == "notebook",
        release_source_git_sha=release.get("source_git_sha"),
    )


def confirm_expected_points_write(
    plan: ExpectedPointsWritePlan,
    league: ExpectedPointsLeague,
    *,
    season: int,
    week: int,
    picks_num: int,
    pick_changes: int,
    play_changes: int,
    input_fn: Callable[[str], str] = input,
) -> bool:
    """Warn for an authorized non-AWS write and confirm interactive notebooks."""

    if not plan.should_write:
        return False
    if plan.is_aws_runtime:
        return True

    warning = (
        "NON-AWS EXPECTED-POINTS WRITE ENABLED\n"
        f"League: {league.value.upper()} | Season/week: {season}/{week}\n"
        f"Version: {plan.model_version} | Picks: {picks_num} | "
        f"Pick changes: {pick_changes} | Play changes: {play_changes}\n"
        f"Local source: {plan.source_git_sha} | "
        f"Release source: {plan.release_source_git_sha or 'unrecorded'}"
    )
    logger.warning(warning.replace("\n", " | "))
    if not plan.requires_interactive_confirmation:
        return True

    phrase = f"WRITE {league.value.upper()} {plan.model_version}"
    response = input_fn(f"{warning}\nType {phrase} to continue: ").strip()
    if response != phrase:
        raise RuntimeError("Non-AWS expected-points write cancelled; no database changes were made")
    return True


__all__ = [
    "ExpectedPointsWritePlan",
    "confirm_expected_points_write",
    "is_aws_lambda_runtime",
    "resolve_expected_points_write_plan",
]
