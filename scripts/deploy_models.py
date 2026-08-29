#!/usr/bin/env python3
"""Interactive, version-aware production deployment for the football models."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_patterns.expected_points.versioning import (
    ModelKey,
    ModelVersion,
    ReleaseDraft,
    build_archive_markdown,
    next_major,
    next_minor,
    parse_release_file,
    parse_version,
)
from src.utils.db.sports_models_db import (
    get_latest_model_release,
    get_model_release,
    insert_model_releases,
)


RECOVERY_PATH = ROOT / ".aws-sam" / "model-release-plan.json"
AWS_REGION = "us-east-1"
TRAINING_FUNCTION_NAME = "sports-models-training-v2"
AWS_VERIFICATION_DELAYS = (0, 1, 2, 4, 8)


@dataclass(frozen=True)
class ModelSpec:
    key: ModelKey
    label: str
    draft_path: Path
    archive_dir: Path
    sam_parameter: str
    lambda_environment_variable: str


@dataclass(frozen=True)
class DeploymentChoice:
    spec: ModelSpec
    current_version: ModelVersion
    action: str
    proposed_version: ModelVersion
    draft: ReleaseDraft | None
    draft_text: str
    draft_hash: str


MODEL_SPECS = (
    ModelSpec(
        ModelKey.NFL_EXPECTED_POINTS,
        "NFL expected points",
        ROOT / "src/sports/football/nfl/expected_points/UNRELEASED.md",
        ROOT / "src/sports/football/nfl/expected_points/releases",
        "NflExpectedPointsVersion",
        "NFL_EXPECTED_POINTS_VERSION",
    ),
    ModelSpec(
        ModelKey.CFB_EXPECTED_POINTS,
        "CFB expected points",
        ROOT / "src/sports/football/cfb/expected_points/UNRELEASED.md",
        ROOT / "src/sports/football/cfb/expected_points/releases",
        "CfbExpectedPointsVersion",
        "CFB_EXPECTED_POINTS_VERSION",
    ),
)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _ensure_clean_tree() -> None:
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "Production deployment requires every change to be committed; "
            f"uncommitted changes found:\n{status}"
        )
    branch = _git("branch", "--show-current")
    if branch != "main":
        raise RuntimeError(
            "Production deployment requires the checked-out branch to be main; "
            f"current branch is {branch or 'detached HEAD'}"
        )


def _draft_snapshot(spec: ModelSpec) -> tuple[ReleaseDraft | None, str, str]:
    if not spec.draft_path.exists():
        raise RuntimeError(f"Missing required release queue file: {spec.draft_path}")
    text = spec.draft_path.read_text(encoding="utf-8")
    draft = parse_release_file(spec.draft_path)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return draft, text, digest


def _choice(prompt: str, valid: tuple[str, ...], input_fn: Callable[[str], str] = input) -> str:
    valid_display = "/".join(valid)
    while True:
        value = input_fn(f"{prompt} [{valid_display}]: ").strip().lower()
        if value in valid:
            return value
        print(f"Please enter one of: {valid_display}")


def build_deployment_choices(
    releases: dict[ModelKey, dict],
    drafts: dict[ModelKey, tuple[ReleaseDraft | None, str, str]],
    *,
    input_fn: Callable[[str], str] = input,
) -> list[DeploymentChoice]:
    choices: list[DeploymentChoice] = []
    for spec in MODEL_SPECS:
        release = releases.get(spec.key)
        if not release:
            raise RuntimeError(f"No baseline release registered for {spec.key.value}")
        current = parse_version(str(release["version"]))
        draft, draft_text, draft_hash = drafts[spec.key]
        if draft is None:
            action = "keep"
            proposed = current
        else:
            print(f"\n{spec.label} has unreleased changes in {spec.draft_path}")
            action = _choice("Release as", ("major", "minor", "abort"), input_fn)
            if action == "abort":
                raise RuntimeError("Deployment aborted during model release selection")
            proposed = next_major(current) if action == "major" else next_minor(current)
        choices.append(DeploymentChoice(spec, current, action, proposed, draft, draft_text, draft_hash))
    return choices


def _print_plan(choices: list[DeploymentChoice], source_git_sha: str) -> None:
    print("\nAWS deployment plan")
    print(f"Git SHA: {source_git_sha}")
    for choice in choices:
        draft_state = "empty" if choice.draft is None else choice.draft.title
        print(f"\n{choice.spec.label}")
        print(f"  Draft: {draft_state}")
        print(f"  Action: {choice.action}")
        print(f"  Version: {choice.current_version} -> {choice.proposed_version}")


def _serialize_choice(choice: DeploymentChoice) -> dict:
    return {
        "model_key": choice.spec.key.value,
        "label": choice.spec.label,
        "draft_path": str(choice.spec.draft_path.relative_to(ROOT)),
        "archive_dir": str(choice.spec.archive_dir.relative_to(ROOT)),
        "action": choice.action,
        "current_version": str(choice.current_version),
        "proposed_version": str(choice.proposed_version),
        "draft_text": choice.draft_text,
        "draft_hash": choice.draft_hash,
        "draft": asdict(choice.draft) if choice.draft else None,
    }


def _write_recovery_payload(payload: dict) -> None:
    RECOVERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RECOVERY_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(RECOVERY_PATH)


def _write_recovery_plan(choices: list[DeploymentChoice], source_git_sha: str) -> None:
    _write_recovery_payload({
        "source_git_sha": source_git_sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deployed_at": None,
        "choices": [_serialize_choice(choice) for choice in choices],
    })


def _draft_from_serialized(value: dict | None) -> ReleaseDraft | None:
    if value is None:
        return None
    return ReleaseDraft(**value)


def _load_recovery_plan() -> tuple[str, str | None, list[dict]]:
    if not RECOVERY_PATH.exists():
        raise RuntimeError(f"No deployment recovery plan exists at {RECOVERY_PATH}")
    payload = json.loads(RECOVERY_PATH.read_text(encoding="utf-8"))
    return payload["source_git_sha"], payload.get("deployed_at"), payload["choices"]


def _record_verified_deployment() -> str:
    if not RECOVERY_PATH.exists():
        raise RuntimeError(f"No deployment recovery plan exists at {RECOVERY_PATH}")
    payload = json.loads(RECOVERY_PATH.read_text(encoding="utf-8"))
    deployed_at = payload.get("deployed_at") or datetime.now(timezone.utc).isoformat()
    payload["deployed_at"] = deployed_at
    _write_recovery_payload(payload)
    return deployed_at


def register_releases(
    source_git_sha: str,
    deployed_at: str,
    serialized_choices: list[dict],
) -> None:
    deployed_at_value = datetime.fromisoformat(deployed_at.replace("Z", "+00:00"))
    pending = []
    for item in serialized_choices:
        if item["action"] == "keep":
            continue
        draft = _draft_from_serialized(item["draft"])
        if draft is None:
            raise RuntimeError(f"Release choice for {item['model_key']} has no draft")
        pending.append((ModelKey(item["model_key"]), item["proposed_version"], draft))
    insert_model_releases(
        pending,
        source_git_sha=source_git_sha,
        deployed_at=deployed_at_value,
    )


def verify_registered_releases(source_git_sha: str, serialized_choices: list[dict]) -> None:
    for item in serialized_choices:
        if item["action"] == "keep":
            continue
        draft = _draft_from_serialized(item["draft"])
        if draft is None:
            raise RuntimeError(f"Release choice for {item['model_key']} has no draft")
        release = get_model_release(item["model_key"], item["proposed_version"])
        if release is None:
            raise RuntimeError(
                f"Supabase release {item['model_key']} {item['proposed_version']} is missing"
            )
        expected = {
            "title": draft.title,
            "public_summary": draft.public_summary,
            "changes_md": draft.changes_md,
            "evaluation_md": draft.evaluation_md,
            "internal_notes_md": draft.internal_notes_md,
            "source_git_sha": source_git_sha,
        }
        mismatches = [
            field for field, value in expected.items() if release.get(field) != value
        ]
        if mismatches:
            raise RuntimeError(
                f"Supabase release {item['model_key']} {item['proposed_version']} "
                f"does not match the deployment plan: {', '.join(mismatches)}"
            )


def finalize_local_drafts(
    source_git_sha: str,
    deployed_at: str,
    serialized_choices: list[dict],
) -> None:
    verify_registered_releases(source_git_sha, serialized_choices)
    for item in serialized_choices:
        if item["action"] == "keep":
            continue
        draft = _draft_from_serialized(item["draft"])
        if draft is None:
            continue
        spec = next(spec for spec in MODEL_SPECS if spec.key.value == item["model_key"])
        current_text = spec.draft_path.read_text(encoding="utf-8")
        if hashlib.sha256(current_text.encode("utf-8")).hexdigest() != item["draft_hash"]:
            print(f"{spec.label}: draft changed during deployment; preserving new notes")
            archive_text = build_archive_markdown(
                draft,
                model_key=spec.key,
                version=parse_version(item["proposed_version"]),
                deployed_at=deployed_at,
                source_git_sha=source_git_sha,
            )
            archive_path = spec.archive_dir / f"v{item['proposed_version']}.md"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_text(archive_text, encoding="utf-8")
            continue
        archive_text = build_archive_markdown(
            draft,
            model_key=spec.key,
            version=parse_version(item["proposed_version"]),
            deployed_at=deployed_at,
            source_git_sha=source_git_sha,
        )
        archive_path = spec.archive_dir / f"v{item['proposed_version']}.md"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_text(archive_text, encoding="utf-8")
        spec.draft_path.write_text("", encoding="utf-8")
        print(f"{spec.label}: archived {archive_path.relative_to(ROOT)} and reset draft")


def _deployment_secret_parameters() -> list[str]:
    parameters = []
    for env_name, parameter in (
        ("ADMIN_API_KEY", "AdminApiKey"),
        ("FRONT_END_API_KEY", "FrontEndApiKey"),
        ("READ_API_KEY", "ReadApiKey"),
        ("NBA_API_KEY", "NbaApiKey"),
        ("AWS_API_KEY", "AwsApiKey"),
        ("CFBD_API_KEY", "CfbdApiKey"),
        ("SUPABASE_DB_URL", "SupabaseDbUrl"),
        ("SUPABASE_SCHEMA", "SupabaseSchema"),
    ):
        value = os.getenv(env_name)
        if not value:
            raise RuntimeError(f"Missing required deployment environment variable {env_name}")
        parameters.append(f"{parameter}={value}")
    return parameters


def _expected_lambda_environment(
    source_git_sha: str,
    serialized_choices: list[dict],
) -> dict[str, str]:
    expected = {"SOURCE_GIT_SHA": source_git_sha}
    specs = {spec.key.value: spec for spec in MODEL_SPECS}
    for item in serialized_choices:
        spec = specs.get(item["model_key"])
        if spec is None:
            raise RuntimeError(f"Unsupported model key in deployment plan: {item['model_key']}")
        expected[spec.lambda_environment_variable] = item["proposed_version"]
    return expected


def verify_aws_deployment(
    source_git_sha: str,
    serialized_choices: list[dict],
    *,
    delays: tuple[int, ...] = AWS_VERIFICATION_DELAYS,
) -> None:
    expected_environment = _expected_lambda_environment(source_git_sha, serialized_choices)
    last_problem = "AWS Lambda configuration was not available"
    for delay in delays:
        if delay:
            time.sleep(delay)
        try:
            result = subprocess.run(
                [
                    "aws",
                    "lambda",
                    "get-function-configuration",
                    "--function-name",
                    TRAINING_FUNCTION_NAME,
                    "--region",
                    AWS_REGION,
                    "--output",
                    "json",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            configuration = json.loads(result.stdout)
        except FileNotFoundError as exc:
            raise RuntimeError("AWS CLI is required to verify the deployed Lambda") from exc
        except (json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            last_problem = f"AWS configuration lookup failed ({type(exc).__name__})"
            continue

        update_status = configuration.get("LastUpdateStatus")
        if update_status == "Failed":
            reason = configuration.get("LastUpdateStatusReason") or "no failure reason returned"
            raise RuntimeError(f"AWS training Lambda update failed: {reason}")

        function_state = configuration.get("State")
        actual_environment = configuration.get("Environment", {}).get("Variables", {})
        mismatches = [
            key
            for key, expected_value in expected_environment.items()
            if actual_environment.get(key) != expected_value
        ]
        if update_status == "Successful" and function_state == "Active" and not mismatches:
            print("AWS training Lambda version and Git SHA verified.")
            return

        problems = []
        if update_status != "Successful":
            problems.append(f"LastUpdateStatus={update_status!r}")
        if function_state != "Active":
            problems.append(f"State={function_state!r}")
        if mismatches:
            problems.append(f"environment mismatch: {', '.join(mismatches)}")
        last_problem = "; ".join(problems)

    raise RuntimeError(
        "AWS deployment could not be verified after bounded retries: " + last_problem
    )


def _sam_deploy(
    source_git_sha: str,
    choices: list[DeploymentChoice],
    secret_parameters: list[str],
) -> None:
    versions = {
        choice.spec.sam_parameter: str(choice.proposed_version)
        for choice in choices
    }
    parameters = [
        "HttpApiName=sports-models-http-api-v2",
        "ApiFunctionName=sports-models-api-v2",
        "TrainingFunctionName=sports-models-training-v2",
        "Localhost=False",
        "EnvironmentName=PROD",
        f"SourceGitSha={source_git_sha}",
        *[f"{key}={value}" for key, value in versions.items()],
        *secret_parameters,
    ]

    subprocess.run(["sam", "build"], cwd=ROOT, check=True)
    subprocess.run(
        [
            "sam", "deploy", "--stack-name", "sports-models-v2", "--region", AWS_REGION,
            "--resolve-s3", "--resolve-image-repos", "--capabilities", "CAPABILITY_IAM",
            "--no-confirm-changeset", "--no-fail-on-empty-changeset",
            "--parameter-overrides", *parameters,
        ],
        cwd=ROOT,
        check=True,
    )


def _main() -> int:
    parser = argparse.ArgumentParser()
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--register-only", action="store_true")
    recovery.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    if args.register_only or args.finalize_only:
        source_git_sha, deployed_at, serialized_choices = _load_recovery_plan()
        if args.register_only:
            verify_aws_deployment(source_git_sha, serialized_choices)
            deployed_at = deployed_at or _record_verified_deployment()
            register_releases(source_git_sha, deployed_at, serialized_choices)
            verify_registered_releases(source_git_sha, serialized_choices)
        if args.finalize_only:
            if deployed_at is None:
                raise RuntimeError(
                    "Recovery plan has no verified AWS deployment timestamp; "
                    "run make sam-register-releases first"
                )
            finalize_local_drafts(source_git_sha, deployed_at, serialized_choices)
            RECOVERY_PATH.unlink(missing_ok=True)
        return 0

    _ensure_clean_tree()
    source_git_sha = _git("rev-parse", "HEAD")
    releases = {spec.key: get_latest_model_release(spec.key) for spec in MODEL_SPECS}
    drafts = {spec.key: _draft_snapshot(spec) for spec in MODEL_SPECS}
    choices = build_deployment_choices(releases, drafts)
    _print_plan(choices, source_git_sha)
    confirmation = input("\nProceed with SAM build and deployment? [y/N]: ").strip().lower()
    if confirmation not in {"y", "yes"}:
        print("Deployment cancelled; no files or database rows changed.")
        return 0

    secret_parameters = _deployment_secret_parameters()
    _write_recovery_plan(choices, source_git_sha)
    _sam_deploy(source_git_sha, choices, secret_parameters)
    serialized_choices = [_serialize_choice(choice) for choice in choices]
    verify_aws_deployment(source_git_sha, serialized_choices)
    deployed_at = _record_verified_deployment()
    register_releases(source_git_sha, deployed_at, serialized_choices)
    finalize_local_drafts(source_git_sha, deployed_at, serialized_choices)
    RECOVERY_PATH.unlink(missing_ok=True)
    print("AWS deployment and model-release registration completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except subprocess.CalledProcessError as exc:
        print(
            f"Deployment command failed with exit code {exc.returncode}; "
            "the recovery plan was retained if confirmation had completed.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except (EOFError, RuntimeError, ValueError) as exc:
        print(f"Deployment failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
