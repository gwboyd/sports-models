#!/usr/bin/env python3
"""Prepare football releases before merge, then deploy their committed versions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_patterns.expected_points.versioning import (
    ModelKey,
    ModelVersion,
    ReleaseDraft,
    next_major,
    next_minor,
    parse_release_markdown,
    parse_version,
)
from src.utils.db.sports_models_db import (
    get_latest_model_release,
    get_model_release,
    initialize_model_release_source,
    insert_model_releases,
    verify_manual_update_schema,
)


RECOVERY_PATH = ROOT / ".aws-sam" / "model-release-plan.json"
VERSIONS_PATH = ROOT / "model-versions.json"
AWS_REGION = "us-east-1"
API_FUNCTION_NAME = "sports-models-api-v2"
TRAINING_FUNCTION_NAME = "sports-models-training-v2"
COORDINATOR_FUNCTION_NAME = "sports-models-schedule-coordinator-v2"
TRAINING_SCHEDULE_GROUP_NAME = "sports-models-training-updates-v2"
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
    draft = parse_release_markdown(text, source=str(spec.draft_path))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return draft, text, digest


def _choice(prompt: str, valid: tuple[str, ...], input_fn: Callable[[str], str] = input) -> str:
    valid_display = "/".join(valid)
    while True:
        value = input_fn(f"{prompt} [{valid_display}]: ").strip().lower()
        if value in valid:
            return value
        print(f"Please enter one of: {valid_display}")


def build_preparation_choices(
    releases: dict[ModelKey, dict],
    drafts: dict[ModelKey, tuple[ReleaseDraft | None, str, str]],
    *,
    prepared_versions: dict[ModelKey, ModelVersion] | None = None,
    input_fn: Callable[[str], str] = input,
) -> list[DeploymentChoice]:
    choices: list[DeploymentChoice] = []
    for spec in MODEL_SPECS:
        release = releases.get(spec.key)
        if not release:
            raise RuntimeError(f"No baseline release registered for {spec.key.value}")
        current = parse_version(str(release["version"]))
        prepared = (prepared_versions or {}).get(spec.key, current)
        draft, draft_text, draft_hash = drafts[spec.key]
        if draft is None:
            action = "keep"
            proposed = prepared
        elif prepared > current:
            action, proposed = "amend", prepared
            existing = _release_snapshot(spec, prepared)[1]
            print(f"\n{spec.label}: amend unpublished {prepared}; review the consolidated notes:")
            print("".join(difflib.unified_diff(
                existing.splitlines(keepends=True), draft_text.splitlines(keepends=True),
                fromfile=f"releases/v{prepared}.md", tofile="UNRELEASED.md",
            )))
        else:
            print(f"\n{spec.label} has unreleased changes in {spec.draft_path}")
            action = _choice("Release as", ("major", "minor", "abort"), input_fn)
            if action == "abort":
                raise RuntimeError("Preparation aborted during model release selection")
            proposed = next_major(current) if action == "major" else next_minor(current)
        choices.append(DeploymentChoice(spec, current, action, proposed, draft, draft_text, draft_hash))
    return choices


def _is_release_review_input(path: str, spec: ModelSpec) -> bool:
    """Broad review signals, never an automatic classification of prediction impact."""
    if path in {"main.py", "Dockerfile", "template.yaml", ".python-version", "pyproject.toml", "uv.lock"}:
        return True
    if path.startswith("requirements") and path.endswith(".txt"):
        return True
    if not path.startswith("src/") or path.endswith(".md"):
        return False
    league = spec.key.value.split("_", 1)[0]
    if path.startswith("src/sports/football/"):
        return not any(path.startswith(f"src/sports/football/{other}/")
                       for other in ("nfl", "cfb") if other != league)
    return not path.startswith("src/sports/basketball/")


def source_change_signals(spec: ModelSpec, source_git_sha: str | None) -> tuple[list[str], str | None]:
    """Compare tracked and untracked inputs with the immutable registered recipe."""
    if not source_git_sha:
        return [], "The registered bootstrap release has no source SHA; review prediction impact manually."
    try:
        paths = _git("diff", "--no-renames", "--name-only", "-z", source_git_sha, "--").split("\0")
        paths += _git("ls-files", "--others", "--exclude-standard", "-z").split("\0")
    except subprocess.CalledProcessError:
        raise RuntimeError(
            f"{spec.label}: cannot inspect registered source {source_git_sha}. "
            "Fetch that Git history before preparing or deploying."
        ) from None
    return sorted({path for path in paths if _is_release_review_input(path, spec)}), None


def review_source_changes(
    choices: list[DeploymentChoice], releases: dict[ModelKey, dict], *,
    input_fn: Callable[[str], str] = input,
) -> None:
    """Make keeping an unchanged version an explicit decision when source changed."""
    for choice in choices:
        paths, warning = source_change_signals(choice.spec, releases[choice.spec.key].get("source_git_sha"))
        if not paths and not warning:
            continue
        print(f"\n{choice.spec.label}: review changes since registered {choice.current_version}:")
        if warning:
            print(f"  {warning}")
        for path in paths:
            print(f"  {path}")
        league = choice.spec.key.value.split("_", 1)[0]
        print(f"Review the release notes and frontend/content/{league}-how-it-works.md. "
              "File changes are signals, not proof that predictions changed.")
        if choice.proposed_version == choice.current_version:
            answer = _choice(
                f"Keep {choice.current_version} only if these changes are prediction-neutral",
                ("neutral", "release", "abort"), input_fn,
            )
            if answer != "neutral":
                raise RuntimeError(
                    f"{choice.spec.label}: add prediction changes to UNRELEASED.md, "
                    "update How It Works, and run make prepare-model-release before commit/merge."
                )


def _print_plan(choices: list[DeploymentChoice], source_git_sha: str | None = None) -> None:
    print("\nAWS deployment plan" if source_git_sha else "\nLocal release preparation")
    if source_git_sha:
        print(f"Git SHA: {source_git_sha}")
    for choice in choices:
        draft_state = "unchanged" if choice.draft is None else choice.draft.title
        print(f"\n{choice.spec.label}")
        print(f"  Notes: {draft_state}")
        print(f"  Action: {choice.action}")
        print(f"  Version: {choice.current_version} -> {choice.proposed_version}")


def load_prepared_versions() -> dict[ModelKey, ModelVersion]:
    """The committed manifest states intent; Supabase states deployment history."""
    payload = json.loads(VERSIONS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {spec.key.value for spec in MODEL_SPECS}:
        raise ValueError("model-versions.json must contain exactly the NFL and CFB model keys")
    versions = {}
    for key, value in payload.items():
        if not isinstance(value, str) or str(parse_version(value)) != value:
            raise ValueError(f"Non-canonical prepared version for {key}: {value!r}")
        versions[ModelKey(key)] = parse_version(value)
    return versions


def _release_snapshot(spec: ModelSpec, version: ModelVersion) -> tuple[ReleaseDraft, str, str]:
    path = spec.archive_dir / f"v{version}.md"
    text = path.read_text(encoding="utf-8")
    # Releases through 2.0 carried deployment metadata. Preserve those files;
    # new prepared notes are exact draft copies with no claims about deployment.
    first, _, rest = text.partition("\n")
    if first.startswith("<!-- model_key:"):
        if not first.startswith(f"<!-- model_key: {spec.key.value}; version: {version};") or not first.endswith("-->"):
            raise ValueError(f"Release metadata does not match {path}")
        text = rest.lstrip("\n")
    draft = parse_release_markdown(text, source=str(path))
    if draft is None:
        raise ValueError(f"Prepared release notes are empty: {path}")
    return draft, text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verify_release_notes(draft: ReleaseDraft, release: dict, label: str) -> None:
    mismatches = [field for field in (
        "title", "public_summary", "changes_md", "evaluation_md", "internal_notes_md",
    ) if getattr(draft, field) != release.get(field)]
    if mismatches:
        raise RuntimeError(f"{label}: registered release notes are immutable ({', '.join(mismatches)})")


def build_deployment_choices(releases: dict[ModelKey, dict]) -> list[DeploymentChoice]:
    """Validate prepared files against the registry before any AWS changes."""
    versions = load_prepared_versions()
    choices = []
    for spec in MODEL_SPECS:
        if _draft_snapshot(spec)[0] is not None:
            raise RuntimeError(f"{spec.label}: run make prepare-model-release before committing/merging")
        release = releases.get(spec.key)
        if not release:
            raise RuntimeError(f"No baseline release registered for {spec.key.value}")
        current, proposed = parse_version(release["version"]), versions[spec.key]
        if proposed < current:
            raise RuntimeError(f"{spec.label}: prepared version {proposed} is older than registered {current}; update from main")
        draft, text, digest = _release_snapshot(spec, proposed)
        if proposed == current:
            _verify_release_notes(draft, release, spec.label)
            action = "keep"
        elif proposed == next_major(current):
            action = "major"
        elif proposed == next_minor(current):
            action = "minor"
        else:
            raise RuntimeError(f"{spec.label}: {current} -> {proposed} is not the next major or minor release")
        choices.append(DeploymentChoice(spec, current, action, proposed, draft, text, digest))
    return choices


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def prepare_model_releases(*, input_fn: Callable[[str], str] = input) -> None:
    """Read registry state, then prepare all selected releases locally together."""
    if RECOVERY_PATH.exists():
        raise RuntimeError("Resolve the existing deployment recovery plan before preparing another release")
    original_manifest = VERSIONS_PATH.read_text(encoding="utf-8")
    versions = load_prepared_versions()
    releases = {spec.key: get_latest_model_release(spec.key) for spec in MODEL_SPECS}
    registered_versions = {key: (row or {}).get("version") for key, row in releases.items()}
    drafts = {spec.key: _draft_snapshot(spec) for spec in MODEL_SPECS}
    original_notes = {
        spec.archive_dir / f"v{versions[spec.key]}.md":
        (spec.archive_dir / f"v{versions[spec.key]}.md").read_text(encoding="utf-8")
        for spec in MODEL_SPECS
    }
    for spec in MODEL_SPECS:
        row = releases.get(spec.key)
        if not row:
            raise RuntimeError(f"No baseline release registered for {spec.key.value}")
        current = parse_version(row["version"])
        if versions[spec.key] < current:
            raise RuntimeError(f"{spec.label}: prepared version is older than registered {current}; update from main")
        existing, _, _ = _release_snapshot(spec, versions[spec.key])
        if versions[spec.key] == current:
            _verify_release_notes(existing, row, spec.label)
        else:
            if versions[spec.key] not in (next_major(current), next_minor(current)):
                raise RuntimeError(f"{spec.label}: invalid prepared version {versions[spec.key]}")
            print(f"{spec.label}: keeping prepared {versions[spec.key]} (not yet registered)")
    choices = build_preparation_choices(releases, drafts, prepared_versions=versions, input_fn=input_fn)
    review_source_changes(choices, releases, input_fn=input_fn)
    updates: dict[Path, str] = {}
    for choice in choices:
        if choice.action == "keep":
            continue
        archive = choice.spec.archive_dir / f"v{choice.proposed_version}.md"
        if archive.exists() and choice.action != "amend":
            raise RuntimeError(f"Refusing to overwrite existing release notes: {archive}")
        updates[archive] = choice.draft_text
        versions[choice.spec.key] = choice.proposed_version
    if not updates:
        print("No unprepared release notes; no files changed.")
        return
    _print_plan(choices)
    if input_fn("\nPrepare these release files locally? [y/N]: ").strip().lower() not in {"y", "yes"}:
        print("Preparation cancelled; no files changed.")
        return
    if VERSIONS_PATH.read_text(encoding="utf-8") != original_manifest or any(
        spec.draft_path.read_text(encoding="utf-8") != drafts[spec.key][1] for spec in MODEL_SPECS
    ) or any(path.read_text(encoding="utf-8") != text for path, text in original_notes.items()):
        raise RuntimeError("Release files changed during preparation; retry with the current files")
    for choice in choices:
        if choice.action in {"major", "minor"} and (choice.spec.archive_dir / f"v{choice.proposed_version}.md").exists():
            raise RuntimeError("Release destination appeared during preparation; refusing to overwrite it")
    for spec in MODEL_SPECS:
        latest = get_latest_model_release(spec.key)
        if latest is None or latest["version"] != registered_versions[spec.key]:
            raise RuntimeError("Registered versions changed during preparation; refresh release intent and retry")
    updates[VERSIONS_PATH] = json.dumps({key.value: str(value) for key, value in versions.items()}, indent=2) + "\n"
    # Save notes, then versions, then clear drafts. Even a process kill must not
    # leave empty drafts paired with the old versions and allow an unversioned deploy.
    for choice in choices:
        if choice.action != "keep":
            updates[choice.spec.draft_path] = ""
    previous = {path: path.read_text(encoding="utf-8") if path.exists() else None for path in updates}
    try:
        for path, text in updates.items():
            _write_text_atomic(path, text)
    except BaseException:
        for path, text in previous.items():
            if text is None:
                path.unlink(missing_ok=True)
            else:
                _write_text_atomic(path, text)
        raise
    print("Release files prepared. Review How It Works, commit the manifest/notes/empty drafts with your changes, then merge once and run make sam-deploy on clean main.")


def _serialize_choice(choice: DeploymentChoice) -> dict:
    return {
        "model_key": choice.spec.key.value,
        "action": choice.action,
        "current_version": str(choice.current_version),
        "proposed_version": str(choice.proposed_version),
        "draft_hash": choice.draft_hash,
        "draft": asdict(choice.draft) if choice.draft else None,
    }


def _write_recovery_payload(payload: dict) -> None:
    _write_text_atomic(RECOVERY_PATH, json.dumps(payload, indent=2) + "\n")


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
    try:
        source_git_sha = payload["source_git_sha"]
        if not isinstance(source_git_sha, str) or not source_git_sha.strip():
            raise ValueError("missing source Git SHA")
        choices = payload["choices"]
        if not isinstance(choices, list) or len(choices) != len(MODEL_SPECS) or {
            item["model_key"] for item in choices
        } != {spec.key.value for spec in MODEL_SPECS}:
            raise ValueError("must contain exactly one NFL and one CFB choice")
        for item in choices:
            for field in ("current_version", "proposed_version"):
                version = item[field]
                if not isinstance(version, str) or str(parse_version(version)) != version:
                    raise ValueError("non-canonical model version")
            if item["action"] not in {"keep", "major", "minor"}:
                raise ValueError("unsupported release action")
            current = parse_version(item["current_version"])
            if parse_version(item["proposed_version"]) != {
                "keep": current, "major": next_major(current), "minor": next_minor(current),
            }[item["action"]]:
                raise ValueError("release action does not match its version increment")
            draft = _draft_from_serialized(item["draft"])
            # Pre-preparation recovery plans have no note snapshot for kept versions.
            if draft is None and item["action"] != "keep":
                raise ValueError("new release has no notes")
            markdown = draft.original_markdown if draft is not None else ""
            if draft != parse_release_markdown(markdown, source="recovery notes"):
                raise ValueError("release fields differ from their Markdown snapshot")
            if item["draft_hash"] != hashlib.sha256(markdown.encode("utf-8")).hexdigest():
                raise ValueError("release note hash mismatch")
        deployed_at = payload.get("deployed_at")
        if deployed_at is not None and datetime.fromisoformat(deployed_at.replace("Z", "+00:00")).utcoffset() is None:
            raise ValueError("deployment time must include a timezone")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid deployment recovery plan: {exc}. Restore the original plan; do not discard it.") from exc
    return source_git_sha, deployed_at, choices


def _verify_deployment_registry(source_git_sha: str, serialized_choices: list[dict]) -> None:
    """Recheck mutable registry state before AWS changes or registration recovery."""
    for item in serialized_choices:
        model = item["model_key"]
        latest = get_latest_model_release(ModelKey(model))
        if latest is None:
            raise RuntimeError(f"No baseline release registered for {model}")
        current, proposed = parse_version(latest["version"]), parse_version(item["proposed_version"])
        if current == proposed:
            draft = _draft_from_serialized(item["draft"])
            if draft is not None:
                _verify_release_notes(draft, latest, model)
            if item["action"] != "keep" and latest.get("source_git_sha") != source_git_sha:
                raise RuntimeError(f"{model} {proposed} was registered from different source; refusing to deploy this plan")
        elif current != parse_version(item["current_version"]) or item["action"] == "keep" or proposed != {
            "major": next_major(current), "minor": next_minor(current),
        }.get(item["action"]):
            raise RuntimeError(f"Registered versions changed: {model} is now {current}; review the deployment plan before retrying")


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
    for item in serialized_choices:
        if item["action"] == "keep":
            initialize_model_release_source(
                item["model_key"],
                item["proposed_version"],
                source_git_sha=source_git_sha,
            )


def verify_registered_releases(source_git_sha: str, serialized_choices: list[dict]) -> None:
    for item in serialized_choices:
        if item["action"] == "keep":
            release = get_model_release(item["model_key"], item["proposed_version"])
            if release is None or not release.get("source_git_sha"):
                raise RuntimeError(
                    f"Supabase release {item['model_key']} {item['proposed_version']} "
                    "is missing its canonical source_git_sha"
                )
            draft = _draft_from_serialized(item.get("draft"))
            if draft is not None:
                _verify_release_notes(draft, release, item["model_key"])
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
    expected_configurations = {
        TRAINING_FUNCTION_NAME: _expected_lambda_environment(source_git_sha, serialized_choices),
        COORDINATOR_FUNCTION_NAME: {
            "SOURCE_GIT_SHA": source_git_sha,
            "SCHEDULE_GROUP_NAME": TRAINING_SCHEDULE_GROUP_NAME,
        },
        API_FUNCTION_NAME: {
            "SOURCE_GIT_SHA": source_git_sha,
            "SCHEDULE_GROUP_NAME": TRAINING_SCHEDULE_GROUP_NAME,
        },
    }
    last_problem = "AWS Lambda configuration was not available"
    for delay in delays:
        if delay:
            time.sleep(delay)
        try:
            configurations = {}
            for function_name in expected_configurations:
                result = subprocess.run(
                    [
                        "aws",
                        "lambda",
                        "get-function-configuration",
                        "--function-name",
                        function_name,
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
                configurations[function_name] = json.loads(result.stdout)
        except FileNotFoundError as exc:
            raise RuntimeError("AWS CLI is required to verify the deployed Lambda") from exc
        except (json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            last_problem = f"AWS configuration lookup failed ({type(exc).__name__})"
            continue

        problems = []
        for function_name, expected_environment in expected_configurations.items():
            configuration = configurations[function_name]
            update_status = configuration.get("LastUpdateStatus")
            if update_status == "Failed":
                reason = configuration.get("LastUpdateStatusReason") or "no failure reason returned"
                raise RuntimeError(f"AWS Lambda {function_name} update failed: {reason}")
            function_state = configuration.get("State")
            actual_environment = configuration.get("Environment", {}).get("Variables", {})
            mismatches = [
                key
                for key, expected_value in expected_environment.items()
                if actual_environment.get(key) != expected_value
            ]
            if update_status != "Successful":
                problems.append(f"{function_name} LastUpdateStatus={update_status!r}")
            if function_state != "Active":
                problems.append(f"{function_name} State={function_state!r}")
            if mismatches:
                problems.append(
                    f"{function_name} environment mismatch: {', '.join(mismatches)}"
                )
        training_arn = configurations[TRAINING_FUNCTION_NAME].get("FunctionArn")
        for function_name in (COORDINATOR_FUNCTION_NAME, API_FUNCTION_NAME):
            environment = configurations[function_name].get("Environment", {}).get("Variables", {})
            if not training_arn or environment.get("TRAINING_FUNCTION_ARN") != training_arn:
                problems.append(f"{function_name} target does not match {TRAINING_FUNCTION_NAME}")
        if not problems:
            print("AWS API, training and coordinator Lambdas, model versions, and Git SHA verified.")
            return
        last_problem = "; ".join(problems)

    raise RuntimeError(
        "AWS deployment could not be verified after bounded retries: " + last_problem
    )


def _sam_deploy(
    source_git_sha: str,
    serialized_choices: list[dict],
    secret_parameters: list[str],
) -> None:
    specs = {spec.key.value: spec for spec in MODEL_SPECS}
    versions = {
        specs[item["model_key"]].sam_parameter: item["proposed_version"]
        for item in serialized_choices
    }
    parameters = [
        "HttpApiName=sports-models-http-api-v2",
        "ApiFunctionName=sports-models-api-v2",
        "TrainingFunctionName=sports-models-training-v2",
        "CoordinatorFunctionName=sports-models-schedule-coordinator-v2",
        "TrainingScheduleGroupName=sports-models-training-updates-v2",
        "Localhost=False",
        "EnvironmentName=PROD",
        f"SourceGitSha={source_git_sha}",
        *[f"{key}={value}" for key, value in versions.items()],
        *secret_parameters,
    ]

    _ensure_deployment_source(source_git_sha)
    subprocess.run(["sam", "build"], cwd=ROOT, check=True)
    _ensure_deployment_source(source_git_sha)
    _verify_deployment_registry(source_git_sha, serialized_choices)
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


def _ensure_deployment_source(source_git_sha: str) -> None:
    _ensure_clean_tree()
    if _git("rev-parse", "HEAD") != source_git_sha:
        raise RuntimeError("Git HEAD changed after release confirmation; restore the planned commit before retrying")


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="Prepare release files locally before commit/merge")
    mode.add_argument("--register-only", action="store_true", help="Verify AWS, recover registration, and clear the local plan")
    args = parser.parse_args()

    if args.prepare:
        prepare_model_releases()
        return 0
    if args.register_only:
        source_git_sha, deployed_at, serialized_choices = _load_recovery_plan()
        _verify_deployment_registry(source_git_sha, serialized_choices)
        verify_aws_deployment(source_git_sha, serialized_choices)
        deployed_at = deployed_at or _record_verified_deployment()
        register_releases(source_git_sha, deployed_at, serialized_choices)
        verify_registered_releases(source_git_sha, serialized_choices)
        RECOVERY_PATH.unlink()
        print("Release registration verified; recovery plan cleared. No tracked files changed.")
        return 0

    _ensure_clean_tree()
    verify_manual_update_schema()
    source_git_sha = _git("rev-parse", "HEAD")
    releases = {spec.key: get_latest_model_release(spec.key) for spec in MODEL_SPECS}
    choices = build_deployment_choices(releases)
    serialized_choices = [_serialize_choice(choice) for choice in choices]
    if RECOVERY_PATH.exists():
        saved_sha, _, saved_choices = _load_recovery_plan()
        signature = lambda items: sorted((i["model_key"], i["proposed_version"], i.get("draft_hash")) for i in items)
        if saved_sha != source_git_sha or signature(saved_choices) != signature(serialized_choices):
            raise RuntimeError("An unfinished deployment plan exists for different source/releases. Recover it with make sam-register-releases, or return to its original clean commit to retry deployment.")
        # A registration retry must retain the original new-release actions and
        # verification timestamp, even if some/all registry writes succeeded.
        serialized_choices = saved_choices
        print("Resuming the existing deployment plan for this commit.")
    review_source_changes(choices, releases, input_fn=input)
    _print_plan(choices, source_git_sha)
    confirmation = input("\nProceed with SAM build and deployment? [y/N]: ").strip().lower()
    if confirmation not in {"y", "yes"}:
        print("Deployment cancelled; no files or database rows changed.")
        return 0

    secret_parameters = _deployment_secret_parameters()
    _verify_deployment_registry(source_git_sha, serialized_choices)
    if not RECOVERY_PATH.exists():
        _write_recovery_plan(choices, source_git_sha)
    _sam_deploy(source_git_sha, serialized_choices, secret_parameters)
    verify_aws_deployment(source_git_sha, serialized_choices)
    deployed_at = _record_verified_deployment()
    register_releases(source_git_sha, deployed_at, serialized_choices)
    verify_registered_releases(source_git_sha, serialized_choices)
    RECOVERY_PATH.unlink()
    print("AWS deployment and model-release registration completed. No tracked files changed.")
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
    except (EOFError, OSError, RuntimeError, ValueError) as exc:
        print(f"Deployment failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
