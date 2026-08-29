from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from scripts import deploy_models
from scripts.deploy_models import MODEL_SPECS, build_deployment_choices
from src.model_patterns.expected_points.versioning import ModelKey, ModelVersion, parse_release_markdown


def draft():
    return parse_release_markdown(
        "# Draft release\n\n## Public Summary\nSummary\n\n## Changes\n- Change\n"
    )


def releases():
    return {
        spec.key: {"version": "1.2"}
        for spec in MODEL_SPECS
    }


def serialized_choices():
    release_draft = draft()
    assert release_draft is not None
    return [
        {
            "model_key": ModelKey.NFL_EXPECTED_POINTS.value,
            "action": "minor",
            "proposed_version": "1.3",
            "draft": asdict(release_draft),
        },
        {
            "model_key": ModelKey.CFB_EXPECTED_POINTS.value,
            "action": "keep",
            "proposed_version": "1.2",
            "draft": None,
        },
    ]


def test_empty_drafts_keep_each_model():
    drafts = {spec.key: (None, "", "empty") for spec in MODEL_SPECS}
    choices = build_deployment_choices(releases(), drafts, input_fn=lambda _prompt: "")
    assert [(choice.action, str(choice.proposed_version)) for choice in choices] == [
        ("keep", "1.2"),
        ("keep", "1.2"),
    ]


def test_populated_draft_requires_major_or_minor_and_only_bumps_that_model():
    drafts = {
        ModelKey.NFL_EXPECTED_POINTS: (draft(), "draft", "hash"),
        ModelKey.CFB_EXPECTED_POINTS: (None, "", "empty"),
    }
    choices = build_deployment_choices(
        releases(), drafts, input_fn=lambda _prompt: "major"
    )
    assert choices[0].action == "major"
    assert choices[0].proposed_version == ModelVersion(2, 0)
    assert choices[1].action == "keep"


def test_populated_draft_cannot_be_kept():
    drafts = {
        spec.key: (draft(), "draft", "hash") if spec is MODEL_SPECS[0] else (None, "", "empty")
        for spec in MODEL_SPECS
    }
    answers = iter(["keep", "abort"])
    with pytest.raises(RuntimeError, match="aborted"):
        build_deployment_choices(
            releases(), drafts, input_fn=lambda _prompt: next(answers)
        )


def test_clean_tree_check_includes_untracked_files(monkeypatch):
    observed = {}

    def fake_git(*args):
        observed["args"] = args
        return "?? untracked_model.py"

    monkeypatch.setattr(deploy_models, "_git", fake_git)
    with pytest.raises(RuntimeError, match="every change to be committed"):
        deploy_models._ensure_clean_tree()
    assert observed["args"] == ("status", "--porcelain", "--untracked-files=all")


def test_aws_verification_accepts_active_lambda_with_exact_metadata(monkeypatch):
    configuration = {
        "State": "Active",
        "LastUpdateStatus": "Successful",
        "Environment": {
            "Variables": {
                "NFL_EXPECTED_POINTS_VERSION": "1.3",
                "CFB_EXPECTED_POINTS_VERSION": "1.2",
                "SOURCE_GIT_SHA": "abc123",
                "UNRELATED_SECRET": "not inspected",
            }
        },
    }
    monkeypatch.setattr(
        deploy_models.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=json.dumps(configuration)),
    )

    deploy_models.verify_aws_deployment(
        "abc123", serialized_choices(), delays=(0,)
    )


def test_aws_verification_retries_a_stale_configuration(monkeypatch):
    configurations = iter(
        [
            {
                "State": "Active",
                "LastUpdateStatus": "Successful",
                "Environment": {"Variables": {}},
            },
            {
                "State": "Active",
                "LastUpdateStatus": "Successful",
                "Environment": {
                    "Variables": {
                        "NFL_EXPECTED_POINTS_VERSION": "1.3",
                        "CFB_EXPECTED_POINTS_VERSION": "1.2",
                        "SOURCE_GIT_SHA": "abc123",
                    }
                },
            },
        ]
    )
    sleeps = []
    monkeypatch.setattr(
        deploy_models.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=json.dumps(next(configurations))
        ),
    )
    monkeypatch.setattr(deploy_models.time, "sleep", sleeps.append)

    deploy_models.verify_aws_deployment(
        "abc123", serialized_choices(), delays=(0, 1)
    )

    assert sleeps == [1]


def test_aws_verification_rejects_failed_lambda_update(monkeypatch):
    configuration = {
        "State": "Active",
        "LastUpdateStatus": "Failed",
        "LastUpdateStatusReason": "container image rejected",
        "Environment": {"Variables": {}},
    }
    monkeypatch.setattr(
        deploy_models.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=json.dumps(configuration)),
    )

    with pytest.raises(RuntimeError, match="container image rejected"):
        deploy_models.verify_aws_deployment(
            "abc123", serialized_choices(), delays=(0,)
        )


def test_release_file_finalization_requires_exact_database_release(monkeypatch):
    choices = serialized_choices()
    release_draft = draft()
    assert release_draft is not None
    row = {
        "title": release_draft.title,
        "public_summary": release_draft.public_summary,
        "changes_md": release_draft.changes_md,
        "evaluation_md": release_draft.evaluation_md,
        "internal_notes_md": release_draft.internal_notes_md,
        "source_git_sha": "different-sha",
    }
    monkeypatch.setattr(deploy_models, "get_model_release", lambda *_args: row)

    with pytest.raises(RuntimeError, match="source_git_sha"):
        deploy_models.verify_registered_releases("abc123", choices)


def test_finalize_recovery_removes_completed_plan(tmp_path, monkeypatch):
    recovery_path = tmp_path / "model-release-plan.json"
    recovery_path.write_text(
        json.dumps(
            {
                "source_git_sha": "abc123",
                "deployed_at": "2026-08-29T12:00:00+00:00",
                "choices": serialized_choices(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(deploy_models, "RECOVERY_PATH", recovery_path)
    monkeypatch.setattr(deploy_models, "finalize_local_drafts", lambda *_args: None)
    monkeypatch.setattr(
        deploy_models.sys,
        "argv",
        ["deploy_models.py", "--finalize-only"],
    )

    assert deploy_models._main() == 0
    assert not recovery_path.exists()


def test_register_recovery_reverifies_aws_before_database_write(tmp_path, monkeypatch):
    recovery_path = tmp_path / "model-release-plan.json"
    recovery_path.write_text(
        json.dumps(
            {
                "source_git_sha": "abc123",
                "deployed_at": None,
                "choices": serialized_choices(),
            }
        ),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(deploy_models, "RECOVERY_PATH", recovery_path)
    monkeypatch.setattr(
        deploy_models,
        "verify_aws_deployment",
        lambda *_args: calls.append("aws"),
    )
    monkeypatch.setattr(
        deploy_models,
        "register_releases",
        lambda *_args: calls.append("database"),
    )
    monkeypatch.setattr(
        deploy_models,
        "verify_registered_releases",
        lambda *_args: calls.append("database_verified"),
    )
    monkeypatch.setattr(
        deploy_models.sys,
        "argv",
        ["deploy_models.py", "--register-only"],
    )

    assert deploy_models._main() == 0
    assert calls == ["aws", "database", "database_verified"]
    payload = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert payload["deployed_at"] is not None
