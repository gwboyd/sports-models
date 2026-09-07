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


def test_production_deploy_requires_main_branch(monkeypatch):
    def fake_git(*args):
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        if args == ("branch", "--show-current"):
            return "feature-model"
        raise AssertionError(args)

    monkeypatch.setattr(deploy_models, "_git", fake_git)
    with pytest.raises(RuntimeError, match="branch to be main"):
        deploy_models._ensure_clean_tree()


def test_aws_verification_accepts_active_lambda_with_exact_metadata(monkeypatch):
    training_configuration = {
        "FunctionArn": "arn:aws:lambda:us-east-1:123:function:sports-models-training-v2",
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
    coordinator_configuration = {
        "State": "Active",
        "LastUpdateStatus": "Successful",
        "Environment": {
            "Variables": {
                "SOURCE_GIT_SHA": "abc123",
                "TRAINING_FUNCTION_ARN": "arn:aws:lambda:us-east-1:123:function:sports-models-training-v2",
                "SCHEDULE_GROUP_NAME": "sports-models-training-updates-v2",
            }
        },
    }

    def get_configuration(args, **_kwargs):
        function_name = args[args.index("--function-name") + 1]
        configuration = (
            coordinator_configuration
            if function_name in (deploy_models.COORDINATOR_FUNCTION_NAME, deploy_models.API_FUNCTION_NAME)
            else training_configuration
        )
        return SimpleNamespace(stdout=json.dumps(configuration))

    monkeypatch.setattr(
        deploy_models.subprocess,
        "run",
        get_configuration,
    )

    deploy_models.verify_aws_deployment(
        "abc123", serialized_choices(), delays=(0,)
    )


def test_aws_verification_retries_a_stale_configuration(monkeypatch):
    training_configurations = iter(
        [
            {
                "FunctionArn": "arn:aws:lambda:us-east-1:123:function:sports-models-training-v2",
                "State": "Active",
                "LastUpdateStatus": "Successful",
                "Environment": {"Variables": {}},
            },
            {
                "FunctionArn": "arn:aws:lambda:us-east-1:123:function:sports-models-training-v2",
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
    coordinator_configuration = {
        "State": "Active",
        "LastUpdateStatus": "Successful",
        "Environment": {
            "Variables": {
                "SOURCE_GIT_SHA": "abc123",
                "TRAINING_FUNCTION_ARN": "arn:aws:lambda:us-east-1:123:function:sports-models-training-v2",
                "SCHEDULE_GROUP_NAME": "sports-models-training-updates-v2",
            }
        },
    }
    sleeps = []

    def get_configuration(args, **_kwargs):
        function_name = args[args.index("--function-name") + 1]
        configuration = (
            coordinator_configuration
            if function_name in (deploy_models.COORDINATOR_FUNCTION_NAME, deploy_models.API_FUNCTION_NAME)
            else next(training_configurations)
        )
        return SimpleNamespace(stdout=json.dumps(configuration))

    monkeypatch.setattr(
        deploy_models.subprocess,
        "run",
        get_configuration,
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


def test_release_registration_initializes_kept_bootstrap_sha(monkeypatch):
    inserted = []
    initialized = []
    monkeypatch.setattr(
        deploy_models,
        "insert_model_releases",
        lambda releases, **kwargs: inserted.append((list(releases), kwargs)),
    )
    monkeypatch.setattr(
        deploy_models,
        "initialize_model_release_source",
        lambda model, version, **kwargs: initialized.append((model, version, kwargs)),
    )

    deploy_models.register_releases(
        "abc123",
        "2026-08-29T12:00:00+00:00",
        serialized_choices(),
    )

    assert len(inserted) == 1
    assert [(model.value, version) for model, version, _draft in inserted[0][0]] == [
        ("nfl_expected_points", "1.3")
    ]
    assert inserted[0][1]["source_git_sha"] == "abc123"
    assert initialized == [
        (
            "cfb_expected_points",
            "1.2",
            {"source_git_sha": "abc123"},
        )
    ]


def test_release_verification_rejects_kept_version_without_canonical_sha(monkeypatch):
    release_draft = draft()
    assert release_draft is not None
    released_row = {
        "title": release_draft.title,
        "public_summary": release_draft.public_summary,
        "changes_md": release_draft.changes_md,
        "evaluation_md": release_draft.evaluation_md,
        "internal_notes_md": release_draft.internal_notes_md,
        "source_git_sha": "abc123",
    }

    def release(model, _version):
        return released_row if model == "nfl_expected_points" else {"source_git_sha": None}

    monkeypatch.setattr(deploy_models, "get_model_release", release)
    with pytest.raises(RuntimeError, match="canonical source_git_sha"):
        deploy_models.verify_registered_releases("abc123", serialized_choices())


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


def test_schema_preflight_blocks_deployment_before_build_or_release_changes(monkeypatch):
    monkeypatch.setattr(deploy_models.sys, 'argv', ['deploy_models.py'])
    monkeypatch.setattr(deploy_models, '_ensure_clean_tree', lambda: None)
    def missing_schema():
        raise RuntimeError('Apply the additive setup SQL before deployment')
    monkeypatch.setattr(deploy_models, 'verify_manual_update_schema', missing_schema)
    monkeypatch.setattr(deploy_models, '_sam_deploy', lambda *_a: pytest.fail('build/deploy started'))
    monkeypatch.setattr(deploy_models, '_write_recovery_plan', lambda *_a: pytest.fail('release plan written'))
    with pytest.raises(RuntimeError, match='setup SQL before deployment'):
        deploy_models._main()
