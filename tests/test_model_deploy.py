from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import deploy_models
from scripts.deploy_models import MODEL_SPECS, build_preparation_choices
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
            "current_version": "1.2",
            "proposed_version": "1.3",
            "draft_hash": hashlib.sha256(release_draft.original_markdown.encode()).hexdigest(),
            "draft": asdict(release_draft),
        },
        {
            "model_key": ModelKey.CFB_EXPECTED_POINTS.value,
            "action": "keep",
            "current_version": "1.2",
            "proposed_version": "1.2",
            "draft_hash": hashlib.sha256(b"").hexdigest(),
            "draft": None,
        },
    ]


def test_empty_drafts_keep_each_model():
    drafts = {spec.key: (None, "", "empty") for spec in MODEL_SPECS}
    choices = build_preparation_choices(releases(), drafts, input_fn=lambda _prompt: "")
    assert [(choice.action, str(choice.proposed_version)) for choice in choices] == [
        ("keep", "1.2"),
        ("keep", "1.2"),
    ]


def test_populated_draft_requires_major_or_minor_and_only_bumps_that_model():
    drafts = {
        ModelKey.NFL_EXPECTED_POINTS: (draft(), "draft", "hash"),
        ModelKey.CFB_EXPECTED_POINTS: (None, "", "empty"),
    }
    choices = build_preparation_choices(
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
        build_preparation_choices(
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


def test_registration_verification_requires_exact_database_release(monkeypatch):
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
    monkeypatch.setattr(deploy_models, "get_latest_model_release", releases().get)
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
    assert not recovery_path.exists()


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


@pytest.fixture
def release_workspace(tmp_path, monkeypatch):
    """Isolate all release files and database reads; never contact AWS or Supabase."""
    specs = tuple(replace(spec, draft_path=tmp_path / spec.key.value / 'UNRELEASED.md',
                          archive_dir=tmp_path / spec.key.value / 'releases') for spec in MODEL_SPECS)
    rows = {}
    for spec in specs:
        spec.archive_dir.mkdir(parents=True)
        spec.draft_path.write_text('')
        (spec.archive_dir / 'v1.2.md').write_text(draft().original_markdown)
        rows[spec.key] = {**asdict(draft()), 'version': '1.2', 'source_git_sha': 'original-sha'}
    monkeypatch.setattr(deploy_models, 'MODEL_SPECS', specs)
    monkeypatch.setattr(deploy_models, 'ROOT', tmp_path)
    monkeypatch.setattr(deploy_models, 'VERSIONS_PATH', tmp_path / 'model-versions.json')
    monkeypatch.setattr(deploy_models, 'RECOVERY_PATH', tmp_path / '.aws-sam/model-release-plan.json')
    deploy_models.VERSIONS_PATH.write_text(json.dumps({s.key.value: '1.2' for s in specs}))
    monkeypatch.setattr(deploy_models, 'get_latest_model_release', rows.get)
    monkeypatch.setattr(deploy_models, 'source_change_signals', lambda *_: ([], None))
    return specs, rows


def snapshot_release_files():
    return {str(p.relative_to(deploy_models.ROOT)): p.read_bytes()
            for p in deploy_models.ROOT.rglob('*') if p.is_file() and '.aws-sam' not in p.parts}


def test_prepare_both_releases_locally_then_deploy_without_version_prompts(release_workspace):
    specs, rows = release_workspace
    exact = draft().original_markdown + '\n'
    for spec in specs:
        spec.draft_path.write_text(exact)
    answers = iter(['major', 'minor', 'yes'])
    deploy_models.prepare_model_releases(input_fn=lambda _: next(answers))
    assert json.loads(deploy_models.VERSIONS_PATH.read_text()) == {
        specs[0].key.value: '2.0', specs[1].key.value: '1.3',
    }
    for spec, version in zip(specs, ('2.0', '1.3')):
        assert spec.draft_path.read_text() == ''
        assert (spec.archive_dir / f'v{version}.md').read_text() == exact
    choices = deploy_models.build_deployment_choices(rows)
    assert [(c.action, str(c.proposed_version)) for c in choices] == [('major', '2.0'), ('minor', '1.3')]
    before = snapshot_release_files()
    deploy_models.prepare_model_releases(input_fn=lambda _: pytest.fail('repeated preparation prompted'))
    assert snapshot_release_files() == before


@pytest.mark.parametrize('answers', [('major', 'abort'), ('major', 'minor', 'no')])
def test_cancel_preparation_leaves_all_files_untouched(release_workspace, answers):
    specs, _ = release_workspace
    for spec in specs:
        spec.draft_path.write_text(draft().original_markdown)
    before = snapshot_release_files()
    answers = iter(answers)
    try:
        deploy_models.prepare_model_releases(input_fn=lambda _: next(answers))
    except RuntimeError as exc:
        assert 'aborted' in str(exc)
    assert snapshot_release_files() == before


def test_preparation_write_failure_restores_all_original_files(release_workspace, monkeypatch):
    specs, _ = release_workspace
    for spec in specs:
        spec.draft_path.write_text(draft().original_markdown)
    before = snapshot_release_files()
    original = deploy_models._write_text_atomic
    failed = False
    def fail_once(path, text):
        nonlocal failed
        if path == deploy_models.VERSIONS_PATH and not failed:
            failed = True
            raise OSError('disk write failed')
        original(path, text)
    monkeypatch.setattr(deploy_models, '_write_text_atomic', fail_once)
    answers = iter(['major', 'minor', 'yes'])
    with pytest.raises(OSError, match='disk write failed'):
        deploy_models.prepare_model_releases(input_fn=lambda _: next(answers))
    assert snapshot_release_files() == before


def test_preparation_records_versions_before_clearing_any_draft(release_workspace, monkeypatch):
    specs, _ = release_workspace
    for spec in specs:
        spec.draft_path.write_text(draft().original_markdown)
    original = deploy_models._write_text_atomic
    def check_order(path, text):
        if path in {spec.draft_path for spec in specs}:
            versions = deploy_models.load_prepared_versions()
            assert [str(versions[spec.key]) for spec in specs] == ['2.0', '1.3']
            for spec in specs:
                assert (spec.archive_dir / f'v{versions[spec.key]}.md').exists()
        original(path, text)
    monkeypatch.setattr(deploy_models, '_write_text_atomic', check_order)
    answers = iter(['major', 'minor', 'yes'])
    deploy_models.prepare_model_releases(input_fn=lambda _: next(answers))


def test_prepare_rejects_changed_draft_during_confirmation(release_workspace):
    specs, _ = release_workspace
    specs[0].draft_path.write_text(draft().original_markdown)
    def answer(prompt):
        if '[y/N]' in prompt:
            specs[0].draft_path.write_text(draft().original_markdown.replace('Summary\n', 'New summary\n'))
            return 'yes'
        return 'major'
    with pytest.raises(RuntimeError, match='changed during preparation'):
        deploy_models.prepare_model_releases(input_fn=answer)
    assert not (specs[0].archive_dir / 'v2.0.md').exists()
    assert deploy_models.load_prepared_versions()[specs[0].key] == ModelVersion(1, 2)


@pytest.mark.parametrize('problem', ['draft', 'downgrade', 'skip_version', 'edited_notes', 'missing_notes'])
def test_deployment_rejects_invalid_release_intent(release_workspace, problem):
    specs, rows = release_workspace
    spec = specs[0]
    if problem == 'draft':
        spec.draft_path.write_text(draft().original_markdown)
    elif problem == 'downgrade':
        rows[spec.key]['version'] = '2.0'
    elif problem == 'skip_version':
        versions = {s.key.value: '1.2' for s in specs}
        versions[spec.key.value] = '4.0'
        deploy_models.VERSIONS_PATH.write_text(json.dumps(versions))
        (spec.archive_dir / 'v4.0.md').write_text(draft().original_markdown)
    elif problem == 'edited_notes':
        (spec.archive_dir / 'v1.2.md').write_text(draft().original_markdown.replace('- Change', '- Different'))
    else:
        (spec.archive_dir / 'v1.2.md').unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        deploy_models.build_deployment_choices(rows)


def test_existing_legacy_archive_is_read_without_rewriting_metadata(release_workspace):
    specs, rows = release_workspace
    path = specs[0].archive_dir / 'v1.2.md'
    path.write_text(f'<!-- model_key: {specs[0].key.value}; version: 1.2; deployed_at: old; source_git_sha: original-sha -->\n\n' + draft().original_markdown)
    before = snapshot_release_files()
    choices = deploy_models.build_deployment_choices(rows)
    assert all(c.action == 'keep' for c in choices)
    assert snapshot_release_files() == before


def test_deploy_and_registration_retry_leave_tracked_release_files_unchanged(release_workspace, monkeypatch):
    specs, rows = release_workspace
    specs[0].draft_path.write_text(draft().original_markdown)
    answers = iter(['major', 'yes'])
    deploy_models.prepare_model_releases(input_fn=lambda _: next(answers))
    before = snapshot_release_files()
    calls = []
    monkeypatch.setattr(deploy_models, '_ensure_clean_tree', lambda: None)
    monkeypatch.setattr(deploy_models, 'verify_manual_update_schema', lambda: None)
    monkeypatch.setattr(deploy_models, '_git', lambda *_: 'deployment-sha')
    monkeypatch.setattr(deploy_models, '_deployment_secret_parameters', lambda: [])
    monkeypatch.setattr('builtins.input', lambda _: 'yes')
    monkeypatch.setattr(deploy_models, '_sam_deploy', lambda *args: calls.append('sam'))
    monkeypatch.setattr(deploy_models, 'verify_aws_deployment', lambda *args: calls.append('aws'))
    monkeypatch.setattr(deploy_models, 'register_releases', lambda *args: calls.append('register'))
    def fail_verification(*args):
        raise RuntimeError('temporary verification failure')
    monkeypatch.setattr(deploy_models, 'verify_registered_releases', fail_verification)
    monkeypatch.setattr(deploy_models.sys, 'argv', ['deploy_models.py'])
    with pytest.raises(RuntimeError, match='temporary verification failure'):
        deploy_models._main()
    assert deploy_models.RECOVERY_PATH.exists()
    assert snapshot_release_files() == before
    saved = json.loads(deploy_models.RECOVERY_PATH.read_text())
    registered = []
    monkeypatch.setattr(deploy_models, 'register_releases', lambda *args: registered.append(args))
    monkeypatch.setattr(deploy_models, 'verify_registered_releases', lambda *args: calls.append('verified'))
    monkeypatch.setattr(deploy_models.sys, 'argv', ['deploy_models.py', '--register-only'])
    assert deploy_models._main() == 0
    assert calls == ['sam', 'aws', 'register', 'aws', 'verified']
    assert registered[0] == (saved['source_git_sha'], saved['deployed_at'], saved['choices'])
    assert not deploy_models.RECOVERY_PATH.exists()
    assert snapshot_release_files() == before


def test_successful_deployment_keeps_existing_release_sha_and_files(release_workspace, monkeypatch):
    _, rows = release_workspace
    before = snapshot_release_files()
    monkeypatch.setattr(deploy_models, '_ensure_clean_tree', lambda: None)
    monkeypatch.setattr(deploy_models, 'verify_manual_update_schema', lambda: None)
    monkeypatch.setattr(deploy_models, '_git', lambda *_: 'new-infrastructure-sha')
    monkeypatch.setattr(deploy_models, '_deployment_secret_parameters', lambda: [])
    monkeypatch.setattr('builtins.input', lambda _: 'yes')
    monkeypatch.setattr(deploy_models, '_sam_deploy', lambda *_: None)
    monkeypatch.setattr(deploy_models, 'verify_aws_deployment', lambda *_: None)
    monkeypatch.setattr(deploy_models, 'insert_model_releases', lambda pending, **kw: pytest.fail('existing versions inserted') if pending else None)
    monkeypatch.setattr(deploy_models, 'initialize_model_release_source', lambda *a, **kw: 'original-sha')
    monkeypatch.setattr(deploy_models, 'get_model_release', lambda model, version: rows[ModelKey(model)])
    monkeypatch.setattr(deploy_models.sys, 'argv', ['deploy_models.py'])
    assert deploy_models._main() == 0
    assert all(row['source_git_sha'] == 'original-sha' for row in rows.values())
    assert snapshot_release_files() == before
    assert not deploy_models.RECOVERY_PATH.exists()


def test_retry_refuses_to_overwrite_another_commit_recovery_plan(release_workspace, monkeypatch):
    _, rows = release_workspace
    deploy_models._write_recovery_plan(deploy_models.build_deployment_choices(rows), 'different-sha')
    before = deploy_models.RECOVERY_PATH.read_bytes()
    monkeypatch.setattr(deploy_models, '_ensure_clean_tree', lambda: None)
    monkeypatch.setattr(deploy_models, 'verify_manual_update_schema', lambda: None)
    monkeypatch.setattr(deploy_models, '_git', lambda *_: 'current-sha')
    monkeypatch.setattr(deploy_models.sys, 'argv', ['deploy_models.py'])
    monkeypatch.setattr(deploy_models, '_sam_deploy', lambda *_: pytest.fail('deployment started'))
    with pytest.raises(RuntimeError, match='unfinished deployment plan'):
        deploy_models._main()
    assert deploy_models.RECOVERY_PATH.read_bytes() == before


def test_retry_same_commit_preserves_original_plan_after_registry_write(release_workspace, monkeypatch):
    specs, rows = release_workspace
    specs[0].draft_path.write_text(draft().original_markdown)
    answers = iter(['major', 'yes'])
    deploy_models.prepare_model_releases(input_fn=lambda _: next(answers))
    choices = deploy_models.build_deployment_choices(rows)
    deploy_models._write_recovery_plan(choices, 'same-sha')
    saved = json.loads(deploy_models.RECOVERY_PATH.read_text())
    rows[specs[0].key].update(version='2.0', source_git_sha='same-sha')
    monkeypatch.setattr(deploy_models, '_ensure_clean_tree', lambda: None)
    monkeypatch.setattr(deploy_models, 'verify_manual_update_schema', lambda: None)
    monkeypatch.setattr(deploy_models, '_git', lambda *_: 'same-sha')
    monkeypatch.setattr(deploy_models, '_deployment_secret_parameters', lambda: [])
    monkeypatch.setattr('builtins.input', lambda _: 'yes')
    monkeypatch.setattr(deploy_models, '_sam_deploy', lambda *_: None)
    monkeypatch.setattr(deploy_models, 'verify_aws_deployment', lambda *_: None)
    monkeypatch.setattr(deploy_models, 'verify_registered_releases', lambda *_: None)
    observed = []
    monkeypatch.setattr(deploy_models, 'register_releases', lambda *args: observed.append(args[2]))
    monkeypatch.setattr(deploy_models.sys, 'argv', ['deploy_models.py'])
    assert deploy_models._main() == 0
    assert observed == [saved['choices']]
    assert observed[0][0]['action'] == 'major'


def test_source_change_during_build_prevents_aws_deploy(monkeypatch):
    calls = []
    monkeypatch.setattr(deploy_models, '_ensure_clean_tree', lambda: None)
    shas = iter(['planned-sha', 'changed-sha'])
    monkeypatch.setattr(deploy_models, '_git', lambda *_: next(shas))
    monkeypatch.setattr(deploy_models.subprocess, 'run', lambda args, **kw: calls.append(args))
    with pytest.raises(RuntimeError, match='HEAD changed'):
        deploy_models._sam_deploy('planned-sha', [], [])
    assert calls == [['sam', 'build']]


def test_sam_receives_both_confirmed_versions_and_source(monkeypatch):
    commands = []
    monkeypatch.setattr(deploy_models, '_ensure_clean_tree', lambda: None)
    monkeypatch.setattr(deploy_models, '_git', lambda *_: 'planned-sha')
    monkeypatch.setattr(deploy_models, 'get_latest_model_release', releases().get)
    monkeypatch.setattr(deploy_models.subprocess, 'run', lambda args, **_: commands.append(args))
    deploy_models._sam_deploy('planned-sha', serialized_choices(), ['SupabaseDbUrl=test-placeholder'])
    assert commands[0] == ['sam', 'build']
    assert commands[1][:2] == ['sam', 'deploy']
    parameters = commands[1][commands[1].index('--parameter-overrides') + 1:]
    assert 'NflExpectedPointsVersion=1.3' in parameters
    assert 'CfbExpectedPointsVersion=1.2' in parameters
    assert 'SourceGitSha=planned-sha' in parameters
    assert 'SupabaseDbUrl=test-placeholder' in parameters


def test_new_notes_after_preparation_block_deploy_and_amend_unpublished_version(release_workspace):
    specs, rows = release_workspace
    spec = specs[0]
    spec.draft_path.write_text(draft().original_markdown)
    answers = iter(['major', 'yes'])
    deploy_models.prepare_model_releases(input_fn=lambda _: next(answers))
    initial = (spec.archive_dir / 'v2.0.md').read_text()
    consolidated = initial + '- Another prediction change\n'
    spec.draft_path.write_text(consolidated)
    with pytest.raises(RuntimeError, match='prepare-model-release'):
        deploy_models.build_deployment_choices(rows)
    prompts = []
    def confirm(prompt):
        prompts.append(prompt)
        return 'yes'
    deploy_models.prepare_model_releases(input_fn=confirm)
    assert len(prompts) == 1  # No second major/minor choice for the same unpublished release.
    assert str(deploy_models.load_prepared_versions()[spec.key]) == '2.0'
    assert (spec.archive_dir / 'v2.0.md').read_text() == consolidated
    assert (spec.archive_dir / 'v1.2.md').read_text() == initial
    assert spec.draft_path.read_text() == ''
    assert deploy_models.build_deployment_choices(rows)[0].action == 'major'


def test_amendment_abort_preserves_prepared_notes_and_new_draft(release_workspace):
    specs, _ = release_workspace
    spec = specs[0]
    spec.draft_path.write_text(draft().original_markdown)
    answers = iter(['major', 'yes'])
    deploy_models.prepare_model_releases(input_fn=lambda _: next(answers))
    spec.draft_path.write_text(draft().original_markdown + '- Later change\n')
    before = snapshot_release_files()
    deploy_models.prepare_model_releases(input_fn=lambda _: 'no')
    assert snapshot_release_files() == before


def test_changes_after_registration_require_a_fresh_version(release_workspace):
    specs, rows = release_workspace
    spec = specs[0]
    spec.draft_path.write_text(draft().original_markdown)
    answers = iter(['major', 'yes'])
    deploy_models.prepare_model_releases(input_fn=lambda _: next(answers))
    rows[spec.key]['version'] = '2.0'
    original = (spec.archive_dir / 'v2.0.md').read_text()
    spec.draft_path.write_text(draft().original_markdown + '- New registered-recipe change\n')
    answers = iter(['minor', 'yes'])
    deploy_models.prepare_model_releases(input_fn=lambda _: next(answers))
    assert str(deploy_models.load_prepared_versions()[spec.key]) == '2.1'
    assert (spec.archive_dir / 'v2.0.md').read_text() == original


def test_other_deployment_during_preparation_cannot_overwrite_newly_registered_release(release_workspace):
    specs, rows = release_workspace
    spec = specs[0]
    spec.draft_path.write_text(draft().original_markdown)
    answers = iter(['major', 'yes'])
    deploy_models.prepare_model_releases(input_fn=lambda _: next(answers))
    spec.draft_path.write_text(draft().original_markdown + '- Later change\n')
    before = snapshot_release_files()
    def confirm(_):
        rows[spec.key]['version'] = '2.0'
        return 'yes'
    with pytest.raises(RuntimeError, match='Registered versions changed'):
        deploy_models.prepare_model_releases(input_fn=confirm)
    assert snapshot_release_files() == before


@pytest.mark.parametrize('content', ['one forgotten note', '<!-- TODO update model -->', '# Draft release\n'])
def test_any_unprepared_nonblank_content_blocks_deployment(release_workspace, content):
    specs, rows = release_workspace
    specs[0].draft_path.write_text(content)
    with pytest.raises((ValueError, RuntimeError)):
        deploy_models.build_deployment_choices(rows)


@pytest.mark.parametrize('answer', ['release', 'abort'])
def test_keeping_changed_source_requires_prediction_neutral_review(release_workspace, monkeypatch, answer):
    _, rows = release_workspace
    monkeypatch.setattr(deploy_models, 'source_change_signals', lambda *_: (['src/shared_training.py'], None))
    with pytest.raises(RuntimeError, match='UNRELEASED.md'):
        deploy_models.review_source_changes(deploy_models.build_deployment_choices(rows), rows, input_fn=lambda _: answer)


def test_prediction_neutral_review_keeps_versions_without_mutations(release_workspace, monkeypatch):
    _, rows = release_workspace
    before = snapshot_release_files()
    monkeypatch.setattr(deploy_models, 'source_change_signals', lambda *_: (['src/shared_training.py'], None))
    choices = deploy_models.build_deployment_choices(rows)
    deploy_models.review_source_changes(choices, rows, input_fn=lambda _: 'neutral')
    assert all(choice.action == 'keep' for choice in choices)
    assert snapshot_release_files() == before


def test_source_signals_cover_shared_deleted_notebook_data_and_untracked_inputs(monkeypatch):
    def git(*args):
        if args[0] == 'diff':
            return ('src/model_patterns/expected_points/training.py\0'
                    'src/sports/football/nfl/expected_points/notebook.ipynb\0'
                    'src/sports/football/cfb/expected_points/old_helper.py\0'
                    'src/sports/football/cfb/starter_pack_data/stats.csv\0'
                    'requirements.txt\0frontend/app/models/nfl/page.tsx\0AGENTS.md\0')
        return 'src/sports/football/transforms/new_helper.py\0'
    monkeypatch.setattr(deploy_models, '_git', git)
    paths, warning = deploy_models.source_change_signals(MODEL_SPECS[1], 'registered-sha')
    assert warning is None
    assert paths == ['requirements.txt', 'src/model_patterns/expected_points/training.py',
                     'src/sports/football/cfb/expected_points/old_helper.py',
                     'src/sports/football/cfb/starter_pack_data/stats.csv',
                     'src/sports/football/transforms/new_helper.py']


def test_missing_source_history_is_not_silently_treated_as_no_changes(monkeypatch):
    import subprocess
    def missing(*args):
        raise subprocess.CalledProcessError(128, ['git', *args])
    monkeypatch.setattr(deploy_models, '_git', missing)
    with pytest.raises(RuntimeError, match='Fetch that Git history'):
        deploy_models.source_change_signals(MODEL_SPECS[0], 'missing-sha')


@pytest.mark.parametrize('problem', [
    'missing_league', 'duplicate_league', 'empty_choices', 'missing_sha',
    'unknown_action', 'invalid_increment', 'missing_notes', 'changed_fields',
    'changed_markdown', 'missing_hash', 'invalid_timestamp',
])
def test_damaged_recovery_plan_cannot_skip_verification_or_write_registry(tmp_path, monkeypatch, problem):
    payload = {'source_git_sha': 'planned-sha', 'deployed_at': None, 'choices': serialized_choices()}
    choice = payload['choices'][0]
    if problem == 'missing_league':
        payload['choices'].pop()
    elif problem == 'duplicate_league':
        payload['choices'][1] = choice.copy()
    elif problem == 'empty_choices':
        payload['choices'] = []
    elif problem == 'missing_sha':
        payload.pop('source_git_sha')
    elif problem == 'unknown_action':
        choice['action'] = 'amend'
    elif problem == 'invalid_increment':
        choice['proposed_version'] = '4.0'
    elif problem == 'missing_notes':
        choice['draft'] = None
    elif problem == 'changed_fields':
        choice['draft']['title'] = 'Different release'
    elif problem == 'changed_markdown':
        changed = parse_release_markdown(draft().original_markdown + '- Extra change\n')
        choice['draft'] = asdict(changed)
    elif problem == 'missing_hash':
        choice.pop('draft_hash')
    else:
        payload['deployed_at'] = '2026-09-06T12:00:00'
    path = tmp_path / 'model-release-plan.json'
    path.write_text(json.dumps(payload))
    original = path.read_bytes()
    monkeypatch.setattr(deploy_models, 'RECOVERY_PATH', path)
    monkeypatch.setattr(deploy_models.sys, 'argv', ['deploy_models.py', '--register-only'])
    monkeypatch.setattr(deploy_models, 'get_latest_model_release', lambda *_: pytest.fail('registry read'))
    monkeypatch.setattr(deploy_models, 'verify_aws_deployment', lambda *_: pytest.fail('AWS contacted'))
    monkeypatch.setattr(deploy_models, 'register_releases', lambda *_: pytest.fail('registry write'))
    with pytest.raises(ValueError, match='Invalid deployment recovery plan'):
        deploy_models._main()
    assert path.read_bytes() == original


@pytest.mark.parametrize('stage', ['confirmation', 'build'])
@pytest.mark.parametrize('competing_version', ['1.3', '2.0'])
def test_competing_release_blocks_aws_deploy(release_workspace, monkeypatch, stage, competing_version):
    specs, rows = release_workspace
    specs[0].draft_path.write_text(draft().original_markdown)
    answers = iter(['major', 'yes'])
    deploy_models.prepare_model_releases(input_fn=lambda _: next(answers))
    original = snapshot_release_files()
    def competing_release():
        rows[specs[0].key].update(version=competing_version, source_git_sha='another-deployment')
    def confirm(_):
        if stage == 'confirmation':
            competing_release()
        return 'yes'
    commands = []
    def command(args, **_):
        commands.append(args)
        assert args == ['sam', 'build'], 'AWS deployment must not start'
        competing_release()
    monkeypatch.setattr(deploy_models, '_ensure_clean_tree', lambda: None)
    monkeypatch.setattr(deploy_models, 'verify_manual_update_schema', lambda: None)
    monkeypatch.setattr(deploy_models, '_git', lambda *_: 'planned-sha')
    monkeypatch.setattr(deploy_models, '_deployment_secret_parameters', lambda: [])
    monkeypatch.setattr('builtins.input', confirm)
    monkeypatch.setattr(deploy_models.subprocess, 'run', command)
    monkeypatch.setattr(deploy_models.sys, 'argv', ['deploy_models.py'])
    with pytest.raises(RuntimeError, match='Registered versions changed|registered from different source'):
        deploy_models._main()
    assert commands == ([['sam', 'build']] if stage == 'build' else [])
    assert snapshot_release_files() == original
    assert deploy_models.RECOVERY_PATH.exists() == (stage == 'build')


def test_retry_cannot_replace_a_release_registered_from_another_commit(release_workspace, monkeypatch):
    specs, rows = release_workspace
    specs[0].draft_path.write_text(draft().original_markdown)
    answers = iter(['major', 'yes'])
    deploy_models.prepare_model_releases(input_fn=lambda _: next(answers))
    deploy_models._write_recovery_plan(deploy_models.build_deployment_choices(rows), 'planned-sha')
    original = deploy_models.RECOVERY_PATH.read_bytes()
    rows[specs[0].key].update(version='2.0', source_git_sha='another-deployment')
    monkeypatch.setattr(deploy_models, '_ensure_clean_tree', lambda: None)
    monkeypatch.setattr(deploy_models, 'verify_manual_update_schema', lambda: None)
    monkeypatch.setattr(deploy_models, '_git', lambda *_: 'planned-sha')
    monkeypatch.setattr(deploy_models, '_deployment_secret_parameters', lambda: [])
    monkeypatch.setattr('builtins.input', lambda _: 'yes')
    monkeypatch.setattr(deploy_models, '_sam_deploy', lambda *_: pytest.fail('AWS deployment started'))
    monkeypatch.setattr(deploy_models.sys, 'argv', ['deploy_models.py'])
    with pytest.raises(RuntimeError, match='registered from different source'):
        deploy_models._main()
    assert deploy_models.RECOVERY_PATH.read_bytes() == original
