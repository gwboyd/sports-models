import pytest

from src.model_patterns.expected_points import write_policy
from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.model_patterns.expected_points.write_policy import (
    confirm_expected_points_write,
    is_aws_lambda_runtime,
    resolve_expected_points_write_plan,
)
from src.model_patterns.expected_points.versioning import ModelKey
from src.utils.db import sports_models_db


def clear_aws_runtime(monkeypatch):
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    monkeypatch.delenv("AWS_SAM_LOCAL", raising=False)
    monkeypatch.delenv("EXPECTED_POINTS_MODEL_VERSION", raising=False)
    monkeypatch.delenv("SOURCE_GIT_SHA", raising=False)


def test_real_lambda_is_automatic_but_sam_local_is_not(monkeypatch):
    clear_aws_runtime(monkeypatch)
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "sports-models-training-v2")
    assert is_aws_lambda_runtime()

    monkeypatch.setenv("AWS_SAM_LOCAL", "true")
    assert not is_aws_lambda_runtime()


def test_aws_write_uses_deployed_version_without_manual_flag(monkeypatch):
    clear_aws_runtime(monkeypatch)
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "sports-models-training-v2")
    monkeypatch.setenv("EXPECTED_POINTS_MODEL_VERSION", "2.3")
    monkeypatch.setenv("SOURCE_GIT_SHA", "deployed-sha")

    plan = resolve_expected_points_write_plan(
        ExpectedPointsLeague.NFL,
        client_name="aws",
        allow_non_aws_write=False,
    )

    assert plan.should_write
    assert plan.model_version == "2.3"
    assert plan.source_git_sha == "deployed-sha"
    assert confirm_expected_points_write(
        plan,
        ExpectedPointsLeague.NFL,
        season=2026,
        week=1,
        picks_num=16,
        pick_changes=0,
        play_changes=0,
        input_fn=lambda _prompt: pytest.fail("AWS must not prompt"),
    )


def test_non_aws_write_defaults_to_read_only_without_database_lookup(monkeypatch):
    clear_aws_runtime(monkeypatch)
    monkeypatch.setattr(
        sports_models_db,
        "get_latest_model_release",
        lambda *_args: pytest.fail("read-only runs must not query releases"),
    )

    plan = resolve_expected_points_write_plan(
        ExpectedPointsLeague.CFB,
        client_name="local-api",
        allow_non_aws_write=False,
    )

    assert not plan.should_write
    assert plan.model_version is None


def test_manual_notebook_write_uses_latest_release_and_exact_confirmation(monkeypatch):
    clear_aws_runtime(monkeypatch)
    observed = {}

    def latest_release(model_key):
        observed["model_key"] = model_key
        return {"version": "1.4", "source_git_sha": "release-sha"}

    monkeypatch.setattr(sports_models_db, "get_latest_model_release", latest_release)
    monkeypatch.setattr(write_policy, "_local_git_identity", lambda: "local-sha-dirty")
    plan = resolve_expected_points_write_plan(
        ExpectedPointsLeague.CFB,
        client_name="notebook",
        allow_non_aws_write=True,
    )

    assert observed["model_key"] is ModelKey.CFB_EXPECTED_POINTS
    assert plan.model_version == "1.4"
    assert plan.source_git_sha == "local-sha-dirty"
    assert plan.requires_interactive_confirmation
    assert confirm_expected_points_write(
        plan,
        ExpectedPointsLeague.CFB,
        season=2026,
        week=2,
        picks_num=10,
        pick_changes=2,
        play_changes=1,
        input_fn=lambda prompt: "WRITE CFB 1.4" if "WRITE CFB 1.4" in prompt else "",
    )


def test_manual_notebook_write_rejects_inexact_confirmation(monkeypatch):
    clear_aws_runtime(monkeypatch)
    monkeypatch.setattr(
        sports_models_db,
        "get_latest_model_release",
        lambda *_args: {"version": "1.0", "source_git_sha": None},
    )
    monkeypatch.setattr(write_policy, "_local_git_identity", lambda: "local-sha")
    plan = resolve_expected_points_write_plan(
        ExpectedPointsLeague.NFL,
        client_name="notebook",
        allow_non_aws_write=True,
    )

    with pytest.raises(RuntimeError, match="no database changes"):
        confirm_expected_points_write(
            plan,
            ExpectedPointsLeague.NFL,
            season=2026,
            week=1,
            picks_num=16,
            pick_changes=1,
            play_changes=1,
            input_fn=lambda _prompt: "yes",
        )


def test_non_notebook_manual_client_uses_request_flag_as_confirmation(monkeypatch):
    clear_aws_runtime(monkeypatch)
    monkeypatch.setattr(
        sports_models_db,
        "get_latest_model_release",
        lambda *_args: {"version": "2.0", "source_git_sha": "release-sha"},
    )
    monkeypatch.setattr(write_policy, "_local_git_identity", lambda: "local-sha")
    plan = resolve_expected_points_write_plan(
        ExpectedPointsLeague.NFL,
        client_name="local-api",
        allow_non_aws_write=True,
    )

    assert not plan.requires_interactive_confirmation
    assert confirm_expected_points_write(
        plan,
        ExpectedPointsLeague.NFL,
        season=2026,
        week=1,
        picks_num=16,
        pick_changes=1,
        play_changes=1,
        input_fn=lambda _prompt: pytest.fail("local API must not prompt"),
    )
