import argparse
import json
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from scripts import backtest_expected_points as cli
from src.model_patterns.expected_points.backtest_artifacts import load_backtest_frame, save_backtest_frame
from src.model_patterns.expected_points.backtest_workflow import historical_frame, preflight_frames, preparation_notebook
from src.model_patterns.expected_points.backtesting import BacktestSpec


def frame():
    return pd.DataFrame({
        "game_id": ["2023a", "2024a", "2025a"], "season": [2023, 2024, 2025], "week": [1, 1, 1],
        "date_time": [f"{y}-09-01-13:00" for y in (2023, 2024, 2025)],
        "home_team": ["a"] * 3, "away_team": ["b"] * 3, "home_score": [21] * 3,
        "away_score": [10] * 3, "spread_line": [-3.5] * 3, "total_line": [41.5] * 3,
        "metric": [1., 2., 3.], "provider_quotes": [[["a", -3.5]]] * 3,
    })


def args(tmp_path, **kwargs):
    return argparse.Namespace(**{
        "command": "compare", "league": "nfl", "profile": "quick", "seasons": "", "cadence": None,
        "frame": str(tmp_path / "source.parquet"), "baseline_frame": None, "candidate_frame": None,
        "baseline": "working-tree", "candidate": "working-tree", "cache_root": str(tmp_path / "cache"),
        "no_cache": False, "cache_working_tree": False, "bootstrap_samples": 20,
        "through_season": None, "preflight_only": False, "output_dir": str(tmp_path / "job"), **kwargs,
    })


def test_named_filtered_bundle_keeps_quotes_and_rebuilds_metadata(tmp_path):
    source = save_backtest_frame(frame(), "nfl", destination=tmp_path / "source.parquet")
    target = save_backtest_frame(historical_frame(load_backtest_frame(source), 2024), "nfl",
                                 destination=tmp_path / "named.parquet")
    assert load_backtest_frame(target).provider_quotes.tolist() == [[["a", -3.5]]] * 2
    assert json.loads(target.with_suffix(".json").read_text())["rows"] == 2
    assert len(load_backtest_frame(source)) == 3


def test_bundle_rejects_missing_stale_and_same_size_replaced_data(tmp_path):
    path = save_backtest_frame(frame(), "nfl", destination=tmp_path / "frame.parquet")
    metadata = path.with_suffix(".json").read_text()
    path.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="Missing frame metadata"):
        load_backtest_frame(path)
    path.with_suffix(".json").write_text(metadata)
    changed = pd.read_parquet(path)
    changed.loc[0, "home_score"] = 99
    changed.to_parquet(path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_backtest_frame(path)


def test_preflight_allows_feature_changes_but_rejects_changed_outcomes():
    baseline = frame()
    candidate = baseline.assign(metric=99., new_feature=0.)
    assert preflight_frames(baseline, candidate, BacktestSpec(profile="quick"))["seasons"] == [2025]
    candidate.loc[0, "home_score"] = 22
    with pytest.raises(ValueError, match='home_score.*2023a'):
        preflight_frames(baseline, candidate, BacktestSpec(profile="quick"))


def test_explicit_season_bound_preserves_warmup_without_intersecting_games():
    baseline = frame()
    candidate = frame()
    candidate.loc[2, "game_id"] = "other"
    with pytest.raises(ValueError, match="candidate_only"):
        preflight_frames(baseline, candidate, BacktestSpec(profile="quick"))
    report = preflight_frames(historical_frame(baseline, 2024), historical_frame(candidate, 2024),
                              BacktestSpec(profile="quick"))
    assert report["rows"] == 2
    assert report["seasons"] == [2024]


def test_compare_fails_before_any_training(tmp_path, monkeypatch):
    source = save_backtest_frame(frame(), "nfl", destination=tmp_path / "source.parquet")
    changed = frame()
    changed.loc[0, "total_line"] = 40.
    candidate = save_backtest_frame(changed, "nfl", destination=tmp_path / "candidate.parquet")
    monkeypatch.setattr(cli, "_load_or_run", lambda *a, **k: pytest.fail("training started"))
    with pytest.raises(ValueError, match="Frame preflight failed"):
        cli._compare(args(tmp_path, frame=str(source), candidate_frame=str(candidate)))
    status = json.loads((tmp_path / "job/status.json").read_text())
    assert status["status"] == "failed"
    assert "total_line" in status["error"]


def test_completed_side_survives_other_side_failure_and_job_cannot_be_overwritten(tmp_path, monkeypatch):
    save_backtest_frame(frame(), "nfl", destination=tmp_path / "source.parquet")
    def run(_args, reference, *, candidate, output):
        if candidate:
            raise RuntimeError("candidate failed")
        output.mkdir()
        (output / "manifest.json").write_text('{"completed": true}')
    monkeypatch.setattr(cli, "_load_or_run", run)
    with pytest.raises(RuntimeError, match="candidate failed"):
        cli._compare(args(tmp_path))
    assert (tmp_path / "job/baseline/manifest.json").exists()
    with pytest.raises(FileExistsError):
        cli._compare(args(tmp_path))


def synthetic_notebook(path):
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell('allow_non_aws_write = True', metadata={"tags": ["parameters"]}),
        nbformat.v4.new_code_cell(
            "assert allow_non_aws_write is False\n"
            "assert current_year == 2026 and current_week == 1\n"
            "import pandas as pd\nfrom types import SimpleNamespace\n"
            "df = pd.DataFrame({'game_id': ['test'], 'season': [2025]})\n"
            "expected_points_recipe = SimpleNamespace(league='nfl')"
        ),
        nbformat.v4.new_code_cell("backtest_frame_path = None\nraise RuntimeError('original save ran')"),
        nbformat.v4.new_code_cell("raise RuntimeError('training or writes ran')"),
    ])
    nbformat.write(notebook, path)


def test_preparation_boundary_is_fail_closed_and_excludes_training(tmp_path):
    path = tmp_path / "notebook.ipynb"
    synthetic_notebook(path)
    notebook = preparation_notebook(path, export_root=tmp_path, current_year=2026, current_week=1)
    source = "\n".join(c.source for c in notebook.cells)
    assert "training or writes ran" not in source
    assert "original save ran" not in source
    assert "allow_non_aws_write = False" in source
    broken = nbformat.read(path, as_version=4)
    broken.cells = broken.cells[:2]
    nbformat.write(broken, path)
    with pytest.raises(ValueError, match="boundary"):
        preparation_notebook(path, export_root=tmp_path, current_year=2026, current_week=1)


def test_preflight_only_writes_reproducible_inputs_without_training(tmp_path, monkeypatch):
    save_backtest_frame(frame(), "nfl", destination=tmp_path / "source.parquet")
    monkeypatch.setattr(cli, "_load_or_run", lambda *a, **k: pytest.fail("training started"))
    output = cli._compare(args(tmp_path, preflight_only=True))
    assert json.loads((output / "preflight.json").read_text())["status"] == "passed"
    assert (output / "resolved_request.json").exists()
    assert (output / "frames/baseline.json").exists()


def test_prepare_executes_only_frame_stage_with_selected_python(tmp_path, monkeypatch):
    """Exercise the actual Papermill process and Jupyter kernel, without feeds."""
    import sys
    source = tmp_path / "synthetic.ipynb"
    synthetic_notebook(source)
    original = cli.preparation_notebook
    monkeypatch.setattr(cli, "preparation_notebook", lambda _path, **kwargs: original(source, **kwargs))
    monkeypatch.setattr(cli, "_reference_python", lambda *_: Path(sys.executable))
    output = cli._prepare(argparse.Namespace(
        command="prepare-frame", league="nfl", recipe="working-tree", current_year=2026, current_week=1,
        through_season=2025, cache_root=str(tmp_path / "cache"), output_dir=str(tmp_path / "prepared"),
    ))
    assert load_backtest_frame(output).game_id.tolist() == ["test"]
    assert (output.parent / "executed.ipynb").exists()
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["provenance"]["resolved_reference"] == "working-tree"
    assert json.loads((output.parent / "status.json").read_text())["status"] == "complete"
    kernel = json.loads((output.parent / "jupyter/kernels/backtest-prepare/kernel.json").read_text())
    assert kernel["argv"][0] == sys.executable


def test_artifact_preflight_rejects_wrong_cutoff_coverage():
    from types import SimpleNamespace
    from src.model_patterns.expected_points.backtest_workflow import preflight_artifact
    from src.model_patterns.expected_points.backtesting import source_bundle_fingerprint
    from src.model_patterns.expected_points.types import ExpectedPointsLeague
    data = frame()
    run = SimpleNamespace(league=ExpectedPointsLeague.NFL, profile="quick", seasons=(2025,),
                          source_fingerprint=source_bundle_fingerprint(data), predictions=data.iloc[-1:].copy())
    preflight_artifact(run, data, BacktestSpec(profile="quick"), "nfl")
    run.predictions = data.iloc[:1]
    with pytest.raises(ValueError, match="Artifact preflight failed"):
        preflight_artifact(run, data, BacktestSpec(profile="quick"), "nfl")


def test_compare_rejects_stale_candidate_provenance(tmp_path, monkeypatch):
    save_backtest_frame(frame(), "nfl", destination=tmp_path / "source.parquet",
                         provenance={"code_identity": "old-code"})
    monkeypatch.setattr(cli, "_load_or_run", lambda *a, **k: pytest.fail("training started"))
    with pytest.raises(ValueError, match="different code"):
        cli._compare(args(tmp_path))


def test_immutable_reference_rejects_dirty_frame_even_with_same_git_sha(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    save_backtest_frame(frame(), "nfl", destination=tmp_path / "source.parquet",
                         provenance={"code_identity": "same-sha:dirty"})
    monkeypatch.setattr(cli, "_resolve_reference", lambda *_: ("same-sha", "version:1.0"))
    monkeypatch.setattr(cli, "recipe_code_identity", lambda *a, **k: "same-sha:clean")
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    @contextmanager
    def checkout(*_args):
        yield tmp_path, "version:1.0"
    monkeypatch.setattr(cli, "_checkout", checkout)
    monkeypatch.setattr(cli, "_load_or_run", lambda *a, **k: pytest.fail("training started"))
    with pytest.raises(ValueError, match="immutable recipe code"):
        cli._compare(args(tmp_path, baseline="deployed"))


@pytest.mark.parametrize("month, expected", [(1, 2024), (2, 2024), (3, 2025), (9, 2025)])
def test_automatic_history_end_is_conservative_during_postseason(month, expected):
    from datetime import datetime, timezone
    assert cli._default_history_end(datetime(2026, month, 1, tzinfo=timezone.utc)) == expected


def test_plain_compare_prepares_each_recipe_then_runs_standard(tmp_path, monkeypatch):
    """Exercise argument parsing and orchestration with different feature schemas."""
    from types import SimpleNamespace
    import sys
    calls = []
    def resolve(reference, _league):
        calls.append(("resolve", reference))
        return "release-sha", "version:1.0"
    monkeypatch.setattr(cli, "_resolve_reference", resolve)
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli, "recipe_code_identity", lambda *a, **k: "candidate-code")
    monkeypatch.setattr(cli, "_default_history_end", lambda: 2025)
    def prepare(request, *, resolved):
        calls.append(("prepare", request.recipe, resolved))
        assert (request.current_year, request.current_week, request.through_season) == (2025, 1, 2025)
        data = frame().drop(columns="metric")
        warmup = data.iloc[:1].assign(game_id="2022a", season=2022, date_time="2022-09-01-13:00")
        data = pd.concat([warmup, data], ignore_index=True)
        data["old_feature" if request.recipe == "deployed" else "new_feature"] = 1.
        return save_backtest_frame(data, "nfl", destination=Path(request.output_dir) / "frame.parquet")
    monkeypatch.setattr(cli, "_prepare", prepare)
    def run(request, reference, *, candidate, output):
        data = load_backtest_frame(request.candidate_frame if candidate else request.baseline_frame)
        assert ("new_feature" in data) == candidate
        assert ("old_feature" in data) != candidate
        assert request.profile == "standard" and request.cache_working_tree is False
        assert (output.parent / "preflight.json").exists()
        calls.append(("run", reference))
        return reference
    monkeypatch.setattr(cli, "_load_or_run", run)
    monkeypatch.setattr(cli, "compare_backtest_runs", lambda a, b, **kw: calls.append(("compare", a, b)))
    monkeypatch.setattr(cli, "write_comparison_artifacts", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv", ["backtest", "compare", "--league", "nfl", "--output-dir", str(tmp_path / "auto")])
    assert cli._main() == 0
    assert calls == [
        ("resolve", "deployed"), ("prepare", "deployed", ("release-sha", "version:1.0")),
        ("prepare", "working-tree", None), ("run", "deployed"), ("run", "working-tree"),
        ("compare", "deployed", "working-tree"),
    ]
    report = json.loads((tmp_path / "auto/preflight.json").read_text())
    assert report["seasons"] == [2023, 2024, 2025]


def test_explicit_frames_skip_preparation(tmp_path, monkeypatch):
    source = save_backtest_frame(frame(), "nfl", destination=tmp_path / "source.parquet")
    monkeypatch.setattr(cli, "_prepare", lambda *a, **k: pytest.fail("unnecessary preparation"))
    request = args(tmp_path, frame=None, baseline_frame=str(source), candidate_frame=str(source))
    cli._comparison_inputs(request, tmp_path / "job")
    assert request.baseline_frame == request.candidate_frame == str(source)
    assert request.through_season is None


def test_one_saved_frame_prepares_only_other_side_for_same_history(tmp_path, monkeypatch):
    source = save_backtest_frame(frame(), "nfl", destination=tmp_path / "source.parquet")
    prepared = []
    def prepare(request, *, resolved):
        prepared.append(request)
        return source
    monkeypatch.setattr(cli, "_prepare", prepare)
    request = args(tmp_path, frame=None, baseline_frame=str(source))
    cli._comparison_inputs(request, tmp_path / "job")
    assert len(prepared) == 1 and prepared[0].output_dir.endswith("preparation/candidate")
    assert prepared[0].current_year == 2025
    assert request.through_season is None  # Preserve the saved artifact's full universe.


def test_artifact_discovers_its_matching_saved_frame(tmp_path, monkeypatch):
    source = save_backtest_frame(frame(), "nfl", destination=tmp_path / "old-job/frames/baseline.parquet")
    monkeypatch.setattr(cli, "_prepare", lambda *a, **k: pytest.fail("artifact was prepared again"))
    request = args(tmp_path, frame=None, baseline=f"artifact:{tmp_path}/old-job/baseline", candidate_frame=str(source))
    cli._comparison_inputs(request, tmp_path / "job")
    assert request.baseline_frame == str(source)
    with pytest.raises(ValueError, match="matching"):
        cli._artifact_frame(f"artifact:{tmp_path}/standalone")


def test_explicit_seasons_set_automatic_preparation_end(tmp_path, monkeypatch):
    requests = []
    def prepare(request, *, resolved):
        requests.append(request)
        return tmp_path / "prepared.parquet"
    monkeypatch.setattr(cli, "_prepare", prepare)
    request = args(tmp_path, frame=None, seasons="2023,2024")
    cli._comparison_inputs(request, tmp_path / "job")
    assert request.through_season == 2024
    assert all(item.current_year == 2024 for item in requests)


def test_make_default_does_not_supply_implicit_latest_frame():
    import subprocess
    result = subprocess.run(["make", "-n", "backtest-expected-points"], cwd=cli.ROOT,
                            check=True, capture_output=True, text=True)
    assert '--profile standard' in result.stdout
    assert '--baseline "deployed"' in result.stdout
    assert '--candidate "working-tree"' in result.stdout
    assert '--frame' not in result.stdout and '--candidate-frame' not in result.stdout


@pytest.mark.parametrize('minor', [4, 5])
def test_preparation_preserves_valid_legacy_and_current_notebook_schemas(tmp_path, minor):
    path = tmp_path / 'source.ipynb'
    synthetic_notebook(path)
    original = nbformat.read(path, as_version=4)
    original.nbformat_minor = minor
    if minor < 5:
        for cell in original.cells:
            cell.pop('id', None)
    nbformat.write(original, path)
    prepared = preparation_notebook(path, export_root=tmp_path / 'export', current_year=2026, current_week=1)
    nbformat.validate(prepared)
    assert prepared.nbformat_minor == minor
    assert all(('id' in cell) == (minor >= 5) for cell in prepared.cells)
    assert prepared.cells[0].source == original.cells[0].source
    assert prepared.cells[2].source == original.cells[1].source
