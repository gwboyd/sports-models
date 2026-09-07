#!/usr/bin/env python3
"""Run or compare read-only expected-points walk-forward simulations."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_patterns.expected_points.backtesting import (
    BacktestSpec,
    dependency_fingerprint,
    load_backtest_frame,
    load_backtest_run,
    run_walk_forward,
    recipe_code_identity,
    write_comparison_artifacts,
    write_run_artifacts,
)
from src.model_patterns.expected_points.backtest_artifacts import save_backtest_frame, write_json_atomic
from src.model_patterns.expected_points.backtest_workflow import (
    historical_frame, preflight_artifact, preflight_frames, preparation_notebook, compare_prepared_runs,
)
from src.model_patterns.expected_points.recipes import get_expected_points_recipe
from src.model_patterns.expected_points.types import ExpectedPointsLeague


def _spec(args: argparse.Namespace) -> BacktestSpec:
    seasons = tuple(int(value) for value in args.seasons.split(",") if value) if args.seasons else ()
    return BacktestSpec(
        profile=args.profile,
        seasons=seasons,
        cadence=args.cadence,
        cache_root=Path(args.cache_root),
        use_cache=not args.no_cache,
        cache_working_tree=args.cache_working_tree,
    )


def _run_current(args: argparse.Namespace, *, version: str, output_dir: Path | None = None):
    frame_path = Path(args.frame).resolve()
    if not frame_path.exists():
        raise FileNotFoundError(
            f"Backtest frame not found: {frame_path}. Run the league notebook's "
            "save_backtest_frame(df, league) cell first or pass --frame."
        )
    frame = load_backtest_frame(frame_path)
    recipe = get_expected_points_recipe(args.league, version=version)
    run = run_walk_forward(frame, recipe, _spec(args))
    if output_dir is not None:
        write_run_artifacts(run, output_dir)
    return run


def _resolve_reference(reference: str, league: ExpectedPointsLeague) -> tuple[str, str]:
    if reference.startswith("sha:"):
        sha = subprocess.run(["git", "rev-parse", f"{reference.removeprefix('sha:')}^{{commit}}"],
                             cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        return sha, reference
    if reference == "deployed" or reference.startswith("version:"):
        from dotenv import load_dotenv

        from src.utils.db.sports_models_db import get_expected_points_recipe_source

        load_dotenv(ROOT / ".env")
        version = reference.removeprefix("version:") if reference.startswith("version:") else None
        resolved = get_expected_points_recipe_source(league, version=version)
        if not resolved or not resolved.get("source_git_sha"):
            raise RuntimeError(f"No runnable production SHA could be resolved for {reference}")
        return str(resolved["source_git_sha"]), f"version:{resolved['version']}"
    raise ValueError(f"Unsupported recipe reference: {reference}")


def _reference_python(worktree: Path, cache_root: Path) -> Path:
    """Reuse this interpreter when possible; otherwise build a cached ref environment."""
    fingerprint = dependency_fingerprint(worktree)
    if fingerprint == dependency_fingerprint(ROOT):
        return Path(sys.executable)
    environment = cache_root / "environments" / fingerprint
    python = environment / "bin" / "python"
    manifest = environment / "environment.json"
    if python.exists() and manifest.exists():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if metadata.get("dependency_fingerprint") == fingerprint:
            return python

    environment.parent.mkdir(parents=True, exist_ok=True)
    lock_path = environment.parent / f".{fingerprint}.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if python.exists() and manifest.exists():
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if metadata.get("dependency_fingerprint") == fingerprint:
                return python

        temporary = Path(tempfile.mkdtemp(
            prefix=f".{fingerprint}.building.",
            dir=environment.parent,
        ))
        try:
            uv = shutil.which("uv")
            if uv:
                subprocess.run([uv, "venv", str(temporary)], check=True)
                installer = [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(temporary / "bin" / "python"),
                ]
            else:
                subprocess.run([sys.executable, "-m", "venv", str(temporary)], check=True)
                installer = [str(temporary / "bin" / "python"), "-m", "pip", "install"]
            requirements = worktree / "requirements-dev.txt"
            if not requirements.exists():
                requirements = worktree / "requirements.txt"
            subprocess.run([*installer, "-r", str(requirements)], cwd=worktree, check=True)
            (temporary / "environment.json").write_text(
                json.dumps({"dependency_fingerprint": fingerprint}, indent=2) + "\n",
                encoding="utf-8",
            )
            if environment.exists():
                shutil.rmtree(environment)
            temporary.replace(environment)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return python


@contextmanager
def _checkout(reference: str, league: str, resolved: tuple[str, str] | None = None):
    if reference == "working-tree":
        yield ROOT, "working-tree"
        return
    sha, label = resolved or _resolve_reference(reference, ExpectedPointsLeague(league))
    subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory(prefix="expected-points-ref-") as directory:
        worktree = Path(directory) / "checkout"
        subprocess.run(["git", "worktree", "add", "--detach", str(worktree), sha], cwd=ROOT, check=True)
        try:
            yield worktree, label
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)],
                           cwd=ROOT, check=False, capture_output=True)


def _stream_command(command: list[str], *, cwd: Path, env: dict | None = None) -> None:
    """Stream child progress to the terminal and the parent job's persistent log."""
    with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        try:
            for line in process.stdout:
                logging.info("%s", line.rstrip())
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def _run_git_reference(args: argparse.Namespace, reference: str, output: Path):
    resolved = args.resolved_references.get(reference)
    with _checkout(reference, args.league, resolved) as (worktree, version_label):
        script = worktree / "scripts/backtest_expected_points.py"
        if not script.exists():
            raise RuntimeError("Reference predates the backtest protocol; use artifact:<path> for its baseline")
        reference_python = _reference_python(worktree, Path(args.cache_root).resolve())
        command = [str(reference_python), str(script), "run", "--league", args.league,
                   "--profile", args.profile, "--frame", str(Path(args.frame).resolve()),
                   "--cache-root", str(Path(args.cache_root).resolve()),
                   "--recipe-version", version_label, "--output-dir", str(output)]
        if args.seasons:
            command.extend(["--seasons", args.seasons])
        if args.cadence:
            command.extend(["--cadence", str(args.cadence)])
        if args.no_cache:
            command.append("--no-cache")
        _stream_command(command, cwd=worktree)
        return load_backtest_run(output)


def _load_or_run(args: argparse.Namespace, reference: str, *, candidate: bool, output: Path):
    selected_frame = args.candidate_frame if candidate else args.baseline_frame
    args = argparse.Namespace(**{**vars(args), "frame": selected_frame})
    if reference.startswith("artifact:"):
        run = load_backtest_run(reference.removeprefix("artifact:"))
        write_run_artifacts(run, output)
        return run
    if reference == "working-tree":
        return _run_current(args, version="working-tree", output_dir=output)
    return _run_git_reference(args, reference, output)


@contextmanager
def _job(output: Path, args: argparse.Namespace):
    # Creating a fresh directory is also an exclusive claim. Never overwrite a
    # prior job's evidence, and never race another invocation in the same job.
    output.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output / "execution.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger = logging.getLogger()
    previous_level = logger.level
    logger.setLevel(min(previous_level, logging.INFO))
    logger.addHandler(handler)
    started = time.monotonic()
    request = {key: value for key, value in vars(args).items() if not key.startswith("_")}
    if args.command == "compare":
        request["bootstrap_random_seed"] = 31
    write_json_atomic(output / "request.json", request)
    write_json_atomic(output / "status.json", {"status": "running"})
    try:
        yield
    except BaseException as error:
        write_json_atomic(output / "status.json", {
            "status": "failed", "error": str(error), "elapsed_seconds": time.monotonic() - started,
        })
        logging.exception("Job failed; completed artifacts and logs remain in %s", output)
        raise
    else:
        write_json_atomic(output / "status.json", {
            "status": "complete", "elapsed_seconds": time.monotonic() - started,
        })
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        handler.close()


def _prepare(args: argparse.Namespace, *, resolved: tuple[str, str] | None = None) -> Path:
    import nbformat
    from dotenv import load_dotenv

    output = Path(args.output_dir).resolve()
    with _job(output, args):
        load_dotenv(ROOT / ".env")
        with _checkout(args.recipe, args.league, resolved) as (checkout, label):
            identity = recipe_code_identity(ExpectedPointsLeague(args.league), root=checkout)
            python = _reference_python(checkout, Path(args.cache_root).resolve())
            notebook_path = checkout / f"src/sports/football/{args.league}/expected_points/notebook.ipynb"
            notebook_sha256 = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
            notebook = preparation_notebook(
                notebook_path,
                export_root=output / "export", current_year=args.current_year, current_week=args.current_week,
            )
            nbformat.write(notebook, output / "input.ipynb")
            kernel = output / "jupyter/kernels/backtest-prepare/kernel.json"
            write_json_atomic(kernel, {
                "argv": [str(python), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Backtest preparation", "language": "python",
            })
            env = {**os.environ, "LOCALHOST": "False", "PYTHONPATH": str(checkout),
                   "JUPYTER_PATH": str(output / "jupyter"), "PYTHONUNBUFFERED": "1"}
            # Never inherit runtime markers that grant deployed write authority.
            for key in ("AWS_LAMBDA_FUNCTION_NAME", "AWS_LAMBDA_RUNTIME_API", "LAMBDA_TASK_ROOT"):
                env.pop(key, None)
            logging.info("Preparing %s frame from %s (%s); stops before training", args.league, label, identity)
            _stream_command([
                str(python), "-m", "papermill", str(output / "input.ipynb"), str(output / "executed.ipynb"),
                "--kernel", "backtest-prepare", "--cwd", str(checkout), "--log-output", "--no-progress-bar",
            ], cwd=checkout, env=env)
            if (recipe_code_identity(ExpectedPointsLeague(args.league), root=checkout) != identity
                    or hashlib.sha256(notebook_path.read_bytes()).hexdigest() != notebook_sha256):
                raise RuntimeError("Recipe changed during frame preparation; retry with a frozen candidate")
            frame = historical_frame(load_backtest_frame(output / f"export/frames/{args.league}/latest.parquet"),
                                     args.through_season)
            destination = save_backtest_frame(frame, args.league, destination=output / "frame.parquet", provenance={
                "reference": args.recipe, "resolved_reference": label, "code_identity": identity,
                "notebook_sha256": notebook_sha256,
                "current_year": args.current_year, "current_week": args.current_week,
                "through_season": args.through_season,
            })
            logging.info("Prepared %s rows: %s", len(frame), destination)
            return destination


def _default_history_end(now: datetime | None = None) -> int:
    """Conservative football season boundary, shared by NFL and CFB.

    Wait until March to call the previous calendar year's season closed. This
    avoids treating a midseason gap with no currently quoted games as completion.
    """
    now = now or datetime.now(timezone.utc)
    return now.year - (2 if now.month < 3 else 1)


def _artifact_frame(reference: str) -> Path:
    run = Path(reference.removeprefix("artifact:")).resolve()
    # Comparison jobs save their two source bundles beside their run artifacts.
    source = run.parent / "frames" / f"{run.name}.parquet"
    if run.name in {"baseline", "candidate"} and source.exists():
        return source
    raise ValueError(
        f"Cannot locate the input frame for {reference}. Pass the matching "
        "--baseline-frame or --candidate-frame; saved predictions are not feature inputs."
    )


def _comparison_inputs(args: argparse.Namespace, output: Path) -> None:
    """Resolve once, then prepare only the sides without explicit saved inputs."""
    args.resolved_references = {}
    selected = {}
    for side in ("baseline", "candidate"):
        reference = getattr(args, side)
        if reference != "working-tree" and not reference.startswith("artifact:"):
            if reference not in args.resolved_references:
                resolved = _resolve_reference(reference, ExpectedPointsLeague(args.league))
                supported = subprocess.run(
                    ["git", "cat-file", "-e", f"{resolved[0]}:scripts/backtest_expected_points.py"],
                    cwd=ROOT, capture_output=True,
                )
                if supported.returncode:
                    raise ValueError(f"{side} reference is unavailable or predates the backtest protocol; use artifact:<path>")
                args.resolved_references[reference] = resolved
                logging.info("Resolved %s: %s -> %s (%s)", side, reference, *resolved)
        source = getattr(args, f"{side}_frame") or args.frame
        if not source and reference.startswith("artifact:"):
            source = _artifact_frame(reference)
        selected[side] = Path(source).resolve() if source else None

    # Validate explicit bundles before spending time preparing their counterpart.
    supplied = [load_backtest_frame(path) for path in selected.values() if path is not None]
    if any(path is None for path in selected.values()):
        if args.through_season is None and not supplied:
            requested = _spec(args).seasons
            args.through_season = max(requested) if requested else _default_history_end()
        history_end = args.through_season or max(int(frame.season.max()) for frame in supplied)
        current_year = getattr(args, "current_year", None) or history_end
        current_week = getattr(args, "current_week", None) or 1
        if current_year < history_end:
            raise ValueError("--current-year must cover the requested history end (--through-season / --seasons)")
        logging.info("Preparing missing inputs through season %s (notebook context %s week %s)",
                     history_end, current_year, current_week)
        for side, path in selected.items():
            if path is not None:
                continue
            reference = getattr(args, side)
            prepare_args = argparse.Namespace(
                command="prepare-frame", league=args.league, recipe=reference,
                current_year=current_year, current_week=current_week,
                through_season=args.through_season, cache_root=args.cache_root,
                output_dir=str(output / "preparation" / side),
            )
            selected[side] = _prepare(prepare_args, resolved=args.resolved_references.get(reference))
    for side, path in selected.items():
        setattr(args, f"{side}_frame", str(path))


def _comparison_code_identity(league: str) -> tuple[str, str]:
    """Include orchestration/feature cells throughout the full comparison job."""
    notebook = ROOT / f"src/sports/football/{league}/expected_points/notebook.ipynb"
    return (recipe_code_identity(ExpectedPointsLeague(league)),
            hashlib.sha256(notebook.read_bytes()).hexdigest())


def _compare(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir).resolve() if args.output_dir else (
        ROOT / ".backtests/expected_points/comparisons" / args.league / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    with _job(output, args):
        spec = _spec(args)
        frames = {}
        current_identity = _comparison_code_identity(args.league)
        logging.info("Comparison job: %s; profile=%s; %s vs %s", output, args.profile, args.baseline, args.candidate)
        _comparison_inputs(args, output)
        write_json_atomic(output / "resolved_request.json", vars(args))
        for side in ("baseline", "candidate"):
            reference = getattr(args, side)
            source = Path(getattr(args, f"{side}_frame")).resolve()
            frame = historical_frame(load_backtest_frame(source), args.through_season)
            metadata = json.loads(source.with_suffix(".json").read_text())
            provenance = metadata.get("provenance", {})
            if metadata.get("league") != args.league:
                raise ValueError(f"{side} frame has the wrong league")
            if reference == "working-tree":
                identity = recipe_code_identity(ExpectedPointsLeague(args.league))
                notebook_path = ROOT / f"src/sports/football/{args.league}/expected_points/notebook.ipynb"
                notebook_changed = (provenance.get("notebook_sha256")
                                    and provenance["notebook_sha256"] != hashlib.sha256(notebook_path.read_bytes()).hexdigest())
                if (provenance.get("code_identity") and provenance["code_identity"] != identity) or notebook_changed:
                    raise ValueError(f"{side} frame was built from different code; regenerate it")
            elif not reference.startswith("artifact:"):
                resolved = args.resolved_references[reference]
                if provenance.get("code_identity") and provenance["code_identity"].split(":")[0] != resolved[0]:
                    raise ValueError(f"{side} frame was built from a different Git SHA; regenerate it")
                if provenance.get("code_identity"):
                    # A dirty frame can share HEAD with an immutable reference.
                    # Compare content identities, not just their Git SHA prefix.
                    with _checkout(reference, args.league, resolved) as (checkout, _label):
                        expected = recipe_code_identity(ExpectedPointsLeague(args.league), root=checkout)
                        notebook_path = checkout / f"src/sports/football/{args.league}/expected_points/notebook.ipynb"
                        notebook_changed = (provenance.get("notebook_sha256") and provenance["notebook_sha256"]
                                            != hashlib.sha256(notebook_path.read_bytes()).hexdigest())
                        if provenance["code_identity"] != expected or notebook_changed:
                            raise ValueError(f"{side} frame differs from the immutable recipe code; regenerate it")
            if not provenance.get("code_identity"):
                logging.warning("%s frame has no code provenance; verify its feature recipe or use prepare-frame", side)
            frames[side] = frame
            path = save_backtest_frame(frame, args.league, destination=output / "frames" / f"{side}.parquet",
                                       provenance={**provenance, "copied_from": str(source),
                                                   "comparison_through_season": args.through_season})
            setattr(args, f"{side}_frame", str(path))
        report = preflight_frames(frames["baseline"], frames["candidate"], spec)
        for side in ("baseline", "candidate"):
            reference = getattr(args, side)
            if reference.startswith("artifact:"):
                preflight_artifact(load_backtest_run(reference.removeprefix("artifact:")), frames[side], spec, args.league)
        write_json_atomic(output / "preflight.json", report)
        write_json_atomic(output / "resolved_request.json", vars(args))
        logging.info("Preflight passed: %s source rows, seasons %s, %s cutoffs per recipe",
                     report["rows"], report["seasons"], len(report["cutoffs"]))
        if report["training_history_differs"]:
            logging.info("Earlier training history differs: baseline=%s; candidate=%s. Evaluation inputs match.",
                         report["baseline_training_seasons"], report["candidate_training_seasons"])
        if args.preflight_only:
            return output
        runs = {}
        for side in ("baseline", "candidate"):
            if "working-tree" in (args.baseline, args.candidate):
                if _comparison_code_identity(args.league) != current_identity:
                    raise RuntimeError("Working-tree code changed during comparison; regenerate frames and retry")
            runs[side] = _load_or_run(args, getattr(args, side), candidate=(side == "candidate"), output=output / side)
        if "working-tree" in (args.baseline, args.candidate):
            if _comparison_code_identity(args.league) != current_identity:
                raise RuntimeError("Working-tree code changed during comparison; completed artifacts are retained")
        baseline, candidate = runs["baseline"], runs["candidate"]
        logging.info("Both runs saved. Calculating paired intervals (%s bootstrap samples)", args.bootstrap_samples)
        comparison = compare_prepared_runs(
            baseline, candidate, frames["baseline"], frames["candidate"], spec,
            bootstrap_samples=args.bootstrap_samples,
        )
        write_comparison_artifacts(comparison, output_dir=output)
    return output


def _parse_leagues(value: str) -> tuple[str, ...]:
    leagues = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if len(leagues) < 2:
        raise argparse.ArgumentTypeError("--leagues requires at least two comma-separated leagues")
    if len(leagues) != len(set(leagues)):
        raise argparse.ArgumentTypeError("--leagues cannot contain duplicates")
    unsupported = sorted(set(leagues) - {"nfl", "cfb"})
    if unsupported:
        raise argparse.ArgumentTypeError(f"Unsupported leagues: {', '.join(unsupported)}")
    return leagues


def _comparison_subprocess_command(
    args: argparse.Namespace,
    *,
    league: str,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "compare",
        "--league", league,
        "--profile", args.profile,
        "--baseline", args.baseline,
        "--candidate", args.candidate,
        "--bootstrap-samples", str(args.bootstrap_samples),
        "--cache-root", str(Path(args.cache_root).resolve()),
        "--output-dir", str(output_dir),
    ]
    for option, value in (
        ("--seasons", getattr(args, "seasons", "")),
        ("--cadence", getattr(args, "cadence", None)),
        ("--through-season", getattr(args, "through_season", None)),
        ("--current-year", getattr(args, "current_year", None)),
        ("--current-week", getattr(args, "current_week", None)),
    ):
        if value not in (None, ""):
            command.extend([option, str(value)])
    for option, enabled in (
        ("--no-cache", args.no_cache),
        ("--cache-working-tree", args.cache_working_tree),
        ("--preflight-only", args.preflight_only),
    ):
        if enabled:
            command.append(option)
    return command


def _compare_many(args: argparse.Namespace) -> int:
    """Run independent league comparisons concurrently without changing fit behavior."""
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    processes: list[tuple[str, Path, subprocess.Popen]] = []
    try:
        for league in args.leagues:
            output = ROOT / ".backtests/expected_points/comparisons" / league / batch_id
            command = _comparison_subprocess_command(args, league=league, output_dir=output)
            logging.info("Starting %s comparison: %s", league, output)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            processes.append((league, output, process))

        failed = False
        for league, output, process in processes:
            return_code = process.wait()
            failed = failed or return_code != 0
            print(f"{league}: {'complete' if return_code == 0 else f'failed ({return_code})'} -> {output}")
        return int(failed)
    except BaseException:
        for _league, _output, process in processes:
            if process.poll() is None:
                process.terminate()
        for _league, _output, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def _add_shared(
    parser: argparse.ArgumentParser,
    *,
    include_league: bool = True,
    include_frame: bool = True,
) -> None:
    if include_league:
        parser.add_argument("--league", choices=("nfl", "cfb"), required=True)
    if include_frame:
        parser.add_argument("--frame", help="Explicit shared saved frame; comparisons otherwise prepare each version automatically")
    parser.add_argument("--profile", choices=("quick", "standard", "full"), default="standard",
                        help="Coverage: standard (default) = 3 seasons weekly; quick = 1 season sampled; full = 4 seasons")
    parser.add_argument("--seasons", default="")
    parser.add_argument("--cadence", type=int)
    parser.add_argument("--cache-root", default=str(ROOT / ".backtests/expected_points"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--cache-working-tree",
        action="store_true",
        help="Enable resumable cutoff caching for the mutable working-tree recipe",
    )


def _add_comparison_options(parser: argparse.ArgumentParser, *, single_league: bool) -> None:
    parser.add_argument("--baseline", default="deployed", help="Recipe reference (default: deployed)")
    parser.add_argument("--candidate", default="working-tree", help="Recipe reference (default: working-tree)")
    if single_league:
        parser.add_argument("--baseline-frame", help="Reuse this saved baseline frame instead of preparing it")
        parser.add_argument("--candidate-frame", help="Reuse this saved candidate frame instead of preparing it")
        parser.add_argument("--output-dir", help="New job directory; default: a timestamped comparison directory")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--through-season", type=int, help="Explicit inclusive upper season bound on both frames")
    parser.add_argument("--preflight-only", action="store_true", help="Prepare/check inputs and stop before fitting")
    parser.add_argument("--current-year", type=int, help="Advanced: notebook loading context; defaults to history end")
    parser.add_argument("--current-week", type=int, default=1, help="Advanced: notebook loading context (default: 1)")


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    _add_shared(run_parser)
    run_parser.add_argument("--recipe-version", default="working-tree")
    run_parser.add_argument("--output-dir", required=True)

    compare_parser = subparsers.add_parser(
        "compare", help="Automatically prepare and compare deployed vs working-tree (standard by default)",
        description="Normal evaluation: compare --league nfl (or cfb). Prepares both versions, checks inputs, "
                    "and runs standard automatically. All other flags are optional.",
    )
    _add_shared(compare_parser)
    _add_comparison_options(compare_parser, single_league=True)
    compare_many_parser = subparsers.add_parser(
        "compare-many",
        help="Run independent league comparisons concurrently",
    )
    compare_many_parser.add_argument("--leagues", required=True, type=_parse_leagues)
    _add_shared(compare_many_parser, include_league=False, include_frame=False)
    _add_comparison_options(compare_many_parser, single_league=False)
    prepare_parser = subparsers.add_parser("prepare-frame", help="Execute loading/features only, using the selected recipe checkout")
    prepare_parser.add_argument("--league", choices=("nfl", "cfb"), required=True)
    prepare_parser.add_argument("--recipe", default="working-tree")
    prepare_parser.add_argument("--current-year", type=int, required=True)
    prepare_parser.add_argument("--current-week", type=int, required=True)
    prepare_parser.add_argument("--through-season", type=int)
    prepare_parser.add_argument("--cache-root", default=str(ROOT / ".backtests/expected_points"))
    prepare_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "prepare-frame":
        print(f"Backtest frame written to {_prepare(args)}")
        return 0
    if args.command == "run" and not args.frame:
        parser.error("run requires --frame; use compare for automatic preparation")
    if args.command in {"compare", "compare-many"} and args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")

    if args.command == "compare-many":
        return _compare_many(args)

    if args.command == "run":
        with _job(Path(args.output_dir).resolve(), args):
            run = _run_current(args, version=args.recipe_version, output_dir=Path(args.output_dir))
        print(
            f"Backtest run written to {args.output_dir}; rows={len(run.predictions)}; "
            f"trained={run.trained_cutoffs}; cache_hits={run.cache_hits}; "
            f"elapsed={run.elapsed_seconds:.1f}s"
        )
        return 0

    output = _compare(args)
    print(f"Backtest {'preflight' if args.preflight_only else 'comparison'} written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
