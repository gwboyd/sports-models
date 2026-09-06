#!/usr/bin/env python3
"""Run or compare read-only expected-points walk-forward simulations."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
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
    compare_backtest_runs,
    dependency_fingerprint,
    load_backtest_frame,
    load_backtest_run,
    run_walk_forward,
    write_comparison_artifacts,
    write_run_artifacts,
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
        return reference.removeprefix("sha:"), reference
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


def _run_git_reference(args: argparse.Namespace, reference: str):
    sha, version_label = _resolve_reference(reference, ExpectedPointsLeague(args.league))
    subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory(prefix="expected-points-ref-") as worktree_text, tempfile.TemporaryDirectory(
        prefix="expected-points-run-"
    ) as output_text:
        worktree = Path(worktree_text)
        output = Path(output_text)
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), sha], cwd=ROOT, check=True
        )
        try:
            script = worktree / "scripts/backtest_expected_points.py"
            if not script.exists():
                raise RuntimeError(
                    f"Recipe SHA {sha} predates the backtest protocol; use artifact:<path> "
                    "for its bootstrapped baseline."
                )
            reference_python = _reference_python(worktree, Path(args.cache_root).resolve())
            command = [
                str(reference_python),
                str(script),
                "run",
                "--league", args.league,
                "--profile", args.profile,
                "--frame", str(Path(args.frame).resolve()),
                "--cache-root", str(Path(args.cache_root).resolve()),
                "--recipe-version", version_label,
                "--output-dir", str(output),
            ]
            if args.seasons:
                command.extend(["--seasons", args.seasons])
            if args.cadence:
                command.extend(["--cadence", str(args.cadence)])
            if args.no_cache:
                command.append("--no-cache")
            subprocess.run(command, cwd=worktree, check=True)
            return load_backtest_run(output)
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=ROOT,
                check=False,
                capture_output=True,
            )


def _load_or_run(args: argparse.Namespace, reference: str, *, candidate: bool):
    selected_frame = (
        args.candidate_frame if candidate else args.baseline_frame
    ) or args.frame
    args = argparse.Namespace(**{**vars(args), "frame": selected_frame})
    if reference.startswith("artifact:"):
        return load_backtest_run(reference.removeprefix("artifact:"))
    if reference == "working-tree":
        return _run_current(args, version="working-tree")
    return _run_git_reference(args, reference)


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--league", choices=("nfl", "cfb"), required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--profile", choices=("quick", "standard", "full"), default="standard")
    parser.add_argument("--seasons", default="")
    parser.add_argument("--cadence", type=int)
    parser.add_argument("--cache-root", default=str(ROOT / ".backtests/expected_points"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--cache-working-tree",
        action="store_true",
        help="Enable resumable cutoff caching for the mutable working-tree recipe",
    )


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    _add_shared(run_parser)
    run_parser.add_argument("--recipe-version", default="working-tree")
    run_parser.add_argument("--output-dir", required=True)

    compare_parser = subparsers.add_parser("compare")
    _add_shared(compare_parser)
    compare_parser.add_argument("--baseline", default="deployed")
    compare_parser.add_argument("--candidate", default="working-tree")
    compare_parser.add_argument("--baseline-frame")
    compare_parser.add_argument("--candidate-frame")
    compare_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    compare_parser.add_argument("--output-dir")
    args = parser.parse_args()

    if args.command == "run":
        run = _run_current(args, version=args.recipe_version, output_dir=Path(args.output_dir))
        print(
            f"Backtest run written to {args.output_dir}; rows={len(run.predictions)}; "
            f"trained={run.trained_cutoffs}; cache_hits={run.cache_hits}; "
            f"elapsed={run.elapsed_seconds:.1f}s"
        )
        return 0

    baseline = _load_or_run(args, args.baseline, candidate=False)
    candidate = _load_or_run(args, args.candidate, candidate=True)
    comparison = compare_backtest_runs(
        baseline,
        candidate,
        bootstrap_samples=args.bootstrap_samples,
    )
    output = write_comparison_artifacts(comparison, output_dir=args.output_dir)
    print(f"Backtest comparison written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
