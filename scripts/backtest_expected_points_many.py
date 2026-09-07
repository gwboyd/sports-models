#!/usr/bin/env python3
"""Convenience launcher for concurrent comparisons; sequential cold runs are generally faster."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/backtest_expected_points.py"
UNSUPPORTED_SHARED_OPTIONS = {"--frame", "--baseline-frame", "--candidate-frame", "--output-dir", "--league"}


def _parse_leagues(value: str) -> tuple[str, ...]:
    leagues = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if set(leagues) != {"nfl", "cfb"} or len(leagues) != 2:
        raise argparse.ArgumentTypeError("--leagues must contain nfl and cfb exactly once")
    return leagues


def _run(leagues: tuple[str, ...], comparison_args: list[str]) -> int:
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    processes: list[tuple[str, Path, subprocess.Popen]] = []
    try:
        for league in leagues:
            output = ROOT / ".backtests/expected_points/comparisons" / league / batch_id
            command = [sys.executable, str(RUNNER), "compare", "--league", league,
                       "--output-dir", str(output), *comparison_args]
            print(f"Starting {league}: {output}", flush=True)
            processes.append((league, output, subprocess.Popen(
                command, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )))

        failed = False
        for league, output, process in processes:
            return_code = process.wait()
            failed |= return_code != 0
            state = "complete" if return_code == 0 else f"failed ({return_code})"
            print(f"{league}: {state} -> {output}")
        return int(failed)
    except BaseException:
        for _league, _output, process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for _league, _output, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leagues", required=True, type=_parse_leagues)
    args, comparison_args = parser.parse_known_args()
    invalid = sorted(option for option in comparison_args
                     if option.split("=", 1)[0] in UNSUPPORTED_SHARED_OPTIONS)
    if invalid:
        parser.error(f"These options require a single league: {', '.join(invalid)}")
    return _run(args.leagues, comparison_args)


if __name__ == "__main__":
    raise SystemExit(_main())
