from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from uuid import uuid4

import papermill as pm
from papermill.exceptions import PapermillExecutionError


logger = logging.getLogger(__name__)


def execute_expected_points_notebook(
    notebook_path: str | Path,
    *,
    season: int,
    week: int,
    client_name: str,
) -> dict:
    run_id = uuid4().hex
    output_path = Path(f"/tmp/expected_points_{run_id}.ipynb")
    result_path = Path(f"/tmp/expected_points_{run_id}.json")
    for path in (output_path, result_path):
        path.unlink(missing_ok=True)

    try:
        try:
            pm.execute_notebook(
                str(notebook_path),
                str(output_path),
                parameters={
                    "current_year": season,
                    "current_week": week,
                    "client_name": client_name,
                    "result_path": str(result_path),
                },
                start_new_kernel=True,
                raise_on_error=True,
            )
        except PapermillExecutionError as exc:
            expected_stop = exc.ename == "SystemExit" and str(exc.evalue) in {"", "0"} and result_path.exists()
            if not expected_stop:
                raise
        if not result_path.exists():
            raise FileNotFoundError(f"Notebook result file was not created: {result_path}")
        with result_path.open("r") as result_file:
            output_data = json.load(result_file)
        output_data["environment"] = output_data.get("environment") or os.getenv("ENVIRONMENT") or "UNKNOWN"
        required = {"write_time", "week", "season", "client_name", "picks_num", "database_updated"}
        missing = sorted(required - output_data.keys())
        if missing:
            raise ValueError(f"Notebook result payload is missing fields: {', '.join(missing)}")
        return output_data
    finally:
        for path in (result_path, output_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not clean up notebook artifact %s", path, exc_info=True)
