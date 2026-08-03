import json

from papermill.exceptions import PapermillExecutionError

from src.model_patterns.expected_points.runtime import execute_expected_points_notebook


def test_notebook_runner_injects_unique_result_path(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    def fake_execute(input_path, output_path, *, parameters, **kwargs):
        captured.update(parameters)
        with open(parameters["result_path"], "w") as result_file:
            json.dump(
                {
                    "write_time": "2026-08-01 00:00:00",
                    "week": 1,
                    "season": 2026,
                    "client_name": "pytest",
                    "picks_num": 10,
                    "database_updated": False,
                },
                result_file,
            )

    monkeypatch.setattr("src.model_patterns.expected_points.runtime.pm.execute_notebook", fake_execute)
    payload = execute_expected_points_notebook(
        tmp_path / "notebook.ipynb",
        season=2026,
        week=1,
        client_name="pytest",
    )

    assert captured["current_year"] == 2026
    assert captured["result_path"].startswith("/tmp/expected_points_")
    assert payload["environment"] == "UNKNOWN"


def test_notebook_runner_accepts_expected_post_persistence_stop(monkeypatch, tmp_path):
    def fake_execute(_input_path, _output_path, *, parameters, **_kwargs):
        with open(parameters["result_path"], "w") as result_file:
            json.dump(
                {
                    "write_time": "2026-08-01 00:00:00",
                    "week": 1,
                    "season": 2026,
                    "client_name": "pytest",
                    "picks_num": 10,
                    "database_updated": True,
                },
                result_file,
            )
        raise PapermillExecutionError(1, 1, "sys.exit(0)", "SystemExit", "0", [])

    monkeypatch.setattr("src.model_patterns.expected_points.runtime.pm.execute_notebook", fake_execute)
    payload = execute_expected_points_notebook(
        tmp_path / "notebook.ipynb",
        season=2026,
        week=1,
        client_name="pytest",
    )
    assert payload["database_updated"] is True
