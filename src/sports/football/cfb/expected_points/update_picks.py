from pathlib import Path

from src.model_patterns.expected_points.runtime import execute_expected_points_notebook


NOTEBOOK_PATH = Path(__file__).with_name("notebook.ipynb")


def main(request_body, client_name):
    return execute_expected_points_notebook(
        NOTEBOOK_PATH,
        season=request_body.season,
        week=request_body.week,
        client_name=client_name,
    )
