from pathlib import Path
import os

from src.model_patterns.expected_points.runtime import execute_expected_points_notebook


NOTEBOOK_PATH = Path(__file__).with_name("notebook.ipynb")


def main(request_body, client_name):
    return execute_expected_points_notebook(
        NOTEBOOK_PATH,
        season=request_body.season,
        week=request_body.week,
        client_name=client_name,
        allow_non_aws_write=request_body.allow_non_aws_write,
        model_version=os.getenv("CFB_EXPECTED_POINTS_VERSION"),
        source_git_sha=os.getenv("SOURCE_GIT_SHA"),
    )
