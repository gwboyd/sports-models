"""Public chart writes must obey the notebook's existing write authority."""
from __future__ import annotations

import ast
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "src/sports/football/nfl/expected_points/notebook.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _publication_source():
    sources = ["".join(cell.get("source", [])) for cell in _notebook()["cells"]
               if cell["cell_type"] == "code"]
    matches = [source for source in sources if "def upload_image_to_s3(" in source]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("with_importance", [False, True])
def test_read_only_chart_cell_never_creates_s3_client(monkeypatch, with_importance):
    def forbidden_client(*args, **kwargs):
        raise AssertionError("Read-only notebook created an S3 client")

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=forbidden_client))
    buffers = {name: BytesIO(name.encode()) for name in
               ("rankings_buf", "offensive_epa_buf", "defensive_epa_buf")}
    importance = BytesIO(b"importance") if with_importance else None
    namespace = {
        **buffers,
        "feature_importance_buf": importance,
        "should_write_database": False,
        # These values cannot override the already resolved write authority.
        "client_name": "api",
        "allow_non_aws_write": True,
    }
    exec(compile(_publication_source(), str(NOTEBOOK), "exec"), namespace)
    assert all(buffer.closed for buffer in buffers.values())
    assert importance is None or importance.closed


@pytest.mark.parametrize("with_importance", [False, True])
def test_authorized_chart_cell_publishes_expected_buffers(monkeypatch, with_importance):
    uploads = []
    clients = []

    class FakeS3:
        def upload_fileobj(self, buffer, bucket, key, *, ExtraArgs):
            uploads.append((bucket, key, buffer.read(), ExtraArgs))

    def client(service):
        clients.append(service)
        return FakeS3()

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=client))
    buffers = {name: BytesIO(name.encode()) for name in
               ("rankings_buf", "offensive_epa_buf", "defensive_epa_buf")}
    importance = BytesIO(b"importance") if with_importance else None
    namespace = {**buffers, "feature_importance_buf": importance,
                 "should_write_database": True}
    exec(compile(_publication_source(), str(NOTEBOOK), "exec"), namespace)
    assert clients == ["s3"]
    expected = [
        ("charts/power_rankings.png", b"rankings_buf"),
        ("charts/offensive_epa.png", b"offensive_epa_buf"),
        ("charts/defensive_epa.png", b"defensive_epa_buf"),
    ]
    if with_importance:
        expected.append(("charts/feature_importance.png", b"importance"))
    assert [(key, data) for _, key, data, _ in uploads] == expected
    assert all(bucket == "nfl-metrics" for bucket, _, _, _ in uploads)
    assert all(args == {"ContentType": "image/png", "CacheControl": "no-cache"}
               for _, _, _, args in uploads)
    assert all(buffer.closed for buffer in buffers.values())
    assert importance is None or importance.closed


def test_nfl_notebook_schema_and_python_cells_remain_valid():
    notebook = _notebook()
    nbformat.validate(notebook, version=notebook["nbformat"],
                      version_minor=notebook["nbformat_minor"])
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell-{index}")
