import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "src/sports/football/nfl/expected_points/notebook.ipynb",
    ROOT / "src/sports/football/cfb/expected_points/notebook.ipynb",
)


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebooks_default_to_read_only_and_use_shared_write_policy():
    for path in NOTEBOOKS:
        source = notebook_source(path)
        assert 'client_name = "notebook"' in source
        assert "allow_non_aws_write = False" in source
        assert "resolve_expected_points_write_plan" in source
        assert "confirm_expected_points_write" in source
        assert "write_authorized=" in source
        assert "client_name != 'hey'" not in source
