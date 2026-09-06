import json
from pathlib import Path

NOTEBOOK_PATH = (
    Path(__file__).parents[1]
    / "src/sports/football/nfl/expected_points/notebook.ipynb"
)


def _code_cells() -> list[dict]:
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_notebook_preserves_inspection_and_guards_full_backtests():
    source = "\n".join("".join(cell["source"]) for cell in _code_cells())
    assert "run_historical_backtest = False" in source
    assert "backtest_cache_working_tree = False" in source
    assert 'backtest_command.append("--cache-working-tree")' in source
    assert "write_backtest_frame = False" in source
    assert "NFLExpectedPointsRecipe" in source
    assert "def inspect_game(game_id):" in source
    assert "health_metrics = health_metrics_frame(ep_run)" in source
    assert 'if client_name != "notebook":' in source
    assert "experiment_df = df.copy(deep=True)" in source
    assert "experiment_config = copy.deepcopy(ep_config)" in source
    assert "if current_year is None:" in source
    assert "if current_week is None:" in source


def test_notebook_backtest_stages_are_independently_runnable():
    cells = ["".join(cell["source"]) for cell in _code_cells()]
    save_index = next(i for i, source in enumerate(cells) if "if write_backtest_frame:" in source)
    train_index = next(i for i, source in enumerate(cells) if "ep_run = run_expected_points" in source)
    inspect_index = next(i for i, source in enumerate(cells) if "def inspect_game(game_id):" in source)
    compare_index = next(i for i, source in enumerate(cells) if "if run_historical_backtest:" in source)
    assert save_index < train_index < inspect_index < compare_index
    assert "run_expected_points" not in cells[save_index]
    assert "subprocess.run" not in cells[train_index]


def test_checked_in_notebook_has_no_stale_execution_outputs():
    for cell in _code_cells():
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []
