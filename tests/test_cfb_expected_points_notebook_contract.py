import json
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).parents[1]
    / "src/sports/football/cfb/expected_points/notebook.ipynb"
)


def _code_cells() -> list[dict]:
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_notebook_uses_the_shared_cfb_market_pipeline():
    source = "\n".join("".join(cell["source"]) for cell in _code_cells())

    assert "assemble_cfb_betting_markets" in source
    assert "scores_to_cfb_bets" in source
    assert 'confidence_scoring="neg_log_loss"' in source
    assert 'betting_transform=scores_to_cfb_bets' in source
    assert "backtest_optional_confidence_feature_groups" in source
    assert '["spread_diff"]' in source
    assert '["total_diff"]' in source
    assert "random.choice" not in source
    assert "preferred_providers" not in source
    assert "moneyline_home" not in source


def test_optional_market_features_are_not_promoted_without_a_gate_result():
    configuration_cell = next(
        "".join(cell["source"])
        for cell in _code_cells()
        if "confidence_market_features =" in "".join(cell["source"])
    )
    configured_features = configuration_cell.split(
        "spread_class_features =", maxsplit=1
    )[0]

    assert "consensus_home_win_prob" not in configured_features
    assert "spread_open_line" not in configured_features
    assert "total_open_line" not in configured_features
    assert "spread_quote_count" not in configured_features
    assert "total_quote_count" not in configured_features


def test_checked_in_notebook_has_no_stale_execution_outputs():
    for cell in _code_cells():
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []
