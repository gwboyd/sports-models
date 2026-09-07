"""Checks for meaningful experiment controls; fitting stays in shared backtests."""
import pandas as pd
import pytest

from scripts.experiment_opponent_adjustment import blend_metric_frames, replace_team_metrics
from src.sports.football.cfb.expected_points.recipe import CFBExpectedPointsRecipe


def test_blend_rejects_changed_population_or_nonmetric_values():
    raw = pd.DataFrame({'game_id': ['a'], 'home_score': [10], 'off_ewma_home': [2.]})
    adjusted = raw.assign(off_ewma_home=4.)
    assert blend_metric_frames(raw, adjusted, .25).off_ewma_home.iloc[0] == 2.5
    with pytest.raises(ValueError, match='outside efficiency'):
        blend_metric_frames(raw, adjusted.assign(home_score=11), .5)
    with pytest.raises(ValueError, match='row and column'):
        blend_metric_frames(raw, adjusted.assign(game_id='b'), .5)


def test_team_metric_join_preserves_population_and_uses_correct_side():
    base = pd.DataFrame({'game_id': [1], 'home_team': ['A'], 'away_team': ['B'], 'home_score': [20]})
    metrics = pd.DataFrame({'game_id': [1, 1], 'team': ['A', 'B'], 'off_ewma': [2., 3.]})
    result = replace_team_metrics(base, metrics)
    assert result.game_id.equals(base.game_id)
    assert result.off_ewma_home.iloc[0] == 2.
    assert result.off_ewma_away.iloc[0] == 3.
    assert result.home_score.equals(base.home_score)


def test_requested_cfb_features_cannot_silently_disappear():
    recipe = CFBExpectedPointsRecipe(efficiency_metrics=('explosiveness', 'ppa'))
    with pytest.raises(ValueError, match='features are missing'):
        recipe.build_config(pd.DataFrame(), season=2025, week=1)


def test_blend_requires_matching_source_history_and_rating_settings(tmp_path):
    import json
    from scripts.experiment_opponent_adjustment import validate_blend_provenance
    raw, full = tmp_path/'raw.parquet', tmp_path/'full.parquet'
    common = dict(league='cfb', alpha=5, carryover=.5, fcs_policy='pooled', history_start=2021,
                  exclude_seasons=[], metrics='ppa', source_sha256={'schedule': 'a'}, code_identity='b')
    raw.with_suffix('.json').write_text(json.dumps({'provenance': {**common, 'strength': 0}}))
    full.with_suffix('.json').write_text(json.dumps({'provenance': {**common, 'strength': 1}}))
    validate_blend_provenance(raw, full)
    full.with_suffix('.json').write_text(json.dumps({'provenance': {**common, 'strength': 1, 'history_start': 2018}}))
    with pytest.raises(ValueError, match='history_start'):
        validate_blend_provenance(raw, full)
