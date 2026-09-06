from src.model_patterns.expected_points.types import ExpectedPointsLeague
from src.sports.football.expected_points_api import build_expected_points_routers


_routers = build_expected_points_routers(ExpectedPointsLeague.NFL)
picks = _routers.picks
pick_results = _routers.results
update = _routers.update
