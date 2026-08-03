from frontend.scripts.sync_team_data import (
    normalize_cfb_team,
    normalize_nfl_team,
    select_cfb_logo_url,
)


def test_cfb_team_normalization_uses_stable_id_aliases_and_standard_logo():
    team, logo_url = normalize_cfb_team(
        {
            "id": 2005,
            "school": "Air Force",
            "mascot": "Falcons",
            "abbreviation": "AF",
            "alternateNames": ["Air Force Academy"],
            "conference": "Mountain West",
            "color": "#003594",
            "alternateColor": "#ffffff",
            "logos": [
                "https://cdn.collegefootballdata.com/logos/256/2005.png",
                "https://cdn.collegefootballdata.com/logos/128/2005.png",
                "https://cdn.collegefootballdata.com/logos-dark/128/2005.png",
            ],
        }
    )

    assert team["id"] == "cfb-2005"
    assert team["logoPath"] == "/teams/cfb/2005.png"
    assert "Air Force Falcons" in team["aliases"]
    assert "Air Force Academy" in team["aliases"]
    assert logo_url == "https://cdn.collegefootballdata.com/logos/128/2005.png"


def test_cfb_logo_selection_avoids_dark_assets_when_possible():
    assert select_cfb_logo_url(
        [
            "https://cdn.collegefootballdata.com/logos-dark/128/1.png",
            "https://cdn.collegefootballdata.com/logos/256/1.png",
        ]
    ) == "https://cdn.collegefootballdata.com/logos/256/1.png"


def test_nfl_team_normalization_ignores_legacy_rows():
    assert normalize_nfl_team({"team_abbr": "LAR", "team_name": "Los Angeles Rams"}) is None
    team, logo_url = normalize_nfl_team(
        {
            "team_abbr": "LA",
            "team_name": "Los Angeles Rams",
            "team_nick": "Rams",
            "team_logo_espn": "https://example.com/rams.png",
        }
    )
    assert team["logoPath"] == "/teams/nfl/la.png"
    assert logo_url == "https://example.com/rams.png"
