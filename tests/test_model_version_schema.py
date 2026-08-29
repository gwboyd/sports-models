from pathlib import Path


SCHEMA_SQL = (
    Path(__file__).resolve().parents[1]
    / "db/sql/001_create_sports_models_schema.sql"
).read_text(encoding="utf-8")

VERSIONED_TABLES = (
    "nfl_expected_points_picks",
    "nfl_expected_points_results",
    "nfl_expected_points_pick_updates",
    "cfb_expected_points_picks",
    "cfb_expected_points_results",
    "cfb_expected_points_pick_updates",
)


def test_schema_setup_is_atomic():
    normalized = SCHEMA_SQL.lower().strip()
    assert normalized.index("begin;") < normalized.index("create schema")
    assert normalized.endswith("commit;")


def test_versioned_tables_keep_legacy_writers_compatible():
    for table in VERSIONED_TABLES:
        statement = (
            f"alter table sports_models.{table}\n"
            "    alter column model_version set default '1.0';"
        )
        assert statement in SCHEMA_SQL


def test_schema_setup_does_not_remove_operational_data():
    normalized = SCHEMA_SQL.lower()
    assert "\ndelete " not in normalized
    assert "\ntruncate " not in normalized
    assert "\ndrop " not in normalized
