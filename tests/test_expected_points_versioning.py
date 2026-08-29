from pathlib import Path

import pytest

from src.model_patterns.expected_points.versioning import (
    ModelKey,
    ModelVersion,
    next_major,
    next_minor,
    parse_release_markdown,
    parse_version,
)


def valid_draft() -> str:
    return """# Player availability

## Public Summary

The model now considers additional information.

## Changes

- Added a new model input.

## Evaluation

The fixed-window comparison is recorded here.

## Internal Notes

Keep this section private.
"""


def test_parse_empty_draft_returns_none():
    assert parse_release_markdown("  \n\n") is None


def test_parse_release_markdown_sections():
    draft = parse_release_markdown(valid_draft())
    assert draft is not None
    assert draft.title == "Player availability"
    assert "new model input" in draft.changes_md
    assert draft.evaluation_md == "The fixed-window comparison is recorded here."
    assert draft.internal_notes_md == "Keep this section private."


def test_parse_release_markdown_requires_public_sections():
    with pytest.raises(ValueError, match="Public Summary"):
        parse_release_markdown("# Incomplete\n\n## Changes\n- change\n")
    with pytest.raises(ValueError, match="Changes"):
        parse_release_markdown("# Incomplete\n\n## Public Summary\nsummary\n")


def test_parse_release_markdown_rejects_unknown_or_duplicate_sections():
    with pytest.raises(ValueError, match="unsupported section"):
        parse_release_markdown("# Draft\n\n## Summary\ntext\n")
    with pytest.raises(ValueError, match="repeats section"):
        parse_release_markdown(
            "# Draft\n\n## Public Summary\ntext\n\n## Public Summary\ntext\n## Changes\nchange"
        )


def test_parse_release_file_uses_utf8(tmp_path: Path):
    path = tmp_path / "UNRELEASED.md"
    path.write_text(valid_draft(), encoding="utf-8")
    assert parse_release_markdown(path.read_text(encoding="utf-8")).title == "Player availability"


def test_version_arithmetic():
    version = parse_version("2.3")
    assert version == ModelVersion(2, 3)
    assert str(next_minor(version)) == "2.4"
    assert str(next_major(version)) == "3.0"
    assert ModelKey.CFB_EXPECTED_POINTS.value == "cfb_expected_points"


@pytest.mark.parametrize("value", ["1", "1.2.3", "0.1", "1.-1", "a.b"])
def test_parse_version_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_version(value)
