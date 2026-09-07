"""Expected-points model release drafts and version arithmetic.

Release notes use a small, strict Markdown format. Prepared versions live in
model-versions.json; this module owns version arithmetic and note parsing shared
by preparation, deployment, and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re


class ModelKey(str, Enum):
    NFL_EXPECTED_POINTS = "nfl_expected_points"
    CFB_EXPECTED_POINTS = "cfb_expected_points"


@dataclass(frozen=True, order=True)
class ModelVersion:
    major: int
    minor: int

    def __post_init__(self) -> None:
        if self.major < 1 or self.minor < 0:
            raise ValueError("Model versions must have major >= 1 and minor >= 0")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


@dataclass(frozen=True)
class ReleaseDraft:
    title: str
    public_summary: str
    changes_md: str
    evaluation_md: str | None
    internal_notes_md: str | None
    original_markdown: str


_VERSION_RE = re.compile(r"^(?P<major>[1-9]\d*)\.(?P<minor>\d+)$")
_SECTION_NAMES = {
    "Public Summary": "public_summary",
    "Changes": "changes_md",
    "Evaluation": "evaluation_md",
    "Internal Notes": "internal_notes_md",
}


def parse_version(value: str) -> ModelVersion:
    match = _VERSION_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Invalid model version {value!r}; expected MAJOR.MINOR")
    return ModelVersion(int(match.group("major")), int(match.group("minor")))


def next_major(version: ModelVersion) -> ModelVersion:
    return ModelVersion(version.major + 1, 0)


def next_minor(version: ModelVersion) -> ModelVersion:
    return ModelVersion(version.major, version.minor + 1)


def _normalize_section(value: str) -> str:
    return value.strip()


def parse_release_markdown(markdown: str, *, source: str = "release draft") -> ReleaseDraft | None:
    """Parse a release draft, returning ``None`` only for an empty file.

    Headings are deliberately exact so a typo cannot silently drop public or
    internal release information during deployment.
    """

    if not markdown.strip():
        return None
    lines = markdown.splitlines()
    first = next((line.strip() for line in lines if line.strip()), "")
    if not first.startswith("# ") or first.startswith("## "):
        raise ValueError(f"{source} must begin with a single '# ' title")
    title = first[2:].strip()
    if not title:
        raise ValueError(f"{source} title cannot be empty")

    sections: dict[str, list[str]] = {}
    current: str | None = None
    title_index = next(index for index, line in enumerate(lines) if line.strip())
    for line in lines[title_index + 1 :]:
        stripped = line.strip()
        if stripped.startswith("# "):
            raise ValueError(f"{source} may contain only one top-level title")
        if stripped.startswith("## "):
            heading = stripped[3:].strip()
            if heading not in _SECTION_NAMES:
                raise ValueError(f"{source} has unsupported section {heading!r}")
            if heading in sections:
                raise ValueError(f"{source} repeats section {heading!r}")
            current = heading
            sections[current] = []
            continue
        if current is None:
            if stripped:
                raise ValueError(f"{source} has content outside a release section")
            continue
        sections[current].append(line)

    public_summary = _normalize_section("\n".join(sections.get("Public Summary", [])))
    changes_md = _normalize_section("\n".join(sections.get("Changes", [])))
    if not public_summary:
        raise ValueError(f"{source} requires a non-empty '## Public Summary' section")
    if not changes_md:
        raise ValueError(f"{source} requires a non-empty '## Changes' section")

    def optional_section(name: str) -> str | None:
        value = _normalize_section("\n".join(sections.get(name, [])))
        return value or None

    return ReleaseDraft(
        title=title,
        public_summary=public_summary,
        changes_md=changes_md,
        evaluation_md=optional_section("Evaluation"),
        internal_notes_md=optional_section("Internal Notes"),
        original_markdown=markdown,
    )


def parse_release_file(path: str | Path) -> ReleaseDraft | None:
    path = Path(path)
    return parse_release_markdown(path.read_text(encoding="utf-8"), source=str(path))
