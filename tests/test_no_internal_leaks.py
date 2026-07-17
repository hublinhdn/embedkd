"""Architecture-boundary guard: the public package must stay domain-agnostic.

The core may never mention the pill domain, internal hostnames or absolute
personal paths. This automates the boundary instead of trusting memory.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "embedkd"
FORBIDDEN = ("pill", "ogyei", "epillid", "labai", "/Users/", "C:\\")


def test_core_package_has_no_internal_leaks():
    offenders = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for token in FORBIDDEN:
            if token.lower() in text:
                offenders.append(f"{path.name}: '{token}'")
    assert not offenders, f"Forbidden internal references in core package: {offenders}"
