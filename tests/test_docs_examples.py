"""Execute every python code block in docs/extend/*.md.

The extensibility guide is a promise the paper makes; running its examples in
CI guarantees the documentation cannot drift from the code.
"""

import re
from pathlib import Path

import pytest

DOCS_EXTEND = Path(__file__).resolve().parent.parent / "docs" / "extend"
PAGES = sorted(DOCS_EXTEND.glob("*.md"))
BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


@pytest.mark.parametrize("page", PAGES, ids=[p.stem for p in PAGES])
def test_extend_page_examples_run(page):
    blocks = BLOCK_RE.findall(page.read_text(encoding="utf-8"))
    assert blocks, f"{page.name} has no python examples"
    namespace: dict = {}
    for block in blocks:
        exec(compile(block, str(page), "exec"), namespace)  # noqa: S102


def test_all_three_extension_points_documented():
    assert {p.stem for p in PAGES} == {
        "custom-objective", "custom-task-loss", "custom-dataset-adapter",
    }
