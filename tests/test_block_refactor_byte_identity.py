"""Byte-identity gate for the typed-block-model refactor (see
``pdfparser-tickets/tickets/typed-block-model-design.md`` and
``plans/`` — the refactor's own plan doc). Each migration phase must not
change ``_assemble_html``'s output at all; the golden files under
``tests/data/golden_html/`` were captured from the pre-refactor code and
must keep matching character-for-character through every phase.
"""

from pathlib import Path

import pytest
from test_dump_replay import _replay

_GOLDEN_DIR = Path(__file__).parent / "data" / "golden_html"

_STEMS = (
    "30592559",
    "31051047",
    "31123167",
    "31298526",
    "32117944",
    "32639976",
)


@pytest.mark.parametrize("stem", _STEMS)
def test_byte_identical_to_pre_refactor_golden(stem: str) -> None:
    golden = (_GOLDEN_DIR / f"{stem}.html").read_text(encoding="utf-8")
    assert _replay(stem) == golden, f"{stem}: diverged from pre-refactor golden output"
