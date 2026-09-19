"""The post-flatten block type: one finished, self-contained HTML element in the
assembled document (a ``<p>``, ``<h2>``, ``<table>``, ``<figure>``, ...).

Leaf module — imports only ``text.py`` — mirroring that module's own
no-pipeline-deps discipline.  ``Block.of`` is the single place a block's shape
is derived from its HTML string; every pass downstream of the
``assemble._assemble_document`` flatten point should read a ``Block``'s fields
instead of re-deriving them, so the same "is this a table/heading/caption"
question is answered identically everywhere instead of drifting across
independently-tuned regexes (see ``pdfparser-tickets/tickets/
typed-block-model-design.md``).

Not a replacement for the pre-flatten ``assemble._Block`` (``_MdBlock |
_FigBlock``) union: that models raw, not-yet-rendered markdown source feeding
the HTML renderer (bbox math, unrendered caption text); this models a
finished, rendered element.  The two stages have non-overlapping degrees of
freedom and are kept deliberately separate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from manuscribe.pipeline.text import (
    _BOLD_LABEL_CAPTURE_RE,
    _ends_sentence,
    _heading_inner,
    _opens_with_caption_label,
    _opens_with_table_label,
    _plain_p_text,
    _visible_text,
)

# Anchored "does this whole block open with this element" checks — the
# canonical home for what were duplicated, independently-defined regexes in
# merge.py (``_TABLE_OPEN_RE``, ``_FIGURE_OPEN_RE``). A post-flatten block is
# always one complete top-level element (the block-splitting invariant), so
# an anchored open-tag match is the correct "is this a table/figure" test —
# unlike ``text._TABLE_TAG_RE``, which deliberately matches *any* table tag
# anywhere (including a stranded continuation fragment) for the pre-flatten,
# not-yet-whole-element case.
_TABLE_OPEN_RE = re.compile(r"^<table[\s>]", re.IGNORECASE)
_FIGURE_OPEN_RE = re.compile(r"^<figure[\s>]", re.IGNORECASE)


class BlockKind(Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    OTHER = "other"  # <ol>/<ul> reference lists, or anything else markdown-it emits


class CaptionLabel(Enum):
    FIGURE = "figure"  # also scheme/supplement, per text._opens_with_caption_label
    TABLE = "table"


def _classify_kind(html: str) -> BlockKind:
    if _heading_inner(html) is not None:
        return BlockKind.HEADING
    if _plain_p_text(html) is not None:
        return BlockKind.PARAGRAPH
    if _TABLE_OPEN_RE.match(html):
        return BlockKind.TABLE
    if _FIGURE_OPEN_RE.match(html):
        return BlockKind.FIGURE
    return BlockKind.OTHER


def _caption_label_of(visible_text: str) -> CaptionLabel | None:
    if _opens_with_table_label(visible_text):
        return CaptionLabel.TABLE
    if _opens_with_caption_label(visible_text):
        return CaptionLabel.FIGURE
    return None


def _bold_label_of(text: str) -> str | None:
    m = _BOLD_LABEL_CAPTURE_RE.match(text)
    return (m.group(1) or m.group(2)) if m else None


@dataclass(frozen=True, slots=True)
class Block:
    """One post-flatten HTML block, with its shape derived once.

    ``html`` is always the original, complete string — the bridge that lets a
    pass not yet migrated to read the derived fields keep working unmodified
    on ``block.html`` during the refactor's incremental rollout.
    """

    html: str
    kind: BlockKind
    heading_level: int | None
    inner: str | None
    visible_text: str
    ends_sentence: bool
    bold_label: str | None
    caption_label: CaptionLabel | None
    source_page: int | None = None

    @staticmethod
    def of(html: str, *, source_page: int | None = None) -> Block:
        kind = _classify_kind(html)
        heading_level: int | None = None
        inner: str | None = None
        if kind is BlockKind.HEADING:
            heading = _heading_inner(html)
            assert heading is not None  # _classify_kind already confirmed this
            heading_level, inner = heading
        elif kind is BlockKind.PARAGRAPH:
            inner = _plain_p_text(html)
        sentence_source = inner if inner is not None else html
        return Block(
            html=html,
            kind=kind,
            heading_level=heading_level,
            inner=inner,
            visible_text=_visible_text(html),
            ends_sentence=_ends_sentence(sentence_source),
            bold_label=_bold_label_of(sentence_source),
            caption_label=_caption_label_of(_visible_text(html)),
            source_page=source_page,
        )
