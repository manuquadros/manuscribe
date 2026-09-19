"""Ingestion path for datalab-to/chandra-ocr-2's raw response: a flat sequence of
labeled ``<div data-bbox="x0 y0 x1 y1" data-label="...">...</div>`` elements, not
markdown. Parses that div-tree directly into ``list[Block]`` — chandra already tells
you each block's kind and crop region, so this is a different constructor
(``Block.from_chandra_div``) from LightOnOCR's markdown-with-inline-bbox path
(``assemble._page_to_html_parts`` / ``figures._FIGURE_PLACEHOLDER_RE``), not a reuse
of its HTML-sniffing.

See ``spike_results/chandra_ocr2_phase1_fidelity.md`` for the evidence this module's
choices are based on: the ``[0, 1000]``-normalized bbox convention (shared with
LightOnOCR — ``figures._denormalize_bbox`` is reused as-is), the ``data-bbox=``/
``data=`` attribute-name split (fixed per page, not per document — both must be
accepted unconditionally), and the one confirmed decode-loop hallucination that
motivates ``_collapse_repeated_divs``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from PIL import Image  # noqa: TC002 — beartype reads annotations at runtime

from manuscribe.pipeline.block import Block
from manuscribe.pipeline.figures import (
    ImageSink,
    _base64_src,
    _denormalize_bbox,
    _figure_html,
)

# Matches both attribute-name variants chandra emits for the same bbox convention
# (see the module docstring) — a parser that accepts only one silently drops every
# block on whichever page came back with the other.
_DIV_RE = re.compile(
    r'<div data(?:-bbox)?="(?P<bbox>[-\d. ]+)" data-label="(?P<label>[^"]+)">'
    r"(?P<inner>.*?)</div>",
    re.DOTALL,
)
_IMG_TAG_RE = re.compile(r"<img\b")

# A run of more than this many byte-identical adjacent divs (same label + content)
# is a decode-loop hallucination, never real repeated content — same bar
# tables.markup._MAX_IDENTICAL_ROW_RUN uses for the table-row-specific case.
_MAX_IDENTICAL_BLOCK_RUN = 3

_FIGURE_LABELS = frozenset({"Figure", "Image", "Chemical-Block", "Diagram"})


@dataclass(frozen=True, slots=True)
class _RawDiv:
    label: str
    bbox: tuple[int, int, int, int]
    inner_html: str


def _iter_divs(raw: str) -> list[_RawDiv]:
    """Parse chandra's flat div sequence, skipping a block whose bbox attribute
    doesn't parse as four numbers rather than raising — a response is untyped
    model output, not a trusted format."""
    divs: list[_RawDiv] = []
    for m in _DIV_RE.finditer(raw):
        coords = [round(float(x)) for x in m.group("bbox").split()]
        if len(coords) != 4:
            continue
        x0, y0, x1, y1 = coords
        divs.append(
            _RawDiv(
                label=m.group("label"),
                bbox=(x0, y0, x1, y1),
                inner_html=m.group("inner"),
            )
        )
    return divs


def _collapse_repeated_divs(divs: list[_RawDiv]) -> list[_RawDiv]:
    """Collapse a decode-loop run of more than ``_MAX_IDENTICAL_BLOCK_RUN``
    byte-identical adjacent divs to the first occurrence.

    Generalizes ``tables.markup._collapse_repeated_rows``'s table-row guard to any
    block type: the confirmed loop (see module docstring) was a repeated ``<p>``
    caption, not a table row, so the row-specific collapse doesn't see it. A short
    run (at or below the bar) is left alone — nothing here claims two or three
    genuinely identical short captions can't coincide."""
    out: list[_RawDiv] = []
    i = 0
    while i < len(divs):
        j = i + 1
        while (
            j < len(divs)
            and divs[j].label == divs[i].label
            and divs[j].inner_html == divs[i].inner_html
        ):
            j += 1
        if j - i > _MAX_IDENTICAL_BLOCK_RUN:
            out.append(divs[i])
        else:
            out.extend(divs[i:j])
        i = j
    return out


def _split_leading_text(inner_html: str) -> str | None:
    """Split a figure-labeled div's content preceding its first ``<img>`` off as a
    standalone paragraph (a panel label chandra fused into the same div, e.g.
    ``<p><b>(A)</b></p><img alt="...">``) — mirrors how ``figures.py`` already
    treats a LightOnOCR panel label as its own block rather than baking fused
    handling into the figure. Returns ``None`` when there is no leading text or no
    ``<img>`` tag at all (the whole div is still figure-shaped; its bbox alone
    drives the crop)."""
    m = _IMG_TAG_RE.search(inner_html)
    if m is None:
        return None
    before = inner_html[: m.start()].strip()
    return before or None


def parse_chandra_response(
    raw: str,
    page_image: Image.Image,
    *,
    source_page: int | None = None,
    encode_src: ImageSink = _base64_src,
) -> list[Block]:
    """Parse one page's raw chandra-ocr-2 response into ``list[Block]``.

    ``page_image`` is the same rendered page image sent to the model — bboxes are
    normalized to its coordinate space (``figures._denormalize_bbox``), and a
    figure-labeled div's crop comes straight from it, not from a second render.
    ``Page-Header``/``Page-Footer`` divs are dropped (see
    ``Block.from_chandra_div``); everything else becomes one ``Block``, in
    document order.
    """
    blocks: list[Block] = []
    for div in _collapse_repeated_divs(_iter_divs(raw)):
        if div.label in _FIGURE_LABELS:
            leading_text = _split_leading_text(div.inner_html)
            if leading_text is not None:
                leading = Block.from_chandra_div(
                    "Text", leading_text, source_page=source_page
                )
                if leading is not None:
                    blocks.append(leading)
            crop = page_image.crop(_denormalize_bbox(div.bbox, page_image))
            figure_html = _figure_html(crop, None, encode_src)
            block = Block.from_chandra_div(
                div.label, figure_html, source_page=source_page
            )
        else:
            block = Block.from_chandra_div(
                div.label, div.inner_html, source_page=source_page
            )
        if block is not None:
            blocks.append(block)
    return blocks
