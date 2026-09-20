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
motivates the ``_ELEMENT_RE`` collapse in ``_iter_divs``.
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
from manuscribe.pipeline.text import _collapse_repeated_elements

# Matches both attribute-name variants chandra emits for the same bbox convention
# (see the module docstring) — a parser that accepts only one silently drops every
# block on whichever page came back with the other.
#
# The content ends at whichever comes first: this div's ``</div>``, the *next*
# div's opener, or the end of the response.  chandra's sequence is flat (never
# nested), so an opener can only follow a close that the model failed to emit —
# a response truncated mid-div, which is what a decode loop exhausting the token
# budget produces.  Without the two extra terminators a lazy ``.*?</div>`` runs on
# to the following div's close tag and swallows that whole block, silently, and on
# a figure-labeled div the swallowed block is then discarded with the inner HTML.
_DIV_RE = re.compile(
    r'<div data(?:-bbox)?="(?P<bbox>[-\d. ]+)" data-label="(?P<label>[^"]+)">'
    r'(?P<inner>.*?)(?:</div>|(?=<div data(?:-bbox)?=")|\Z)',
    re.DOTALL,
)
_IMG_TAG_RE = re.compile(r"<img\b")
# Any element together with its matching close tag.  The observed decode loop
# repeats a <p>, but the model gets stuck on whatever element it was emitting, so
# the tag is captured and back-referenced rather than listed: the run comparison
# then stays tag-for-tag whichever element looped.  Non-greedy and outermost-first,
# so a <table> or <ol> is one match and its rows/items are not mistaken for
# siblings of the surrounding text.
_ELEMENT_RE = re.compile(
    r"<(?P<tag>[a-zA-Z][\w-]*)\b[^>]*>.*?</(?P=tag)\s*>",
    re.DOTALL,
)

_FIGURE_LABELS = frozenset({"Figure", "Image", "Chemical-Block", "Diagram"})


@dataclass(frozen=True, slots=True)
class _RawDiv:
    label: str
    bbox: tuple[int, int, int, int]
    inner_html: str


def _iter_divs(raw: str) -> list[_RawDiv]:
    """Parse chandra's flat div sequence, skipping a block whose bbox attribute
    doesn't parse as four numbers rather than raising — a response is untyped
    model output, not a trusted format.

    A div the model never closed (a truncated response) is kept, carrying the
    content that preceded the truncation, and is terminated at the next div's
    opener so the following block survives intact (see ``_DIV_RE``). The partial
    content is genuine transcription of that region and its bbox is complete —
    dropping it would hide real body text to no end, and a truncated div whose
    content is a decode loop is already reduced by the collapse below.

    Each div's inner HTML passes through the decode-loop guard here, where every
    consumer of a ``_RawDiv`` gets it: the confirmed hallucination (see the module
    docstring) repeats one element *inside* a single div — a figure's ``<p>`` label
    emitted 342 times until the token budget ran out — so the run to collapse is
    between an element and its siblings, not between whole divs. A short run (at or
    below ``_MAX_IDENTICAL_ELEMENT_RUN``) is left alone; nothing here claims three
    identical short labels can't genuinely coincide."""
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
                inner_html=_collapse_repeated_elements(m.group("inner"), _ELEMENT_RE),
            )
        )
    return divs


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
    for div in _iter_divs(raw):
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
