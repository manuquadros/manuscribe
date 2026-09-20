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

from manuscribe.pipeline.block import _FIGURE_LABELS, Block
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
_IMG_TAG_RE = re.compile(r"<img\b[^>]*>")
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

# Any tag, open or close, with its self-closing slash captured — the node splitter
# below tracks nesting depth with it, so a ``<p>`` that *contains* an image is one
# node and is never cut in half.
_TAG_RE = re.compile(r"<(?P<close>/?)(?P<tag>[a-zA-Z][\w-]*)\b[^>]*?(?P<void>/?)>")
_VOID_TAGS = frozenset({"br", "col", "hr", "img", "input", "source", "wbr"})


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


def _top_level_nodes(inner_html: str) -> list[str]:
    """Split a div's inner HTML into its top-level nodes, in document order.

    A node is one complete element with its subtree, one void tag, or a run of
    bare text between them; nesting depth is tracked so an element is never cut
    in half.  Everything the model left unbalanced (a truncated response, a stray
    close tag) lands in the surrounding node rather than raising — this is
    untyped model output.  Empty/whitespace nodes are dropped.
    """
    nodes: list[str] = []
    depth = 0
    start = 0
    for m in _TAG_RE.finditer(inner_html):
        if m.group("close"):
            if depth:
                depth -= 1
                if depth == 0:
                    nodes.append(inner_html[start : m.end()])
                    start = m.end()
        elif m.group("void") or m.group("tag").lower() in _VOID_TAGS:
            if depth == 0:
                nodes.append(inner_html[start : m.start()])
                nodes.append(m.group())
                start = m.end()
        else:
            if depth == 0:
                nodes.append(inner_html[start : m.start()])
                start = m.start()
            depth += 1
    nodes.append(inner_html[start:])
    return [n for n in nodes if n.strip()]


def _figure_div_blocks(
    div: _RawDiv,
    page_image: Image.Image,
    encode_src: ImageSink,
    source_page: int | None,
) -> list[Block]:
    """Blocks for one figure-labeled div: its content as a *sequence*, with the
    crop standing in for the image.

    chandra fuses into a figure div whatever shares the region — a panel label
    before the image, and after it prose, further images with their compound
    labels, or a complete ``<table>`` it already transcribed.  Each top-level
    node therefore reaches the block stream in document order, so a transcribed
    table arrives as a table instead of as a picture of itself, and only the
    ``<img>`` tags themselves are consumed by the crop (chandra's carry no
    ``src`` — an inline one left in kept text would render as a broken image).

    **One crop per div, covering the whole div bbox**, wherever the first image
    sits: a div's bbox is the only geometry chandra gives (in the twelve spike
    dumps only 11 of 113 images inside a figure div carry a ``data-bbox`` of
    their own).  So a div interleaving several images with text yields one crop
    of all of them, and a div that is mostly text yields a crop that re-shows
    that text as pixels beside the blocks carrying it.  Over-including rather
    than clipping is ``figures.py``'s standing trade-off, and the text is
    recovered either way.

    A node's kind is sniffed from its HTML (:meth:`Block.of`) because content
    *inside* a div carries no label of its own — unlike the div, which
    :meth:`Block.from_chandra_div` maps from the label chandra gave it.
    """
    crop = page_image.crop(_denormalize_bbox(div.bbox, page_image))
    figure = Block.from_chandra_div(
        div.label, _figure_html(crop, None, encode_src), source_page=source_page
    )
    nodes = _top_level_nodes(div.inner_html)
    if not nodes:
        return [figure] if figure is not None else []
    # No image at all (never seen in the dumps, not assumed away): the div is
    # still figure-labeled, so its region leads and its text follows.
    crop_at = next((i for i, n in enumerate(nodes) if _IMG_TAG_RE.search(n)), 0)
    blocks: list[Block] = []
    for i, node in enumerate(nodes):
        if i == crop_at and figure is not None:
            blocks.append(figure)
        text = _IMG_TAG_RE.sub("", node).strip()
        if not text:
            continue
        block = (
            Block.of(text, source_page=source_page)
            if text.startswith("<")
            else Block.from_chandra_div("Text", text, source_page=source_page)
        )
        if block is not None:
            blocks.append(block)
    return blocks


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
    ``Block.from_chandra_div``); every other div becomes one ``Block`` in
    document order, except a figure-labeled one, which becomes its crop
    *together with* the blocks its content holds (:func:`_figure_div_blocks`).
    """
    blocks: list[Block] = []
    for div in _iter_divs(raw):
        if div.label in _FIGURE_LABELS:
            blocks.extend(_figure_div_blocks(div, page_image, encode_src, source_page))
            continue
        block = Block.from_chandra_div(
            div.label, div.inner_html, source_page=source_page
        )
        if block is not None:
            blocks.append(block)
    return blocks
