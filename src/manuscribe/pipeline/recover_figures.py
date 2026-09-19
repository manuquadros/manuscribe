"""Figure re-OCR: recover a figure LightOnOCR drops *entirely* from a page.

The full-page pass occasionally emits neither the ``![image]`` bbox placeholder
nor the "Figure N" caption for a figure — the figure vanishes from the document,
and re-OCRing the whole page reproduces the omission (it is systematic, not a
sporadic miss).  Re-OCRing a tight crop of just the figure recovers it, exactly as
:mod:`manuscribe.pipeline.tables` recovers content dropped from dense tables.

Unlike a dropped table — text the PDF text layer can localize directly — a figure
is an *image*, invisible to the text layer; only its caption is text.  So we
(1) enumerate the figure numbers the text layer carries as caption labels,
(2) subtract the numbers the OCR actually emitted (a gap, e.g. 1,2,3,_,5,6,7),
and (3) for each missing number localize its caption via the text layer, crop the
band spanning the figure (the text-free area above the caption) plus the caption,
and re-OCR that tight crop.  The recovered ``![image]`` placeholder's box is
crop-relative, so it is mapped back to page coordinates and spliced into the page
markdown as an ordinary placeholder — the assembler then crops and renders it
through the unchanged figure path.

Detection only *nominates* candidates; the re-OCR confirms one by actually
returning a figure box.  A spurious candidate (a caption-shaped line that is not a
figure, or a caption-below-figure layout this above-the-caption crop can't reach)
re-OCRs to no placeholder and is silently dropped — so the step never injects a
bogus figure.

Leaf module: touches the PDF (text layer + render) and the GPU (via the injected
``ocr_region`` callback), keeping the pure ``_assemble_html`` core model-free.
"""

from __future__ import annotations

import re
from collections.abc import Callable  # noqa: TC003 — beartype reads annotations

import pypdfium2 as pdfium  # noqa: TC002 — beartype reads annotations at runtime
from PIL import Image  # noqa: TC002 — beartype reads annotations at runtime

from manuscribe.pipeline.classify import _leading_pages_to_skip_md
from manuscribe.pipeline.figures import (
    _BBOX_NORM_MAX,
    _is_bare_figure_label,
    _opens_with_panel_label,
    _safe_ocr_region,
)
from manuscribe.pipeline.layers import _Box, _DocumentLayers, _normalize
from manuscribe.pipeline.tables import _group_lines, _scaled_crop, _union
from manuscribe.pipeline.text import _split_md_blocks

# A figure caption *label* at the start of a text line: "FIG 1", "Figure 4.",
# "**Figure 6.**", "FIGURE 1 |" (Frontiers), "Fig 1. Effect…".  After the number
# we require end-of-line, a separator (.:|)), or whitespace then a Capital/paren —
# so an in-prose reference reflowed to a line start ("Fig. 2, it was predicted…",
# "(Figure 4A)") does not match: a comma or a lowercase word after the number, or a
# leading "(", all fail the tail, and "4A" (a panel reference) keeps the digit run
# from being followed by an accepted separator.
_FIG_CAPTION_LABEL_RE = re.compile(
    r"^[ \t]*(?:\*{1,2}[ \t]*)?fig(?:ure|\.)?\.?[ \t]*(\d+)"
    r"(?:[ \t]*\r?$|[ \t]*[.:|)]|[ \t]+[A-Z(])",
    re.IGNORECASE | re.MULTILINE,
)
# The recovered placeholder anywhere in the crop re-OCR (not line-anchored: the
# model sometimes prefixes a stray panel label, e.g. "(A) ![image](…)x0,y0,x1,y1").
_PLACEHOLDER_BBOX_RE = re.compile(
    r"!\[[^\]]*\]\([^)]*\)[ \t]*"
    r"(\d+)[ \t]*,[ \t]*(\d+)[ \t]*,[ \t]*(\d+)[ \t]*,[ \t]*(\d+)"
)

# Vertical band kept below the caption label so the crop captures a multi-line
# caption / panel description (and not just the "Figure N" line); body text pulled
# in past it is discarded when the caption blocks are extracted.
_CAPTION_BAND_FRAC = 0.12

# A normalized caption header must be at least this many chars before its presence
# in the page text is taken as "the OCR already emitted this caption".  A bare
# "Figure N" folds to ~8 chars and recurs in prose (in-text references), so the
# dedup guard requires a header carrying a title, not just the label.
_MIN_CAPTION_MATCH_LEN = 12


def _emitted_figure_numbers(pages_md: list[str]) -> set[int]:
    """Figure numbers the OCR emitted as caption labels across all pages."""
    return {
        int(m.group(1)) for md in pages_md for m in _FIG_CAPTION_LABEL_RE.finditer(md)
    }


def _textlayer_caption_pages(pdf: pdfium.PdfDocument) -> dict[int, list[int]]:
    """Map each figure number whose caption label appears in the text layer to the
    page indices carrying it, so a missing number can be localized to its page."""
    pages: dict[int, list[int]] = {}
    for i in range(len(pdf)):
        # Cheap text-view scan for caption labels; the box-aligned char-array layer
        # (_PageLayer) is overkill here, so don't force its full extraction.  Close the
        # native handle (pdfium's PdfTextPage has no context-manager protocol).
        textpage = pdf[i].get_textpage()
        try:
            text = textpage.get_text_range()
        finally:
            textpage.close()
        for m in _FIG_CAPTION_LABEL_RE.finditer(text):
            pages.setdefault(int(m.group(1)), []).append(i)
    return {num: sorted(set(ps)) for num, ps in pages.items()}


def _column_bounds(
    lines: list[_Box], cap_box: _Box, cap_top: float, page_w: float
) -> tuple[float, float]:
    """Horizontal crop bounds (left, right) for the caption's text column.

    Full page width for a single-column page; clamped to the caption's half for a
    two-column one, so the figure crop doesn't reach across the gutter and pull in
    the neighbouring column's figure.  Two-column is inferred from the body lines
    *below* the caption: a single-column body has lines spanning the page centre,
    a two-column body never does.  Absent that evidence the full width is kept,
    erring toward over-inclusion (a slightly wide crop is cosmetic; a clipped one
    loses figure the reader can't recover)."""
    mid = page_w / 2
    body = [ln for ln in lines if ln[3] <= cap_top]  # lines wholly below the caption
    if not body or any(ln[0] < mid < ln[2] for ln in body):
        return 0.0, page_w
    cap_cx = (cap_box[0] + cap_box[2]) / 2
    return (0.0, mid) if cap_cx < mid else (mid, page_w)


def _figure_crop_box(
    text: str,
    boxes: list[_Box | None],
    rotations: list[int | None],
    num: int,
    page_size: tuple[float, float],
) -> tuple[_Box, float] | None:
    """Locate the crop spanning figure ``num`` and its caption from a page's
    text/box arrays (a ``_PageLayer``'s ``page_text``/``char_boxes``/``char_rotations``,
    shared via the document-level cache across that page's missing figures and the
    table passes).

    Returns the crop box (PDF points) and the caption label's top ``y`` (used to
    order intra-page placement), or ``None`` when the caption label is not found.
    The figure is assumed to sit above its caption (the scientific-journal norm):
    the box runs from the bottom of the nearest text line above the caption (the
    running head, or the page top when none) down through a band below the caption
    label wide enough to hold a multi-line caption, and is clamped horizontally to
    the caption's column (:func:`_column_bounds`).  A caption-above-figure or other
    layout leaves no figure in the band, so the re-OCR returns no placeholder and
    recovery declines it — there is no wrong-crop to splice.

    Lines are built from *upright* glyphs only: a sideways margin watermark
    ("Downloaded from …", rotated 90°) running up the gutter would otherwise drop a
    stray line into the figure↔caption gap and collapse the box onto the caption."""
    label = next(
        (m for m in _FIG_CAPTION_LABEL_RE.finditer(text) if int(m.group(1)) == num),
        None,
    )
    if label is None:
        return None
    label_boxes = [b for b in boxes[label.start() : label.end()] if b is not None]
    if not label_boxes:
        return None
    cap_box = _union(label_boxes)
    cap_top = cap_box[3]

    upright: list[_Box | None] = [
        b if r in (0, None) else None for b, r in zip(boxes, rotations, strict=True)
    ]
    lines = _group_lines(upright)
    # y-up coords: a line sits above the caption when its bottom (_Box[1]) is higher
    # up the page than the caption's top, so its bottom exceeds cap_top.  Require a
    # full label-line-height of clearance: a faux-bold caption renders the label's
    # own first line *twice* with a sub-point vertical offset, and the ghost copy
    # (bottom a fraction above cap_top) would otherwise count as "text above" and
    # collapse the region onto the caption — clipping the figure out of the crop.
    label_h = cap_box[3] - cap_box[1]
    above = [ln[1] for ln in lines if ln[1] > cap_top + label_h]
    page_w, page_h = page_size
    region_top = min(above) if above else page_h
    region_bottom = max(0.0, cap_top - _CAPTION_BAND_FRAC * page_h)
    if region_top - region_bottom < 1.0:
        return None
    left, right = _column_bounds(lines, cap_box, cap_top, page_w)
    return (left, region_bottom, right, region_top), cap_top


def _remap_bbox_to_page(
    bbox: tuple[int, int, int, int], region: _Box, page_size: tuple[float, float]
) -> tuple[int, int, int, int]:
    """Map a crop-relative ``[0, 1000]`` box to a page-relative ``[0, 1000]`` box.

    ``bbox`` is normalized to the re-OCR crop (origin top-left, y down); ``region``
    is the crop's PDF-points rectangle on the page (origin bottom-left, y up).  The
    result is the page-normalized box the assembler's figure path expects."""
    page_w, page_h = page_size
    left, bottom, right, top = region
    fx0, fy0, fx1, fy1 = (v / _BBOX_NORM_MAX for v in bbox)
    crop_h = top - bottom
    px0 = left + fx0 * (right - left)
    px1 = left + fx1 * (right - left)
    # Crop's top edge is page-y ``top`` (high y); image y grows downward from it.
    py0_from_top = (page_h - top) + fy0 * crop_h
    py1_from_top = (page_h - top) + fy1 * crop_h
    return (
        round(px0 / page_w * _BBOX_NORM_MAX),
        round(py0_from_top / page_h * _BBOX_NORM_MAX),
        round(px1 / page_w * _BBOX_NORM_MAX),
        round(py1_from_top / page_h * _BBOX_NORM_MAX),
    )


def _starts_new_section(block: str) -> bool:
    """True when ``block`` begins a new structural element — a heading, a table, or
    another figure's caption — rather than continuing the current caption.  Used to
    stop a bare "FIG. N" label from claiming a following heading/table/figure block
    (body the generous crop also caught) as its title."""
    s = block.lstrip()
    return (
        s.startswith(("#", "|", "<table")) or _FIG_CAPTION_LABEL_RE.match(s) is not None
    )


def _extract_recovered_figure(
    crop_md: str, num: int
) -> tuple[tuple[int, int, int, int], str] | None:
    """Pull the recovered figure's crop-relative box and caption from the crop
    re-OCR, or ``None`` when no figure placeholder came back.

    The crop re-OCR emits the caption either *after* the ``![image]`` placeholder or
    *before* it (the model orders the caption and the image inconsistently for an
    above-the-image caption), so the placeholder line is dropped and *this* figure's
    caption is located across the whole crop by its **numbered** label — a tight crop
    can also catch a neighbouring figure's caption tail above the image, which a
    number-agnostic test would mistake for this figure's caption.  The caption is
    that label block, then — when the label is bare (the crop re-OCR often splits the
    label, its title and the panels into separate blocks) — the title block that
    follows, then any parenthesised panel descriptions; the scan stops at the first
    block that starts a new section (heading/table/another figure) or is neither
    title nor panel, so the body prose the generous crop also captured is not
    re-spliced.  No numbered label found (a caption-below-figure layout, or a garbled
    re-OCR) yields an empty caption — the figure renders without a ``<figcaption>``
    rather than not at all."""
    m = _PLACEHOLDER_BBOX_RE.search(crop_md)
    if m is None:
        return None
    bbox = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))

    blocks = [
        b.strip()
        for b in _split_md_blocks(_PLACEHOLDER_BBOX_RE.sub("", crop_md))
        if b.strip()
    ]
    start = next(
        (
            i
            for i, block in enumerate(blocks)
            if (label := _FIG_CAPTION_LABEL_RE.match(block)) is not None
            and int(label.group(1)) == num
        ),
        None,
    )
    if start is None:
        return bbox, ""
    # A bare label carries no title on its own block; the next block is its title.
    expect_title = _is_bare_figure_label(blocks[start])
    caption_blocks = [blocks[start]]
    for block in blocks[start + 1 :]:
        if expect_title and not _starts_new_section(block):
            caption_blocks.append(block)
            expect_title = False
            continue
        if not _opens_with_panel_label(block):
            break
        caption_blocks.append(block)
    return bbox, "\n\n".join(caption_blocks)


def _figure_block(bbox: tuple[int, int, int, int], caption: str) -> str:
    placeholder = f"![image](image_1.png){bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    return f"{placeholder}\n\n{caption}" if caption else placeholder


def _caption_already_present(caption: str, page_md: str) -> bool:
    """True when the recovered caption's header already appears in ``page_md``.

    A figure number is judged missing when no *line-anchored* caption label was
    emitted, but the OCR may have transcribed the caption in a shape the label
    regex misses — an em-dash separator ("Figure 4 — …"), or the caption glued
    onto the end of a prose line — leaving the caption in the body even though the
    image placeholder was dropped.  Re-emitting the recovered caption would then
    show it twice, so the figure is spliced image-only in that case.  Matched on
    the NFKD-folded text, so the separator/casing differences that defeated the
    label regex collapse away; a bare "Figure N" with no title folds too short to
    match (``_MIN_CAPTION_MATCH_LEN``) and is left to splice normally.

    The caption arrives as the label and its title in *separate* blocks (joined by
    a blank line), so the match runs on the whole caption flattened to one line —
    keying on the first block alone would be the bare "FIG. N" label, which folds
    too short and never matches even when the full caption is already on the page."""
    header = caption.replace("\n", " ").strip()
    norm = _normalize(header)
    return len(norm) >= _MIN_CAPTION_MATCH_LEN and norm in _normalize(page_md)


def _splice_figures_into_page(
    page_md: str, figures: list[tuple[float, str]], page_h: float
) -> str:
    """Splice recovered figure blocks into the page in top-to-bottom reading order.

    Each ``(cap_top, block)`` is placed by its caption's vertical position: figures
    captioned in the page's top half lead the page, lower ones trail it, and within
    each group the higher caption comes first (``cap_top`` descending, y-up).
    Ordering the whole set at once — rather than prepending/appending one at a time
    onto an already-modified string — keeps two figures dropped from the same page
    in their on-page order."""
    top = sorted(
        (f for f in figures if (page_h - f[0]) / page_h < 0.5),
        key=lambda f: f[0],
        reverse=True,
    )
    bottom = sorted(
        (f for f in figures if (page_h - f[0]) / page_h >= 0.5),
        key=lambda f: f[0],
        reverse=True,
    )
    return "\n\n".join([b for _, b in top] + [page_md] + [b for _, b in bottom])


def _attempt_page_figure(
    layers: _DocumentLayers,
    p: int,
    num: int,
    ocr_region: Callable[[Image.Image], str],
    page_md: str,
) -> tuple[float, str] | None:
    """Localize, crop and re-OCR figure ``num`` on page ``p``; return its
    ``(caption_top, figure_block)`` or ``None`` if this page yields no figure
    (a running caption reference localizes none above it, or the crop re-OCR fails
    on a transient GPU OOM — recovery is best-effort, so a failed crop declines that
    figure rather than aborting the document)."""
    page = layers.pdf[p]
    page_size = page.get_size()
    layer = layers.page_layer(p)
    located = _figure_crop_box(
        layer.page_text, layer.char_boxes, layer.char_rotations, num, page_size
    )
    if located is None:
        return None
    region, cap_top = located
    crop_md = _safe_ocr_region(ocr_region, _scaled_crop(page, region, page_size))
    if crop_md is None:
        return None
    recovered = _extract_recovered_figure(crop_md, num)
    if recovered is None:
        return None
    bbox, caption = recovered
    if _caption_already_present(caption, page_md):
        caption = ""
    block = _figure_block(_remap_bbox_to_page(bbox, region, page_size), caption)
    return cap_top, block


def _recover_dropped_figures(
    layers: _DocumentLayers,
    pages_md: list[str],
    ocr_region: Callable[[Image.Image], str],
) -> list[str]:
    """Recover figures the page pass dropped whole and splice them into the page
    markdown; return updated markdown, one entry per input page.

    A figure number present in the text layer's caption labels but absent from the
    OCR's emitted captions is localized on its page, its figure+caption band is
    cropped and re-OCR'd, and the recovered placeholder (remapped to page
    coordinates) plus caption is spliced in.  Documents with no missing figure
    incur no OCR.  Order and length match ``pages_md``.

    A figure number whose caption recurs on several pages is attempted page by page
    until one yields a figure (the others — running references — localize no figure
    above the caption and are skipped).  Leading pages ``_assemble_html`` will drop
    are not attempted, so no re-OCR is spent on a figure that would be discarded."""
    emitted = _emitted_figure_numbers(pages_md)
    truth = _textlayer_caption_pages(layers.pdf)
    missing = sorted(num for num in truth if num not in emitted)
    if not missing:
        return pages_md
    skip = _leading_pages_to_skip_md(pages_md)

    pages_md = list(pages_md)
    recovered_by_page: dict[int, list[tuple[float, str]]] = {}
    for num in missing:
        for p in truth[num]:
            if p < skip:
                continue
            found = _attempt_page_figure(layers, p, num, ocr_region, pages_md[p])
            if found is None:
                continue
            recovered_by_page.setdefault(p, []).append(found)
            break

    for p, figs in recovered_by_page.items():
        pages_md[p] = _splice_figures_into_page(
            pages_md[p], figs, layers.pdf[p].get_size()[1]
        )
    return pages_md
