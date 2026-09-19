"""Table-cell bold recovery from the PDF text layer.

LightOnOCR transcribes table cells as plain ``<td>``/``<th>`` with no emphasis
markup, even when the source is boldfaced (section/category rows, total rows).  The
bold is a genuine font property in the PDF text layer (per-glyph font weight); this
pass localizes each transcribed ``<table>`` against the layer, reads the weight of
the glyphs each cell matched, and re-wraps a predominantly-bold ``<td>`` in
``<strong>``.

Pure CPU + text-layer work — no OCR, no network.  Best-effort enrichment: a cell the
OCR altered (so its text matches no glyph run), a rotated table, or an image-only
page with no text layer all leave the markup unchanged.  Runs after the table re-OCR
/ text-layer-rebuild passes so the bold lands on the table HTML that actually
survives.  Header ``<th>`` cells are left alone (the render's CSS already
differentiates them); only data ``<td>`` cells are bolded.
"""

from __future__ import annotations

import re

from manuscribe.pipeline.layers import _Box, _DocumentLayers, _normalize, _PageLayer
from manuscribe.pipeline.tables.localize import (
    _anchor_texts,
    _glyph_centers,
    _locate_bbox,
)
from manuscribe.pipeline.tables.markup import (
    _TABLE_RE,
    _cell_texts,
    _collapse_repeated_rows,
)
from manuscribe.pipeline.text import _visible_text

# CSS-style font weight: 400 normal, 700 bold.  >=600 captures semibold/bold while
# excluding normal/medium — the 31123167 fixture's bold section rows read 708, plain
# cells 315 (see plans/table-cell-bold-recovery.md).
_BOLD_WEIGHT_MIN = 600
# A cell counts as bold when at least this fraction of its matched glyphs are bold, so
# a lone bold superscript can't bold a whole cell, nor one thin glyph un-bold a
# mostly-bold one.
_CELL_BOLD_FRACTION = 0.6

# Capture the open tag, inner HTML, and close tag separately so a bold cell is
# re-wrapped without disturbing inline markup it carries (``R<sub>merge</sub>``).
_CELL_SUB_RE = re.compile(r"(<t[dh]\b[^>]*>)(.*?)(</t[dh]>)", re.DOTALL | re.IGNORECASE)
_ALREADY_BOLD_RE = re.compile(r"<(?:strong|b)\b", re.IGNORECASE)


def _is_bold_glyph(weight: int | None) -> bool:
    """A glyph is bold when its font weight is at/above the bold threshold (a ``None``
    weight — pdfium reported an error for that glyph — counts as not bold)."""
    return weight is not None and weight >= _BOLD_WEIGHT_MIN


def _bbox_glyph_run(
    bbox: _Box,
    norm: str,
    idx_map: list[int],
    centers: list[tuple[float, float] | None],
    weights: list[int | None],
) -> tuple[str, list[int | None]]:
    """The in-bbox glyph stream in reading order plus its per-char font weight.

    The folded page text restricted to glyphs whose centre lies inside ``bbox`` (runs
    of out-of-bbox glyphs collapse to one space, preserving word boundaries), with the
    weight aligned to each kept char (``None`` at the separator spaces).  Cells are
    matched against *this* stream, not the whole page, so a cell's text can't bind to
    an identical string elsewhere on the page."""
    left, bottom, right, top = bbox
    chars: list[str] = []
    run_weights: list[int | None] = []
    prev_space = True
    for pos, ch in enumerate(norm):
        center = centers[pos]
        if (
            ch != " "
            and center is not None
            and left <= center[0] <= right
            and bottom <= center[1] <= top
        ):
            chars.append(ch)
            run_weights.append(weights[idx_map[pos]])
            prev_space = False
        elif not prev_space:
            chars.append(" ")
            run_weights.append(None)
            prev_space = True
    return "".join(chars), run_weights


def _rewrap_bold_cells(
    table_html: str, bbox_norm: str, bbox_weights: list[int | None]
) -> str:
    """Re-wrap each predominantly-bold ``<td>`` of ``table_html`` in ``<strong>``.

    Cells are matched against ``bbox_norm`` in document order with a monotonic cursor,
    so a cell string repeated within the table (e.g. "Resolution" in two sections)
    binds to its own occurrence rather than always the first.  A cell whose normalized
    text isn't found past the cursor (the OCR altered it) leaves the cursor and the
    cell unchanged — bold recovery never guesses."""
    cursor = 0

    def rewrap(m: re.Match[str]) -> str:
        nonlocal cursor
        open_tag, inner, close_tag = m.group(1), m.group(2), m.group(3)
        cell_norm = _normalize(_visible_text(inner))
        if not cell_norm:
            return m.group(0)
        idx = bbox_norm.find(cell_norm, cursor)
        if idx == -1:
            return m.group(0)
        weights = [w for w in bbox_weights[idx : idx + len(cell_norm)] if w is not None]
        cursor = idx + len(cell_norm)
        if (
            open_tag[:3].lower() == "<td"
            and weights
            and sum(_is_bold_glyph(w) for w in weights) / len(weights)
            >= _CELL_BOLD_FRACTION
            and not _ALREADY_BOLD_RE.search(inner)
        ):
            return f"{open_tag}<strong>{inner}</strong>{close_tag}"
        return m.group(0)

    return _CELL_SUB_RE.sub(rewrap, table_html)


def _bold_one_table(
    table_html: str,
    layer: _PageLayer,
    centers: list[tuple[float, float] | None],
    page_size: tuple[float, float],
) -> str:
    """Bold the cells of one ``<table>`` whose source glyphs are bold, or return it
    unchanged when it can't be localized or is rotated (the first cut skips rotated
    tables — their glyph reading axis differs from an upright cell walk)."""
    # Collapse a decode-loop row explosion first: bolding row 1 but missing its
    # identical copies (no second glyph occurrence to match) would leave them differing,
    # defeating _collapse_repeated_rows_md's later de-duplication.  Idempotent.
    table_html = _collapse_repeated_rows(table_html)
    cells = _cell_texts(table_html)
    if not cells:
        return table_html
    located = _locate_bbox(
        _anchor_texts(cells),
        layer.norm,
        layer.idx_map,
        layer.char_boxes,
        layer.char_rotations,
        page_size,
    )
    if located is None:
        return table_html
    bbox, rot = located
    if rot:
        return table_html
    bbox_norm, bbox_weights = _bbox_glyph_run(
        bbox, layer.norm, layer.idx_map, centers, layer.char_weights
    )
    return _rewrap_bold_cells(table_html, bbox_norm, bbox_weights)


def _apply_table_bold(md: str, layers: _DocumentLayers, page_index: int) -> str:
    """Re-apply ``<strong>`` to bold cells of every ``<table>`` on the page.

    The page text layer is extracted lazily and at most once (via the shared cache):
    no ``<table>`` means no work, and a page whose text layer carries no font-weight
    metadata (image-only / scanned) leaves every table unchanged."""
    if "<table" not in md.lower():
        return md
    layer = layers.page_layer(page_index)
    if not any(w is not None for w in layer.char_weights):
        return md
    centers = _glyph_centers(layer.norm, layer.idx_map, layer.char_boxes)
    page_size = layers.pdf[page_index].get_size()
    return _TABLE_RE.sub(
        lambda m: _bold_one_table(m.group(0), layer, centers, page_size), md
    )


def _recover_table_cell_bold(layers: _DocumentLayers, pages_md: list[str]) -> list[str]:
    """Re-apply ``<strong>`` to bold table cells across all pages — one entry per input
    page.  Runs after the table re-OCR / rebuild passes so it bolds the surviving
    table HTML (see :func:`_apply_table_bold`)."""
    return [_apply_table_bold(md, layers, i) for i, md in enumerate(pages_md)]
