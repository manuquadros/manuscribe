"""Table re-OCR: recover content LightOnOCR drops from dense full-page tables.

The full-page OCR pass silently drops small in-table content — e.g. a column-
spanning subheader like ``Relative activity (%)`` — at *every* render resolution,
yet re-OCRing the table on its own as a tight crop recovers it (and produces the
``colspan``/``rowspan`` structure the full-page pass omits).  So for each table
region we locate it on the page (``localize``), re-OCR just that crop, and
substitute the result.

Crops are planned for the whole document first (text-layer localization plus a
coverage gate that skips regions the page already captured in full), then re-OCR'd
in one batched ``ocr_regions`` call so the server processes them concurrently.

Leaf pass: touches the PDF (render + text layer, via ``layers``/``localize``) and
the GPU (via the injected ``ocr_regions`` callback), so the pure ``_assemble_html``
core stays model-free."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — beartype reads annotations
from dataclasses import dataclass

from PIL import Image  # noqa: TC002 — beartype reads annotations at runtime

from manuscribe.pipeline.layers import _DocumentLayers, _normalize
from manuscribe.pipeline.markdown import _render_inline
from manuscribe.pipeline.tables.localize import (
    _adjacent_para_tokens,
    _anchor_texts,
    _glyph_centers,
    _locate_bbox,
    _region_fully_captured,
    _scaled_crop,
)
from manuscribe.pipeline.tables.markup import (
    _cell_texts,
    _close_unclosed_tables,
    _collapse_repeated_rows,
    _extract_tables,
    _nonempty_cell_count,
    _table_regions,
)

_LEGEND_MAX_LEN = 200  # a recovered legend/source note is short, not body prose


def _legend_footnote_html(legend: str) -> str:
    """Render a table legend the crop recovered as a footnote paragraph.

    The legend is raw OCR markup — a ``<sup>a</sup>`` footnote marker, ``*emphasis*``
    (organism names), an inline ``$…$`` script — rendered tag-aware: blanket
    ``html.escape`` would surface a literal ``<sup>a</sup>``, while a raw passthrough
    would let a stray ``<``/``&`` (a footnote ``n<5``) start a bogus tag."""
    return f'<p class="footnote">{_render_inline(legend)}</p>'


def _crop_trailing(crop_md: str) -> str:
    """Text the crop re-OCR emitted *after* the last table — a legend the page OCR
    placed below the table (and may have truncated)."""
    idx = crop_md.rfind("</table>")
    return crop_md[idx + len("</table>") :].strip() if idx != -1 else ""


def _splice_region(md: str, start: int, end: int, new_tables: list[str]) -> str:
    return md[:start] + "\n\n".join(new_tables) + md[end:]


@dataclass(frozen=True)
class _RegionPlan:
    """A table region whose crop re-OCR is worth attempting: its char span in the
    page's markdown, the original table blocks, and the rendered crop image.  Held
    per page (the caller keeps the page partition), so it carries no page index, and
    the re-OCR markdown flows as a separate value rather than mutating the plan."""

    start: int
    end: int
    tables: list[str]
    crop: Image.Image


def _plan_page_tables(
    md: str, layers: _DocumentLayers, page_index: int
) -> tuple[str, list[_RegionPlan]]:
    """Localize each table region on the page and render its re-OCR crop, dropping
    regions the page already captured in full (the coverage gate) or that cannot be
    localized.  Pure CPU work — no OCR.  Returns the page markdown with open tables
    closed (so the apply pass splices into the same string) and the crop plans.

    Closing unclosed tables first means a table that overran the page (and was thus
    truncated mid-row) is a closed region the crop re-OCR can locate and recover in
    full, not an unmatched fragment."""
    md = _close_unclosed_tables(md)
    regions = _table_regions(md)
    if not regions:
        return md, []

    layer = layers.page_layer(page_index)
    centers = _glyph_centers(layer.norm, layer.idx_map, layer.char_boxes)
    page = layers.pdf[page_index]
    page_size = page.get_size()

    plans: list[_RegionPlan] = []
    for start, end, tables in regions:
        cells = [c for tbl in tables for c in _cell_texts(tbl)]
        located = _locate_bbox(
            _anchor_texts(cells),
            layer.norm,
            layer.idx_map,
            layer.char_boxes,
            layer.char_rotations,
            page_size,
        )
        if located is None:
            continue
        bbox, rot = located
        captured = {t for c in cells for t in _normalize(c).split()}
        captured |= _adjacent_para_tokens(md, start, end)
        if _region_fully_captured(bbox, layer.norm, centers, captured):
            continue
        crop = _scaled_crop(page, bbox, page_size)
        # Turn a sideways table upright before re-OCR: the model reads an upright
        # table's column structure correctly, but mis-groups a rotated one (a 4-wide
        # column header collapses to colspan=2).  rotate() is CCW and the glyph angle
        # is the CCW rotation, so rotating by it cancels the page rotation.
        if rot:
            crop = crop.rotate(rot, expand=True)
        plans.append(_RegionPlan(start, end, tables, crop))
    return md, plans


def _apply_page_results(md: str, results: list[tuple[_RegionPlan, str]]) -> str:
    """Splice each planned region's re-OCR markup back into the page markdown.

    Each ``(plan, crop_md)`` pairs a region with its batched re-OCR output.  A region
    is replaced only when its re-OCR yields tables with at least as many non-empty
    cells as the originals, so a crop that came back worse (or empty) leaves the
    full-page transcription untouched.  Splices run from the back so earlier char
    offsets stay valid."""
    source_norm = _normalize(md)  # to tell a recovered legend from one already in md
    for plan, crop_md in sorted(results, key=lambda pc: pc[0].start, reverse=True):
        new_tables = [_collapse_repeated_rows(t) for t in _extract_tables(crop_md)]
        if not new_tables:
            continue
        old_cells = sum(_nonempty_cell_count(t) for t in plan.tables)
        new_cells = sum(_nonempty_cell_count(t) for t in new_tables)
        if new_cells < old_cells:
            continue
        # Fold a legend the crop recovered from below the table into the table block
        # (so it rides as one float and stays under its table) — but only when it is
        # not already in the page markdown, else a normal table would duplicate the
        # following content the crop happens to also see.
        legend = _crop_trailing(crop_md)
        if (
            legend
            and len(legend) <= _LEGEND_MAX_LEN
            and _normalize(legend) not in source_norm
        ):
            new_tables[-1] += "\n" + _legend_footnote_html(legend)
        md = _splice_region(md, plan.start, plan.end, new_tables)
    return md


def _recover_dropped_tables(
    layers: _DocumentLayers,
    pages_md: list[str],
    ocr_regions: Callable[[list[Image.Image]], list[str]],
) -> list[str]:
    """Re-OCR the table regions the page pass may have under-captured and splice the
    richer markup back in; return updated markdown, one entry per input page.

    Every page's crops are planned first (text-layer localization + the coverage
    gate, pure CPU work), then OCR'd in **one batched call** so the server can
    process them concurrently, then spliced back per page.  Pages without a table —
    and regions the gate found already complete — incur no OCR at all.  Order and
    length match ``pages_md`` so the caller's page-to-image alignment is preserved."""
    closed_md: list[str] = []
    per_page_plans: list[list[_RegionPlan]] = []
    for i, md in enumerate(pages_md):
        if "<table" not in md.lower():
            closed_md.append(md)
            per_page_plans.append([])
            continue
        page_md, plans = _plan_page_tables(md, layers, i)
        closed_md.append(page_md)
        per_page_plans.append(plans)

    flat = [plan for plans in per_page_plans for plan in plans]
    if not flat:
        return closed_md

    crop_mds = iter(ocr_regions([plan.crop for plan in flat]))
    for i, plans in enumerate(per_page_plans):
        if plans:
            pairs = [(plan, next(crop_mds)) for plan in plans]
            closed_md[i] = _apply_page_results(closed_md[i], pairs)
    return closed_md
