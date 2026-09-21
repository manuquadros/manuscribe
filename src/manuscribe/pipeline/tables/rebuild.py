"""Text-layer table rebuild.

A dense two-column statistics table (crystallography refinement stats) is mangled by
the OCR in a way it can't self-correct: it drops the empty top-left header cell,
shifting the first rows off by one and *losing* a value, and truncates the tail —
and the failure is deterministic (same every run).  The PDF text layer holds the
table's full content at fixed positions, so for such a table we rebuild the
*structure* from the text layer (rows by y, label|value by the column gap) and keep
the OCR's good cell *formatting* by matching each cell's text — recovering the dropped
value/header with correct sub/superscripts, deterministically.  Crosses the usual
"text layer = geometry only" rule, scoped to a two-column table the OCR got wrong."""

from __future__ import annotations

import html
import re

from manuscribe.pipeline.layers import _Box, _DocumentLayers, _normalize, _PageLayer
from manuscribe.pipeline.tables.localize import (
    _GAP_FACTOR,
    _anchor_texts,
    _collect_seeds,
    _grow_run_down,
    _median,
    _union,
)
from manuscribe.pipeline.tables.markup import (
    _CELL_RE,
    _COLSPAN_RE,
    _ROW_RE,
    _TABLE_RE,
    _cell_texts,
    _collapse_repeated_rows,
    _table_columns,
)
from manuscribe.pipeline.text import _visible_text

# Breathing room (pt) added around the seed column — both its x-bounds and its top
# y-bound — so a glyph just outside the anchors' tight box is still captured.
_RECON_PAD = 6.0
_RECON_ROW_TOL = 6.0  # merge super/subscript-shifted glyphs into their row
_RECON_MIN_SEEDS = 4  # unique anchors needed to localize the column confidently
_RECON_SPACE_FRAC = 0.3  # insert a space when a glyph gap exceeds this * glyph width
_RECON_LABEL_GAP = 10.0  # min label|value separation (pt)
_RECON_VALUE_OFFSET = 25.0  # a row starting this far right of the labels has no label
_RECON_FOOTNOTE_LEN = 34  # a long value-less row past the body is a footnote: stop
# A long value-less row only counts as a footnote-stop once this many data rows have
# been seen; before that it is the table's title above the body, which is skipped.
_RECON_MIN_DATA_ROWS_BEFORE_FOOTNOTE = 3
# A value-only cell this short and non-alphanumeric is a stray rule glyph (a lone "_"
# the header underline leaves behind), dropped rather than emitted as a row.
_RECON_RULE_GLYPH_MAXLEN = 1
# The space heuristic over-splits a digit run (a narrow "1" reads as a gap); rejoin a
# space that sits between two digits/decimal points.
_RECON_NUM_SPACE = re.compile(r"(?<=[\d.])\s+(?=[\d.])")
# char, x0, x1, y-center, y-bottom, y-top — the y-extents (not just the centre) let
# the below-table trim measure edge-to-edge whitespace, matching _locate_bbox's gap.
_Glyph = tuple[str, float, float, float, float, float]


def _glyph_cell_text(seg: list[_Glyph]) -> str:
    """Join a glyph run into text — the text layer drops space glyphs, so insert a space
    at each inter-glyph gap wider than a fraction of the glyph width, then rejoin a
    digit run the heuristic over-split."""
    if not seg:
        return ""
    cw = _median([g[2] - g[1] for g in seg]) or 1.0
    out = seg[0][0]
    for prev, cur in zip(seg, seg[1:], strict=False):
        if cur[1] - prev[2] > _RECON_SPACE_FRAC * cw:
            out += " "
        out += cur[0]
    return _RECON_NUM_SPACE.sub("", out.strip())


def _group_glyph_rows(glyphs: list[_Glyph]) -> list[list[_Glyph]]:
    """Group glyphs into rows top-to-bottom, merging a baseline-shifted super/subscript
    glyph into its row (within ``_RECON_ROW_TOL``); each row is sorted left to right."""
    rows: list[list[_Glyph]] = []
    anchors: list[float] = []
    for g in sorted(glyphs, key=lambda g: (-g[3], g[1])):
        if anchors and abs(anchors[-1] - g[3]) <= _RECON_ROW_TOL:
            rows[-1].append(g)
        else:
            rows.append([g])
            anchors.append(g[3])
    return [sorted(row, key=lambda g: g[1]) for row in rows]


def _trim_rows_below_table(
    rows: list[list[_Glyph]], seeds: list[_Box]
) -> list[list[_Glyph]]:
    """Drop the body-prose rows the unbounded column sweep collected below the table.

    The glyph filter has only a top bound (the seeds' top), so it runs to page bottom;
    a 2-column page's body prose below the table would then be swept in as extra rows,
    inflating the rebuild past the OCR table and winning the row-count substitution
    gate with a wrong table.  Mirror ``_locate_bbox``: take the rows the seed box
    overlaps (cross-axis interval test, not a centre) as the table body, measure
    spacing as the *edge-to-edge whitespace* of that seeded run — so ``_GAP_FACTOR``,
    tuned for whitespace rather than centre-to-centre pitch, applies unchanged — then
    grow down through the trailing data rows (the shared ``_grow_run_down``) while the
    gap stays within the prose margin, cutting at the wider gap to body prose."""
    if len(rows) < 2 or not seeds:
        return rows
    tops = [max(g[5] for g in row) for row in rows]
    bottoms = [min(g[4] for g in row) for row in rows]
    gaps = [bottoms[i] - tops[i + 1] for i in range(len(rows) - 1)]
    _, seed_bottom, _, seed_top = _union(seeds)
    touched = [
        i for i in range(len(rows)) if bottoms[i] < seed_top and tops[i] > seed_bottom
    ]
    if not touched:
        return rows
    lo, hi = min(touched), max(touched)
    region = gaps[lo:hi]
    spacing = _median(region) if region else _median(gaps)
    hi = _grow_run_down(gaps, hi, spacing * _GAP_FACTOR)
    return rows[: hi + 1]


def _label_left(rows: list[list[_Glyph]]) -> float:
    """The x where the label column starts: the leftmost start-x among rows that
    actually *split* (carry a clear label|value gap), so a value-only row doesn't drag
    the column boundary left.  Falls back to the first row's start if none split."""
    return min(
        (
            row[0][1]
            for row in rows
            if any(
                row[i + 1][1] - row[i][2] >= _RECON_LABEL_GAP
                for i in range(len(row) - 1)
            )
        ),
        default=rows[0][0][1],
    )


def _classify_row(row: list[_Glyph], label_left: float) -> tuple[str, str]:
    """Split one row into ``(label, value)``: a row starting well right of the label
    column is value-only (empty label — the data-column header or a centered divider); a
    row with a clear gap splits at the widest gap; a gapless row is a single label."""
    gaps = [(row[i + 1][1] - row[i][2], i) for i in range(len(row) - 1)]
    maxgap, gi = max(gaps, default=(0.0, -1))
    if row[0][1] > label_left + _RECON_VALUE_OFFSET:
        return "", _glyph_cell_text(row)
    if maxgap >= _RECON_LABEL_GAP:
        return _glyph_cell_text(row[: gi + 1]), _glyph_cell_text(row[gi + 1 :])
    return _glyph_cell_text(row), ""


def _rows_to_cells(rows: list[list[_Glyph]]) -> list[tuple[str, str]]:
    """Turn text-layer rows into ``(label, value)`` cells.  A row that starts in the
    value column has an empty label (the data-column header, or a centered divider); a
    row with a clear gap splits there; a gapless row is a single cell.  Stops at the
    first footnote (a long value-less row below the table body)."""
    if not rows:
        return []
    label_left = _label_left(rows)
    cells: list[tuple[str, str]] = []
    seen_data = 0
    for row in rows:
        label, value = _classify_row(row, label_left)
        # A long value-less row is the table's title (above the body) or a footnote
        # (below it): skip it above the body, stop at it once data rows have started.
        if not value and len(label) > _RECON_FOOTNOTE_LEN:
            if seen_data >= _RECON_MIN_DATA_ROWS_BEFORE_FOOTNOTE:
                break
            continue
        # skip a stray rule glyph (a lone "_" the header underline leaves behind)
        if not label and len(value) <= _RECON_RULE_GLYPH_MAXLEN and not value.isalnum():
            continue
        if label or value:
            cells.append((label, value))
            seen_data += bool(value)
    return cells


def _cell_format_map(table_html: str) -> dict[str, str]:
    """Map each OCR cell's space-stripped normalized text to its formatted inner HTML,
    so a reconstructed cell can reuse the OCR's sub/superscript markup."""
    fmt: dict[str, str] = {}
    for inner in _CELL_RE.findall(table_html):
        key = _normalize(_visible_text(inner)).replace(" ", "")
        if key and key not in fmt:
            fmt[key] = inner.strip()
    return fmt


def _format_cell(plain: str, fmt: dict[str, str]) -> str:
    # The format-map branch is already escaped HTML from the OCR cell; only the raw
    # text-layer fallback needs escaping, or a value carrying '<'/'>'/'&' (a "<0.001"
    # statistic, an "&") injects markup / breaks the <td>.
    return fmt.get(_normalize(plain).replace(" ", "")) or html.escape(plain)


def _leading_caption_rows(table_html: str) -> tuple[list[str], str]:
    """The OCR table's leading spanning rows (the ``<th colspan>`` title/caption) and
    their combined normalized text — kept verbatim (they carry the table title) and used
    to drop the title fragments the text-layer pass picks up from the wide centered
    caption."""
    rows: list[str] = []
    for row in _ROW_RE.findall(table_html):
        if len(_CELL_RE.findall(row)) == 1 and _COLSPAN_RE.search(row):
            rows.append(row.strip())
        else:
            break
    return rows, _normalize(_visible_text("".join(rows)))


def _build_reconstructed_table(
    cells: list[tuple[str, str]], fmt: dict[str, str], caption_rows: list[str]
) -> str:
    rows: list[str] = list(caption_rows)
    for label, value in cells:
        if label and not value:
            rows.append(f'<tr><td colspan="2">{_format_cell(label, fmt)}</td></tr>')
        else:
            rows.append(
                f"<tr><td>{_format_cell(label, fmt)}</td>"
                f"<td>{_format_cell(value, fmt)}</td></tr>"
            )
    return "<table>\n" + "\n".join(rows) + "\n</table>"


def _reconstruct_table_from_text_layer(
    layer: _PageLayer, table_html: str
) -> str | None:
    """Rebuild a two-column table from the page text layer, or ``None`` if it can't be
    localized.  Localizes the table *column* from the OCR cells used as anchors, so the
    body prose in the other column of a two-column page is excluded."""
    seeds, _ = _collect_seeds(
        _anchor_texts(_cell_texts(table_html)),
        layer.norm,
        layer.idx_map,
        layer.char_boxes,
        layer.char_rotations,
    )
    if len(seeds) < _RECON_MIN_SEEDS:
        return None
    col_left = min(s[0] for s in seeds) - _RECON_PAD
    col_right = max(s[2] for s in seeds) + _RECON_PAD
    y_top = max(s[3] for s in seeds) + _RECON_PAD
    glyphs = [
        (ch, b[0], b[2], (b[1] + b[3]) / 2, b[1], b[3])
        for ch, b in zip(layer.page_text, layer.char_boxes, strict=True)
        if b is not None
        and not ch.isspace()
        and col_left <= (b[0] + b[2]) / 2 <= col_right
        and (b[1] + b[3]) / 2 <= y_top
    ]
    rows = _trim_rows_below_table(_group_glyph_rows(glyphs), seeds)
    caption_rows, caption_text = _leading_caption_rows(table_html)
    # Drop the wrapped title fragments the wide centered caption leaves in the column,
    # but only *above* the table body — a value-less row mid-table is a real divider
    # whose label can legitimately be a substring of the caption (e.g. "Refinement").
    cells: list[tuple[str, str]] = []
    seen_data = False
    for label, value in _rows_to_cells(rows):
        seen_data = seen_data or bool(value)
        if (
            not seen_data
            and not value
            and _normalize(label)
            and (_normalize(label) in caption_text)
        ):
            continue
        cells.append((label, value))
    if not cells:
        return None
    return _build_reconstructed_table(cells, _cell_format_map(table_html), caption_rows)


def _table_tokens(table_html: str) -> set[str]:
    """Every word the table's cells hold, folded through ``_normalize`` so a
    superscript/dash encoding gap or a line wrap between the OCR and the text layer
    doesn't read as recovered content."""
    return {w for cell in _cell_texts(table_html) for w in _normalize(cell).split()}


def _repair_tables_from_text_layer(
    layers: _DocumentLayers, pages_md: list[str]
) -> list[str]:
    """Replace a two-column table the OCR mangled with a deterministic text-layer
    reconstruction, when that has more rows *and* recovers text the OCR table lacks
    (it was truncated/off-by-one); a healthy table keeps its better-formatted OCR
    markup.  One entry per input page."""
    out: list[str] = []
    for i, md in enumerate(pages_md):
        if "<table" not in md.lower():
            out.append(md)
            continue
        out.append(_repair_page_tables(md, layers, i))
    return out


def _repair_page_tables(md: str, layers: _DocumentLayers, page_index: int) -> str:
    # Extract the page text layer lazily and at most once: _TABLE_RE.sub calls repair
    # for every complete <table>, and the extraction is thousands of native calls — so
    # share it (via the document-level cache) across a page's 2-column tables, but skip
    # it entirely on a page whose tables are all non-2-column (the common case) or whose
    # <table> is unclosed (no _TABLE_RE match), where no reconstruction is ever
    # attempted.
    def repair(m: re.Match[str]) -> str:
        table = m.group()
        if _table_columns(table) != 2:  # the repair only understands label|value tables
            return table
        layer = layers.page_layer(page_index)
        recon = _reconstruct_table_from_text_layer(layer, table)
        if recon is None:
            return table  # couldn't localize
        # Compare against the *collapsed* OCR table: a decode-loop explosion inflates
        # its raw <tr> count (collapsed only later in _assemble_html), which would
        # otherwise mask a genuinely shorter (off-by-one/truncated) table.
        collapsed = _collapse_repeated_rows(table)
        if recon.count("<tr") <= collapsed.count("<tr"):
            return table  # the OCR table is no shorter
        # Extra rows alone don't mean the OCR lost content: every text-layer line is a
        # row, so a row label wrapping onto two lines outnumbers a *correct* OCR table.
        # Substitute only when the rebuild also carries words that table doesn't.
        if not _table_tokens(recon) - _table_tokens(collapsed):
            return table
        return recon

    return _TABLE_RE.sub(repair, md)
