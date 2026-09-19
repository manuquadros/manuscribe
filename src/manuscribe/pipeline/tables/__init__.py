"""Table re-OCR and text-layer repair, split into cohesive submodules:

- :mod:`.markup` — pure ``<table>`` string/regex manipulation (parse, collapse, group).
- :mod:`.localize` — match cells against the PDF text layer to seed + grow a bbox,
  render the re-OCR crop, and the coverage gate.
- :mod:`.rebuild` — deterministic two-column reconstruction from the text layer.
- :mod:`.recover` — the batched crop re-OCR pass that splices richer markup back in.

The dependency runs one way ``rebuild``/``recover`` → ``localize``/``markup`` →
``layers`` (the foundational text-layer cache).  This package re-exports the names
callers and tests already import, so ``from manuscribe.pipeline.tables import _X``
keeps working unchanged; **monkeypatching** a re-OCR internal must target its owning
submodule (e.g. ``manuscribe.pipeline.tables.recover._region_fully_captured``), not
this package, since the caller resolves it in that submodule's namespace.
"""

from manuscribe.pipeline.tables.enrich import (
    _apply_table_bold,
    _bbox_glyph_run,
    _bold_one_table,
    _is_bold_glyph,
    _recover_table_cell_bold,
    _rewrap_bold_cells,
)
from manuscribe.pipeline.tables.localize import (
    _adjacent_para_tokens,
    _glyph_centers,
    _group_lines,
    _in_bbox_tokens,
    _locate_bbox,
    _region_fully_captured,
    _scaled_crop,
    _union,
)
from manuscribe.pipeline.tables.markup import (
    _cell_texts,
    _close_unclosed_tables,
    _collapse_repeated_rows,
    _collapse_repeated_rows_md,
    _extract_tables,
    _nonempty_cell_count,
    _table_regions,
)
from manuscribe.pipeline.tables.rebuild import (
    _cell_format_map,
    _format_cell,
    _glyph_cell_text,
    _group_glyph_rows,
    _leading_caption_rows,
    _reconstruct_table_from_text_layer,
    _repair_page_tables,
    _repair_tables_from_text_layer,
    _rows_to_cells,
    _trim_rows_below_table,
)
from manuscribe.pipeline.tables.recover import (
    _crop_trailing,
    _legend_footnote_html,
    _recover_dropped_tables,
)

__all__ = [
    "_adjacent_para_tokens",
    "_apply_table_bold",
    "_bbox_glyph_run",
    "_bold_one_table",
    "_cell_format_map",
    "_cell_texts",
    "_close_unclosed_tables",
    "_collapse_repeated_rows",
    "_collapse_repeated_rows_md",
    "_crop_trailing",
    "_extract_tables",
    "_format_cell",
    "_glyph_cell_text",
    "_glyph_centers",
    "_group_glyph_rows",
    "_group_lines",
    "_in_bbox_tokens",
    "_is_bold_glyph",
    "_leading_caption_rows",
    "_legend_footnote_html",
    "_locate_bbox",
    "_nonempty_cell_count",
    "_reconstruct_table_from_text_layer",
    "_recover_dropped_tables",
    "_recover_table_cell_bold",
    "_region_fully_captured",
    "_rewrap_bold_cells",
    "_repair_page_tables",
    "_repair_tables_from_text_layer",
    "_rows_to_cells",
    "_scaled_crop",
    "_table_regions",
    "_trim_rows_below_table",
    "_union",
]
