"""HTML/markdown table-string manipulation: the ``<table>`` regexes plus the pure
string passes that parse, normalize, collapse and group table blocks — no geometry,
no PDF, no model.  The leaf the rest of the ``tables`` package builds on."""

from __future__ import annotations

import re

from manuscribe.pipeline.text import _visible_text

_TABLE_RE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
_OPEN_TABLE_RE = re.compile(r"<table\b", re.IGNORECASE)
_CLOSE_TABLE_RE = re.compile(r"</table\s*>", re.IGNORECASE)
_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
_CAPTION_RE = re.compile(r"<caption\b[^>]*>.*?</caption>", re.DOTALL | re.IGNORECASE)
_ROW_RE = re.compile(r"<tr\b.*?</tr>", re.DOTALL | re.IGNORECASE)
_CELL_OPEN_RE = re.compile(r"<t[dh]\b([^>]*)>", re.IGNORECASE)
_COLSPAN_RE = re.compile(r"colspan\s*=\s*[\"']?(\d+)", re.IGNORECASE)
_THEAD_OPEN_RE = re.compile(r"<thead\b[^>]*>", re.IGNORECASE)
_TABLE_OPEN_RE = re.compile(r"\s*<table\b[^>]*>", re.IGNORECASE)
# Sub-table labels ("A. Effect…", "B. ICP-MS analysis") that the crop re-OCR lifts
# out as level-2+ markdown headings; level-1 is the table's overall caption, which
# the pipeline already carries as its own block, so it is left out here.
_SUBHEADING_RE = re.compile(r"^#{2,6}\s+(.*\S)\s*$", re.MULTILINE)

# More than this many byte-identical consecutive rows is a decode repetition loop,
# not real data — a genuine table never repeats one row this many times in a row.
_MAX_IDENTICAL_ROW_RUN = 3


def _close_unclosed_tables(md: str) -> str:
    """Append a closing tag for any ``<table>`` the page OCR left open.

    LightOnOCR transcribes a table row by row and, when the table runs past the
    bottom of the page, stops mid-table without emitting ``</table>``.  Left open,
    the table absorbs everything after it — most visibly the *next* page's opening
    prose, which renders inside the table.  Closing the surplus opens per page (the
    unit at which the model transcribes) keeps following content out.  A balanced
    page is returned unchanged, so this is idempotent."""
    missing = len(_OPEN_TABLE_RE.findall(md)) - len(_CLOSE_TABLE_RE.findall(md))
    if missing <= 0:
        return md
    return md.rstrip() + "</table>" * missing


def _cell_texts(table_html: str) -> list[str]:
    """Visible text of every non-empty cell in a ``<table>`` block."""
    return [t for m in _CELL_RE.findall(table_html) if (t := _visible_text(m).strip())]


def _nonempty_cell_count(table_html: str) -> int:
    return len(_cell_texts(table_html))


def _table_columns(table_html: str) -> int:
    """Widest row's column count (honoring ``colspan``), at least 1."""
    best = 0
    for row in _ROW_RE.findall(table_html):
        cols = sum(
            int(m.group(1)) if (m := _COLSPAN_RE.search(attrs)) else 1
            for attrs in _CELL_OPEN_RE.findall(row)
        )
        best = max(best, cols)
    return best or 1


def _fold_subheading(heading: str, table_html: str) -> str:
    """Prepend ``heading`` as a full-width spanning header row.

    The crop re-OCR lifts a sub-table label out of the table into a heading; the
    full-page pass had it as a spanning ``<th>`` row, so we restore it as one — both
    keeping the label and the cell it contributes (so the substitution guard sees no
    regression)."""
    row = f'<tr><th colspan="{_table_columns(table_html)}">{heading}</th></tr>'
    thead = _THEAD_OPEN_RE.search(table_html)
    if thead:
        return table_html[: thead.end()] + row + table_html[thead.end() :]
    open_tag = _TABLE_OPEN_RE.match(table_html)
    at = open_tag.end() if open_tag else 0
    return f"{table_html[:at]}<thead>{row}</thead>{table_html[at:]}"


def _extract_tables(md: str) -> list[str]:
    """``<table>`` blocks from re-OCR output, inner ``<caption>`` stripped (the
    pipeline colocates captions separately, so an inline one would double), and any
    immediately-preceding sub-table label folded back in as a spanning header row."""
    tables: list[str] = []
    prev_end = 0
    for m in _TABLE_RE.finditer(md):
        table = _CAPTION_RE.sub("", m.group()).strip()
        labels = _SUBHEADING_RE.findall(md[prev_end : m.start()])
        if labels:
            table = _fold_subheading(labels[-1], table)
        tables.append(table)
        prev_end = m.end()
    return tables


def _collapse_repeated_rows(table_html: str) -> str:
    """Collapse a degenerate run of identical consecutive ``<tr>`` rows to one.

    A tight table crop sometimes drives the model into a decode repetition loop:
    it emits one row over and over (the "RAMS Deviations" explosion — dozens of
    identical rows trailing a real table).  A run of more than
    ``_MAX_IDENTICAL_ROW_RUN`` byte-identical adjacent rows is that pathology, never
    real tabular data, so it is reduced to its first occurrence.  Left in, the loop's
    cell count also beats the truncated original and wins the substitution gate."""
    matches = list(_ROW_RE.finditer(table_html))
    drop = [False] * len(matches)
    i = 0
    while i < len(matches):
        key = matches[i].group()
        j = i + 1
        while j < len(matches) and matches[j].group() == key:
            j += 1
        if j - i > _MAX_IDENTICAL_ROW_RUN:
            for k in range(i + 1, j):
                drop[k] = True
        i = j
    if not any(drop):
        return table_html
    out: list[str] = []
    cursor = 0
    for m, dropped in zip(matches, drop, strict=True):
        if dropped:
            gap = table_html[cursor : m.start()]
            if gap.strip():
                out.append(gap)
            cursor = m.end()
    out.append(table_html[cursor:])
    return "".join(out)


def _collapse_repeated_rows_md(md: str) -> str:
    """Collapse decode-loop row runs in every ``<table>`` of a page's markdown.

    The crop re-OCR path collapses its own substitutions, but the *page* pass has
    its own re-OCR (``_ocr_page`` retries a length-truncated page over the full
    context window) that can decode-loop on a dense table and land the explosion
    straight in ``pages_md`` — a path the crop guard never sees.  Running the
    collapse over the assembled page markdown catches the loop whatever its source.
    Idempotent: a table with no degenerate run is returned byte-for-byte."""
    return _TABLE_RE.sub(lambda m: _collapse_repeated_rows(m.group()), md)


def _table_regions(md: str) -> list[tuple[int, int, list[str]]]:
    """Group the page's ``<table>`` blocks into regions of consecutive tables
    (separated by whitespace only).  Returns ``(start, end, [table_html, ...])``
    char spans, so stacked tables the model split (e.g. a Table 2 "A"/"B") are
    re-OCR'd together from one crop."""
    regions: list[tuple[int, int, list[str]]] = []
    start = end = -1
    tables: list[str] = []
    for m in _TABLE_RE.finditer(md):
        if tables and md[end : m.start()].strip() == "":
            end = m.end()
            tables.append(m.group())
        else:
            if tables:
                regions.append((start, end, tables))
            start, end, tables = m.start(), m.end(), [m.group()]
    if tables:
        regions.append((start, end, tables))
    return regions
