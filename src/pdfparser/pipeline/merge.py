"""Block-stream stitching: re-join paragraphs split by two-column PDF layout and
fold free-standing table captions into their ``<table>``.

Pure, no GPU.  Operates on the flat list of block-HTML strings produced per page.
"""

from __future__ import annotations

import re

from pdfparser.pipeline.block import Block, BlockKind
from pdfparser.pipeline.classify import (
    _LEADING_SUP_RE,
    _REF_HEADING_RE,
    _UNICODE_SUP_MARKER_RE,
    _is_stray_metadata,
)
from pdfparser.pipeline.dehyphenate import _dehyphenate_join
from pdfparser.pipeline.text import (
    _BOLD_LABEL_RE,
    _TABLE_CAPTION_RE,
    _ends_sentence,
    _heading_inner,
    _opens_with_caption_label,
    _plain_p_text,
    _visible_text,
)

_FLOAT_RE = re.compile(r"^<(?:table|figure)[\s>]", re.IGNORECASE)
_ENUM_RE = re.compile(
    r"^\s*(?:\d+[.)]\s|\[\d|[•\-]\s|\([a-z0-9ivx]+\)\s)", re.IGNORECASE
)
# A fragment ending with a function word is *definitively* grammatically
# incomplete: its continuation must be a predicate, object, or complement,
# which in normal prose starts lowercase.  If the next region starts with an
# uppercase letter in this context the real continuation was likely dropped by
# OCR, so we refuse the merge rather than joining unrelated sentences.
# A trailing comma arms the guard only after a clause-introducer
# (that/which/…) — "revealed that, with no additives present, …" — where the
# comma opens a fronted parenthetical and the real continuation is still
# lowercase.  After a preposition/conjunction a trailing comma before a capital
# is more often a genuine (proper-noun) continuation, so it must not arm the
# guard.
_FUNCTION_WORD_END_RE = re.compile(
    r"\b(?:a|an|the|is|are|was|were|be|been|being|have|has|had|"
    r"will|would|can|could|should|may|might|must|do|does|did|"
    r"of|in|on|at|by|for|with|to|from|and|or|but|nor)\s*$"
    r"|\b(?:that|which|who|whom|this|these|those)[\s,;]*$",
    re.IGNORECASE,
)
# A scientific identifier opening the continuation — an all-caps acronym ("TRII",
# "DNA", "NAD") or a mixed-case gene/protein name ("SpRDH", "PtTRI") — is part of
# the same sentence, not a new-sentence capital, so it must not trip the
# function-word guard; otherwise a clause split across a column/page break
# ("…TRI and" / "TRII compete…", or "…carbon metabolism. The" / "SpRDH operon…")
# is wrongly left as two paragraphs.  The tell is an uppercase letter *inside* the
# first token (an ordinary sentence-opening word is capitalized only on its first
# letter), which also subsumes the all-caps acronym case.
_MIDSENTENCE_HEAD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*[A-Z]")
# A numbered bibliography entry whose marker the OCR left *without* the trailing
# period markdown needs to list-ify it ("9 Peck, S.C., …" not "9. Peck"): a small
# leading integer (optionally with a list delimiter or brackets) followed by a
# capitalised author surname.  Such an entry is a plain <p> that — like every
# entry, trailing off in a DOI rather than terminal punctuation — would otherwise
# be merged onto the previous one; in the references section it marks a new entry
# just as a capital-led continuation does.  The required capital after the number
# keeps a genuinely wrapped entry that resumes with a figure ("…, 2016, 12, …")
# from matching.
_NUMBERED_REF_HEAD_RE = re.compile(r"^\[?\d{1,3}[.)\]]?\s+[A-Z]")
# A continuation opening with one of these is a page footnote, not prose (used by the
# continuation guard); also the table-footnote marker set (``_is_footnote_label``).
_FOOTNOTE_SYMBOLS = frozenset("*†‡§¶#")
# A column/page break can strand a paragraph's continuation behind a whole float
# cluster — observed as two figures plus a table between the two halves — so the
# skip window must clear such a cluster.  The grammatical guards below (the
# fragment lacks terminal punctuation, the continuation isn't a new sentence /
# caption / enumeration) are what keep the merge honest; this bound only limits
# how far floats are relocated after the joined paragraph.
_MAX_FLOATS_TO_SKIP = 3

_TABLE_OPEN_TAG_RE = re.compile(r"^<table\b[^>]*>", re.IGNORECASE)
_FIRST_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
# A row that is a single cell spanning ``colspan`` (>= 2) columns — the shape a
# table title takes when the model bakes it into the table instead of leaving it a
# free-standing caption.
_SPANNING_TITLE_CELL_RE = re.compile(
    r"^<(th|td)\b[^>]*\bcolspan\s*=\s*[\"']?([2-9]\d*)[^>]*>(.*?)</\1>$",
    re.IGNORECASE | re.DOTALL,
)
_EMPTY_THEAD_RE = re.compile(r"<thead\b[^>]*>\s*</thead>", re.IGNORECASE)
# A *bare* table label: the "Table <id>" caption header with no descriptive text
# after it ("TABLE I", "Table 1.", "Supplementary Table 2:").  Unlike
# _TABLE_CAPTION_RE this rejects a label carrying its own title ("Table 4 X"),
# because only a header with the title stranded in the *next* paragraph needs
# rejoining.
_BARE_TABLE_LABEL_RE = re.compile(
    r"^\*{0,2}\s*"
    r"(?i:supp(?:l(?:ementary)?)?\.?\s+)?"
    r"(?i:table)\b\s*"
    r"\w+"
    r"\s*[.:]?\s*\*{0,2}\s*$"
)
# A table label whose identifier is *number-like* — a digit, a roman numeral, or a
# supplement letter+digit ("TABLE 2", "Table IV", "Table S1") — rather than a word.
# Used to gate promoting a *heading* to a table caption: "Table of Contents" matches
# the looser _TABLE_CAPTION_RE but is a real section heading, not a caption.
_TABLE_LABEL_HEADING_RE = re.compile(
    r"^\*{0,2}\s*(?i:supp(?:l(?:ementary)?)?\.?\s+)?(?i:table)\b\s*[0-9A-Z(]"
)


def _bare_table_label_inner(part: str) -> str | None:
    """The inner HTML of a block that is a *bare* table label ("TABLE IV"), whether
    the OCR left it a plain ``<p>`` or promoted it to a heading
    (``<h2>TABLE IV</h2>``).  ``None`` when the block isn't a lone table label.

    A label rendered as a heading must be recognised too: otherwise its title,
    stranded in the next paragraph, never folds into the table and — lacking
    terminal punctuation — is mistaken for a body fragment and glued onto the
    prose that resumes after the table.
    """
    inner = _plain_p_text(part)
    if inner is None and (heading := _heading_inner(part)) is not None:
        inner = heading[1]
    if inner is not None and _BARE_TABLE_LABEL_RE.match(_visible_text(inner).strip()):
        return inner
    return None


def _join_split_table_caption_labels(parts: list[str]) -> list[str]:
    """Rejoin a table caption OCR split into a bare label paragraph and its title.

    A caption rendered as ``<p>TABLE I</p>`` followed by ``<p>Selected
    substrates…</p>`` leaves the title stranded as its own block: colocation
    folds a caption into its ``<table>`` only when the caption is one block
    adjacent to the table, and the cross-table paragraph merge can't see past
    the stray ``<p>`` either.  A bare label (nothing after the table identifier)
    is never a body sentence, so the following plain paragraph is its title and
    the two are one caption.  The label may arrive as a ``<p>`` or as a heading
    the OCR promoted it to (``<h2>TABLE IV</h2>``).
    """
    out: list[str] = []
    i = 0
    while i < len(parts):
        label = _bare_table_label_inner(parts[i])
        if label is not None and i + 1 < len(parts):
            title = _plain_p_text(parts[i + 1])
            if title is not None and not _opens_with_caption_label(title):
                out.append(f"<p>{label.rstrip()} {title.lstrip()}</p>")
                i += 2
                continue
        out.append(parts[i])
        i += 1
    return out


# A table's abbreviation legend ("MW: molecular weight, NR: Not reported"): a
# short block of two-or-more "ABBR: expansion" entries the OCR drops right after a
# table.  When the table carries no <sup> markers, _colocate_table_footnotes can't
# anchor it (no marker run) and it is not a source/attribution note, so it is left
# loose — wedged between the table and the prose resuming after it.  The two-plus
# short uppercase-led "X:" entries are the discriminator: ordinary prose never
# carries such a run, so this can't swallow a genuine sentence continuation.
_LEGEND_NOTE_MAX_LEN = 200
_ABBREVIATION_ENTRY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]{1,4}:\s")
_MIN_LEGEND_ENTRIES = 2


def _is_table_legend(part: str) -> bool:
    """True for a short abbreviation-key legend the OCR stranded after a table
    (see ``_ABBREVIATION_ENTRY_RE``).  The cross-table merge steps over such a
    block instead of splicing it into the sentence split across the table."""
    inner = _plain_p_text(part)
    if inner is None:
        return False
    text = _visible_text(inner).strip()
    return (
        len(text) <= _LEGEND_NOTE_MAX_LEN
        and len(_ABBREVIATION_ENTRY_RE.findall(text)) >= _MIN_LEGEND_ENTRIES
    )


def _is_open_fragment(part: str, inner: str, stripped: str, visible: str) -> bool:
    """True when a plain ``<p>`` is an incomplete paragraph fragment eligible to
    absorb a following continuation.

    The fragment lacks terminal punctuation (the sentence-end test runs on
    ``stripped`` — terminal punctuation can sit *inside* a closing inline tag,
    "…carboxylase.</em>", or behind a trailing citation superscript,
    "…humans.<sup>15–18</sup>", which ``_ends_sentence`` handles) and is none of the
    self-contained block shapes that merely happen to end without a period:

    - a bold front-matter label, or a figure/table caption label;
    - a metadata footer (DOI / "Published online …" / correspondence) — complete
      even unterminated; backstops the page-0 stray sweep;
    - an affiliation/footnote line opening with a leading superscript marker
      ("¹ Department of …, South Korea") — a multi-affiliation run overruns
      ``_is_stray_metadata``'s length cap, so it is guarded directly or the abstract
      that follows is glued on and hidden with it;
    - a table abbreviation legend ("MW: molecular weight, NR: Not reported").
    """
    return (
        not _ends_sentence(stripped)
        and not _BOLD_LABEL_RE.match(inner)
        and not _opens_with_caption_label(inner)
        and not _is_stray_metadata(part)
        and not _LEADING_SUP_RE.match(visible.lstrip())
        and not _is_table_legend(part)
    )


def _skip_floats(parts: list[str], start: int) -> tuple[int, list[str]]:
    """Step over the tables/figures (up to ``_MAX_FLOATS_TO_SKIP``) and any table
    legend the OCR stranded between a fragment and its continuation, collecting them
    to re-emit *after* the rejoined paragraph.

    Returns the index of the first block that is neither a float nor a legend, and
    the collected blocks.  A legend is furniture, not a float, so it does not consume
    the float budget (else the legend is taken as the continuation and spliced into
    the sentence)."""
    j = start
    floats: list[str] = []
    skipped_floats = 0
    while j < len(parts):
        if _is_table_legend(parts[j]):
            floats.append(parts[j])
            j += 1
        elif _FLOAT_RE.match(parts[j]) and skipped_floats < _MAX_FLOATS_TO_SKIP:
            floats.append(parts[j])
            skipped_floats += 1
            j += 1
        else:
            break
    return j, floats


def _is_valid_continuation(
    cont: str, cont_head: str, visible: str, in_refs: bool
) -> bool:
    """True when ``cont`` is a genuine continuation of an open fragment, not a new
    block that merely follows it.

    Rejects an enumeration item, a bold label, a caption label, and a page footnote
    opening with a footnote marker ("¹http://…" or a "*"/"†" note — matched by
    footnote *shape*, not by any leading superscript, so an isotope-led continuation
    "³⁵S methionine …" still merges).  Two head-capital guards:

    - after a function word the fragment is grammatically incomplete, so an
      uppercase-led continuation (not a mid-token scientific identifier) is a dropped
      continuation, not the real one — refuse it;
    - in the references section a capital-led continuation is the next bibliography
      entry (author surname, even an internal-capital "McKenzie"); a period-less
      numbered marker ("9 Peck, …") opens a new entry too but leads with its digit,
      so it is matched explicitly. Only a lowercase continuation is a wrapped entry."""
    return (
        not _ENUM_RE.match(cont)
        and not _BOLD_LABEL_RE.match(cont)
        and not _opens_with_caption_label(cont)
        and not _UNICODE_SUP_MARKER_RE.match(cont_head)
        and cont_head[:1] not in _FOOTNOTE_SYMBOLS
        and not (
            _FUNCTION_WORD_END_RE.search(visible)
            and cont_head[:1].isupper()
            and not _MIDSENTENCE_HEAD_RE.match(cont_head)
        )
        and not (
            in_refs
            and (cont_head[:1].isupper() or _NUMBERED_REF_HEAD_RE.match(cont_head))
        )
    )


def _merge_split_paragraphs(parts: list[str]) -> list[str]:
    """Stitch paragraph fragments broken by two-column PDF layout.

    When a plain ``<p>`` ends without terminal punctuation (``_is_open_fragment``)
    the next plain ``<p>`` is treated as a continuation (``_is_valid_continuation``).
    Intervening tables and figures (up to ``_MAX_FLOATS_TO_SKIP``) are collected by
    ``_skip_floats`` and re-emitted *after* the merged paragraph so the float stays
    near its reference text.

    Headings, footnote paragraphs, enumeration items, and figure/table captions act
    as merge barriers and are never absorbed into an adjacent paragraph.

    Inside the references section the rule tightens: a bibliography entry rarely ends
    in terminal punctuation (it trails off in a DOI — "…1997.00426.x"), so the generic
    "no period ⇒ fragment" test would chain the whole list into one paragraph; there a
    capital-led continuation is taken as the next entry (``_is_valid_continuation``).
    """
    # Keyed on the references *heading*, not "<p>[1]": a body block opening "[1] …"
    # (a numbered list, an inline citation) must not switch the guard on early and
    # suppress legitimate body-prose merges.  Recomputed each pass because merges
    # shift indices.
    ref_start = next(
        (k for k, p in enumerate(parts) if _REF_HEADING_RE.match(p)), len(parts)
    )
    out: list[str] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        inner = _plain_p_text(part)
        if inner is not None:
            stripped = inner.rstrip()
            visible = _visible_text(stripped).rstrip()
            if _is_open_fragment(part, inner, stripped, visible):
                j, floats = _skip_floats(parts, i + 1)
                if j < len(parts):
                    cont = _plain_p_text(parts[j])
                    cont_head = _visible_text(cont).lstrip() if cont is not None else ""
                    if cont is not None and _is_valid_continuation(
                        cont, cont_head, visible, i >= ref_start
                    ):
                        joined = _dehyphenate_join(stripped, cont)
                        out.append(f"<p>{joined}</p>")
                        out.extend(floats)
                        i = j + 1
                        continue
        out.append(part)
        i += 1
    return out


def _merge_split_paragraphs_stable(parts: list[str]) -> list[str]:
    """Run ``_merge_split_paragraphs`` until it reaches a fixpoint.

    A single pass joins each fragment to its immediate continuation, but a
    paragraph split into three-plus pieces by column/page breaks only collapses
    fully once the earlier joins expose the next adjacency.  Iterating to a
    fixpoint stitches the whole chain regardless of how many pieces it was in.
    """
    while True:
        merged = _merge_split_paragraphs(parts)
        if merged == parts:
            return merged
        parts = merged


def _table_caption_inner(part: str) -> str | None:
    """The inner HTML of a free-standing table caption ("TABLE 2 Comparison …"),
    whether the OCR left it a plain ``<p>`` or promoted the whole caption to a
    heading (``<h2>TABLE 2 …</h2>``).  ``None`` when the block isn't a table
    caption.

    The heading form must be recognised too: otherwise a caption the model
    levelled as a section heading is left stranded in the heading hierarchy while
    its table renders captionless beneath it.  A heading is promoted only when its
    identifier is number-like (``_TABLE_LABEL_HEADING_RE``), so a "Table of Contents"
    section heading is not mistaken for a caption."""
    inner = _plain_p_text(part)
    from_heading = False
    if inner is None and (heading := _heading_inner(part)) is not None:
        inner = heading[1]
        from_heading = True
    if inner is None:
        return None
    text = _visible_text(inner).lstrip()
    if not _TABLE_CAPTION_RE.match(text):
        return None
    if from_heading and not _TABLE_LABEL_HEADING_RE.match(text):
        return None
    return inner


def _is_table_caption(part: str) -> bool:
    """True when a block is a stand-alone table caption ("Table 1 …")."""
    return _table_caption_inner(part) is not None


def _inject_table_caption(table_html: str, caption_part: str) -> str:
    """Insert the caption's inner HTML as a ``<caption>`` first child of the table.

    A lambda replacement keeps any backslash in the caption text (e.g. a literal
    ``\\frac``) from being read as a regex backreference."""
    inner = _table_caption_inner(caption_part)
    if inner is None:
        return table_html
    # Insert after the *whole* opening tag so attributes ("<table class=…>")
    # aren't split, which would orphan them and break the table.
    return _TABLE_OPEN_TAG_RE.sub(
        lambda m: f"{m.group(0)}<caption>{inner}</caption>", table_html, count=1
    )


def _hoist_inline_table_caption(table_html: str) -> str:
    """Hoist a table's leading "Table N …" title row into a ``<caption>``.

    When the OCR (or the text-layer rebuild) bakes the title into the first row as a
    lone spanning ``<th colspan=N>Table N …</th>`` cell instead of leaving it a
    free-standing block, ``_colocate_table_captions`` can't fold it. Recognise that
    row — the first row, one cell spanning multiple columns, caption-shaped text — and
    move it out into a ``<caption>`` so the title renders semantically like every other
    table's. The caption-text guard (``_TABLE_CAPTION_RE``) leaves a genuine spanning
    *section* header (e.g. "A. Effect of EDTA …") in place as a row.
    """
    if "<caption" in table_html.lower():
        return table_html
    row = _FIRST_ROW_RE.search(table_html)
    if row is None:
        return table_html
    cell = _SPANNING_TITLE_CELL_RE.match(row.group(1).strip())
    if cell is None:
        return table_html
    inner = cell.group(3).strip()
    if not _TABLE_CAPTION_RE.match(_visible_text(inner).lstrip()):
        return table_html
    body = table_html[: row.start()] + table_html[row.end() :]
    body = _EMPTY_THEAD_RE.sub("", body)
    return _TABLE_OPEN_TAG_RE.sub(
        lambda m: f"{m.group(0)}<caption>{inner}</caption>", body, count=1
    )


def _claim_table_caption(
    table_idx: int,
    step: int,
    is_caption: list[bool],
    is_figure: list[bool],
    used: list[bool],
) -> int | None:
    """Index of the nearest unused caption from ``table_idx`` in direction ``step``,
    skipping intervening figures and stopping at prose or another table."""
    k = table_idx + step
    while 0 <= k < len(is_caption):
        if is_caption[k] and not used[k]:
            return k
        if is_figure[k]:
            k += step
            continue
        return None
    return None


# A table whose first header cell is a panel sub-header ("A. Effect of EDTA …",
# "B. ICP-MS analysis") — the shape a composite multi-panel table takes when the
# model splits its panels into separate <table>s on the internal blank line between
# them (which _split_md_blocks then breaks into separate blocks).
_TH_CELL_RE = re.compile(r"<th\b[^>]*>(.*?)</th>", re.IGNORECASE | re.DOTALL)
_PANEL_LABEL_RE = re.compile(r"^([A-Z])\.\s")


def _table_panel_letter(table_html: str) -> str | None:
    """The single-letter panel label a table's first ``<th>`` opens with ("A" for
    ``<th>A. Effect of EDTA …</th>``), or ``None`` when it carries no panel header."""
    m = _TH_CELL_RE.search(table_html)
    if m is None:
        return None
    label = _PANEL_LABEL_RE.match(_visible_text(m.group(1)).strip())
    return label.group(1) if label else None


def _merge_panel_tables(first: str, second: str) -> str:
    """Concatenate the inner rows of two panel ``<table>``s under the first's opening
    tag, so the two panels render as one table (the shared caption is folded onto it
    later by ``_colocate_table_captions``)."""
    open_end = first.index(">") + 1
    first_inner = first[open_end : first.rindex("</table>")]
    second_inner = second[second.index(">") + 1 : second.rindex("</table>")]
    return f"{first[:open_end]}{first_inner}{second_inner}</table>"


def _merge_split_panel_tables(blocks: list[Block]) -> list[str]:
    """Re-fuse a composite multi-panel table the model split into adjacent ``<table>``
    blocks (one per panel) on the blank line between panels.

    A run of adjacent **captionless** tables whose first ``<th>`` carries *sequential*
    panel labels (A, then B, then C, …) is one logical table — only a composite table's
    panels label their header rows that way and sit back-to-back — so they are merged
    into the first block.  Gated tightly (captionless + sequential single-letter
    headers) so two genuinely distinct adjacent tables are never fused.  Runs before
    ``_colocate_table_captions`` so the single "Table N …" caption attaches to the
    merged table, not just its first panel."""
    parts = [b.html for b in blocks]
    out: list[str] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        letter = (
            _table_panel_letter(part)
            if blocks[i].kind is BlockKind.TABLE and "<caption" not in part.lower()
            else None
        )
        if letter is not None:
            merged = part
            expect = chr(ord(letter) + 1)
            j = i + 1
            while (
                j < len(parts)
                and blocks[j].kind is BlockKind.TABLE
                and "<caption" not in parts[j].lower()
                and _table_panel_letter(parts[j]) == expect
            ):
                merged = _merge_panel_tables(merged, parts[j])
                expect = chr(ord(expect) + 1)
                j += 1
            out.append(merged)
            i = j
            continue
        out.append(part)
        i += 1
    return out


def _colocate_table_captions(blocks: list[Block]) -> list[str]:
    """Fold a free-standing "Table N …" caption into its ``<table>`` as a
    ``<caption>`` first child so it renders with the table rather than drifting
    in the block stream.

    Each captionless table claims the nearest unused table caption — preferring
    one just *before* it (the usual convention), else just *after* — skipping
    only intervening figures and never reaching across prose or another table.
    A caption with no table nearby is left untouched as its own block.

    A table that carries its title as an inline spanning row (rather than a
    free-standing block) is first rewritten so the title becomes its ``<caption>``;
    it then reads as already-captioned here and claims no neighbouring caption.
    """
    parts = [
        _hoist_inline_table_caption(b.html) if b.kind is BlockKind.TABLE else b.html
        for b in blocks
    ]
    n = len(parts)
    is_caption = [_is_table_caption(p) for p in parts]
    is_figure = [b.kind is BlockKind.FIGURE for b in blocks]
    needs_caption = [
        b.kind is BlockKind.TABLE and "<caption" not in p.lower()
        for b, p in zip(blocks, parts, strict=True)
    ]
    used = [False] * n
    attached: dict[int, int] = {}

    def claim(table_idx: int, step: int) -> int | None:
        return _claim_table_caption(table_idx, step, is_caption, is_figure, used)

    tables = [t for t in range(n) if needs_caption[t]]
    # Backward first for every table, so each secures its own leading caption
    # before a neighbouring table can forward-grab it: a caption between two
    # tables belongs to the one it precedes, not the one it follows.
    for t in tables:
        c = claim(t, -1)
        if c is not None:
            used[c] = True
            attached[t] = c
    for t in tables:
        if t not in attached and (c := claim(t, 1)) is not None:
            used[c] = True
            attached[t] = c

    out: list[str] = []
    for idx, part in enumerate(parts):
        if used[idx]:
            continue
        out.append(
            _inject_table_caption(part, parts[attached[idx]])
            if idx in attached
            else part
        )
    return out


# A table note is a single line ("Molecule structures are shown in Fig. 3.") before
# the superscript-marker lines; bounding the leading non-marker run to one keeps a
# body paragraph an OCR mis-order may have stranded after a table from being
# swallowed as a note.
_MAX_TABLE_NOTE_LINES = 1

# A source/attribution note ("Data adapted from Clark et al. [7].") closes a table
# legend without any superscript marker, so the marker run can't anchor it the way
# it anchors a leading note.  It is recognised lexically instead: a short line that
# opens by attributing the data to a source.  The cue is anchored at the block
# start and the line is length-bounded so body prose that merely mentions a source
# mid-sentence can't match; the check also only ever runs on the line right after a
# table (or its markers).
_TABLE_SOURCE_NOTE_MAX_LEN = 200
# Verbs that specifically credit a source: they rarely open a body sentence as a
# bare participle, so they qualify even without a data subject ("Adapted from …").
_ATTRIBUTION_VERB = r"(?:adapted|reproduced|reprinted|redrawn|modified)"
# Verbs generic enough to open ordinary prose ("Obtained from a supplier, …",
# "Calculated from Eq. 3, …"): they qualify only when introduced by a data subject
# ("Data obtained from …"), never on their own.
_GENERIC_SOURCE_VERB = r"(?:taken|obtained|derived|calculated|compiled)"
_SOURCE_SUBJECT = r"(?:data|values?|results?|means?)"
_TABLE_SOURCE_NOTE_RE = re.compile(
    r"^\s*(?:"
    r"sources?\s*:"
    rf"|{_SOURCE_SUBJECT}\s+(?:(?:were|are|was|is)\s+)?"
    rf"(?:{_ATTRIBUTION_VERB}|{_GENERIC_SOURCE_VERB})\s+from\b"
    rf"|{_ATTRIBUTION_VERB}\s+from\b"
    rf"|{_SOURCE_SUBJECT}\s+from\b"
    r")",
    re.IGNORECASE,
)
# A footnote marker: a SHORT superscript ("<sup>a</sup>", "<sup>*</sup>").  Same
# bound as classify's _SUP_MARKER_RE, but capturing the label so a trailing
# footnote can be matched to the marker it annotates inside the table.
_SUP_LABEL_RE = re.compile(r"<sup>([^<]{1,3})</sup>")


def _is_footnote_label(label: str) -> bool:
    """True for a footnote-style superscript label (letter or footnote symbol).

    A purely numeric/sign superscript inside a table is an exponent or charge
    ("cm²", "10⁻¹"), not a footnote referent, so it must not let a numbered
    article footnote that follows the table be mistaken for the table's own.
    """
    return any(c.isalpha() or c in _FOOTNOTE_SYMBOLS for c in label)


def _leading_sup_label(inner: str) -> str | None:
    """The footnote-marker label a block opens with ("a" for ``<sup>a</sup>…``)."""
    m = _SUP_LABEL_RE.match(inner)
    return m.group(1).strip() if m else None


def _table_sup_labels(table_html: str) -> set[str]:
    """The footnote-style superscript labels a table carries (exponents excluded)."""
    return {
        label
        for m in _SUP_LABEL_RE.finditer(table_html)
        if _is_footnote_label(label := m.group(1).strip())
    }


def _as_table_footnote(part: str) -> str:
    inner = _plain_p_text(part)
    return f'<p class="footnote">{inner}</p>' if inner is not None else part


def _is_table_source_note(part: str) -> bool:
    """True for a short source/attribution note that closes a table legend
    ("Data adapted from Clark et al. [7].") — see ``_TABLE_SOURCE_NOTE_RE``."""
    inner = _plain_p_text(part)
    if inner is None:
        return False
    text = _visible_text(inner).strip()
    return (
        len(text) <= _TABLE_SOURCE_NOTE_MAX_LEN
        and _TABLE_SOURCE_NOTE_RE.match(text) is not None
    )


def _colocate_table_footnotes(blocks: list[Block]) -> list[str]:
    """Absorb a table's trailing footnote run into its ``<table>`` block.

    Right after ``</table>`` the model emits the table's footnotes — superscript
    marker lines (``<sup>a</sup> …``) and any short note sentence wedged between
    the table and those markers — as free-standing paragraphs.  Folding them onto
    the end of the table block keeps them rendering under their table, stops the
    classifier from sweeping them into the article's footnote section, and lets
    the cross-table paragraph merge skip the table (now a single float) to rejoin
    prose split across it.

    A marker line is only this table's footnote when its label is one the table
    actually carries (``<sup>a</sup>`` inside a header/cell) — otherwise the
    superscript line is an article footnote that merely follows the table, and is
    left for the classifier to route to the footnote section.  A plain legend note
    that sits *before* the markers folds only when those markers anchor it; a
    source/attribution note ("Data adapted from …") folds by its lexical shape,
    whether it trails the markers or stands alone directly after the table.  A run
    with neither a matching marker nor a source note is the body resuming and is
    left untouched.
    """
    parts = [b.html for b in blocks]
    n = len(parts)
    out: list[str] = []
    i = 0
    while i < n:
        part = parts[i]
        if blocks[i].kind is not BlockKind.TABLE:
            out.append(part)
            i += 1
            continue
        labels = _table_sup_labels(part)
        # A single legend note may precede the markers; it is a table note only
        # when markers follow to anchor it, so it is tracked apart from the marker
        # run and never folded on its own — that is what keeps a body line stranded
        # between the table and a later source note out of the footnotes.
        leading_start = j = i + 1
        while (
            j < n
            and j - leading_start < _MAX_TABLE_NOTE_LINES
            and (inner := _plain_p_text(parts[j])) is not None
            and _leading_sup_label(inner) is None
            and not _is_table_source_note(parts[j])
        ):
            j += 1
        has_leading = j > leading_start
        marker_start = j
        while j < n and (inner := _plain_p_text(parts[j])) is not None:
            marker = _leading_sup_label(inner)
            if marker is None or marker not in labels:
                break
            j += 1
        seen_marker = j > marker_start
        # A source note carries no marker, so it trails the marker run or stands
        # directly after the table; it is absorbed by its lexical shape, never by
        # position alone, so the body prose that resumes after the footnotes stays.
        source_start = j
        while j < n and _is_table_source_note(parts[j]):
            j += 1
        has_source = j > source_start
        if seen_marker:
            fold_start: int | None = leading_start
        elif has_source and not has_leading:
            fold_start = source_start
        else:
            fold_start = None
        if fold_start is not None:
            out.append(
                part
                + "".join(_as_table_footnote(parts[k]) for k in range(fold_start, j))
            )
            i = j
        else:
            out.append(part)
            i += 1
    return out
