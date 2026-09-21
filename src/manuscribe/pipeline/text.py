"""Shared block-level text helpers and regexes.

Leaf module: the small predicates and patterns that several pipeline stages
(``merge``, ``classify``, ``assemble``) all need to inspect a block's HTML —
strip tags, recognise a plain ``<p>``, a heading, or a caption label.
"""

from __future__ import annotations

import re

_STRIP_TAGS_RE = re.compile(r"<[^>]+>")
# More than this many byte-identical consecutive sibling elements is a decode
# repetition loop, not real content — no document repeats one row, paragraph or
# list item this many times in a row.  Shared by every decode-loop guard
# (``_collapse_repeated_elements``) so the bar can't drift between them.
_MAX_IDENTICAL_ELEMENT_RUN = 3
_SENTENCE_END_RE = re.compile(r"[.!?;:]\s*$")
# The canonical superscript-digit class *content* (to interpolate inside ``[…]`` and
# ``(?!…)``): ``¹²³`` are Latin-1 (U+00B9/B2/B3), ``⁰⁴-⁹`` are U+2070/2074-2079.
# Deliberately NOT ``⁰-⁹`` (U+2070-2079), which also spans ``ⁱ`` (U+2071) and
# unassigned codepoints — the documented footgun (see CLAUDE.md).  Shared so the four
# call sites (here, latex unit runs, classify marker/author patterns) can't drift
# back into ``⁰-⁹``.
_SUP_DIGITS = "¹²³⁰⁴-⁹"
# A trailing citation superscript ("…harmless to humans.<sup>15–18</sup>",
# "…PyMOL software.³²") hides the sentence-terminal period from the visible-text
# end test, so a finished paragraph looks like an unterminated fragment that the
# paragraph merge then glues the next paragraph onto.  Both shapes occur: an HTML
# <sup> citation (a LaTeX/markdown superscript) and a bare Unicode superscript run.
# A citation is digits with list/range separators only — a charge or isotope
# ("Mg²⁺", "³⁵S") ends in a non-digit superscript, so the digit-anchored patterns
# leave it untouched.
_TRAILING_SUP_CITATION_RE = re.compile(r"(?:\s*<sup>[\d\s,–—-]+</sup>)+\s*$")
_TRAILING_UNICODE_SUP_CITATION_RE = re.compile(rf"[{_SUP_DIGITS}]+\s*$")
# Paragraphs that open with a bold label ("Keywords:", "Abbreviations:", "Note:")
# are structured metadata, never mid-sentence continuations.
_BOLD_LABEL_RE = re.compile(r"^<strong>[^<]+:</strong>")
# Matches — and captures the name of — a leading bold label with the colon inside
# *or* outside the bold ("<strong>Keywords:</strong>" vs "<strong>Keywords</strong>:"):
# OCR emits both shapes for the same label, and matching colon-inside only stranded
# the colon-outside keyword line in the body instead of relocating it to the panel.
# This is the *single* either-colon matcher — every leading-bold-label check
# (the metadata/glossary/front-matter predicates via ``_bold_label_in``, and the
# abstract terminator) goes through it, so the colon convention has one home.  Distinct
# from ``_BOLD_LABEL_RE`` (colon-inside only), which stays stricter for the
# merge/furniture guards that must not treat a colon-outside run as a label.
_BOLD_LABEL_CAPTURE_RE = re.compile(
    r"^<strong>([^<]+):</strong>|^<strong>([^<]+)</strong>\s*:"
)
_HEADING_TAG_RE = re.compile(r"^<h([1-6])>(.*)</h\1>$", re.DOTALL)

# A caption opens with a figure/table label ("FIG. 2 …", "**Table 1.** …").
_CAPTION_RE = re.compile(
    r"^\*{0,2}(?:fig(?:ure|\.|\b)|table|scheme|supplement)", re.IGNORECASE
)
# A figure placeholder only ever owns a *figure* caption; a "Table …" block sat
# beside it belongs to its table (see _colocate_table_captions), so the
# figure-caption test deliberately excludes the table label.
_FIGURE_CAPTION_RE = re.compile(r"^\*{0,2}(?:fig(?:ure|\.|\b)|scheme)", re.IGNORECASE)
# The head of a *numbered* figure label — an optional bold opener, the label word,
# and the number (group 1): "FIG 1", "**Figure 6", "Fig. 4".  The one home for that
# shape: the bare-label test (figures) and the caption-label scan (recover_figures)
# both build on it, so the two cannot disagree about what a label looks like, and
# each appends its own tail — which is what separates a caption from an in-prose
# reference.  As in _TABLE_CAPTION_RE, only the label word is case-folded: a tail's
# capitalised-title test *is* the signal keeping "Figure 3 shows …" out, so a global
# re.IGNORECASE would fold away the one rule doing the work.  Line-bound ([ \t], not
# \s): every call site matches a label within one line.
_FIGURE_LABEL_HEAD = r"\*{0,2}[ \t]*(?i:fig(?:ure|\.)?)[ \t]*\.?[ \t]*(\d+)"
# A table caption ("Table 1 …", "Supplementary Table 2 …").  Matched against a
# block's *visible* text so it's recognised through a <strong> wrapper.  After
# the "Table <id>" label a true caption is followed by punctuation, a
# capitalised title word, or nothing — *not* a lowercase word, which marks a
# running reference sentence ("Table 1 summarizes …") that must stay in the body.
# A pipe ("TABLE 2 | Kinetic parameters …") is the Frontiers caption separator, but
# only when a title follows it — "Table 1 | 2 | 3" is a stray prose row, not a
# caption — so the pipe branch carries the same capitalised-title requirement as the
# space-separated one rather than accepting any character after the pipe.  Only
# "table"/"supplementary" are case-folded (OCR casing is unreliable); the
# capitalised-title test stays case-sensitive, as that is the whole signal.
_TABLE_CAPTION_RE = re.compile(
    r"^\*{0,2}\s*"
    r"(?i:supp(?:l(?:ementary)?)?\.?\s+)?"
    r"(?i:table)\b\s*"
    r"\w+"
    r"(?:\s*[.:)–—-]|\s*\|\s*[A-Z(]|\s+[A-Z(]|\s*\*{0,2}\s*$)"
)
# A table tag anywhere in a block — the opening ``<table>`` *or* a continuation
# fragment ``_split_md_blocks`` strands when a blank line splits a table
# (``</table>``, ``<tr>…`` carry no opening ``<table``).  Shared by reconcile (a
# table-like block is skipped for prose recovery) and assemble (a split tail is not
# caption prose / is a table block).
_TABLE_TAG_RE = re.compile(r"</?(?:table|tr|td|th|tbody|thead)\b", re.IGNORECASE)


_BLOCK_SPLIT_RE = re.compile(r"\n[ \t]*\n")

# Case-insensitive: an HTML parser reads <IMG SRC=…> as an image just as it reads
# the lowercase spelling, so a case-sensitive strip would leave the tag standing.
_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)


def _split_md_blocks(md: str) -> list[str]:
    """Split markdown into blocks on blank lines (a run of whitespace-only lines).

    The single definition of "block" shared by the page-block parser and the
    figure-recovery crop parser, so both segment a page identically."""
    return _BLOCK_SPLIT_RE.split(md.strip())


def _visible_text(html: str) -> str:
    """The text a reader sees: ``html`` with every tag removed."""
    return _STRIP_TAGS_RE.sub("", html)


def _visible_text_folded(html: str) -> str:
    """``_visible_text`` normalized for label comparison — trimmed and lowercased.

    OCR casing is unreliable, so document-type and metadata-heading labels are
    matched case-insensitively against this form.
    """
    return _visible_text(html).strip().lower()


def _ends_sentence(inner: str) -> bool:
    """True when a block's visible text ends with terminal punctuation, looking
    past a trailing citation superscript that would otherwise hide it
    ("…humans.<sup>15–18</sup>", "…software.³²")."""
    html = _TRAILING_SUP_CITATION_RE.sub("", inner.rstrip())
    visible = _TRAILING_UNICODE_SUP_CITATION_RE.sub("", _visible_text(html).rstrip())
    # _SENTENCE_END_RE tolerates a trailing-whitespace tail (\s*$), so no re-strip.
    return bool(_SENTENCE_END_RE.search(visible))


def _plain_p_text(s: str) -> str | None:
    """Return the inner content of a plain ``<p>…</p>`` block, or ``None``.

    Returns ``None`` for footnote/class paragraphs, multi-paragraph strings
    (reference lists), headings, tables, figures, and any other element.
    """
    if s.startswith("<p>") and s.endswith("</p>") and s.count("</p>") == 1:
        return s.removeprefix("<p>").removesuffix("</p>")
    return None


def _heading_inner(part: str) -> tuple[int, str] | None:
    m = _HEADING_TAG_RE.match(part)
    return (int(m.group(1)), m.group(2).strip()) if m else None


def _looks_like_figure_caption(block: str) -> bool:
    return bool(_FIGURE_CAPTION_RE.match(block.strip()))


def _opens_with_caption_label(text: str) -> bool:
    """True when text's visible content opens with a figure/table caption label,
    even when wrapped in inline markup (``<strong>Table 1</strong> …``).

    A caption is a merge barrier: gluing it onto an adjacent paragraph both
    garbles the sentence and strands the caption away from its float, so the
    label must be recognised through the ``<strong>`` the model often wraps it
    in, which a raw match of ``_CAPTION_RE`` against the markdown would miss."""
    return bool(_CAPTION_RE.match(_visible_text(text).lstrip()))


def _opens_with_table_label(text: str) -> bool:
    """True when text's visible content opens with a *table* caption label
    ("Table 2 …", "TABLE 2 | …") — the figure-caption case excluded."""
    return bool(_TABLE_CAPTION_RE.match(_visible_text(text).lstrip()))


def _collapse_repeated_elements(html: str, element_re: re.Pattern[str]) -> str:
    """Collapse each run of more than ``_MAX_IDENTICAL_ELEMENT_RUN`` byte-identical
    adjacent ``element_re`` matches down to its first occurrence.

    The shared mechanism behind every decode-loop guard: the caller supplies the
    element the loop repeats — table rows for a looped table crop
    (``tables.markup._collapse_repeated_rows``), any element for chandra's div inner
    HTML (``chandra._iter_divs``) — and everything between kept matches (whitespace,
    surrounding tags) is preserved, so a document with no degenerate run comes back
    byte-for-byte and the pass is idempotent."""
    matches = list(element_re.finditer(html))
    drop = [False] * len(matches)
    i = 0
    while i < len(matches):
        key = matches[i].group()
        j = i + 1
        while j < len(matches) and matches[j].group() == key:
            j += 1
        if j - i > _MAX_IDENTICAL_ELEMENT_RUN:
            for k in range(i + 1, j):
                drop[k] = True
        i = j
    if not any(drop):
        return html
    out: list[str] = []
    cursor = 0
    for m, dropped in zip(matches, drop, strict=True):
        if dropped:
            gap = html[cursor : m.start()]
            if gap.strip():
                out.append(gap)
            cursor = m.end()
    out.append(html[cursor:])
    return "".join(out)


def _strip_img_tags(html: str) -> str:
    """Remove every ``<img>`` tag from a fragment of OCR-sourced HTML.

    The trust boundary for ``<img src>``: an image element is *pipeline* output
    (``figures._figure_html``, whose src comes from the ``ImageSink`` the caller
    supplied), never transcription — LightOnOCR emits figures as ``![image]``
    placeholders consumed before markdown rendering and chandra's own ``<img>``
    carries no src at all.  So an ``<img>`` in OCR-sourced HTML is the model
    echoing markup a crafted PDF printed as visible text, and its src would make
    every reader of the assembled document fetch an attacker-chosen URL.  Removing
    it here, upstream of the pipeline's own figures, is what lets the final
    allow-list pass keep accepting a sink's served URL.

    Substitution is iterated to a fixed point because one pass can splice a fresh
    tag out of the text surrounding a removed one (``<im<img x>g src=…>``)."""
    while (stripped := _IMG_TAG_RE.sub("", html)) != html:
        html = stripped
    return html
