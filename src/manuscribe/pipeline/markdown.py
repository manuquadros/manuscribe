"""Markdown → top-level block-HTML conversion.

LightOnOCR emits whole-page markdown mixed with raw HTML (``<table>``, ``<sup>``)
and inline LaTeX.  CommonMark + the table plugin, with raw-HTML passthrough,
covers the structure; LaTeX sub/superscripts are converted to HTML beforehand by
``latex._latex_to_html``.
"""

from __future__ import annotations

import re

from markdown_it import MarkdownIt

from manuscribe.pipeline.dehyphenate import _dehyphenate_join
from manuscribe.pipeline.latex import _latex_to_html
from manuscribe.pipeline.text import _CAPTION_RE, _plain_p_text, _visible_text

_MD = MarkdownIt("commonmark", {"html": True}).enable("table")

# Inline fragments — figure captions, table cells — need only emphasis and the raw
# inline HTML ``_latex_to_html`` already emitted (``<sup>``/``<sub>``/``<em>``).
# Their OCR prose can carry backticks (mis-read primes "3`"), bracket+paren
# adjacencies ("[12](2019)"), and bare ``<addr>`` tokens that a full CommonMark
# pass would turn into ``<code>``/``<a href>``/autolinks; those rules are disabled
# so such text stays literal, while balanced-emphasis and ``\\*``-escape handling
# (the reason for using the parser over a regex sub) are kept.
_MD_INLINE = MarkdownIt("commonmark", {"html": True}).disable(
    ["backticks", "link", "image", "autolink"]
)

# When LightOnOCR transcribes a page column-by-column it preserves the visual
# line wrapping, emitting one paragraph as many soft-wrapped lines (CommonMark
# keeps these as ``\n`` inside the rendered <p>).  Two artefacts follow from
# that, both repaired by ``_reflow_paragraph``: a word split across the wrap
# keeps its soft hyphen ("Unfortu-\nnately", de-hyphenated by ``_dehyphenate_join``),
# and a paragraph break that fell on a line boundary is lost (the next paragraph
# just continues on the next line).
# A wrapped line ends a paragraph only when it ends a sentence; ``;:`` and commas
# never do, so they stay mid-paragraph.  Closing brackets/quotes may follow the
# terminal mark.
_LINE_SENTENCE_END_RE = re.compile(r"[.!?][)\]\"'»]*$")

# LightOnOCR emits tables as raw HTML, which CommonMark passes through verbatim —
# so ``*emphasis*`` the model leaves inside a cell (organism names: ``*Klebsiella
# aerogenes*``) never reaches the inline parser.  Render it cell-by-cell instead.
_TABLE_CELL_RE = re.compile(
    r"(<t[dh]\b[^>]*>)(.*?)(</t[dh]>)", re.DOTALL | re.IGNORECASE
)


def _render_inline_html(text: str) -> str:
    """Inline-render a fragment whose ``$…$`` spans were already converted.

    The CommonMark inline parser is the tag-aware counterpart to a regex sub: it
    passes valid inline HTML through (``<sup>a</sup>`` markers, ``<sub>`` from
    ``_latex_to_html``), escapes a stray ``<``/``&``, and — unlike a hand-rolled
    bold/italic pass — forms *balanced* emphasis, nesting ``**bold *italic***``
    correctly and honouring ``\\*`` escapes instead of mis-ordering the closing
    tags (``…<em>X</strong></em>``).  Use this for a fragment that has already been
    through ``_latex_to_html`` (figure captions); :func:`_render_inline` is the
    variant that applies that pass first."""
    rendered: str = _MD_INLINE.renderInline(text)
    return rendered.strip()


# A bolded figure-caption title ("**Figure 1. …**") runs straight into the legend
# that follows it ("(A) …"); markdown renders both as one inline run, so break the
# title onto its own line.  The captured group spans *all* adjacent leading bold spans
# (a number + title the model emits as two runs, "**Fig. 2.** **Title.**"), and the
# legend is required to be non-bold (negative lookahead) so an all-bold two-run title
# with no legend isn't split between its runs.  (Each <strong>…</strong> closes its own
# nested </em> before its </strong>, so .*? stops at the right tag.)
_CAPTION_BOLD_TITLE_RE = re.compile(
    r"^(<strong>.*?</strong>(?:\s+<strong>.*?</strong>)*)\s+((?!<strong>)\S.*)$",
    re.DOTALL,
)


def _break_caption_title(caption_html: str) -> str:
    """Break a figure caption's leading bold *title* onto its own line with a ``<br>``,
    so the legend the model bolded the title straight into starts fresh.  A no-op unless
    the caption opens with a bold figure/table *title* (``_CAPTION_RE`` — not a bare
    emphasised word or a panel label "(A)") and non-bold legend prose follows it."""
    m = _CAPTION_BOLD_TITLE_RE.match(caption_html)
    if m is None or not _CAPTION_RE.match(_visible_text(m.group(1)).lstrip()):
        return caption_html
    return f"{m.group(1)}<br>{m.group(2)}"


def _caption_inner_html(caption_text: str) -> str:
    """Inline-render a figure caption and break its bold title off from the legend."""
    return _break_caption_title(_render_inline_html(caption_text))


def _render_inline(text: str) -> str:
    """Render a fragment of OCR markup to HTML through CommonMark's *inline*
    parser, the tag-aware counterpart to a regex sub.

    It passes valid inline HTML through (``<sup>a</sup>`` markers, ``<sub>`` from
    ``_latex_to_html``), escapes a stray ``<``/``&`` (so a footnote ``n<5`` stays
    safe rather than starting a bogus tag), and forms emphasis only on properly
    flanked ``*`` (so a cell ``5 * 10 * 3`` is not spuriously italicised).
    """
    return _render_inline_html(_latex_to_html(text))


def _render_cell_markdown(table_html: str) -> str:
    return _TABLE_CELL_RE.sub(
        lambda m: m.group(1) + _render_inline(m.group(2)) + m.group(3),
        table_html,
    )


def _join_wrapped_lines(lines: list[str]) -> str:
    """Join the soft-wrapped lines of one paragraph back into a single run,
    de-hyphenating words split across a wrap (see ``_dehyphenate_join``)."""
    out = lines[0]
    for line in lines[1:]:
        out = _dehyphenate_join(out, line)
    return out


def _reflow_paragraph(p_html: str, inner: str) -> list[str]:
    """Reflow a soft-wrapped ``<p>`` block: de-hyphenate broken words and split
    where the model dropped a paragraph break onto a line boundary.

    ``inner`` is the block's plain ``<p>`` content the caller already extracted (via
    ``_plain_p_text``), so it is not re-derived here.

    A line that ends a sentence marks a paragraph break only when the next line's
    first word would have fit on it — i.e. the break was forced by the paragraph
    ending, not by the line filling up.  Width is read off the block's own widest
    line, so the test is self-calibrating per column.  Blocks with no soft wrap,
    or with explicit ``<br>`` hard breaks (affiliation lists, addresses), are
    left untouched.
    """
    if "\n" not in inner or "<br" in inner:
        return [p_html]
    lines = inner.split("\n")
    visibles = [_visible_text(line).strip() for line in lines]
    fill = max(len(v) for v in visibles)
    paragraphs: list[list[str]] = [[]]
    for idx, line in enumerate(lines):
        paragraphs[-1].append(line)
        if idx + 1 >= len(lines):
            continue
        first_word = visibles[idx + 1].split(" ", 1)[0]
        would_fit = len(visibles[idx]) + 1 + len(first_word) <= fill
        if _LINE_SENTENCE_END_RE.search(visibles[idx]) and would_fit:
            paragraphs.append([])
    return [f"<p>{_join_wrapped_lines(par)}</p>" for par in paragraphs]


def _md_to_html_blocks(md_text: str) -> list[str]:
    """Convert a page's markdown to a list of top-level block HTML strings.

    One string per top-level block (heading, paragraph, list, raw-HTML table…),
    so downstream cleanup (merge, header/footer strip, footnote reordering) can
    operate block-by-block.  Thematic breaks (``---``) are dropped.
    """
    tokens = _MD.parse(_latex_to_html(md_text))
    blocks: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # Defensive: a top-level walk should only ever land on a level-0 token (the
        # depth tracking below consumes each nested group whole), but skip a stray
        # nested token rather than mis-group it if the stream is ever unbalanced.
        if token.level != 0:
            i += 1
            continue
        if token.nesting == 1:
            depth, j = 0, i
            while j < len(tokens):
                depth += tokens[j].nesting
                j += 1
                if depth == 0:
                    break
            group, i = tokens[i:j], j
        else:
            group, i = [token], i + 1
        html = _MD.renderer.render(group, _MD.options, {}).strip()
        if not html or html.startswith("<hr"):
            continue
        if "<td" in html or "<th" in html:
            blocks.append(_render_cell_markdown(html))
        elif (inner := _plain_p_text(html)) is not None:
            blocks.extend(_reflow_paragraph(html, inner))
        else:
            blocks.append(html)
    return blocks
