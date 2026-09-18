"""Document assembly: per-page block parts → cleaned, classified HTML shell.

Design B-prime (see plans/replace-falcon-with-lightonocr.md):
render → per-page markdown → block HTML → cleanup/merge → document shell.

``_assemble_html`` is the pure (no-GPU) core — render-free, model-free — so it is
unit-testable by feeding synthetic markdown + images.  ``lightonocr_pdf_to_html``
wires in rendering and OCR around it.
"""

from __future__ import annotations

import html as _html
import re
from collections.abc import Callable  # noqa: TC003 — beartype reads annotations
from dataclasses import dataclass
from pathlib import Path

import nh3
from PIL import Image  # noqa: TC002 — beartype reads annotations at runtime

from pdfparser.pipeline.block import Block, BlockKind
from pdfparser.pipeline.classify import (
    _FOOTNOTE_MARKER_CHARS,
    _UNICODE_SUP_MARKER_RE,
    _classify_parts,
    _extract_front_matter,
    _extract_named_metadata_sections,
    _extract_stray_metadata,
    _leading_pages_to_skip_md,
    _normalize_heading_levels,
    _recover_headingless_abstract,
    _split_abstract_citation,
)
from pdfparser.pipeline.doi import _extract_doi
from pdfparser.pipeline.figures import (
    _FIGURE_MERGE_GAP_FRAC,
    _FIGURE_NOTE_RE,
    ImageSink,
    _base64_src,
    _cluster_figure_boxes,
    _denormalize_bbox,
    _figure_html,
    _file_image_writer,
    _is_bare_figure_label,
    _is_panel_label,
    _opens_with_panel_label,
    _parse_figure_placeholder,
    _safe_crop,
    _union_box,
)
from pdfparser.pipeline.furniture import (
    _capture_license_footer,
    _strip_running_furniture,
)
from pdfparser.pipeline.latex import _latex_to_html
from pdfparser.pipeline.layers import _DocumentLayers, _leading_image_only_pages
from pdfparser.pipeline.markdown import _caption_inner_html, _md_to_html_blocks
from pdfparser.pipeline.merge import (
    _colocate_table_captions,
    _colocate_table_footnotes,
    _join_split_table_caption_labels,
    _merge_split_panel_tables,
    _merge_split_paragraphs_stable,
)
from pdfparser.pipeline.model import OcrModel, _ocr_page, _ocr_pages, load_ocr_model
from pdfparser.pipeline.reconcile import _reconcile_text_layer
from pdfparser.pipeline.recover_figures import _recover_dropped_figures
from pdfparser.pipeline.render import _render_page_images
from pdfparser.pipeline.tables import (
    _close_unclosed_tables,
    _collapse_repeated_rows_md,
    _recover_dropped_tables,
    _recover_table_cell_bold,
    _repair_tables_from_text_layer,
)
from pdfparser.pipeline.text import (
    _TABLE_TAG_RE,
    _looks_like_figure_caption,
    _opens_with_caption_label,
    _opens_with_table_label,
    _plain_p_text,
    _split_md_blocks,
    _visible_text,
)

_WRAPPER_CSS = """
body {
    font-family: Georgia, serif;
    max-width: 860px;
    margin: 2rem auto;
    padding: 0 1.5rem;
    color: #222;
    line-height: 1.7;
    font-size: 16px;
}
header {
    border-bottom: 1px solid #ccc; margin-bottom: 2rem; padding-bottom: 1rem; }
header h1 { font-size: 1.4rem; margin: 0 0 .4rem; }
h1 { font-size: 1.3rem; margin: 1.5em 0 .4em; }
h2 { font-size: 1.15rem; margin: 1.5em 0 .4em; border-bottom: 1px solid #e0e0e0; }
h3 { font-size: 1rem; margin: 1.2em 0 .3em; }
p  { margin: .6em 0; }
section.abstract { background: #f7f7f7; padding: 1em 1.2em; border-radius: 4px;
    margin: 1.5em 0; }
details.metadata { margin: 1.5em 0; border: 1px solid #e0e0e0; border-radius: 4px;
    padding: .3em 1em; background: #fafafa; font-size: .9rem; color: #555; }
details.metadata > summary { cursor: pointer; font-weight: bold; color: #333;
    list-style: disclosure-closed; }
details.metadata[open] > summary { list-style: disclosure-open;
    margin-bottom: .4em; }
figure { margin: 1.5em 0; }
figcaption { font-size: .875em; color: #555; }
p.footnote { font-size: .8rem; color: #666; border-top: 1px solid #eee;
    padding-top: .3em; margin-top: .3em; }
table { border-collapse: collapse; width: 100%; overflow-x: auto;
    display: block; font-size: .9rem; margin: 1em 0; }
caption { font-weight: bold; text-align: left; padding: .4em 0 .6em;
    font-size: .9rem; color: #333; }
td, th { padding: .4em .7em; border: 1px solid #ccc; }
hr { border: none; border-top: 1px solid #ddd; margin: 2rem 0; }
"""


@dataclass
class _MdBlock:
    """A run of page markdown destined for ``_md_to_html_blocks``."""

    text: str


@dataclass
class _FigBlock:
    """A figure placeholder: its ``[0, 1000]``-normalized box (``None`` for the
    bbox-less base-variant placeholder) and the caption markdown, if any."""

    bbox_norm: tuple[int, int, int, int] | None
    caption: str | None


_Block = _MdBlock | _FigBlock


def _starts_figure(block: str) -> bool:
    """True when ``block``'s first line is a figure placeholder."""
    block = block.strip()
    return (
        bool(block) and _parse_figure_placeholder(block.splitlines()[0]).is_placeholder
    )


# A caption's final clause: the run after its last sentence terminator, with no
# terminator of its own (a heading the OCR echoed onto the caption trails off without
# punctuation, unlike a closed legend sentence).
_SENTENCE_TAIL_RE = re.compile(r"[.!?]\s+([^.!?]+?)\s*$")
# This stream is raw markdown (pre-HTML), so a heading is "## Title", not "<h2>…".
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")
_WORD_RE = re.compile(r"\w+")
_HEADING_ECHO_MIN_TOKENS = 3


def _strip_trailing_heading_echo(caption: str, next_block: str) -> str:
    """Drop a section heading the OCR echoed onto the end of a caption.

    The model sometimes emits a section heading twice — once glued (no ``#``, no
    terminal punctuation) onto the tail of a figure's last panel description, and once
    as the real heading block that follows (``…green, respectively. Crystal structure
    of BkTauF`` before a ``# Crystal structures of BkTauF``).  The panel fold then bakes
    the echo into the ``<figcaption>``.  When the caption's final clause duplicates the
    *immediately-following* heading block — at least three words, almost all of them
    (all but one, to tolerate a singular/plural slip) in that heading — strip the
    clause.  Keyed on the real heading text, so a genuine trailing legend sentence
    (which shares few words with the next heading) is left intact."""
    heading = _MD_HEADING_RE.match(next_block.strip())
    if heading is None:
        return caption
    stripped = caption.rstrip()
    m = _SENTENCE_TAIL_RE.search(stripped)
    if m is None:
        return caption
    tail_tokens = set(_WORD_RE.findall(m.group(1).lower()))
    head_tokens = set(_WORD_RE.findall(heading.group(1).lower()))
    if (
        len(tail_tokens) >= _HEADING_ECHO_MIN_TOKENS
        and len(tail_tokens - head_tokens) <= 1
    ):
        return stripped[: m.start(1)].rstrip()
    return caption


def _is_caption_continuation(block: str) -> bool:
    """A block that continues a caption: running prose, not the start of a new
    structural element (another float, a heading, a table).

    Every line is checked, not just the first: when the OCR merges a caption
    paragraph with a following heading/table (no blank-line separator) into one
    block, folding it whole onto the caption would swallow that section — so a
    block carrying any such boundary line is not a clean continuation."""
    block = block.strip()
    if not block:
        return False
    for line in block.splitlines():
        s = line.lstrip()
        if (
            s.startswith(("#", "|"))
            or _TABLE_TAG_RE.match(s)
            or _parse_figure_placeholder(s).is_placeholder
        ):
            return False
    return not _opens_with_caption_label(block)


def _is_title_only_figure_caption(caption: str) -> bool:
    """True for a figure caption that is just a "Figure N. Title" header carrying no
    legend sentence of its own — its title trails off on a word ("… close homologs
    of *BkTauF*"), so the caption ends on an alphanumeric character.  Such a header
    means a following prose block is the descriptive legend the OCR split off, not
    body text.  A caption ending in *any* punctuation is treated as complete: a
    period/!/? closes a legend sentence the caption already states, and a ")"/":"/","
    closes a parenthetical or list ("… (maximum likelihood)") rather than trailing
    off — after either, a prose block is the body resuming, which must not be folded
    in.  Requiring an alphanumeric end (not merely "no sentence terminator") is what
    keeps those punctuation-closed captions from absorbing the next paragraph."""
    text = caption.strip().rstrip("*").rstrip()
    return bool(text) and text[-1].isalnum()


# Footnote-marker symbols that open a stranded note ("† Present address …", "‡ These
# authors contributed equally") — the shared ``_FOOTNOTE_MARKER_CHARS`` set minus
# '*', which is excluded here because a '*'-led block is an italicised organism/gene
# name opening a legend ("*Bk*TauF is shown in …"), not a footnote.  ('#' is already
# rejected as a heading by _is_caption_continuation.)
_LEGEND_FOOTNOTE_MARKERS = _FOOTNOTE_MARKER_CHARS.replace("*", "")


def _is_legend_continuation(block: str) -> bool:
    """A block safe to fold onto a title-only figure caption: caption-continuation
    prose that is not a stranded footnote or footer-metadata line.

    The cross-page paragraph merge applies the same exclusions before it stitches a
    fragment to its continuation (``_merge_split_paragraphs`` guards against leading
    superscript / footnote markers and DOI/URL footer lines); the caption fold needs
    them too, or a footnote, a "Downloaded from http://…" stamp, or a DOI line the
    OCR dropped right after a title-only caption is baked into the figcaption and
    never reaches the footnote-section / running-furniture routing downstream."""
    if not _is_caption_continuation(block):
        return False
    head = block.strip()
    return not (
        _UNICODE_SUP_MARKER_RE.match(head)
        or head[:1] in _LEGEND_FOOTNOTE_MARKERS
        or _FIGURE_NOTE_RE.search(head)
    )


def _opens_panel_continuation(block: str) -> bool:
    """A multi-panel sub-description ("(A) … (B) …") the model split into its own
    paragraph: caption text the OCR detached from the header, foldable onto the
    caption.  Excludes a panel block that merged with a following heading/table
    (``_is_caption_continuation``), which must not be folded whole."""
    return _opens_with_panel_label(block) and _is_caption_continuation(block)


def _fold_next(
    raw_blocks: list[str],
    k: int,
    caption: str,
    next_ok: Callable[[str], bool],
) -> tuple[int, str]:
    """Fold the block after ``k`` onto ``caption`` when ``next_ok`` accepts it,
    advancing the cursor past the consumed block; otherwise return both unchanged."""
    if k + 1 < len(raw_blocks) and next_ok(raw_blocks[k + 1]):
        return k + 1, f"{caption} {raw_blocks[k + 1].strip()}"
    return k, caption


def _parse_page_blocks(md: str) -> list[_Block]:
    """Split a page's markdown into an ordered stream of prose and figure blocks.

    Pure: no image needed.  A figure's caption is taken from the placeholder's
    own trailing text, else the following caption-like block (which is then
    consumed); a bare ``FIG. N`` label has its following descriptive block
    rejoined onto it.  Lone single-letter panel labels the model split out of a
    multi-panel figure are dropped when they abut a placeholder."""
    blocks: list[_Block] = []
    raw_blocks = _split_md_blocks(md)
    k = 0
    while k < len(raw_blocks):
        block = raw_blocks[k].strip()
        lines = block.splitlines()
        fig = _parse_figure_placeholder(lines[0]) if lines else None
        if fig is None or not fig.is_placeholder:
            next_is_fig = k + 1 < len(raw_blocks) and _starts_figure(raw_blocks[k + 1])
            prev_is_fig = bool(blocks) and isinstance(blocks[-1], _FigBlock)
            if _is_panel_label(block) and (next_is_fig or prev_is_fig):
                k += 1
                continue
            blocks.append(_MdBlock(block))
            k += 1
            continue
        rest = "\n".join(lines[1:]).strip()
        caption: str | None = rest or None
        if (
            caption is None
            and k + 1 < len(raw_blocks)
            and _looks_like_figure_caption(raw_blocks[k + 1])
        ):
            k += 1
            caption = raw_blocks[k].strip()
        if caption is not None and _is_bare_figure_label(caption):
            k, caption = _fold_next(raw_blocks, k, caption, _is_caption_continuation)
        # Fold each multi-panel sub-description ("(A) … (B) … (C) …") onto the
        # caption.  Gated on a caption header having been found (caption is not
        # None): otherwise a headerless figure would absorb a genuine body
        # enumeration, or the next figure's caption.
        while caption is not None:
            new_k, caption = _fold_next(
                raw_blocks, k, caption, _opens_panel_continuation
            )
            if new_k == k:
                break
            k = new_k
        # A legend the OCR split after the "Figure N. Title" header — a plain
        # descriptive sentence, not a "(A) …" panel run — is folded onto a
        # title-only caption.  Stranded between its figure and the paragraph the
        # figure interrupts, such a legend otherwise reads as body prose that the
        # cross-page merge glues onto the previous column.  Gated on a title-only
        # caption so a figure whose legend the caption already states does not also
        # absorb the body resuming after it.
        if (
            caption is not None
            and _looks_like_figure_caption(caption)
            and _is_title_only_figure_caption(caption)
        ):
            k, caption = _fold_next(raw_blocks, k, caption, _is_legend_continuation)
        # A folded panel/legend block can carry a duplicate of the *next* section
        # heading the OCR glued onto its tail; drop that echo so the heading isn't
        # repeated inside the figcaption (the real heading still follows as its block).
        if caption is not None and k + 1 < len(raw_blocks):
            caption = _strip_trailing_heading_echo(caption, raw_blocks[k + 1])
        blocks.append(_FigBlock(fig.bbox_norm, caption))
        k += 1
    return blocks


def _is_table_md(block: _Block | None) -> bool:
    return (
        isinstance(block, _MdBlock)
        and _TABLE_TAG_RE.match(block.text.lstrip()) is not None
    )


def _is_table_caption_md(block: _Block | None) -> bool:
    """A markdown block that is a free-standing "TABLE N …" caption (not the table
    itself) — the form the model emits when it doesn't bake the caption onto the
    figure placeholder line."""
    return (
        isinstance(block, _MdBlock)
        and not _is_table_md(block)
        and _opens_with_table_label(block.text)
    )


def _dedup_table_figures(blocks: list[_Block]) -> list[_Block]:
    """Drop a figure that is really a table boxed as an image.

    A model whose bbox head reads a sideways or graphical table as a picture emits
    *both* a figure placeholder and the ``<table>`` it also transcribed, so the table
    renders twice — once as a cropped image, once as the real table.  The figure is
    the duplicate when an actual ``<table>`` follows it, optionally with the table's
    "TABLE N …" caption in between (the model emits that caption either baked onto the
    placeholder line — held on the ``_FigBlock`` — or as its own following block).

    The image is dropped; the caption is preserved as a standalone block so
    :func:`_colocate_table_captions` folds it into the real table as its
    ``<caption>``.  When the caption already stands as its own block it is left in
    place; only when it rode on the placeholder is it re-emitted."""
    out: list[_Block] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if isinstance(block, _FigBlock):
            j = i + 1
            standalone_caption = _is_table_caption_md(
                blocks[j] if j < len(blocks) else None
            )
            if standalone_caption:
                j += 1
            baked_caption = (
                block.caption
                if block.caption is not None and _opens_with_table_label(block.caption)
                else None
            )
            table_follows = _is_table_md(blocks[j] if j < len(blocks) else None)
            # The standalone-table-caption path dedups only a *caption-less* figure: a
            # table the model boxed as an image carries no caption of its own (its
            # caption is the standalone "TABLE N" block, or — the baked_caption path —
            # the table label baked onto the placeholder).  A figure that carries any
            # caption of its own is kept as a genuine figure that merely precedes a
            # separate table (the OCR dropping the "---" between them).  Deliberately
            # "any caption", not "a FIG-labelled caption": per the figures-over-include
            # trade, leaving a boxed table's stray duplicate image beats dropping a
            # real figure whose caption the model didn't prefix with "FIG N".
            if table_follows and (
                baked_caption is not None
                or (standalone_caption and block.caption is None)
            ):
                if baked_caption is not None:
                    # The caption rode on the placeholder line, not as its own block;
                    # re-emit it so _colocate_table_captions can fold it in.
                    out.append(_MdBlock(baked_caption))
                i += 1  # drop the figure image; keep the caption/table blocks after it
                continue
        out.append(block)
        i += 1
    return out


def _resolve_figure_clusters(
    blocks: list[_Block], image: Image.Image
) -> tuple[dict[int, tuple[int, int, int, int]], dict[int, str | None], set[int]]:
    """Resolve figure boxes the model over-segmented into stacked crops.

    Pure (reads only ``image.size``, never crops).  Returns, keyed by figure
    block index: ``box_at`` the pixel crop box (the union for a merged cluster),
    ``caption_at`` the cluster's caption (its first non-empty member's), both
    held on the surviving lowest index, and ``drop`` the merged-away members.
    Figure blocks without a bbox are absent from all three, so the caller leaves
    them be."""
    figures = [
        (i, b.caption, _denormalize_bbox(b.bbox_norm, image))
        for i, b in enumerate(blocks)
        if isinstance(b, _FigBlock) and b.bbox_norm is not None
    ]
    boxes = [box for _, _, box in figures]
    gap = _FIGURE_MERGE_GAP_FRAC * image.size[1]

    box_at: dict[int, tuple[int, int, int, int]] = {}
    caption_at: dict[int, str | None] = {}
    drop: set[int] = set()
    for group in _cluster_figure_boxes(boxes, gap):
        members = sorted(group, key=lambda g: figures[g][0])
        survivor = figures[members[0]][0]
        box_at[survivor] = _union_box([boxes[g] for g in members])
        caption_at[survivor] = next(
            (figures[g][1] for g in members if figures[g][1]), None
        )
        drop.update(figures[g][0] for g in members[1:])
    return box_at, caption_at, drop


def _figcaption_only(caption_html: str) -> str:
    inner = _caption_inner_html(caption_html)
    return f"<figure><figcaption>{inner}</figcaption></figure>"


def _page_to_html_parts(
    md: str,
    image: Image.Image,
    ocr_region: Callable[[Image.Image], str] | None = None,
    encode_image: ImageSink = _base64_src,
) -> list[str]:
    """Convert one page's markdown to block HTML, replacing each ``![image]``
    placeholder with a cropped ``<figure>`` and stitching consecutive prose into
    a single markdown render.  ``ocr_region`` (re-OCR a sub-image), when supplied,
    lets the crop trim a caption the model baked into a text-bodied figure box.
    ``encode_image`` turns each crop into its ``<img src>`` (inline data URI by
    default; a sidecar PNG path when a file writer is supplied)."""
    blocks = _dedup_table_figures(_parse_page_blocks(md))
    box_at, caption_at, drop = _resolve_figure_clusters(blocks, image)

    parts: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            parts.extend(_md_to_html_blocks("\n\n".join(pending)))
            pending.clear()

    for i, block in enumerate(blocks):
        if isinstance(block, _MdBlock):
            pending.append(block.text)
            continue
        flush()
        if i in drop:
            continue
        box = box_at.get(i)
        caption = caption_at.get(i, block.caption)
        caption_html = _latex_to_html(caption) if caption else None
        crop = (
            _safe_crop(image, box, caption_text=caption, ocr_region=ocr_region)
            if box is not None
            else None
        )
        if crop is not None:
            parts.append(_figure_html(crop, caption_html, encode_image))
        elif caption_html is not None:
            parts.append(_figcaption_only(caption_html))
    flush()
    return parts


# LightOnOCR transcribes whatever the source PDF's visible text/markup shows, so a
# corrupted or deliberately crafted PDF can make the model emit a <script>/<iframe>
# or an on*-handler attribute that markdown.py's html:True passthrough would
# otherwise carry verbatim into HTML the D3 Annotation Hub stores and serves. This
# is the final allow-list boundary all OCR-sourced content HTML passes through
# before reaching the document shell (title, byline, abstract, metadata, body —
# the shell's own <html>/<head>/<style>/wrapper markup is never run through it).
# Anything outside `_CONTENT_TAGS` is unwrapped (tag dropped, text kept); the tags
# in `_CONTENT_CLEAN_TAGS` drop their content too, since script/style source or an
# iframe's off-screen document isn't reader-visible prose to preserve. No <a>/href:
# nothing in this package emits one deliberately — the block-level markdown parser
# (unlike the inline one used for captions/cells) still has CommonMark's
# link/autolink rules enabled, so OCR prose shaped like a citation adjacency
# ("[12](2019)") could otherwise resolve to a live link — and no fixture needs an
# outbound link. "data" is allowed only as a URL scheme (for <img src>): a data:
# URI decoded in an <img> context cannot execute embedded script (unlike
# <object>/<iframe>), even for image/svg+xml.
_CONTENT_TAGS = frozenset(
    {
        "p",
        "br",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "ol",
        "ul",
        "li",
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
        "caption",
        "sup",
        "sub",
        "em",
        "strong",
        "figure",
        "figcaption",
        "img",
    }
)
_CONTENT_CLEAN_TAGS = frozenset(
    {"script", "style", "iframe", "noscript", "object", "embed", "svg", "template"}
)
_CONTENT_ATTRIBUTES: dict[str, set[str]] = {
    "td": {"colspan", "rowspan"},
    "th": {"colspan", "rowspan"},
    "ol": {"start"},
    "img": {"src", "alt"},
}
# The only content-level class pdfparser itself emits (classify.py/merge.py/
# tables/recover.py footnote paragraphs); anything else stays unstyled rather than
# letting OCR-crafted markup pick a class the wrapper CSS targets.
_CONTENT_ALLOWED_CLASSES: dict[str, set[str]] = {"p": {"footnote"}}
_CONTENT_URL_SCHEMES = frozenset(nh3.ALLOWED_URL_SCHEMES) | {"data"}

_CONTENT_SANITIZER = nh3.Cleaner(
    tags=_CONTENT_TAGS,
    clean_content_tags=_CONTENT_CLEAN_TAGS,
    attributes=_CONTENT_ATTRIBUTES,
    allowed_classes=_CONTENT_ALLOWED_CLASSES,
    url_schemes=_CONTENT_URL_SCHEMES,
)


def _sanitize_content_html(html: str) -> str:
    """Run one block of OCR-sourced content HTML through the final allow-list pass."""
    return _CONTENT_SANITIZER.clean(html)


def _abstract_section(abstract: list[str]) -> str:
    if not abstract:
        return ""
    return "<section class='abstract'>\n" + "\n".join(abstract) + "\n</section>"


def _metadata_panel(metadata: list[str]) -> str:
    # Front matter is kept right after the abstract, collapsed by default in a
    # toggleable <details> panel, so the body opens with prose.
    if not metadata:
        return ""
    return (
        "<details class='metadata'>\n<summary>Metadata</summary>\n"
        + "\n".join(metadata)
        + "\n</details>"
    )


# Block-based equivalents of classify._REF_HEADING_RE / _REF_SECTION_RE, read from
# a heading Block's already-derived kind/inner instead of re-matching the raw
# "<h\d[^>]*>…" string against every block. merge._merge_split_paragraphs still
# uses the raw-string regexes directly (a separate, larger migration — see
# BlockKind.TABLE/FIGURE's deferred _FLOAT_RE call site in the same function).
_REF_LABEL_RE = re.compile(
    r"^(?:\d+\.?\s+)?(?:references|bibliography|literature\s+cited|works\s+cited)\b",
    re.IGNORECASE,
)


def _is_ref_heading_block(block: Block) -> bool:
    return block.kind is BlockKind.HEADING and bool(
        _REF_LABEL_RE.match(block.inner or "")
    )


def _is_ref_section_start_block(block: Block) -> bool:
    """The looser "References" heading or a "[1]"-led paragraph — the fallback
    anchor for a heading-less numbered bibliography (see
    ``_insert_footnotes_before_refs``)."""
    if block.kind is BlockKind.HEADING:
        return (block.inner or "").strip().lower() == "references"
    if block.kind is BlockKind.PARAGRAPH:
        return (block.inner or "").startswith("[1]")
    return False


def _insert_footnotes_before_refs(body: list[Block], footnotes: list[str]) -> list[str]:
    """Splice footnote paragraphs in just before the references section (or at the
    end, if there is none) so they read after the prose but before the bibliography.

    Anchors on the references *heading* first; the looser "<p>[1]" anchor
    (``_is_ref_section_start_block``) is only the fallback for a heading-less
    numbered bibliography, so a stray "[1]"-led body paragraph (an inline citation,
    a numbered list) that precedes the real bibliography can't strand the
    footnotes mid-body."""
    if not footnotes:
        return [b.html for b in body]
    heading_idx: int | None = None
    section_idx: int | None = None
    for i, block in enumerate(body):
        if _is_ref_heading_block(block):
            heading_idx = i
            break
        if section_idx is None and _is_ref_section_start_block(block):
            section_idx = i
    if heading_idx is not None:
        ref_idx = heading_idx
    elif section_idx is not None:
        ref_idx = section_idx
    else:
        ref_idx = len(body)
    html = [b.html for b in body]
    return html[:ref_idx] + footnotes + html[ref_idx:]


# A bibliography entry the OCR emitted as a plain <p> because it dropped the period
# the markdown list needs ("<p>9 Peck, …</p>" not a "9." list item): a leading 1–3
# digit number, optional list delimiter, then a capitalised author surname.  The
# number and the entry text are captured so the number (which the <ol> renders
# itself) can be dropped when the entry is re-attached as an <li>.
_NUMBERED_REF_P_RE = re.compile(r"^<p>\s*(\d{1,3})[.)\]]?\s+([A-Z].*)</p>$", re.DOTALL)
_OL_BLOCK_RE = re.compile(r"^<ol\b[^>]*>.*</ol>$", re.IGNORECASE | re.DOTALL)


def _is_reference_continuation(block: str) -> bool:
    """A bibliography entry's tail the OCR stranded after a page break: a plain ``<p>``,
    not itself a numbered entry, opening mid-sentence (lowercase) — the head (and its
    number) was dropped at the break, so it lands loose between the list and the next
    entry's ``<ol>``.  Used only right after an ``<ol>`` in the references section."""
    inner = _plain_p_text(block)
    if inner is None or _NUMBERED_REF_P_RE.match(block):
        return False
    head = _visible_text(inner).lstrip()
    return bool(head) and head[0].islower()


def _consolidate_numbered_references(blocks: list[Block]) -> list[str]:
    """Fold period-less numbered reference entries into one ``<ol>``.

    Markdown turns "1. Author …" into ``<ol><li>`` but leaves "9 Author …" — the
    period dropped by OCR on a continuation page — a plain ``<p>``.  The references
    merge guard keeps each such entry its own block; this pass then re-attaches a run
    of them as ``<li>`` items (dropping the now-redundant leading number the ``<ol>``
    renders itself), either extending an immediately preceding ``<ol>`` so the list
    continues its numbering, or wrapping a free-standing run in a new ``<ol start=N>``.
    Without it a bibliography split across pages renders as an ``<ol>`` for the first
    entries followed by loose numbered paragraphs for the rest.  Scoped to the
    references section so a numbered ``<p>`` run elsewhere in the body is untouched."""
    parts = [b.html for b in blocks]
    ref_start = next(
        (i for i, block in enumerate(blocks) if _is_ref_heading_block(block)), None
    )
    if ref_start is None:
        return parts
    out: list[str] = []
    i = 0
    while i < len(parts):
        # A page-break-stranded entry tail (its number dropped) lands as a loose <p>
        # after the list; fold it back into the last <li> so it reads inside the
        # bibliography rather than as a stray paragraph between numbered entries.
        if (
            i > ref_start
            and out
            and _OL_BLOCK_RE.match(out[-1])
            and _is_reference_continuation(parts[i])
        ):
            inner = _plain_p_text(parts[i])
            cut = out[-1].rfind("</p>")
            out[-1] = f"{out[-1][:cut]} {inner}{out[-1][cut:]}"
            i += 1
            continue
        m = _NUMBERED_REF_P_RE.match(parts[i])
        if i > ref_start and m is not None:
            first_num = m.group(1)
            items: list[str] = []
            while i < len(parts) and (m := _NUMBERED_REF_P_RE.match(parts[i])):
                items.append(f"<li>\n<p>{m.group(2).strip()}</p>\n</li>")
                i += 1
            body = "\n".join(items)
            if out and _OL_BLOCK_RE.match(out[-1]):
                prev = out[-1]
                out[-1] = prev[: prev.rfind("</ol>")] + body + "\n</ol>"
            else:
                out.append(f'<ol start="{first_num}">\n{body}\n</ol>')
            continue
        out.append(parts[i])
        i += 1
    return out


def _document_shell(
    *, title_html: str, byline_html: str, abstract: str, metadata: str, body: str
) -> str:
    title_safe = _html.escape(_visible_text(title_html))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title_safe}</title>
<style>{_WRAPPER_CSS}</style>
</head>
<body>
<header>
  <h1>{title_html}</h1>
  <p>{byline_html}</p>
</header>
{abstract}
{metadata}
<div class="body">
{body}
</div>
</body>
</html>"""


def _assemble_document(
    pages_md: list[str],
    images: list[Image.Image],
    ocr_region: Callable[[Image.Image], str] | None = None,
    encode_image: ImageSink = _base64_src,
) -> tuple[str, str, str]:
    """The pure assembly core: returns the document HTML plus its plain-text title
    and byline, flattened from the same header HTML the shell renders so a consumer
    gets them without re-parsing the output.  :func:`_assemble_html` is the
    ``.html``-only view kept for the existing string callers."""
    start = _leading_pages_to_skip_md(pages_md)
    pages_md = pages_md[start:]
    images = images[start:]

    # Close any table the OCR left open at a page's bottom before block-splitting,
    # so a table that overran the page does not swallow the next page's prose; and
    # collapse any decode-loop row explosion the page re-OCR may have produced
    # (the crop re-OCR collapses its own output, but the page pass does not).
    per_page_parts = [
        _page_to_html_parts(
            _collapse_repeated_rows_md(_close_unclosed_tables(md)),
            img,
            ocr_region,
            encode_image,
        )
        for md, img in zip(pages_md, images, strict=True)
    ]
    # Front matter the OCR scattered into the article's first page — a glossary
    # section (Abbreviations / Nomenclature) past the leading-run scan, and
    # self-contained footer lines (correspondence, submission/DOI, supporting-info
    # notes) — is pulled here, scoped to that first page.  The stray-line sweep runs
    # before the paragraph-merge so a footer line ending in ")" is removed before
    # the merge can glue following body prose onto it.
    named_metadata: list[str] = []
    stray_metadata: list[str] = []
    if per_page_parts:
        named_metadata, per_page_parts[0] = _extract_named_metadata_sections(
            per_page_parts[0]
        )
        stray_metadata, per_page_parts[0] = _extract_stray_metadata(per_page_parts[0])

    # From here on, a block is the typed post-flatten `Block` (see block.py), not a
    # bare str. Every pass below this point still takes/returns list[str] (they are
    # migrated to read Block fields directly in later, separate changes — see
    # pdfparser-tickets/tickets/typed-block-model-design.md) so each call is wrapped
    # with a thin unwrap-to-html / rewrap-to-Block shim at its boundary. This proves
    # list[Block] can flow end to end with zero behavior change before any pass's
    # internals are touched. source_page is exact on construction here and
    # best-effort (None) on any block a shim below creates by merging/splitting —
    # nothing downstream reads it yet, so an approximate value is behavior-neutral.
    blocks: list[Block] = [
        Block.of(part, source_page=page_idx)
        for page_idx, page in enumerate(per_page_parts)
        for part in page
    ]
    # Table-cleanup passes, in a load-bearing order: panel-fusion must precede
    # caption colocation so the single "Table N …" caption attaches to the *merged*
    # table; footnote colocation must precede both classify (so a table's footnotes
    # stay with it rather than being swept into the article footnote run) and the
    # paragraph merge (so the table is one float the cross-table merge steps over).
    # The first pass isn't yet migrated to read Block fields, so it still takes/
    # returns list[str]; the other three read block.kind directly.
    blocks = [
        Block.of(s) for s in _join_split_table_caption_labels([b.html for b in blocks])
    ]
    blocks = [Block.of(s) for s in _merge_split_panel_tables(blocks)]
    blocks = [Block.of(s) for s in _colocate_table_captions(blocks)]
    blocks = [Block.of(s) for s in _colocate_table_footnotes(blocks)]

    meta = _classify_parts([b.html for b in blocks])

    abstract = [Block.of(s) for s in _merge_split_paragraphs_stable(meta.abstract)]
    # One copy of a recurring copyright/open-access license footer, captured before the
    # furniture strip drops the per-page repeats, so it lands in the panel not nowhere.
    meta_body_blocks = [Block.of(s) for s in meta.body]
    license_footer = _capture_license_footer(meta_body_blocks)
    stripped_body = _strip_running_furniture(meta_body_blocks)
    # Relocate it only if the strip actually dropped it; a footer the strip kept (too
    # few repeats to count as furniture) stays in the body — capturing would duplicate.
    if license_footer is not None and license_footer in stripped_body:
        license_footer = None
    body = [Block.of(s) for s in _merge_split_paragraphs_stable(stripped_body)]
    body = [Block.of(s) for s in _consolidate_numbered_references(body)]
    body = [Block.of(s) for s in _insert_footnotes_before_refs(body, meta.footnotes)]
    leading_metadata_str, body_str = _extract_front_matter([b.html for b in body])
    leading_metadata = [Block.of(s) for s in leading_metadata_str]
    body = [Block.of(s) for s in body_str]
    # A headingless, label-less abstract (Frontiers, Bioscience Reports) the
    # classifier left atop the body — recovered once the leading front matter is gone.
    if not abstract:
        abstract_str, body_str = _recover_headingless_abstract([b.html for b in body])
        abstract = [Block.of(s) for s in abstract_str]
        body = [Block.of(s) for s in body_str]
    # Re-level body section headings the OCR's ##/### jitter left inconsistent, using
    # only high-confidence signals (section numbering, canonical section names) so a
    # real section is never demoted.  Runs last, once the body's heading set is settled.
    body = [Block.of(s) for s in _normalize_heading_levels([b.html for b in body])]
    # A copyright/journal-citation clause the OCR ran onto the abstract's end is front
    # matter; move it to the panel so it doesn't read as abstract prose.
    abstract_citation_str: list[str]
    abstract_str, abstract_citation_str = _split_abstract_citation(
        [b.html for b in abstract]
    )
    abstract = [Block.of(s) for s in abstract_str]
    abstract_citation = [Block.of(s) for s in abstract_citation_str]
    named_metadata_blocks = [Block.of(s) for s in named_metadata]
    stray_metadata_blocks = [Block.of(s) for s in stray_metadata]
    metadata = (
        leading_metadata
        + named_metadata_blocks
        + stray_metadata_blocks
        + abstract_citation
    )
    if license_footer is not None:
        metadata = metadata + [Block.of(license_footer)]

    title_html = _sanitize_content_html(meta.title_html) or "Untitled"
    byline_html = _sanitize_content_html(meta.byline_html)
    html = _document_shell(
        title_html=title_html,
        byline_html=byline_html,
        abstract=_abstract_section(
            [_sanitize_content_html(part.html) for part in abstract]
        ),
        metadata=_metadata_panel(
            [_sanitize_content_html(part.html) for part in metadata]
        ),
        body="\n".join(_sanitize_content_html(part.html) for part in body),
    )
    # markdown-it entity-escapes the header HTML (`&` → `&amp;`), so unescape the
    # tag-stripped text back to the reader-visible title/byline a consumer stores.
    # Derived from the sanitized HTML so a stripped <script>'s inner text can't leak
    # into the plain-text title/byline the Hub stores even though the rendered HTML
    # dropped it.
    title = _html.unescape(_visible_text(title_html)).strip()
    byline = _html.unescape(_visible_text(byline_html)).strip()
    return html, title, byline


def _assemble_html(
    pages_md: list[str],
    images: list[Image.Image],
    ocr_region: Callable[[Image.Image], str] | None = None,
    encode_image: ImageSink = _base64_src,
) -> str:
    """The ``str`` view of :func:`_assemble_document` — the document HTML only."""
    return _assemble_document(pages_md, images, ocr_region, encode_image)[0]


@dataclass
class ParsedDocument:
    """A parsed PDF's HTML plus the metadata a consumer needs without re-parsing it.

    ``title``/``byline`` are the reader-visible (tag-stripped, entity-unescaped) forms
    of the header the HTML renders — empty strings when the document carried none.
    ``doi`` is the best-effort article DOI, ``None`` when none was found."""

    html: str
    title: str
    byline: str
    doi: str | None


def _recover_from_text_layer(
    layers: _DocumentLayers,
    pages_md: list[str],
    ocr_region: Callable[[Image.Image], str],
    ocr_regions: Callable[[list[Image.Image]], list[str]],
) -> tuple[list[str], str | None]:
    """Run the post-OCR PDF-text-layer passes over the page markdown, returning the
    recovered markdown and the best-effort DOI.

    Every pass here — plus the DOI scan — reads the PDF text layer via the shared
    ``layers`` (a lazy per-page ``_PageLayer`` cache), which the orchestrator opens
    **once** for the whole pre-OCR-skip → recover span and closes; a page several passes
    localize against is walked char-by-char only once.  The pass order is load-bearing
    (each step's own docstring explains why), so move steps only with that in mind."""
    pages_md = _recover_dropped_tables(layers, pages_md, ocr_regions)
    # Rebuild a two-column table the OCR mangled (off-by-one header, dropped
    # cells) from the deterministic PDF text layer, keeping the OCR's cell
    # formatting.
    pages_md = _repair_tables_from_text_layer(layers, pages_md)
    # Re-apply <strong> to table cells the OCR transcribed plain but that are
    # boldfaced in the PDF (section/total rows).  After the table passes so it
    # bolds whatever table HTML survives (re-OCR'd or rebuilt).
    pages_md = _recover_table_cell_bold(layers, pages_md)
    pages_md = _recover_dropped_figures(layers, pages_md, ocr_region)
    # Recover short tails the OCR truncated, from the PDF text layer.  Runs
    # *after* table/figure recovery so an appended tail can neither feed the
    # table coverage gate's adjacent-token check nor make a figure number look
    # already-emitted.  No-op on a PDF without a usable text layer.
    pages_md = _reconcile_text_layer(layers, pages_md)
    # DOI from the article's first-page text layer (deterministic), else that
    # page's OCR text — both keyed on the same leading-ad skip the assembly uses,
    # so an ad-prefixed PDF's empty page 0 is stepped over.
    start = _leading_pages_to_skip_md(pages_md)
    doi = _extract_doi(
        layers.page_raw_text(start),
        pages_md[start] if start < len(pages_md) else "",
    )
    return pages_md, doi


def _ocr_document_pages(
    images: list[Image.Image], layers: _DocumentLayers, ocr: OcrModel
) -> list[str]:
    """OCR the article pages, skipping the leading image-only ad/cover pages.

    An ad-prefixed PDF (a full-page-image cover before the article) OCRs its cover to
    garbage anyway — 1–2 wasted ~10-24 s round trips.  ``_leading_image_only_pages``
    reads the (shared) text layer to find those pages *before* any OCR, so
    ``_ocr_pages`` is handed only the pages from the first text-bearing one onward.
    Each skipped page gets empty markdown, so ``pages_md`` stays positionally aligned
    with ``images`` (the assembly ``zip``) and with the PDF pages the recovery passes
    index into — the padding keeps every index absolute, so no downstream pass has to
    thread a skip offset.  The empty leading pages are then dropped during assembly by
    ``_leading_pages_to_skip_md`` — a *distinct, coarser* skip that re-scans for the
    first page carrying an article-start heading — just as an OCR'd-but-heading-less
    cover would have been.  A document with no leading ad (``skip`` == 0) OCRs every
    page as before."""
    skip = _leading_image_only_pages(layers)
    return [""] * skip + _ocr_pages(images[skip:], ocr)


def lightonocr_pdf_to_document(
    pdf_path: Path | str,
    *,
    ocr: OcrModel | None = None,
    base_url: str | None = None,
    model: str | None = None,
    image_dir: Path | None = None,
    encode_image: ImageSink | None = None,
) -> ParsedDocument:
    """Parse a PDF with LightOnOCR-2-1B-bbox into HTML plus structured metadata.

    Args:
        pdf_path: Path to the input PDF.
        ocr: Open connection bundle.  ``None`` calls ``load_ocr_model()``.
        base_url: vLLM endpoint root, used only when ``ocr`` is ``None``.
        model: Served model name, used only when ``ocr`` is ``None``.
        image_dir: When given (and ``encode_image`` is not), figure crops are written
            here as sidecar PNGs and referenced by a path relative to its parent (so
            the HTML, written into that parent, links them) instead of inlined as
            base64 — quicker to regenerate and live-editable in a browser.
        encode_image: The image-delivery sink ``(png_bytes, mime) -> src``; pdfparser
            writes the returned value into each figure's ``<img src>``.  ``None``
            (default) inlines the crops as base64 data URIs, or writes sidecar PNGs
            when ``image_dir`` is set.  Pass a sink to store the bytes elsewhere (e.g.
            an asset store) and return a served URL.  Takes precedence over
            ``image_dir``.

    Returns:
        A :class:`ParsedDocument` carrying the HTML, plain-text title/byline, and the
        best-effort DOI.

    Raises:
        OcrUnavailableError: The OCR server was unreachable, unhealthy, or timed out.
        OcrResponseError: The OCR server replied, but the payload was unconsumable.
        PdfInputError: The input PDF could not be opened or rendered.
    """
    owns_ocr = ocr is None
    if ocr is None:
        ocr = load_ocr_model(base_url=base_url, model=model)
    try:
        images = _render_page_images(Path(pdf_path))
        # Single-region re-OCR for figure caption-trimming; batched (concurrent)
        # re-OCR for the document's table crops.
        ocr_region = lambda region: _ocr_page(region, ocr)  # noqa: E731
        ocr_regions = lambda regions: _ocr_pages(regions, ocr)  # noqa: E731
        # The pre-OCR ad-skip and the post-OCR recovery passes both read the PDF text
        # layer and bracket the OCR phase, so one shared _DocumentLayers spans the pair
        # (opened once, not once each); assembly needs no text layer, so it closes
        # before assembly runs.
        with _DocumentLayers.open(pdf_path) as layers:
            pages_md = _ocr_document_pages(images, layers, ocr)
            pages_md, doi = _recover_from_text_layer(
                layers, pages_md, ocr_region, ocr_regions
            )
        if encode_image is None:
            encode_image = (
                _file_image_writer(image_dir) if image_dir is not None else _base64_src
            )
        html, title, byline = _assemble_document(
            pages_md, images, ocr_region, encode_image
        )
        return ParsedDocument(html=html, title=title, byline=byline, doi=doi)
    finally:
        if owns_ocr:
            ocr.close()


def lightonocr_pdf_to_html(
    pdf_path: Path | str,
    *,
    ocr: OcrModel | None = None,
    base_url: str | None = None,
    model: str | None = None,
    image_dir: Path | None = None,
    encode_image: ImageSink | None = None,
) -> str:
    """Parse a PDF to HTML with LightOnOCR-2-1B-bbox — the ``.html`` of
    :func:`lightonocr_pdf_to_document` (which see for the arguments).  Kept as the
    thin string entry point for the CLI and existing callers."""
    return lightonocr_pdf_to_document(
        pdf_path,
        ocr=ocr,
        base_url=base_url,
        model=model,
        image_dir=image_dir,
        encode_image=encode_image,
    ).html
