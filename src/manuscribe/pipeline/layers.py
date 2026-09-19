"""PDF text-layer extraction and the geometry/text-folding primitives it shares
with table localization.

This is the foundational leaf the four post-OCR text-layer passes build on —
table recovery/repair (``tables``), figure recovery (``recover_figures``) and
reconciliation (``reconcile``).  It deliberately imports nothing from the rest of
the pipeline so it can sit *under* ``tables`` (which used to host this code and so
became a false hub for the three passes that only wanted the cache): the dependency
now runs one way, ``tables``/``reconcile``/``recover_figures``/``assemble`` →
``layers``.

The cache (:class:`_DocumentLayers`) holds one open ``PdfDocument`` and a lazy
per-page :class:`_PageLayer`, so a page several passes localize against is walked
char by char only once."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — beartype reads annotations at runtime

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

_Box = tuple[float, float, float, float]  # PDF points: left, bottom, right, top

# A leading page whose text layer holds fewer than this many alphanumeric characters
# is treated as image-only furniture (an ad/cover) rather than article content: a real
# article page carries ~1500+ alnum chars in its text layer, while a full-page-image
# cover carries ~0 (see ``_leading_image_only_pages``).  Deliberately well below any
# real page — the skip must never drop content, so a page with even a sentence of text
# (a graphical-abstract caption, a masthead) stays above the bar and is left to the OCR
# and the model's own article-start detection.
_MIN_TEXT_LAYER_ALNUM = 32


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Fold text to matchable form and map each output char to its source index.

    NFKD plus alnum-only (everything else collapses to a single space) erases the
    encoding gaps between the OCR'd cell and the text layer — superscripts
    (``²⁺`` → ``2``), the micro sign (``µ`` vs Greek ``μ``), and the assorted
    dashes NFKD leaves as a U+2212 minus where the text layer has ASCII ``-``.
    The index map lets a match in the folded string recover the original char
    range, and thus the glyph boxes."""
    out: list[str] = []
    idx_map: list[int] = []
    prev_space = True
    for i, ch in enumerate(text):
        for d in unicodedata.normalize("NFKD", ch):
            if d.isalnum():
                out.append(d.lower())
                idx_map.append(i)
                prev_space = False
            elif not prev_space:
                out.append(" ")
                idx_map.append(i)
                prev_space = True
    while out and out[-1] == " ":
        out.pop()
        idx_map.pop()
    return "".join(out), idx_map


def _normalize(text: str) -> str:
    return _normalize_with_map(text)[0]


def _snap_rotation(angle_deg: float) -> int:
    """Snap a glyph rotation to the nearest quarter turn (0/90/180/270)."""
    return round(angle_deg / 90) % 4 * 90


def _page_text_and_boxes(
    textpage: pdfium.PdfTextPage,
) -> tuple[str, list[_Box | None], list[int | None], list[int | None]]:
    """Page text paired with one glyph box, rotation and font weight per character,
    index-aligned.

    Built char by char in the same index domain so that position *p* in the
    returned text always indexes ``boxes[p]``, ``rotations[p]`` and ``weights[p]``.
    ``get_text_range()`` (the text view) and ``count_chars()`` (the char-array)
    disagree on real PDFs — pdfium drops or inserts characters between the two — so
    deriving the text from one and the boxes from the other would silently misalign
    every glyph past the first dropped char, locating the table from the wrong
    region.

    The rotation (snapped to a quarter turn) is what lets a sideways table — laid
    out at 90°/270° on the page — be localized along its true reading axis and the
    crop turned upright before re-OCR; ``FPDFText_GetCharAngle`` returns the glyph's
    counter-clockwise rotation in radians (a negative value signals an error).

    The font weight (CSS-style: 400 normal, 700 bold; ``FPDFText_GetFontWeight``
    returns ``-1`` on error → ``None``) backs the table-cell bold recovery, which
    re-applies ``<strong>`` to cells the OCR transcribed as plain text."""
    parts: list[str] = []
    boxes: list[_Box | None] = []
    rotations: list[int | None] = []
    weights: list[int | None] = []
    raw = textpage.raw
    for i in range(textpage.count_chars()):
        ch = textpage.get_text_range(i, 1)
        try:
            left, bottom, right, top = textpage.get_charbox(i)
            box: _Box | None = (
                (left, bottom, right, top) if right > left and top > bottom else None
            )
        except Exception:
            # Broad on purpose: this runs per glyph over thousands of chars, and a
            # single malformed box (any pdfium quirk) must drop that glyph's box, not
            # abort the page's extraction — the char keeps its index with box=None.
            box = None
        angle = pdfium_c.FPDFText_GetCharAngle(raw, i)
        rot = _snap_rotation(math.degrees(angle)) if angle >= 0 else None
        weight = pdfium_c.FPDFText_GetFontWeight(raw, i)
        parts.append(ch)
        # a glyph may decode to several text chars — replicate its metadata per char
        boxes.extend([box] * len(ch))
        rotations.extend([rot] * len(ch))
        weights.extend([weight if weight >= 0 else None] * len(ch))
    return "".join(parts), boxes, rotations, weights


@dataclass(frozen=True)
class _PageLayer:
    """A page's text layer extracted once: the raw text + per-char geometry, plus the
    normalized text and its index map back to the raw stream.

    Building it walks the page char by char through pdfium's native API (a
    ``get_text_range`` + ``get_charbox`` + ``FPDFText_GetCharAngle`` +
    ``FPDFText_GetFontWeight`` round-trip each — thousands of ctypes calls), so the
    localization passes share one instance per page rather than re-extracting it per
    table."""

    page_text: str
    char_boxes: list[_Box | None]
    char_rotations: list[int | None]
    char_weights: list[int | None]
    norm: str
    idx_map: list[int]


def _page_layer(page: pdfium.PdfPage) -> _PageLayer:
    """Extract ``page``'s text layer once (see ``_PageLayer``), closing the native
    text page handle rather than leaking it to GC."""
    textpage = page.get_textpage()
    try:
        page_text, char_boxes, char_rotations, char_weights = _page_text_and_boxes(
            textpage
        )
    finally:
        textpage.close()
    norm, idx_map = _normalize_with_map(page_text)
    return _PageLayer(
        page_text, char_boxes, char_rotations, char_weights, norm, idx_map
    )


class _DocumentLayers:
    """One open ``PdfDocument`` plus a lazy per-page :class:`_PageLayer` cache.

    The post-OCR text-layer passes — table recovery/repair (``tables``), figure
    recovery (``recover_figures``) and reconciliation (``reconcile``) — each used to
    re-open the PDF and re-extract a page's text layer independently.  Holding one
    document open across the whole phase and memoizing each page's ``_PageLayer`` means
    a page several passes touch is extracted once, not once per pass (each extraction
    is the thousands-of-native-calls char-by-char walk in :func:`_page_layer`).

    The cache is lazy: ``page_layer`` triggers extraction only when a pass actually
    needs a page's geometry, so a page no pass localizes against (e.g. one with no
    table and no missing figure) is never walked.  Used as a context manager so the
    native document handle is released; ownership of the passed-in document transfers
    to the cache (``__exit__`` closes it)."""

    def __init__(self, pdf: pdfium.PdfDocument) -> None:
        self.pdf = pdf
        self._layers: dict[int, _PageLayer] = {}

    @classmethod
    def open(cls, pdf_path: Path | str) -> _DocumentLayers:
        return cls(pdfium.PdfDocument(str(pdf_path)))

    def __enter__(self) -> _DocumentLayers:
        return self

    def __exit__(self, *exc: object) -> None:
        self.pdf.close()

    def __len__(self) -> int:
        return len(self.pdf)

    def page_layer(self, index: int) -> _PageLayer:
        """The page's text layer, extracted on first request and cached thereafter."""
        layer = self._layers.get(index)
        if layer is None:
            layer = _page_layer(self.pdf[index])
            self._layers[index] = layer
        return layer

    def page_raw_text(self, index: int) -> str:
        """The page's text-view layer — raw text only, no per-glyph geometry.

        The cheap native read (``get_text_range``), not the char-by-char
        :class:`_PageLayer` walk: used where a pass needs only a page's text (the DOI
        scan) and not its glyph boxes.  Out-of-range pages return ``""``.  Not cached
        (callers want it for a page or two), and the native text-page handle is closed
        explicitly — pypdfium2's ``PdfTextPage`` has no context-manager protocol."""
        if not 0 <= index < len(self.pdf):
            return ""
        textpage = self.pdf[index].get_textpage()
        try:
            return str(textpage.get_text_range())
        finally:
            textpage.close()


def _is_trivial_text_layer(page_text: str) -> bool:
    """Whether a page's text layer is essentially empty — an image-only page.

    Keyed on the alphanumeric-character count (not raw length) so whitespace, form
    feeds and stray punctuation a scanner may leave don't read as content.  A page
    below ``_MIN_TEXT_LAYER_ALNUM`` alnum chars carries no extractable text worth
    OCRing (its OCR would be a garbled transcription of a picture)."""
    return sum(ch.isalnum() for ch in page_text) < _MIN_TEXT_LAYER_ALNUM


def _leading_image_only_count(trivial: list[bool]) -> int:
    """Length of the leading run of image-only (empty-text-layer) pages.

    ``trivial[i]`` is whether page *i*'s text layer is essentially empty.  The index of
    the first text-bearing page is the count of leading empty pages to skip (never an
    interior page — mirrors ``_leading_pages_to_skip_md``'s leading-run semantics, and
    is the same ``next((i for …), 0)`` idiom).  The default-0 covers the all-empty case:
    a fully-scanned paper has no usable text layer anywhere, so there is nothing to
    distinguish a cover from the article and the model must OCR it all (skip 0)."""
    return next((i for i, t in enumerate(trivial) if not t), 0)


def _leading_image_only_pages(layers: _DocumentLayers) -> int:
    """Count the leading image-only pages that can be skipped before OCR.

    A full-page-image ad/cover before the article (e.g. ``31051047.pdf``) has an empty
    text layer and OCRs to garbage the leading-page skip discards anyway — one or two
    wasted ~10-24 s round trips.  Reading the (shared) text layer here decides, *before*
    any OCR, which leading pages needn't be sent at all.  Stops at the first
    text-bearing page (the leading run can't extend past it), so a normal document costs
    a single cheap text-view read; a *fully* image-only document (a scanned paper) has
    no such page, so it reads every page and — via ``_leading_image_only_count``'s
    all-empty guard — skips nothing.

    **Tradeoff (deliberate).** The empty-text-layer signal cannot tell an ad/cover from
    a *scanned* leading page that is genuine article content: a mixed document whose
    leading page is an image carrying the real title/abstract, followed by text-bearing
    pages, would have that page skipped and dropped (its image is never OCR'd).
    Accepted because it is vanishingly rare — a modern article's first page carries a
    text layer — and the only alternative is to OCR every page to find out, which is
    exactly the cost this skip exists to avoid; the all-empty guard still protects a
    wholly-scanned paper.  Do **not** widen the skip to a text-*bearing* page — deciding
    ad-vs-article there stays the model's call via ``_is_article_page_md``."""
    trivial: list[bool] = []
    for i in range(len(layers)):
        is_trivial = _is_trivial_text_layer(layers.page_raw_text(i))
        trivial.append(is_trivial)
        if not is_trivial:
            break  # first text-bearing page ends the leading run — read no further
    return _leading_image_only_count(trivial)
