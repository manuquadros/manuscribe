"""Recover short OCR truncations from the PDF text layer.

LightOnOCR occasionally drops the tail of a text block — most often the last
line above a page's footer, sometimes a phrase it has already emitted earlier in
the document (an autoregressive early-stop).  The dropped run is still present in
the PDF's embedded text layer, so this pass anchors each OCR block's tail in that
layer and re-attaches the short interstitial run up to the next confident anchor.

It is a *second witness*, not a merge: the text layer is unreliable (broken glyph
maps, scrambled order in multi-column layouts, its own furniture), so every
candidate passes a stack of gates tuned to fire only on genuine short tails —
anchor-bounded gap, a length ceiling, a prose ratio, a recurrence-based furniture
filter, a weak-bound ambiguity guard, and a glyph-sanity reject.  Born-digital
PDFs with a usable layer benefit; a scanned PDF (no layer) is a silent no-op.

Stress-measured against the six fixtures by re-injecting 298 short truncations:
~80% of the geometrically recoverable tails are restored, zero wrong splices.
The unrecoverable remainder are blocks the OCR reordered relative to the layer
(figure captions, second-column prose) — out of reach without full reading-order
alignment, and deliberately left for the gates to decline.
"""

import re
import unicodedata
from collections import Counter

import pypdfium2 as pdfium

from manuscribe.pipeline.layers import _DocumentLayers
from manuscribe.pipeline.text import (
    _STRIP_TAGS_RE,
    _TABLE_TAG_RE,
    _looks_like_figure_caption,
    _split_md_blocks,
)

# Tail/head anchors are this many tokens; 6 is distinctive enough that a prose
# anchor rarely collides spuriously in a single page's layer.
_ANCHOR_TOKENS = 6
# A next block shorter than this (a one-word "FUNDING" heading) is a weak bound:
# its lone token can match a word inside the span being recovered.
_MIN_HEAD_TOKENS = 3
# Below this many normalized chars the recovered slice is too little signal to trust
# as a genuine truncated tail (vs an incidental fragment); the floor to _MAX_GAP_CHARS.
_MIN_GAP_CHARS = 12
# A truncation tail is a clause; a larger gap is almost always a block the OCR
# reordered relative to the layer, so decline it rather than splice foreign text.
_MAX_GAP_CHARS = 60
# A genuine prose tail is mostly letters; below majority-alpha the slice is a digit /
# punctuation run (a table-number column, a furniture stamp), not prose — decline it.
_MIN_ALPHA_RATIO = 0.55
# A line recurring on at least this many pages is running furniture.
_FURNITURE_MIN_PAGES = 3

_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)[\d,]*")
_MD_RE = re.compile(r"[*_`#>|]+")
_DASH_RE = re.compile(r"[‐-―−]")
_WS_RE = re.compile(r"\s+")
# Distinct from furniture._DIGITS_RE: that strips a folio digit (``sub("")``) from
# an HTML part; this masks every digit run (``sub("#")``) on a raw layer line.
_DIGIT_RUN_RE = re.compile(r"\d+")
# A figure/table caption label opening a block or a recovered gap.
_LABEL_START_RE = re.compile(r"^(?:Fig(?:ure)?|Table|Scheme)\b", re.I)
# Trailing/inline page furniture that must not be recovered as prose.
_FURNITURE_RE = re.compile(
    r"©|doi\.org|https?://|www\.|downloaded from|licen[sc]ed under|"
    r"creative commons|all rights reserved",
    re.I,
)
# Replacement char + the two noncharacters pdfium emits for an unmapped glyph;
# recovering a run that carries one would import a fresh defect.
_BAD_GLYPHS = "�￾￿"
# Raw layer text is spliced into the *pre-markdown* stream, so its markdown-active
# characters must become entities or markdown-it/_latex_to_html re-read them as
# emphasis/code/links (mirrors latex._MD_EMPHASIS_ESCAPE for the splice path).
_SPLICE_ESCAPE = str.maketrans(
    {
        "*": "&#42;",
        "_": "&#95;",
        "`": "&#96;",
        "[": "&#91;",
        "]": "&#93;",
        "\\": "&#92;",
    }
)


def _norm(s: str) -> str:
    """Normalize OCR markdown (or a furniture line) for anchor matching."""
    s = unicodedata.normalize("NFKC", s)
    s = _IMG_RE.sub(" ", s)
    s = _STRIP_TAGS_RE.sub(" ", s)
    s = _MD_RE.sub(" ", s)
    s = _DASH_RE.sub("-", s)
    return _WS_RE.sub(" ", s).strip()


def _norm_mapped(raw: str) -> tuple[str, list[int]]:
    """Normalize layer text, returning the normalized string and, per normalized
    character, the source index in ``raw``.

    The map lets every boundary located in normalized space (a tail end, a head
    start, a furniture cut) be projected back so the *raw* layer slice — original
    casing, en-dashes, sub/superscripts — is what gets spliced.  The returned map
    has one trailing entry (``len(raw)``) so an end-exclusive bound at the string
    end resolves.

    Deliberately distinct from ``layers._normalize_with_map`` (which shares only
    the index-map shape): that folds to a lossy alnum-only *match key* (NFKD,
    lower-cased, punctuation dropped), whereas this keeps case, dashes, and
    punctuation so the projected raw slice is faithful to splice back."""
    chars: list[str] = []
    src: list[int] = []
    for i, ch in enumerate(raw):
        for c in unicodedata.normalize("NFKC", ch):
            chars.append("-" if _DASH_RE.match(c) else c)
            src.append(i)

    out: list[str] = []
    out_src: list[int] = []
    prev_ws = False
    for c, i in zip(chars, src, strict=True):
        if c.isspace():
            if not prev_ws:
                out.append(" ")
                out_src.append(i)
            prev_ws = True
        else:
            out.append(c)
            out_src.append(i)
            prev_ws = False

    lo, hi = 0, len(out)
    while lo < hi and out[lo] == " ":
        lo += 1
    while hi > lo and out[hi - 1] == " ":
        hi -= 1
    return "".join(out[lo:hi]), out_src[lo:hi] + [len(raw)]


def _layer_furniture_key(line: str) -> str:
    """Key a raw layer line for recurrence so a running head/footer that differs
    only by page number, date, or volume (``…org 1 February 2020 Volume 8``) keys
    consistently.

    Named apart from ``furniture._furniture_key`` (which keys HTML parts and
    *strips* the folio digit) to keep the two recurrence heuristics from being
    mistaken for one — this masks every digit run on a raw text-layer line."""
    return _DIGIT_RUN_RE.sub("#", _norm(line)).lower()


def _recurring_furniture(layer_texts: list[str]) -> set[str]:
    """Lines that recur across pages — running heads/footers the OCR rightly drops."""
    seen: Counter[str] = Counter()
    for raw in layer_texts:
        page_lines = {_layer_furniture_key(ln) for ln in raw.splitlines() if ln.strip()}
        for ln in page_lines:
            if len(ln) >= 6:
                seen[ln] += 1
    return {ln for ln, n in seen.items() if n >= _FURNITURE_MIN_PAGES}


def _is_recoverable_block(block: str) -> bool:
    """A prose block eligible to receive a recovered tail — not an image, a table
    (including a fragment a blank line split off the table), a markdown table row,
    or a figure/table caption (appending to a caption would corrupt the label the
    figure/table passes match on)."""
    if block.startswith(("![", "|")):
        return False
    if _TABLE_TAG_RE.search(block):
        return False
    if _LABEL_START_RE.match(_norm(block)):
        return False
    return not _looks_like_figure_caption(block)


def _alpha_ratio(s: str) -> float:
    dense = s.replace(" ", "")
    return sum(c.isalpha() for c in dense) / max(1, len(dense))


def _is_furniture(gap: str, furniture: set[str]) -> bool:
    g = _layer_furniture_key(gap)
    return any(g in f or f in g for f in furniture)


def _splits_source_char(b: int, src: list[int]) -> bool:
    """Whether normalized boundary ``b`` falls *inside* a source char's NFKC
    expansion (a 1→many fold maps every expanded char to the same source index),
    where a raw slice cannot represent the boundary faithfully."""
    return 0 < b < len(src) and src[b] == src[b - 1]


def _recover_tail(
    block: str,
    next_block: str | None,
    layer: str,
    low: str,
    src: list[int],
    raw: str,
    furniture: set[str],
) -> str | None:
    """Return the raw layer run that continues ``block``, or ``None`` to decline.

    ``block``/``next_block`` are already normalized; ``layer``/``low`` are the
    normalized layer and its lower-cased twin; ``src`` maps ``layer`` indices back
    to ``raw``."""
    tokens = block.split(" ")
    if len(tokens) <= _ANCHOR_TOKENS:
        return None
    anchor = " ".join(tokens[-_ANCHOR_TOKENS:]).lower()
    # An anchor that repeats on the page is ambiguous — we cannot tell which
    # occurrence is this block, so decline rather than splice the wrong tail.
    if low.count(anchor) != 1:
        return None
    end = low.index(anchor) + len(anchor)

    # A weak (short) next block is a collision-prone bound — trust it only when its
    # anchor occurs exactly once (as a whole word) in the gap window; a substring
    # count would mis-handle "project" inside "projection".  Otherwise run to the
    # page end.
    if next_block is not None and len(next_block.split(" ")) < _MIN_HEAD_TOKENS:
        weak = " ".join(next_block.split(" ")[:_ANCHOR_TOKENS]).lower()
        window = low[end : end + 2 * _MAX_GAP_CHARS]
        pat = rf"(?<!\w){re.escape(weak)}(?!\w)"
        hits = len(re.findall(pat, window)) if weak else 0
        if hits != 1:
            next_block = None

    if next_block is not None:
        head_anchor = " ".join(next_block.split(" ")[:_ANCHOR_TOKENS]).lower()
        head = low.find(head_anchor, end)
        if head <= end:
            return None
    else:
        head = len(layer)

    furn = _FURNITURE_RE.search(layer, end, head)
    cut = furn.start() if furn else head
    if _splits_source_char(end, src) or _splits_source_char(cut, src):
        return None
    return _validate_gap_splice(layer, end, cut, src, raw, furniture)


def _validate_gap_splice(
    layer: str,
    end: int,
    cut: int,
    src: list[int],
    raw: str,
    furniture: set[str],
) -> str | None:
    """Gate the recovered ``[end:cut]`` gap and return the escaped raw slice, or
    ``None`` to decline.

    Gate on a fully-cleaned view (the offset-mapped ``layer`` keeps markdown
    punctuation like ``|`` that ``_norm`` — and the furniture set — strip), but
    splice the faithful raw slice."""
    gap_clean = _norm(layer[end:cut])
    if not (_MIN_GAP_CHARS <= len(gap_clean) <= _MAX_GAP_CHARS):
        return None
    if _alpha_ratio(gap_clean) < _MIN_ALPHA_RATIO:
        return None
    if _LABEL_START_RE.match(gap_clean):
        return None
    if _is_furniture(gap_clean, furniture):
        return None

    gap_raw = raw[src[end] : src[cut]].strip()
    if any(c in gap_raw for c in _BAD_GLYPHS):
        return None
    return gap_raw.translate(_SPLICE_ESCAPE) or None


def _reconcile_page(page_md: str, layer_raw: str, furniture: set[str]) -> str:
    """Splice recovered tails into one page's markdown (pure; no IO)."""
    if not layer_raw.strip():
        return page_md
    blocks = _split_md_blocks(page_md)
    layer, src = _norm_mapped(layer_raw)
    low = layer.lower()

    out = list(blocks)
    for i, block in enumerate(blocks):
        if not _is_recoverable_block(block):
            continue
        # The bound is the immediately-following block of any kind (a table or
        # caption bounds the gap tightly); only the *target* must be prose.
        nxt = _norm(blocks[i + 1]) if i + 1 < len(blocks) else None
        tail = _recover_tail(_norm(block), nxt, layer, low, src, layer_raw, furniture)
        if tail:
            out[i] = f"{block.rstrip()} {tail}"
    return "\n\n".join(out)


def _reconcile_text_layer(layers: _DocumentLayers, pages_md: list[str]) -> list[str]:
    """Recover short OCR truncations across a document; one entry per input page.

    A born-digital PDF's text layer is read once per page, running furniture is
    learned from cross-page recurrence, and each page is reconciled against its own
    layer.  Pages whose layer is empty (scanned, or no text) pass through unchanged.
    Order and length match ``pages_md``.

    Reads the cheap text-view (``get_text_range``) per page rather than the cached
    ``_PageLayer`` (the char-by-char walk): reconciliation runs on every document and
    needs only the raw text, so forcing full-layer extraction of every page here would
    cost far more than it saves."""
    page_texts = [_page_raw_text(layers.pdf, p) for p in range(len(pages_md))]
    furniture = _recurring_furniture(page_texts)
    return [
        _reconcile_page(md, layer, furniture)
        for md, layer in zip(pages_md, page_texts, strict=True)
    ]


def _page_raw_text(pdf: pdfium.PdfDocument, p: int) -> str:
    """The page's text-view layer, closing the native text-page handle (pdfium's
    ``PdfTextPage`` has no context-manager protocol, so close explicitly)."""
    if p >= len(pdf):
        return ""
    textpage = pdf[p].get_textpage()
    try:
        return str(textpage.get_text_range())
    finally:
        textpage.close()
