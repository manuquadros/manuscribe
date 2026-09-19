"""Figure extraction: parse LightOnOCR's bbox placeholders, recover the true
crop from the rendered page, merge over-segmented boxes, and emit ``<figure>``.

Pure image-processing, no GPU.  All boxes are LightOnOCR's ``[0, 1000]``-normalized
``x0,y0,x1,y1`` until ``_denormalize_bbox`` scales them to page pixels.

Design bias: a clipped figure loses image the reader never recovers, while a crop
that runs into surrounding margin is merely cosmetic, so when the model's box and
the real figure disagree this module leans toward **including too much rather than
too little** — it grows a clipped box out to the figure's true edges on all four
sides (``_extend_edge``) and unions over-segmented stacked boxes back into one
(``_cluster_figure_boxes``).  The one
check on that bias is ambiguity: growth stops (rather than guessing) when the
figure's end isn't marked by a clear whitespace gap, so a correct box is never
extended blindly into caption, body text, or a neighbouring column.  See the
"Figures" section of the design notes.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from collections.abc import Callable  # noqa: TC003 — beartype reads annotations
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — beartype reads annotations at runtime

import numpy as np
from PIL import Image  # noqa: TC002 — beartype reads annotations at runtime

from manuscribe.pipeline.markdown import _caption_inner_html

_log = logging.getLogger(__name__)

# LightOnOCR-bbox emits figures as a markdown image placeholder with the crop
# box appended as bare ``x0,y0,x1,y1`` integers **normalized to [0, 1000]** (per
# the model card), e.g. ``![image](image_1.png)122,89,877,614``.  The base
# variant omits the coordinates, so they are optional.
_BBOX_NORM_MAX = 1000
_FIGURE_PLACEHOLDER_RE = re.compile(
    r"^!\[[^\]]*\]\([^)]*\)"
    r"(?:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+))?\s*$"
)
# A figure label the model emitted as its own block with no caption sentence after
# the number ("FIG. 2", "Figure 3.", "**Fig 4**").  When the descriptive caption
# arrives as the *following* block, it must be rejoined onto this label — otherwise
# the figure owns only the label, the caption is stranded in the body, and the
# baked-caption trim never receives the words it needs to recognise the caption.
_BARE_FIGURE_LABEL_RE = re.compile(
    r"^\*{0,2}\s*fig(?:ure|\.|\b)\s*\.?\s*\d+[a-z]?\s*[.:]?\s*\*{0,2}\s*$",
    re.IGNORECASE,
)
# A single-letter panel label ("A", "(B)", "C.") the model split out of a
# multi-panel figure as its own text block.  It belongs to the figure (baked into
# the crop), not the prose, so it is dropped when adjacent to a figure placeholder.
_PANEL_LABEL_RE = re.compile(r"^\(?[A-Z]\)?[.:]?$")
# A multi-panel caption continuation the model split into its own paragraph,
# opening with a parenthesised panel label and its description ("(A) Gene clusters
# … (B) … (C) …").  This is caption text the model detached from the "Figure N …"
# header, not body prose — body prose effectively never opens "(A) ".  Capital-only
# so a lowercase roman enumeration in prose ("(i) … (ii) …") is not mistaken for it.
_PANEL_DESC_RE = re.compile(r"^\([A-Z]\)\s")

_MIN_FIGURE_HEIGHT = 50  # pixels — gaps smaller than this are not figures
# The model's box often clips a figure's bottom or right edge.  Rather than pad
# blindly (which grabs the caption, or a neighbouring column, when the box was
# already correct), grow the clipped edge over figure content and stop at the
# whitespace gap beyond it — the gap before the caption below, or the page margin
# / inter-column gutter to the right.  A line (row or column) is *blank* when
# fewer than this fraction of its pixels are ink — kept low (≈empty) so a sparse
# figure line (a thin axis, or content narrower than the box) still counts as
# content rather than ending growth early.  A run of blank lines this deep is the
# gap that ends growth; growth is capped at this fraction of the page dimension.
_FIGURE_BLANK_LINE_FRAC = 0.005
_FIGURE_INK_LEVEL = 250  # pixel value below which a grayscale pixel is "ink"
_FIGURE_GAP_FRAC = 0.012
_FIGURE_MAX_GROW_FRAC = 0.10
# Growth recovers only figure content *contiguous* with the box: a leading run of
# this many blank lines before any ink means the box already ends at the figure
# boundary, and what follows — a caption below, a neighbouring column to the right
# — is separated from it by whitespace, so growth is declined rather than reaching
# across the gap to pull it in.  Kept well below the trailing-gap threshold so a
# tight figure-to-caption margin still reads as a boundary (the clip that motivated
# this leaves no leading gap at all — the box cuts straight through figure ink).
_FIGURE_LEAD_GAP_FRAC = 0.004
# When the OCR emitted a caption block for a figure, a recovered bottom band that
# is actually that caption (the model boxed the figure correctly but growth ran
# into the prose below) is trimmed back.  The band is judged as a whole by its mean
# horizontal ink-run length normalized to the band width: caption prose is short
# letter runs, while figure content growth legitimately recovers here (shaded
# panels, gel lanes, sequence alignments) is long continuous runs.  This threshold
# sits at the midpoint of the observed gap — caption bands measured ≤ 0.046, figure
# bands ≥ 0.10.  Sparse line-art (thin axes, tick labels) shares prose's short runs,
# so a clipped sparse tail sitting just above its caption can be trimmed with it —
# an accepted minor loss: the caption is always re-emitted as <figcaption>, and the
# alternative (baking a whole caption into the image) is worse.
_FIGURE_PROSE_RUN_FRAC = 0.07
# When a figure is itself text — a sequence alignment, a data table — its caption
# is pixel-identical to it, so the run-length test above can't find a caption the
# model baked *inside* the box.  The tie-breaker is the document, not the pixels:
# the OCR already emitted that caption as its own text block, so a trailing band is
# the caption when re-OCRing it reproduces caption words.  ``_trim_baked_caption``
# only runs on text-like (not dense-figure) trailing bands and only in the bottom
# of the crop, so the extra OCR is bounded to genuinely doubtful cases.  A band is
# the caption when at least this fraction of its words appear in the caption text.
# Kept high: a caption line re-OCRs to ~all caption words, whereas a figure row that
# the caption happens to label (a sequence alignment names its own rows — BsSDH,
# HmSDH …) is one matching label among non-caption data, well under this bar.
_FIGURE_CAPTION_WORD_FRAC = 0.7
_FIGURE_CAPTION_MIN_WORDS = 3  # ignore one/two-word figure labels that match by luck
# A figure the model can't read (a sequence alignment) can OCR into a repeated-token
# wall — e.g. "BMSDH BMSDH BMSDH …" — which scores ~1.0 against a caption that names
# that row.  Reject it by requiring this many *distinct* caption words in the band:
# real caption prose carries several distinct caption words, a wall (of any length)
# collapses to one or two types, and a short repeat ("panel panel panel") likewise.
_FIGURE_CAPTION_MIN_DISTINCT = 3
# only hunt a baked caption in the bottom half of a crop
_FIGURE_CAPTION_SCAN_FRAC = 0.5
# A caption is a minority of a figure; if trimming would leave less than this
# fraction of the figure, the "caption" bands were really figure content that
# echoes the caption words (a GC-MS trace labelled with the retention times its
# caption lists, a sequence alignment naming its own rows) — reject the trim and
# keep the full crop rather than annihilate a real figure.  Its own literal (not an
# alias of the scan-window fraction): a keep-ratio is an unrelated knob that happens
# to share the 0.5 value, so tuning one must not move the other.
_FIGURE_CAPTION_MIN_KEEP_FRAC = 0.5
_FIGURE_CAPTION_NOTE_BANDS = 2  # trailing note lines (a DOI) tolerated below a caption
_FIGURE_BLOCK_GAP_FRAC = 0.008  # blank run separating caption / figure / note blocks
_FIGURE_OCR_MIN_BAND_PX = 48  # pad thinner bands so the vision model can patch them
_WORD_RE = re.compile(r"[a-z0-9]+")
_FIGURE_NOTE_RE = re.compile(r"doi\.org|https?://", re.IGNORECASE)
# The model sometimes over-segments one figure into stacked boxes (a tall figure
# split into a main box + a thin strip).  Two boxes are the same figure when they
# share a column (substantial horizontal overlap) and are vertically adjacent.
_FIGURE_MERGE_GAP_FRAC = 0.03  # of page height


@dataclass(frozen=True)
class _FigurePlaceholder:
    """The classification of a markdown line as a figure placeholder.

    ``is_placeholder`` is ``False`` only when the line is not a placeholder at all.
    ``bbox_norm`` is the ``[0, 1000]``-normalized ``(x0, y0, x1, y1)`` crop box, or
    ``None`` for a bbox-less placeholder (the base-variant fallback).
    """

    is_placeholder: bool
    bbox_norm: tuple[int, int, int, int] | None


def _parse_figure_placeholder(line: str) -> _FigurePlaceholder:
    """Classify a single markdown line as a figure placeholder.

    Returns the crop box when the placeholder carries one, a bbox-less result for
    the base-variant fallback, or a non-placeholder result when the line is not a
    figure placeholder at all.
    """
    m = _FIGURE_PLACEHOLDER_RE.match(line.strip())
    if m is None:
        return _FigurePlaceholder(False, None)
    if m.group(1) is None:
        return _FigurePlaceholder(True, None)
    bbox = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
    return _FigurePlaceholder(True, bbox)


def _is_bare_figure_label(block: str) -> bool:
    """True when ``block`` is a figure label with no caption sentence after it."""
    return bool(_BARE_FIGURE_LABEL_RE.match(block.strip()))


def _is_panel_label(block: str) -> bool:
    """True when ``block`` is a lone single-letter panel label ("A", "(B)")."""
    return bool(_PANEL_LABEL_RE.match(block.strip()))


def _opens_with_panel_label(block: str) -> bool:
    """True when ``block`` opens with a panel label and its description ("(A) …") —
    a multi-panel caption continuation the model split off the "Figure N" header."""
    return bool(_PANEL_DESC_RE.match(block.strip()))


# The image-delivery seam: a figure crop, already encoded to image bytes, plus its
# MIME type → the value manuscribe writes into ``<img src>``.  Inline base64
# (:func:`_base64_src`) is the self-contained default; a sidecar-file writer
# (:func:`_file_image_writer`) or a caller-supplied sink (e.g. one that stores the
# bytes in an asset store and returns its served URL) plug in the same way.  The
# crop→PNG encode happens once at the call site (:func:`_figure_html`), so a sink
# never touches Pillow.
ImageSink = Callable[[bytes, str], str]
"""The image-delivery seam — ``(png_bytes, mime) -> src``.

Given a figure crop's encoded PNG bytes and its MIME type, returns the string written
into the figure's ``<img src>`` — an inline ``data:`` URI (:func:`_base64_src`), a
sidecar-PNG path (:func:`_file_image_writer`), or a served URL from a caller-supplied
store.  The crop is encoded to PNG once, in :func:`_figure_html`, so a sink never
touches Pillow.
"""


def _base64_src(image_bytes: bytes, mime: str) -> str:
    """Encode image bytes as an inline ``data:`` URI (the self-contained default)."""
    return f"data:{mime};base64," + base64.b64encode(image_bytes).decode()


def _file_image_writer(image_dir: Path) -> ImageSink:
    """Return an :data:`ImageSink` that writes each crop as a PNG into ``image_dir``
    and references it by a path relative to ``image_dir``'s parent.

    So when the HTML is written into that parent directory it links the sidecar
    PNGs (quick to regenerate, live-editable in a browser) instead of inlining a
    base64 data URI.  The counter is per-document, shared across pages."""
    image_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    def encode(image_bytes: bytes, mime: str) -> str:
        nonlocal count
        count += 1
        name = f"fig_{count:03d}.png"
        (image_dir / name).write_bytes(image_bytes)
        return f"{image_dir.name}/{name}"

    return encode


def _figure_html(
    crop: Image.Image,
    caption_text: str | None,
    encode_src: ImageSink = _base64_src,
) -> str:
    """Return a ``<figure>`` element; ``encode_src`` turns the crop's PNG bytes into
    the ``<img src>`` (an inline data URI by default, a sidecar PNG path with
    :func:`_file_image_writer`)."""
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    src = encode_src(buf.getvalue(), "image/png")
    caption_html = (
        f"<figcaption>{_caption_inner_html(caption_text)}</figcaption>"
        if caption_text
        else ""
    )
    return f'<figure><img src="{src}" alt="">{caption_html}</figure>'


def _denormalize_bbox(
    bbox: tuple[int, int, int, int], image: Image.Image
) -> tuple[int, int, int, int]:
    """Scale a ``[0, 1000]``-normalized box to ``image``'s pixel coordinates."""
    w, h = image.size
    return (
        round(bbox[0] / _BBOX_NORM_MAX * w),
        round(bbox[1] / _BBOX_NORM_MAX * h),
        round(bbox[2] / _BBOX_NORM_MAX * w),
        round(bbox[3] / _BBOX_NORM_MAX * h),
    )


def _ink_run_end(ink_per_line: np.ndarray, gap: int, lead_gap: int) -> int | None:
    """Offset (1-based, past the edge) of the last inked line before the first
    whitespace gap of ``gap`` blank lines in a 1-D ink profile, or ``None`` to
    leave the edge unchanged.

    ``None`` is returned both when the ink reaches the search cap with no trailing
    gap (ambiguous) and when a *leading* gap of ``lead_gap`` blank lines precedes
    any ink: content the box is already separated from by whitespace is the
    caption / neighbouring column, not a clipped continuation of the figure, so it
    is not pulled in.
    """
    last_ink = 0
    blank_run = 0
    seen_ink = False
    for offset, fraction in enumerate(ink_per_line, start=1):
        if fraction < _FIGURE_BLANK_LINE_FRAC:
            blank_run += 1
            if not seen_ink and blank_run >= lead_gap:
                return None
            if blank_run >= gap:
                return last_ink
        else:
            seen_ink = True
            blank_run = 0
            last_ink = offset
    return None


def _grow_edge(
    ink_profile: np.ndarray, dim: int, gap_frac: float = _FIGURE_GAP_FRAC
) -> int:
    """Offset to extend an edge along ``ink_profile`` (the ink profile of the strip
    past the edge), or 0 to leave it; the gap thresholds scale to page ``dim``.

    ``gap_frac`` sizes the trailing whitespace run that ends growth — the caller
    passes a smaller one (``_FIGURE_BLOCK_GAP_FRAC``) when a caption follows so
    growth halts at the figure↔caption gap instead of stepping over it."""
    run = _ink_run_end(
        ink_profile,
        max(1, round(gap_frac * dim)),
        max(1, round(_FIGURE_LEAD_GAP_FRAC * dim)),
    )
    return run if run is not None else 0


def _grow_edge_offset(
    image: Image.Image,
    box: tuple[int, int, int, int],
    dim: int,
    *,
    axis: int,
    reverse: bool,
    gap_frac: float = _FIGURE_GAP_FRAC,
) -> int:
    """Non-negative offset to grow one edge over the contiguous ink in ``box``.

    Crops the strip past the edge, reduces it to a 1-D ink profile (mean over
    ``axis``), reverses it when growing toward 0 (the left edge) so the scan always
    runs outward from the edge, and returns the run length before the whitespace
    gap.  The three ``_extend_*`` wrappers clamp the search ``box`` and apply the
    direction; this holds the shared strip→profile→``_grow_edge`` body."""
    strip = np.asarray(image.crop(box).convert("L"))
    profile = (strip < _FIGURE_INK_LEVEL).mean(axis=axis)
    if reverse:
        profile = profile[::-1]
    return _grow_edge(profile, dim, gap_frac)


# Per-edge parameters for _extend_edge: the box-coordinate index the edge moves
# (x0,y0,x1,y1), the array axis to reduce when building the ink profile (0 = over
# rows → a per-column profile for the vertical left/right edges; 1 = over columns →
# a per-row profile for the horizontal top/bottom edges), and whether the edge
# grows toward 0 (top/left, which also reverse the profile so the scan runs outward
# from the edge) or away from it (bottom/right).
_EDGE_PARAMS: dict[str, tuple[int, int, bool]] = {
    "left": (0, 0, True),
    "right": (2, 0, False),
    "top": (1, 1, True),
    "bottom": (3, 1, False),
}


def _extend_edge(
    image: Image.Image,
    box: tuple[int, int, int, int],
    edge: str,
    gap_frac: float = _FIGURE_GAP_FRAC,
) -> int:
    """Grow one edge of ``box`` outward over figure ink and return its new
    coordinate, or the edge unchanged when growth is declined.

    ``edge`` selects which side moves (``left``/``right``/``top``/``bottom``).
    Growth recovers a figure edge the model's box clipped — a clipped bottom, a
    wide figure's right edge, a plot's left axis line, a top panel label or frame
    line — and stops at the whitespace gap beyond it: the space before a caption
    below, the page margin, or the inter-column gutter (so a column figure never
    reaches its neighbour).  It is declined (edge unchanged) when the ink runs to
    the search cap with no trailing gap (ambiguous) or a leading gap already
    separates the box from what lies beyond (a caption / adjacent column / prose
    above), so no text is pulled in.  Growth is capped at ``_FIGURE_MAX_GROW_FRAC``
    of the page dimension; the caller tightens ``gap_frac`` for a bottom edge with a
    caption below so growth halts at the figure↔caption gap rather than stepping
    past it.
    """
    coord_i, axis, toward_zero = _EDGE_PARAMS[edge]
    coord = box[coord_i]
    dim = image.size[axis]
    grow_max = round(_FIGURE_MAX_GROW_FRAC * dim)
    if toward_zero:
        limit = max(0, coord - grow_max)
        if coord <= limit:
            return coord
        lo, hi = limit, coord
    else:
        limit = min(dim, coord + grow_max)
        if coord >= limit:
            return coord
        lo, hi = coord, limit
    x0, y0, x1, y1 = box
    strip = (lo, y0, hi, y1) if axis == 0 else (x0, lo, x1, hi)
    offset = _grow_edge_offset(
        image, strip, dim, axis=axis, reverse=toward_zero, gap_frac=gap_frac
    )
    return coord - offset if toward_zero else coord + offset


def _mean_norm_run_length(mask: np.ndarray) -> float:
    """Mean length of horizontal ink runs over a boolean ink ``mask``'s inked
    rows, normalized by mask width; 0 when no row carries ink.  Short runs mark
    prose (letterforms); long runs mark shaded / continuous figure content."""
    width = mask.shape[1]
    if width == 0:
        return 0.0
    means: list[float] = []
    for row in mask:
        if row.mean() <= _FIGURE_BLANK_LINE_FRAC:
            continue
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))))
        runs = edges[1::2] - edges[::2]
        if runs.size:
            means.append(float(runs.mean()))
    return float(np.mean(means) / width) if means else 0.0


def _trim_swallowed_caption(
    image: Image.Image, x0: int, x1: int, box_bottom: int, grown_y1: int
) -> int:
    """Trim a recovered bottom band back to ``box_bottom`` when it reads as a
    swallowed caption rather than recovered figure content.

    Called only when the OCR emitted a caption block for the figure, so prose is
    known to sit below it.  The band (``box_bottom``..``grown_y1``) is judged as a
    whole by :func:`_mean_norm_run_length`: a prose-reading band is dropped, a
    figure-reading one is kept whole — so a band mixing a clipped figure tail with
    a little caption is kept rather than risk clipping the figure.
    """
    if grown_y1 <= box_bottom:
        return grown_y1
    mask = np.asarray(image.crop((x0, box_bottom, x1, grown_y1)).convert("L"))
    if _mean_norm_run_length(mask < _FIGURE_INK_LEVEL) < _FIGURE_PROSE_RUN_FRAC:
        return box_bottom
    return grown_y1


def _ink_bands(mask: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Contiguous inked-row bands ``(top, bottom)`` separated by runs of at least
    ``gap`` blank rows, top to bottom."""
    row_ink = mask.mean(axis=1) > _FIGURE_BLANK_LINE_FRAC
    bands: list[tuple[int, int]] = []
    start: int | None = None
    blank = 0
    for i, ink in enumerate(row_ink):
        if ink:
            if start is None:
                start = i
            blank = 0
        elif start is not None:
            blank += 1
            if blank >= gap:
                bands.append((start, i - blank + 1))
                start = None
    if start is not None:
        bands.append((start, len(row_ink)))
    return bands


def _ocr_band(
    image: Image.Image,
    box: tuple[int, int, int, int],
    ocr_region: Callable[[Image.Image], str],
) -> str:
    """OCR a band crop, padding it with white to ``_FIGURE_OCR_MIN_BAND_PX`` tall
    first — the vision model needs at least two patch rows and raises on a band
    only one patch tall (a thin separator or a single text line)."""
    crop = image.crop(box).convert("RGB")
    if crop.height < _FIGURE_OCR_MIN_BAND_PX:
        canvas = Image.new(
            "RGB", (crop.width, _FIGURE_OCR_MIN_BAND_PX), (255, 255, 255)
        )
        canvas.paste(crop, (0, (_FIGURE_OCR_MIN_BAND_PX - crop.height) // 2))
        crop = canvas
    return ocr_region(crop)


def _band_is_caption(text: str, caption_words: set[str]) -> bool:
    """True when ``text`` re-OCRs to (part of) the figure caption: at least
    ``_FIGURE_CAPTION_WORD_FRAC`` of its words appear in ``caption_words`` and it
    carries at least ``_FIGURE_CAPTION_MIN_DISTINCT`` distinct caption words — the
    latter rejecting a degenerate repeated-token wall that the caption merely names."""
    words = _WORD_RE.findall(text.lower())
    if len(words) < _FIGURE_CAPTION_MIN_WORDS:
        return False
    matched = [w for w in words if w in caption_words]
    if len(set(matched)) < _FIGURE_CAPTION_MIN_DISTINCT:
        return False
    return len(matched) / len(words) >= _FIGURE_CAPTION_WORD_FRAC


def _trim_baked_caption(
    image: Image.Image,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    caption_text: str,
    ocr_region: Callable[[Image.Image], str],
) -> int:
    """Trim a caption the model baked into the *interior* of the figure box.

    When the figure is itself text (a sequence alignment, a table) the caption is
    pixel-identical to it, so :func:`_trim_swallowed_caption` can't see it.  Here
    the trailing text bands are re-OCRed and dropped when they reproduce the
    figure's own ``caption_text`` (which the page OCR already emitted as a block).
    Dense (non-text) bands are never OCRed and end the scan.  The hunt for the
    caption's first band is confined to the bottom ``_FIGURE_CAPTION_SCAN_FRAC`` of
    the crop (so the extra OCR stays on genuinely ambiguous text-bodied figures),
    but once found the caption is followed up past that floor so a tall caption is
    trimmed whole.
    """
    mask = np.asarray(image.crop((x0, y0, x1, y1)).convert("L")) < _FIGURE_INK_LEVEL
    floor = round((1 - _FIGURE_CAPTION_SCAN_FRAC) * (y1 - y0))
    bands = _ink_bands(mask, max(1, round(_FIGURE_BLOCK_GAP_FRAC * image.size[1])))
    caption_words = set(_WORD_RE.findall(caption_text.lower()))
    caption_top: int | None = None
    notes = 0
    for top, bot in reversed(bands):
        if caption_top is None and bot <= floor:
            break  # hunted down past the bottom slab with no caption — give up
        if _mean_norm_run_length(mask[top:bot]) >= _FIGURE_PROSE_RUN_FRAC:
            break  # dense figure content — stop without OCR
        text = _ocr_band(image, (x0, y0 + top, x1, y0 + bot), ocr_region)
        if _band_is_caption(text, caption_words):
            caption_top = y0 + top
        elif caption_top is not None:
            break  # text above the caption that isn't caption → figure text
        elif _FIGURE_NOTE_RE.search(text):
            notes += 1
            if notes > _FIGURE_CAPTION_NOTE_BANDS:
                break
        else:
            break  # a non-caption, non-note text tail → no baked caption here
    return caption_top if caption_top is not None else y1


def _safe_ocr_region(
    ocr_region: Callable[[Image.Image], str], image: Image.Image
) -> str | None:
    """Run an injected ``ocr_region`` re-OCR, degrading to ``None`` (logged) on any
    failure instead of letting it abort the document.

    ``ocr_region`` is an opaque network call to the vLLM server on a small shared
    GPU, so a transient OOM or connection blip can raise mid-document.  The single-call
    re-OCR sites that use this — figure-crop recovery — are best-effort refinements of
    an already-usable result (the figure was already dropped), so one failure should
    decline that single refinement, not crash the conversion.  The failure type is not
    nameable here (the callable is injected), so the guard is necessarily broad; it logs
    with a traceback so a real defect isn't silently masked.

    (The composite baked-caption band trim in :func:`_trim_caption_band` applies the
    same policy inline, because its fallback is the pixel-only trim rather than ``None``
    and it issues *several* OCR calls — it can't reduce to this single-call form.)"""
    try:
        return ocr_region(image)
    except Exception:
        _log.warning("re-OCR of a sub-region failed; declining it", exc_info=True)
        return None


def _trim_caption_band(
    image: Image.Image,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    grown_y1: int,
    caption_text: str,
    ocr_region: Callable[[Image.Image], str] | None,
) -> int:
    """Trim the caption out of the bottom of a grown crop, returning the kept ``y``.

    When ``ocr_region`` is available, the OCR-based band trim is authoritative: it
    cuts at a baked/abutting caption's true top, and — finding none — leaves the
    crop as grown.  Bottom growth already halts at the figure↔caption gap, so a
    "no caption found" result means the grown band is the recovered figure tail;
    keeping it is correct, whereas the cheap run-length trim would mistake a sparse
    tail (axis tick labels) for prose and clip it.  The run-length trim is only the
    backstop when OCR is unavailable or raises (a band re-OCR can fail on a
    transient OOM on the shared GPU, or a degenerate strip — the trim is cosmetic,
    so the failure is logged and the run-length trim takes over)."""
    if ocr_region is not None:
        try:
            return _trim_baked_caption(
                image, x0, x1, y0, grown_y1, caption_text, ocr_region
            )
        except Exception:
            # ocr_region is an opaque injected callable (network OCR), so the failure
            # type isn't nameable here; degrade to the pixel-only trim rather than
            # abort the document, but log it so a real defect isn't silently masked.
            _log.warning(
                "baked-caption trim failed; using run-length trim", exc_info=True
            )
    return _trim_swallowed_caption(image, x0, x1, y1, grown_y1)


def _safe_crop(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    caption_text: str | None = None,
    ocr_region: Callable[[Image.Image], str] | None = None,
) -> Image.Image | None:
    """Crop a pixel-space ``bbox`` from ``image``, clamped to bounds and grown
    out to the figure's true top, left, right and bottom edges; ``None`` if
    degenerate.

    ``caption_text`` (the caption block the OCR emitted for this figure) enables
    trimming the caption back out of the crop: cheaply when it was a band growth
    pulled in from *below* the box, and — when ``ocr_region`` is supplied — by
    re-OCRing trailing text bands the model baked *inside* the box."""
    w, h = image.size
    x0 = max(0, min(bbox[0], w))
    y0 = max(0, min(bbox[1], h))
    x1 = max(0, min(bbox[2], w))
    y1 = max(0, min(bbox[3], h))
    if x1 - x0 < _MIN_FIGURE_HEIGHT or y1 - y0 < _MIN_FIGURE_HEIGHT:
        return None
    x0 = _extend_edge(image, (x0, y0, x1, y1), "left")
    x1 = _extend_edge(image, (x0, y0, x1, y1), "right")
    box_y0 = y0  # the model's box top, kept for the caption-trim keep-ratio guard
    y0 = _extend_edge(image, (x0, y0, x1, y1), "top")
    # With a caption below, halt bottom growth at the figure↔caption gap (the
    # smaller block gap) so it recovers the clipped tail without leaping into the
    # caption; without one, the larger gap steps over inter-row gaps as before.
    has_caption = caption_text is not None
    bottom_gap = _FIGURE_BLOCK_GAP_FRAC if has_caption else _FIGURE_GAP_FRAC
    grown_y1 = _extend_edge(image, (x0, y0, x1, y1), "bottom", bottom_gap)
    if caption_text is not None:
        untrimmed_y1 = grown_y1
        grown_y1 = _trim_caption_band(
            image, x0, x1, y0, y1, grown_y1, caption_text, ocr_region
        )
        # Measured against the model's box (box_y0..y1), not the grown top/bottom,
        # so a legitimately swallowed caption pulled in by _extend_bottom_to_content
        # can still be trimmed off a short figure without tripping the guard.
        if grown_y1 - box_y0 < _FIGURE_CAPTION_MIN_KEEP_FRAC * (y1 - box_y0):
            grown_y1 = untrimmed_y1  # trim ran away on a text-bodied figure
    if grown_y1 - y0 < _MIN_FIGURE_HEIGHT:  # a trim left too little to be a figure
        return None
    return image.crop((x0, y0, x1, grown_y1))


def _union_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """Smallest box enclosing every box — how over-segmented crops are merged."""
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _figures_same(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: float
) -> bool:
    x_overlap = min(a[2], b[2]) - max(a[0], b[0])
    if x_overlap < 0.5 * min(a[2] - a[0], b[2] - b[0]):
        return False
    y_gap = max(a[1], b[1]) - min(a[3], b[3])  # ≤ 0 when the boxes overlap
    return y_gap <= gap


def _cluster_figure_boxes(
    boxes: list[tuple[int, int, int, int]], gap: float
) -> list[list[int]]:
    """Group indices of boxes that belong to the same figure (transitive)."""
    parent = list(range(len(boxes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _figures_same(boxes[i], boxes[j], gap):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
