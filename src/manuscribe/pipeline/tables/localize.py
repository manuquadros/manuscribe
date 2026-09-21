"""Table localization: match the OCR's cells against the PDF text layer to seed a
bounding box, grow it along the reading axis to the whole table (rotation-aware for
sideways tables), render that box to a re-OCR crop, and judge — via the coverage
gate — whether the page already captured the region in full.

Text layer = **geometry only** here: cell *content* is never read into the
document; the box it seeds is.  Depends only on the foundational ``layers``
primitives plus ``render``/``text``."""

from __future__ import annotations

from collections import Counter

import pypdfium2 as pdfium  # noqa: TC002 — beartype reads annotations at runtime
from PIL import Image  # noqa: TC002 — beartype reads annotations at runtime

from manuscribe.pipeline.layers import _Box, _normalize
from manuscribe.pipeline.render import _downscale_to_long_side
from manuscribe.pipeline.text import _visible_text

# A normalized cell must be at least this many chars to seed localization — short
# numeric/symbol cells ("1 mM", "None") recur in prose and would match the wrong
# spot, so only distinctive multi-word cells anchor the box; the flood then fills
# the rest of the table around them.
_MIN_ANCHOR_LEN = 4

# The coverage gate ignores one-letter/one-symbol noise, but a multi-char numeric
# token ("100", "26621") is real table data — a dropped data column is exactly the
# kind of loss the gate must not skip over — so digit-bearing tokens count as
# evidence from length 2, while plain words must reach _COVERAGE_MIN_TOKEN_LEN
# (shorter words like "of"/"the" recur in prose and would misjudge completeness).
_COVERAGE_MIN_TOKEN_LEN = 4
_COVERAGE_MIN_NUMERIC_LEN = 2

# A table's caption (above) and legend (below) are short; a block longer than this
# is body prose the located bbox happened to overrun, and folding it into the gate's
# "captured" set would mask a genuine drop, so it is not treated as adjacent.
_ADJACENT_PARA_MAX_LEN = 300

# Vertical growth stops when the gap to the next text line exceeds this multiple
# of the table's own line spacing — large enough to step over inter-row gaps,
# small enough to halt at the wider margin separating the table from body prose.
# Derived from spacing (not a fixed size) so it adapts to font and render scale.
_GAP_FACTOR = 1.5
# A table legend sits one line below the body at a gap wider than the row gaps but
# narrower than the margin to prose; this factor (vs _GAP_FACTOR) reaches it.
_LEGEND_GAP_FACTOR = 2.5
_MAX_LEGEND_LINES = 1
_PAD_PT = 4.0  # small margin so glyph edges are not clipped by the crop

# Crop render scale: aim for this long side in px, clamped to a sane DPI band.
_TARGET_CROP_PX = 1500
_MIN_SCALE = 200 / 72
_MAX_SCALE = 600 / 72

# Quarter-turn rotations whose text reads along the *vertical* page axis — a table
# at either is "sideways" and localized along that axis; 0°/180° keep horizontal rows.
_SIDEWAYS_ROTATIONS = frozenset({90, 270})


def _anchor_texts(cell_texts: list[str]) -> list[str]:
    """Normalized cell strings distinctive enough to seed localization."""
    seen: set[str] = set()
    anchors: list[str] = []
    for cell in cell_texts:
        norm = _normalize(cell)
        if len(norm) >= _MIN_ANCHOR_LEN and norm not in seen:
            seen.add(norm)
            anchors.append(norm)
    return anchors


def _union(boxes: list[_Box]) -> _Box:
    ls, bs, rs, ts = zip(*boxes, strict=True)
    return (min(ls), min(bs), max(rs), max(ts))


def _group_lines(char_boxes: list[_Box | None], vertical: bool = False) -> list[_Box]:
    """Cluster glyph boxes into text lines, in reading order.

    Boxes whose cross-flow intervals overlap are the same line; a clean gap starts a
    new one.  Each returned box is one line's bounds (PDF points).  For upright text
    a line is a horizontal run (boxes sharing a vertical interval), ordered top of
    page first.  For a sideways table (text at 90°/270°) the reading lines run
    *vertically* — a line is a column-strip of glyphs sharing a horizontal interval —
    so clustering and ordering switch to the x-axis (left of page first)."""
    lo_i, hi_i = (0, 2) if vertical else (1, 3)
    key = (lambda b: b[0]) if vertical else (lambda b: -b[3])
    boxes = sorted((b for b in char_boxes if b is not None), key=key)
    lines: list[list[float]] = []
    for box in boxes:
        if lines and box[lo_i] < lines[-1][hi_i] and box[hi_i] > lines[-1][lo_i]:
            ln = lines[-1]
            ln[0], ln[1] = min(ln[0], box[0]), min(ln[1], box[1])
            ln[2], ln[3] = max(ln[2], box[2]), max(ln[3], box[3])
        else:
            lines.append(list(box))
    return [tuple(ln) for ln in lines]  # type: ignore[misc]


def _median(values: list[float]) -> float:
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _grow_run_down(gaps: list[float], start: int, threshold: float) -> int:
    """Extend a line/row run toward page bottom while the gap to the next line stays
    within ``threshold`` of the prose margin; return the new last index.

    Shared by table-crop localization (``_locate_bbox``) and the text-layer rebuild's
    below-table trim (``tables.rebuild._trim_rows_below_table``) so retuning the cut
    can't desync the two.  ``gaps[i]`` is the gap between line/row ``i`` and ``i+1``,
    so it has one fewer entry than the lines."""
    hi = start
    while hi < len(gaps) and gaps[hi] <= threshold:
        hi += 1
    return hi


def _seed_rotation(rotations: list[int]) -> int:
    """The dominant quarter-turn among a region's seed glyphs (0 when none rotate).

    A sideways table carries the same rotation on every glyph, so the mode of the
    seed glyphs' rotations identifies it; ties and an empty input fall back to 0 (the
    upright path), so a stray rotated glyph can never flip an upright table sideways."""
    if not rotations:
        return 0
    counts = Counter(rotations)
    best = max(counts.values())
    return 0 if counts[0] == best else max(counts, key=lambda r: (counts[r], -r))


def _collect_seeds(
    anchors: list[str],
    page_norm: str,
    idx_map: list[int],
    char_boxes: list[_Box | None],
    char_rotations: list[int | None],
) -> tuple[list[_Box], list[int]]:
    """Seed boxes and their rotations from each anchor that occurs exactly once
    (a repeated anchor is ambiguous — it may be prose — so it is skipped)."""
    seeds: list[_Box] = []
    seed_rotations: list[int] = []
    for anchor in anchors:
        first = page_norm.find(anchor)
        if first == -1 or page_norm.find(anchor, first + 1) != -1:
            continue  # absent, or ambiguous (recurs elsewhere on the page)
        i0, i1 = idx_map[first], idx_map[first + len(anchor) - 1]
        spanned = [b for b in char_boxes[i0 : i1 + 1] if b is not None]
        if spanned:
            seeds.append(_union(spanned))
            seed_rotations.extend(
                r for r in char_rotations[i0 : i1 + 1] if r is not None
            )
    return seeds, seed_rotations


def _locate_bbox(
    anchors: list[str],
    page_norm: str,
    idx_map: list[int],
    char_boxes: list[_Box | None],
    char_rotations: list[int | None],
    page_size: tuple[float, float],
) -> tuple[_Box, int] | None:
    """Bounding box (PDF points) and rotation of the table region, or ``None``.

    Anchors that occur exactly once in the folded page text locate the table's
    lines (a repeated anchor is ambiguous — it may be prose — so it is skipped).
    The line run is then grown along the reading axis while the inter-line gap stays
    within ``_GAP_FACTOR`` of the table's own spacing, so the box reaches rows no
    anchor covered (e.g. trailing data rows) yet halts at the wider gap to body
    prose.  Cross-axis extent is the union of the included lines.

    A sideways table (its anchor glyphs rotated 90°/270°) is localized along its
    *true* reading axis: lines run vertically, growth runs left↔right, and only
    glyphs sharing the table's rotation are clustered — so the box grows over the
    table's columns and stops at the gutter to the upright body text beside it
    instead of sweeping a neighbouring column's heading into the crop."""
    seeds, seed_rotations = _collect_seeds(
        anchors, page_norm, idx_map, char_boxes, char_rotations
    )
    if not seeds:
        return None

    rot = _seed_rotation(seed_rotations)
    vertical = rot in _SIDEWAYS_ROTATIONS
    lo_i, hi_i = (0, 2) if vertical else (1, 3)
    # For a sideways table, cluster only the sideways glyphs so the upright body text
    # beside it can't bridge the column gutter into the box.  Keep *both* 90° and
    # 270° (not just the modal rotation): a uniformly-rotated table can still have a
    # cell glyph pdfium snaps to the opposite quarter-turn, and dropping it would
    # clip that column from the crop.  Upright (0°) and inverted (180°) body text is
    # still excluded.
    boxes = (
        [
            b
            for b, r in zip(char_boxes, char_rotations, strict=True)
            if r in _SIDEWAYS_ROTATIONS
        ]
        if vertical
        else char_boxes
    )
    lines = _group_lines(boxes, vertical)
    if not lines:
        return None
    if vertical:
        gaps = [lines[i + 1][0] - lines[i][2] for i in range(len(lines) - 1)]
    else:
        gaps = [lines[i][1] - lines[i + 1][3] for i in range(len(lines) - 1)]

    seed_box = _union(seeds)
    touched = [
        i
        for i, ln in enumerate(lines)
        if ln[lo_i] < seed_box[hi_i] and ln[hi_i] > seed_box[lo_i]
    ]
    if not touched:
        return None
    lo, hi = min(touched), max(touched)

    # Median (not max) of the table's own line gaps: robust to a single wide
    # internal separation (a row-group break) that would otherwise inflate the
    # threshold enough to swallow the following section.
    anchor_gaps = gaps[lo:hi]
    spacing = _median(anchor_gaps) if anchor_gaps else (_median(gaps) if gaps else 0.0)
    threshold = spacing * _GAP_FACTOR
    while lo > 0 and gaps[lo - 1] <= threshold:
        lo -= 1
    hi = _grow_run_down(gaps, hi, threshold)

    # Legend allowance: a table's footnote/legend ("MW: molecular weight, …") sits
    # one line below the body, set off by a gap a little wider than the row gaps —
    # past the body threshold but well short of the margin to prose.  Pull in that
    # single line so the crop re-OCR can recover a legend the page OCR truncated.
    # Skipped for a sideways table: its legend is colocated as a separate caption,
    # and the line-gap model that anchors it assumes upright rows.
    legend_threshold = spacing * _LEGEND_GAP_FACTOR
    added = 0
    while (
        not vertical
        and hi < len(lines) - 1
        and added < _MAX_LEGEND_LINES
        and gaps[hi] <= legend_threshold
    ):
        hi += 1
        added += 1

    left, bottom, right, top = _union(list(lines[lo : hi + 1]))
    page_w, page_h = page_size
    bbox = (
        max(0.0, left - _PAD_PT),
        max(0.0, bottom - _PAD_PT),
        min(page_w, right + _PAD_PT),
        min(page_h, top + _PAD_PT),
    )
    return bbox, rot


def _scaled_crop(
    page: pdfium.PdfPage, bbox: _Box, page_size: tuple[float, float]
) -> Image.Image:
    """Render only ``bbox`` of the page, scaled for a clean table re-OCR."""
    page_w, page_h = page_size
    left, bottom, right, top = bbox
    long_pt = max(right - left, top - bottom)
    scale = min(_MAX_SCALE, max(_MIN_SCALE, _TARGET_CROP_PX / long_pt))
    img: Image.Image = (
        page.render(scale=scale, crop=(left, bottom, page_w - right, page_h - top))
        .to_pil()
        .convert("RGB")
    )
    return _downscale_to_long_side(img)


def _glyph_centers(
    page_norm: str, idx_map: list[int], char_boxes: list[_Box | None]
) -> list[tuple[float, float] | None]:
    """Per-``page_norm``-position glyph centre (``None`` for a space or box-less
    char), computed once per page so the coverage gate's per-region containment test
    is a cheap lookup rather than re-deriving every centre for every table region."""
    centers: list[tuple[float, float] | None] = []
    for pos, ch in enumerate(page_norm):
        box = None if ch == " " else char_boxes[idx_map[pos]]
        centers.append(
            None if box is None else ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        )
    return centers


def _in_bbox_tokens(
    bbox: _Box, page_norm: str, centers: list[tuple[float, float] | None]
) -> list[str]:
    """Folded text-layer tokens whose glyphs fall inside ``bbox``.

    Walks ``page_norm`` (its single spaces preserved as word boundaries) and keeps
    an alnum char only when its precomputed glyph centre lies in ``bbox``, so words
    outside the box drop out without gluing onto their in-box neighbours.  The
    boundary must come from ``page_norm`` rather than from filtering glyph boxes
    directly: a text-layer space carries no glyph box (pdfium reports none), so
    dropping box-less chars would weld adjacent words into one unmatchable token."""
    left, bottom, right, top = bbox
    kept: list[str] = []
    for ch, center in zip(page_norm, centers, strict=True):
        if (
            ch != " "
            and center is not None
            and left <= center[0] <= right
            and bottom <= center[1] <= top
        ):
            kept.append(ch)
        else:
            kept.append(" ")
    return "".join(kept).split()


def _adjacent_para_tokens(md: str, start: int, end: int) -> set[str]:
    """Folded tokens of the short paragraphs flanking a table region — the caption
    above and any legend below.  The located bbox grows to include these lines, yet
    they are captured in the page markdown though not in the cells, so the coverage
    gate must count them as already-present; otherwise every captioned table would
    look like it had dropped content and the gate would never fire.

    Only a genuinely short flanking block (a caption/legend) is folded in: a long
    block — or the whole pre-table content when no blank line separates it — is body
    prose the bbox overran, and adding it to ``captured`` would mask a real drop."""
    before = md[:start].rstrip().rsplit("\n\n", 1)[-1]
    after = md[end:].lstrip().split("\n\n", 1)[0]
    tokens: set[str] = set()
    for chunk in (before, after):
        text = _visible_text(chunk)
        if len(text) <= _ADJACENT_PARA_MAX_LEN:
            tokens.update(_normalize(text).split())
    return tokens


def _is_distinctive(token: str) -> bool:
    """A token strong enough to judge table completeness: a real word (length
    ``_COVERAGE_MIN_TOKEN_LEN`` or more) or a multi-char number (table data)."""
    return len(token) >= _COVERAGE_MIN_TOKEN_LEN or (
        len(token) >= _COVERAGE_MIN_NUMERIC_LEN and any(c.isdigit() for c in token)
    )


def _region_fully_captured(
    bbox: _Box,
    page_norm: str,
    centers: list[tuple[float, float] | None],
    captured: set[str],
) -> bool:
    """Whether the page already captured every distinctive text-layer token inside
    the region — in which case the crop re-OCR has nothing to recover and is skipped.

    Conservative by construction: only the *provably complete* case skips (no
    distinctive in-box token missing from ``captured``).  Distinctive covers real
    words and multi-char numbers, so a dropped data column counts and is not skipped.
    A scanned or empty text layer yields no distinctive tokens, so it returns
    ``False`` and the region is re-OCR'd as before — the gate is an opt-out for
    proven completeness, never an opt-out on absence of evidence.  This is
    deliberately strict: the cross-fixture data shows the located bbox pulls in body
    prose, so a loose threshold cannot separate a genuine drop (often a single
    token) from that prose noise; only the zero-missing case is safe to skip (the
    text-layer coverage gate."""
    distinctive = [
        t for t in _in_bbox_tokens(bbox, page_norm, centers) if _is_distinctive(t)
    ]
    if not distinctive:
        return False
    return all(t in captured for t in distinctive)
