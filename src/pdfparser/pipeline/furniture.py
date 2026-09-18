"""Running-furniture stripping: detect and remove the headers, footers, page
numbers, per-page license/copyright footers and degenerate OCR repetition that recur
across a document's pages, so they don't break the cross-page paragraph merge or leak
into the body.

Predicates over the already-OCR'd, typed block list; depends on ``block`` and
``text``.  ``_strip_running_furniture`` / ``_capture_license_footer`` (consumed by
``assemble``) and ``_is_degenerate_repetition`` (by ``classify``) are the public
names; the rest support them."""

from __future__ import annotations

import re
from collections import Counter

from pdfparser.pipeline.block import Block, BlockKind
from pdfparser.pipeline.text import _visible_text

# Running header/footer: a short, terminal-punctuation-free line that recurs
# across pages.  Page numbers vary per page, so they are stripped before the
# recurrence is counted (e.g. "Biotechnology … 601" / "… 602" share a key).
# The cap only bounds which lines are *considered*; the recurrence guards (a
# candidate must repeat, a sentence-terminated one at least
# _SENTENCE_LIKE_FURNITURE_MIN_REPEAT times) are what actually distinguish
# furniture from prose.  It must clear a
# full-sentence open-access/copyright footer ("© 2019 The Author(s). … (CC BY).",
# ~200 chars) and a gutter download stamp ("Downloaded from http://…/…​.pdf by
# guest on …"), both of which recur on every page and otherwise break the
# cross-page paragraph merge by gluing onto the prose they interrupt.
_FURNITURE_MAX_LEN = 256
_DIGITS_RE = re.compile(r"\d+")
# A footer/header is identified by its digit-stripped text recurring across
# pages ("… 601" / "… 602" share a key).  When the line carries digits, require
# that text to be substantial so stripping them can't collapse short enumerated
# labels ("Fig 1" / "Fig 2", "Step 1" / "Step 2") into one key and delete them as
# furniture.  A digit-free line can't suffer that collision — its key is the
# verbatim text — so a recurring short one (an author-surname running foot,
# "Clark") only needs to clear a small floor that rules out a stray initial.
_MIN_FURNITURE_KEY_LEN = 12
_MIN_DIGIT_FREE_FURNITURE_KEY_LEN = 3
# A recurring line that reads like a finished sentence (ends in terminal
# punctuation) is normally real prose, not furniture — but a running head ending
# in an abbreviation ("… lichen-associated Sphingomonas sp.") only *looks* like
# one.  Genuine prose virtually never repeats verbatim across this many pages, so
# a sentence-terminated line recurring at least this often is a running head.
_SENTENCE_LIKE_FURNITURE_MIN_REPEAT = 3
# A folio is a block whose entire visible text is a bare number.  Bounded length
# keeps it to plausible page numbers and away from longer numeric data.
_PAGE_NUMBER_RE = re.compile(r"\d{1,4}")

# Even though LightOnOCR-bbox usually boxes figures (so they never reach the text
# stream), a diagram it misses can still be OCRed into one label repeated dozens
# of times ("AaTRI, AaTRI, …") — many tokens, almost no diversity.  This drops
# such a paragraph from the body; real prose (even with some repetition) stays.
_MIN_REPEAT_TOKENS = 8
_MAX_REPEAT_SHARE = 0.6
_TOKEN_RE = re.compile(r"\w+")

# Text-normalization for the furniture key (the only consumer): drop punctuation
# and collapse whitespace so two header/footer lines compare equal.
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _furniture_key(inner: str) -> str:
    text = _DIGITS_RE.sub("", _visible_text(inner))
    return _WHITESPACE_RE.sub(" ", _PUNCT_RE.sub("", text)).strip().lower()


# A running-furniture candidate is a plain paragraph or a heading — the two kinds
# Block.inner populates. OCR transcribes the same marginal line as a <p> on dense
# body pages but promotes it to a heading on sparse pages (last page, after
# references), so both forms must feed the recurrence count to be stripped
# consistently; a table/figure/other block is never a candidate (block.inner is
# None), matching the former _furniture_inner's blanket exclusion of those shapes.
_FURNITURE_CANDIDATE_KINDS = (BlockKind.PARAGRAPH, BlockKind.HEADING)


def _is_furniture_candidate(block: Block) -> str | None:
    if block.kind not in _FURNITURE_CANDIDATE_KINDS or block.inner is None:
        return None
    inner = block.inner
    plain = _visible_text(inner)
    if len(plain) > _FURNITURE_MAX_LEN:
        return None
    key = _furniture_key(inner)
    min_len = (
        _MIN_FURNITURE_KEY_LEN
        if _DIGITS_RE.search(plain)
        else _MIN_DIGIT_FREE_FURNITURE_KEY_LEN
    )
    return key if len(key) >= min_len else None


def _ends_like_sentence(block: Block) -> bool:
    # block.ends_sentence is already _ends_sentence(block.inner) for these two
    # kinds (computed once in Block.of) — a line whose terminal period hides
    # behind a trailing citation superscript still reads as a sentence there.
    return block.kind in _FURNITURE_CANDIDATE_KINDS and block.ends_sentence


def _is_standalone_page_number(block: Block) -> bool:
    """A folio printed alone in the margin that OCR emitted as its own block.

    The recurrence pass can't catch it: ``_furniture_key`` strips digits before
    keying, so a number-only block has an empty key, and each page's number is
    distinct anyway.  A block whose only content is a bare number is the folio
    itself, so it is dropped directly."""
    if block.kind not in _FURNITURE_CANDIDATE_KINDS or block.inner is None:
        return False
    return bool(_PAGE_NUMBER_RE.fullmatch(_visible_text(block.inner).strip()))


# A copyright / open-access license footer ("© 2019 The Author(s). … (CC BY).") the
# journal prints on every page.  _strip_running_furniture rightly drops the per-page
# repeats from the body, but the article's license is front matter worth keeping, so
# one copy is captured into the Metadata panel first (see _capture_license_footer).
_LICENSE_FOOTER_RE = re.compile(
    r"©\s*\d{4}\b.*?(?:open[\s-]?access|creative commons|\bcc by\b|licen[sc]ed under)",
    re.IGNORECASE | re.DOTALL,
)


def _capture_license_footer(blocks: list[Block]) -> str | None:
    """One copy of a *recurring* copyright/open-access license footer, for the Metadata
    panel, or ``None`` when there is none.  Only a footer that repeats (≥2 blocks) is
    captured — it is the per-page furniture ``_strip_running_furniture`` then removes
    from the body, so taking one copy relocates it rather than losing it; a
    single-occurrence copyright is left in place (it is not running furniture)."""
    matches = [
        block.html
        for block in blocks
        if block.kind is BlockKind.PARAGRAPH
        and block.inner is not None
        and _LICENSE_FOOTER_RE.search(_visible_text(block.inner))
    ]
    return matches[0] if len(matches) > 1 else None


def _strip_running_furniture(blocks: list[Block]) -> list[str]:
    """Drop short, recurring header/footer lines (page-number-insensitive) and
    standalone page-number blocks.

    A heading form normally counts as furniture only when the same line also
    appears as a plain paragraph somewhere: a running header/footer is body text
    the OCR promotes to a heading on sparse pages, so it shows up in both forms.  A
    line that recurs *only* as a heading is usually a genuine section heading the
    article legitimately repeats (e.g. "Purification of X" under both Methods and
    Results), not furniture, and must be kept.  The exception is a heading that
    carried *digits* whose *verbatim* text recurs ("Bioscience Reports (2019) 39
    BSR20190715" on every page): a journal citation / folio running head, constant
    across pages, whose paragraph form may differ (it also carries a DOI line, so
    its digit-stripped key never matches the bare-heading key).  Requiring the
    verbatim text — not just the digit-stripped key — to recur keeps distinct
    numbered headings that merely collide after digit-stripping ("Step 1: X" /
    "Step 2: X") from being mistaken for one running head and deleted.

    A sentence-terminated line is treated as real prose unless it recurs often
    enough (``_SENTENCE_LIKE_FURNITURE_MIN_REPEAT``) to be a running head whose
    trailing abbreviation only mimics a sentence end ("… Sphingomonas sp.")."""
    counts: Counter[str] = Counter()
    as_paragraph: set[str] = set()
    not_sentence_like: set[str] = set()
    verbatim_counts: Counter[str] = Counter()
    # (key, verbatim text) for each digit-bearing candidate — used to find a heading
    # whose *exact* text recurs (a journal citation / folio constant across pages),
    # which qualifies for the heading-only relaxation; distinct numbered headings
    # ("Step 1: X" / "Step 2: X") only share a digit-stripped key, not verbatim text.
    digit_bearing: list[tuple[str, str]] = []
    for block in blocks:
        key = _is_furniture_candidate(block)
        if key is None:
            continue
        counts[key] += 1
        if block.kind is BlockKind.PARAGRAPH:
            as_paragraph.add(key)
        if not _ends_like_sentence(block):
            not_sentence_like.add(key)
        text = _visible_text(block.inner).strip() if block.inner is not None else ""
        verbatim_counts[text] += 1
        if _DIGITS_RE.search(text):
            digit_bearing.append((key, text))
    digit_citation = {key for key, text in digit_bearing if verbatim_counts[text] > 1}
    repeated = {
        key
        for key, n in counts.items()
        if n > 1
        and (key in as_paragraph or key in digit_citation)
        and (key in not_sentence_like or n >= _SENTENCE_LIKE_FURNITURE_MIN_REPEAT)
    }
    return [
        block.html
        for block in blocks
        if _is_furniture_candidate(block) not in repeated
        and not _is_standalone_page_number(block)
    ]


def _is_degenerate_repetition(text: str) -> bool:
    tokens = _TOKEN_RE.findall(_visible_text(text))
    if len(tokens) < _MIN_REPEAT_TOKENS:
        return False
    top = Counter(tokens).most_common(1)[0][1]
    return top / len(tokens) >= _MAX_REPEAT_SHARE
