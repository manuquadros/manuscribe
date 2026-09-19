"""Soft line-break de-hyphenation, shared by the markdown reflow and the
block-merge stitcher.

LightOnOCR (and two-column PDF layout) split a paragraph across visual lines and
across page/column block boundaries; a word broken at such a break keeps a soft
hyphen ("Unfortu-" + "nately").  Dropping *every* line-break hyphen would corrupt
a genuine compound that happens to wrap at its own hyphen ("well-known"), so the
hyphen is kept unless we are confident the break is syllabic: the merged form is
a real English word, or both sides are lowercase non-words ("At-" + "ropa").
"""

from __future__ import annotations

import re

from spellchecker import SpellChecker

_HYPHEN_TAIL_RE = re.compile(r"-\s*$")
_TRAILING_WORD_RE = re.compile(r"([A-Za-z]+)$")
_LEADING_WORD_RE = re.compile(r"([A-Za-z]+)")

# Productive prefixes that attach without a hyphen in modern scientific writing
# ("over" + "expression" -> "overexpression").  When one heads the left side of a
# wrap break it is a soft hyphen to drop, even though the solid form is usually
# absent from a general-English dictionary — the failure mode the dictionary test
# alone gets wrong.  Compound-forming words that keep their hyphen ("self-",
# "well-", "high-", "cross-", "time-") are deliberately excluded; the
# both-halves-are-words rule handles those.  This is a closed list of an open
# class, so it generalises far better than enumerating solid forms term by term.
_SOLID_PREFIXES = frozenset(
    {
        "anti",
        "auto",
        "bio",
        "co",
        "counter",
        "cyto",
        "de",
        "down",
        "electro",
        "endo",
        "exo",
        "extra",
        "hetero",
        "homo",
        "hydro",
        "hyper",
        "hypo",
        "immuno",
        "inter",
        "intra",
        "iso",
        "macro",
        "mega",
        "meta",
        "micro",
        "mono",
        "multi",
        "nano",
        "neuro",
        "non",
        "over",
        "photo",
        "poly",
        "post",
        "pre",
        "pseudo",
        "radio",
        "semi",
        "sub",
        "super",
        "supra",
        "thermo",
        "trans",
        "ultra",
        "under",
        "up",
    }
)

_spell_singleton: SpellChecker | None = None


def _spell() -> SpellChecker:
    # The English frequency dictionary loads once (~6 MB); defer it off the
    # import path so unrelated pipeline imports stay cheap.
    global _spell_singleton
    if _spell_singleton is None:
        _spell_singleton = SpellChecker(distance=0)
    return _spell_singleton


def _is_word(word: str) -> bool:
    """True when ``word`` is in the English frequency dictionary (case-folded)."""
    return word.lower() in _spell()


def _keep_line_break_hyphen(left_word: str, right_word: str) -> bool:
    """Decide a hyphen at a line/block break: keep it (real compound, range,
    formula, acronym) or drop it (syllabic word split).

    The default is to keep — a hyphen is dropped only when the break is
    confidently syllabic: the two halves merge into a real word ("Unfortu" +
    "nately" → "unfortunately"), the left half is a productive solid prefix on a
    lowercase continuation ("over" + "expression" → "overexpression"), or, when
    neither the merged form nor both halves are words, the boundary is
    lowercase-to-lowercase ("At" + "ropa" → "Atropa").  A prefix before a capital
    or number keeps its hyphen ("anti" + "CRISPR" → "anti-CRISPR").
    """
    if not left_word or not right_word:
        return True
    if _is_word(left_word + right_word):
        return False
    if left_word.lower() in _SOLID_PREFIXES and right_word[0].islower():
        return False
    if _is_word(left_word) and _is_word(right_word):
        return True
    return not (left_word[-1].islower() and right_word[0].islower())


def _dehyphenate_join(left: str, right: str) -> str:
    """Join a wrapped fragment ``left`` to its continuation ``right``.

    A trailing soft hyphen is dropped (no space) for a syllabic break and kept
    (no space) for a real compound; a break with no trailing hyphen joins with a
    single space.  The word tests run on the bare letters either side of the
    hyphen, so inline tags around the break ("…<em>At-") do not perturb them.
    """
    right = right.lstrip()
    hyphen = _HYPHEN_TAIL_RE.search(left)
    if hyphen is None:
        return left.rstrip() + " " + right
    stem = left[: hyphen.start()]
    left_word_m = _TRAILING_WORD_RE.search(stem)
    right_word_m = _LEADING_WORD_RE.match(right)
    left_word = left_word_m.group(1) if left_word_m else ""
    right_word = right_word_m.group(1) if right_word_m else ""
    if _keep_line_break_hyphen(left_word, right_word):
        return stem + "-" + right
    return stem + right
