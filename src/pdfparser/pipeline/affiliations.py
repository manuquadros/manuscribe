"""Affiliation-line recognition: the structural cue that a comma-separated
address line is an institutional affiliation (so a byline whose superscript marker
the OCR dropped is still routed out of the body), plus the country lexicon and
postal/place-tail regexes it leans on.

Pure string predicates over already-OCR'd text; depends only on ``text`` for the
sentence-end test.  ``_is_affiliation_line`` is the one name consumed elsewhere
(``classify``); the rest support it."""

from __future__ import annotations

import re

from pdfparser.pipeline.text import _SENTENCE_END_RE

# An affiliation names an institution: a department, university, institute,
# laboratory, college, hospital, faculty, academy or school.  Stems
# ("universit", "institut", "laborato") cover the common international spellings
# (University/Université/Universität, Institute/Institut/Instituto,
# Laboratory/Laboratoire/Laboratorio).
_AFFILIATION_RE = re.compile(
    r"\b(?:department|universit\w*|institut\w*|laborato\w*|college|hospital|"
    r"faculty|academy|polytechnic|school\s+of)\b",
    re.IGNORECASE,
)
# The same institution stems (mirroring _AFFILIATION_RE) anchored at the line
# *start*, so a no-postcode international address ("Department of …, City,
# Country") is still recognised.  Anchored, and paired with a place-like tail
# (see _has_place_like_tail): each relaxation alone would also match an
# OCR-truncated prose clause that merely names an institution.
_AFFILIATION_HEAD_RE = re.compile(
    r"^(?:the\s+)?(?:department|centers?\s+for|centres?\s+for|universit\w*|"
    r"institut\w*|laborato\w*|college|hospital|faculty|academy|polytechnic|"
    r"school\s+of|division\s+of)\b",
    re.IGNORECASE,
)
# A place tail must *open* on a capital (a lowercase connector like "of" may
# follow): that is what rejects a list/clause continuation ("… and two partner
# clinics") while accepting "South Korea" / "Republic of Korea".
_PLACE_TAIL_MAX_WORDS = 4
_PLACE_TAIL_CONNECTORS = frozenset({"of"})
# An address almost always closes on a country/region — so a country tail
# recognises the affiliation regardless of the *institution* word's language
# (German "Labor", Czech "Ústav", …), which no stem lexicon covers.  This
# language-independent path backstops the stem-based ones below; either signal
# alone marks the line an affiliation.  Bare "us"/"uk" are omitted as too
# ambiguous a closing token.
_AFFILIATION_COUNTRIES = frozenset(
    {
        "afghanistan",
        "albania",
        "algeria",
        "argentina",
        "armenia",
        "australia",
        "austria",
        "österreich",
        "azerbaijan",
        "bahrain",
        "bangladesh",
        "belarus",
        "belgium",
        "belgië",
        "belgique",
        "bolivia",
        "bosnia and herzegovina",
        "brazil",
        "brasil",
        "bulgaria",
        "cambodia",
        "cameroon",
        "canada",
        "chile",
        "china",
        "pr china",
        "p.r. china",
        "people's republic of china",
        "colombia",
        "costa rica",
        "croatia",
        "hrvatska",
        "cuba",
        "cyprus",
        "czech republic",
        "czechia",
        "česká republika",
        "česko",
        "denmark",
        "danmark",
        "ecuador",
        "egypt",
        "estonia",
        "ethiopia",
        "finland",
        "suomi",
        "france",
        "georgia",
        "germany",
        "deutschland",
        "ghana",
        "greece",
        "hungary",
        "magyarország",
        "iceland",
        "india",
        "indonesia",
        "iran",
        "iraq",
        "ireland",
        "israel",
        "italy",
        "italia",
        "japan",
        "jordan",
        "kazakhstan",
        "kenya",
        "korea",
        "south korea",
        "north korea",
        "republic of korea",
        "kuwait",
        "latvia",
        "lebanon",
        "lithuania",
        "luxembourg",
        "malaysia",
        "malta",
        "mexico",
        "méxico",
        "moldova",
        "mongolia",
        "montenegro",
        "morocco",
        "myanmar",
        "nepal",
        "netherlands",
        "the netherlands",
        "nederland",
        "new zealand",
        "nigeria",
        "norway",
        "norge",
        "oman",
        "pakistan",
        "palestine",
        "panama",
        "peru",
        "philippines",
        "poland",
        "polska",
        "portugal",
        "qatar",
        "romania",
        "russia",
        "russian federation",
        "saudi arabia",
        "serbia",
        "singapore",
        "slovakia",
        "slovensko",
        "slovenia",
        "south africa",
        "spain",
        "españa",
        "sri lanka",
        "sudan",
        "sweden",
        "sverige",
        "switzerland",
        "schweiz",
        "suisse",
        "svizzera",
        "syria",
        "taiwan",
        "tanzania",
        "thailand",
        "tunisia",
        "turkey",
        "türkiye",
        "uganda",
        "ukraine",
        "united arab emirates",
        "uae",
        "united kingdom",
        "u.k.",
        "united states",
        "united states of america",
        "usa",
        "u.s.a.",
        "uruguay",
        "uzbekistan",
        "venezuela",
        "vietnam",
        "viet nam",
        "yemen",
        "zimbabwe",
    }
)
_POSTAL_DIGITS_RE = re.compile(r"\b\d[\d-]*\b")
# An affiliation OCR'd without its author's superscript marker (J. Biol. Chem.'s
# "From the Department of …, City, Region, postcode") is recognised structurally:
# it names an institution, is laid out as a comma-separated address ending in a
# postal code, and — being a noun phrase, not a sentence — carries no terminal
# punctuation.  The postal-code tail is the load-bearing signal: "no terminal
# punctuation" on its own also matches an OCR-truncated prose clause that happens
# to name a university ("… with the Department of Biology, the School of Medicine,
# and several partner hospitals across the region"), but such a clause does not
# end in a postcode.  The trade is deliberately safe — an address without a
# postcode is left visible rather than risk hiding prose.
_AFFILIATION_MIN_COMMAS = 2
_AFFILIATION_MAX_LEN = 300
# A postal code closing the address: a 4–6 digit run (US ZIP and most national
# codes).  Searched only in the trailing comma-segments so a number earlier in a
# prose clause ("enrolled 250 patients …") cannot stand in for the address tail.
_POSTAL_TAIL_RE = re.compile(r"\b\d{4,6}\b")
_AFFILIATION_TAIL_SEGMENTS = 2


def _has_place_like_tail(plain: str) -> bool:
    """True when the last comma-segment looks like a place name — a short run of
    capitalised words, opening on a capital (see ``_PLACE_TAIL_MAX_WORDS``)."""
    words = plain.rsplit(",", 1)[-1].split()
    if not 1 <= len(words) <= _PLACE_TAIL_MAX_WORDS or not words[0][:1].isupper():
        return False
    return all(w[:1].isupper() or w.lower() in _PLACE_TAIL_CONNECTORS for w in words)


# A comma-separated address reads as a list of place/institution names: at most a
# short connector word ("of", "für", "der", ...) interrupts two capitalised names,
# so a lone non-English common noun stands alone between them ("Ústav organické
# chemie" has a run of two lowercase-led words). A body sentence that merely
# *closes* on a place name is still a finite-verb clause beforehand, and reads as a
# much longer run of lowercase-led words ("were collected from farms near").
_PROSE_RUN_MIN = 3


def _has_prose_clause(plain: str) -> bool:
    """True when ``plain`` contains a run of ``_PROSE_RUN_MIN`` or more consecutive
    lowercase-led words — the shape of a finite-verb clause, as opposed to a
    comma-separated list of place/institution names (see above)."""
    run = 0
    for word in plain.split():
        if word[:1].islower():
            run += 1
            if run >= _PROSE_RUN_MIN:
                return True
        else:
            run = 0
    return False


def _ends_with_country(plain: str) -> bool:
    """True when the final comma-segment is a recognised country/region — the
    language-independent affiliation cue (see ``_AFFILIATION_COUNTRIES``) — and the
    line is not itself a prose clause that merely happens to close on a place name
    (see ``_has_prose_clause``: a column-split Methods/Results sentence can end
    "…, City, City2, Country" with no institution word to reject it on).  Any
    postal digits sharing the segment ("South Korea 08826") are stripped first."""
    tail = _POSTAL_DIGITS_RE.sub("", plain.rsplit(",", 1)[-1]).strip()
    return tail.lower() in _AFFILIATION_COUNTRIES and not _has_prose_clause(plain)


def _is_affiliation_line(plain: str) -> bool:
    """A bare affiliation address that lost its author's superscript marker.

    Identified structurally (a comma-separated address layout with no terminal
    punctuation) so it is recognised without a leading ``¹``/``*``, which
    ``_LEADING_SUP_RE`` already handles.  Any one of the closing signals marks it:
    a country/region tail (language-independent), or — needing an institution
    name, so limited to the stem spellings — a place-name tail behind a
    keyword-headed line, or a postal-code tail."""
    if (
        len(plain) > _AFFILIATION_MAX_LEN
        or plain.count(",") < _AFFILIATION_MIN_COMMAS
        or _SENTENCE_END_RE.search(plain)
    ):
        return False
    if _ends_with_country(plain):
        return True
    if not _AFFILIATION_RE.search(plain):
        return False
    if _AFFILIATION_HEAD_RE.match(plain) and _has_place_like_tail(plain):
        return True
    tail = ",".join(
        plain.rsplit(",", _AFFILIATION_TAIL_SEGMENTS)[-_AFFILIATION_TAIL_SEGMENTS:]
    )
    return bool(_POSTAL_TAIL_RE.search(tail))
