"""Document-structure classification of the block-HTML stream.

Pure, no GPU.  Pulls title/byline/abstract/footnotes out of the flat block list,
detects the article's first page, strips running headers/footers and degenerate
OCR repetition, and separates leading front matter (affiliations, keywords,
correspondence, dates) from the body.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from pdfparser.pipeline.affiliations import _is_affiliation_line
from pdfparser.pipeline.furniture import _is_degenerate_repetition
from pdfparser.pipeline.text import (
    _BOLD_LABEL_CAPTURE_RE,
    _BOLD_LABEL_RE,
    _SENTENCE_END_RE,
    _SUP_DIGITS,
    _ends_sentence,
    _heading_inner,
    _plain_p_text,
    _visible_text,
    _visible_text_folded,
)

# A leading footnote marker is a SHORT superscript ("<sup>a</sup>", "<sup>1</sup>").
# Bounding the marker length stops a body paragraph that merely opens with a
# reconstructed multi-character superscript from being mistaken for a footnote.
_SUP_MARKER_RE = re.compile(r"^<sup>[^<]{1,3}</sup>")
# A numbered page footnote emitted as raw unicode superscripts instead of a <sup>
# tag ("¹http://…/home.htm" — a footnote the model didn't wrap).  Restricted to
# superscript *digits*: an asterisk/dagger marker (``*†‡§``) is the shape a table
# note takes ("*Each value represents the mean …"), which belongs under its table,
# not in the article footnote run.  A digit run *immediately* (no space) followed by
# an uppercase letter is an isotope/mass-number — "²H NMR", "³⁵S-labeled", "¹⁸O" —
# i.e. body prose, not a marker, so it is excluded; a marker is instead followed by
# whitespace or non-uppercase content.  It shares its leading class with affiliation
# lines ("¹ Department of …"), so it is only consulted in the body — see
# ``seen_body_heading`` in ``_classify_parts`` — never on the first-page affiliations.
_UNICODE_SUP_MARKER_RE = re.compile(
    rf"^[{_SUP_DIGITS}]{{1,3}}(?![{_SUP_DIGITS}])(?:\s+|(?![A-Z]))\S"
)
_DOCUMENT_TYPE_LABELS = frozenset(
    {
        "abstract",
        "article",
        "research article",
        "original article",
        "letter",
        "review",
        "communication",
        "report",
        "brief communication",
        "short communication",
    }
)
# re.IGNORECASE: OCR output casing is unreliable ("REFERENCES", "References").
# <h\d[^>]*> tolerates class/id attributes the model may inject.
_REF_SECTION_RE = re.compile(
    r"^(?:<h\d[^>]*>\s*References\s*</h\d>|<p>\[1\])", re.IGNORECASE
)
# The references *heading* alone (no "<p>[1]" branch), tolerant of a leading
# section number ("7. References") and the common synonyms.  Used where a body
# block that merely opens "[1] …" (a numbered list, an inline citation) must not be
# mistaken for the start of the bibliography — e.g. the paragraph-merge's
# references guard, which only ever applies to author–year (un-numbered) entries;
# numbered "[n]" entries are already kept apart as enumeration items.
_REF_HEADING_RE = re.compile(
    r"^<h\d[^>]*>\s*(?:\d+\.?\s+)?"
    r"(?:references|bibliography|literature\s+cited|works\s+cited)\b",
    re.IGNORECASE,
)

# The article starts at the first page carrying an "Abstract"/"Introduction"
# heading; a cover ad / masthead has neither.
_ARTICLE_HEADING_RE = re.compile(
    r"^\s*(?:\d+[.)]?\s+)?(?:abstract|introduction)\b", re.IGNORECASE
)
# A markdown ATX heading line ("## Title"), capturing the heading text.
_MD_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+(.*)")
# Headings that may legitimately sit *inside* the front matter (between the
# abstract and the article body).  Everything up to the first heading that is
# neither one of these nor a document-type label is treated as front matter,
# so the boundary doesn't depend on the body's opening section being literally
# named "Introduction" (it may be "Background", numbered, non-English, etc.).
# A glossary the journal prints as a column-bottom footnote on the first page —
# "**Abbreviations:** ACT, …" / "**Nomenclature:** …".  OCR drops it inline,
# mid-section, where it splits the surrounding paragraph in two; the pre-classify
# stray sweep relocates it (before the paragraph merge) so the prose rejoins.
# Keywords is excluded *here*: it doubles as the abstract terminator the classifier
# reads, so removing it pre-classify leaks the following prose into the abstract;
# a stranded keyword line is instead relocated post-classify in
# ``_extract_front_matter`` (see ``_is_inline_frontmatter_label``).
_GLOSSARY_METADATA_LABELS = frozenset({"abbreviations", "nomenclature"})
# Front-matter section labels recognised as headings: the glossary labels plus the
# keyword labels, derived so _GLOSSARY_METADATA_LABELS provably stays a subset.
_FRONTMATTER_HEADING_LABELS = _GLOSSARY_METADATA_LABELS | frozenset(
    {"keywords", "key words"}
)
_SECTION_NUMBER_RE = re.compile(r"^\d+(?:[.)]\d*)*[.)]?\s+")
# A plain <p> is positively front matter when it carries a metadata label
# ("Keywords:", reuses _BOLD_LABEL_RE), opens with an affiliation/footnote
# superscript marker, or is a submission/correspondence/copyright line.
# `_SUP_DIGITS` (the canonical class, see text.py) + footnote symbols.
_LEADING_SUP_RE = re.compile(rf"^[{_SUP_DIGITS}*†‡§]")
_FRONTMATTER_TEXT_RE = re.compile(
    r"^(?:received|accepted|published|revised|doi|https?://|©|copyright|e-?mail|"
    r"(?:address\s+for\s+)?correspond(?:ence|ing\s+author))\b",
    re.IGNORECASE,
)
# A month, abbreviated or spelled out, fenced with ``(?![a-z])`` so a longer word
# that merely opens with a month prefix ("decided", "Mayor", "augment", "novel")
# is not read as a month — a bare ``(?:jan…dec)[a-z]*`` would swallow the whole
# word and flag "decided 5 2019" as a date.
_MONTH_NAME = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?"
    r"|dec(?:ember)?)\.?(?![a-z])"
)
# A calendar date in either ordering — "26 March 2019" (day-first, used by many
# journals) or "May 25, 2019" (month-first, ACS and other US journals).  The
# trailing year is fenced with ``(?!\d)`` rather than ``\b`` so it still matches
# when OCR glues the next line's first word straight onto the year ("2019Revised").
_DATE_RE = (
    r"\b(?:"
    r"\d{1,2}\s+" + _MONTH_NAME + r"\s+\d{4}"
    r"|" + _MONTH_NAME + r"\s+\d{1,2},?\s+\d{4}"
    r")(?!\d)"
)
_DATE_TOKEN_RE = re.compile(_DATE_RE, re.IGNORECASE)
# A token only metadata carries: an e-mail, a DOI/URL, a submission date, or a
# phone/fax label.  Its presence separates a genuine multi-clause metadata line
# ("Received … DOI: …", "Address for correspondence: … e-mail: …") from a body
# sentence that merely opens with a front-matter keyword ("Published reports
# indicate …"), which has none of these.
_METADATA_TOKEN_RE = re.compile(
    r"""
      \S+@\S+\.\S                                # e-mail address
    | doi:\s*10\.\d{4,}                          # DOI
    | https?://                                  # URL
    | """
    + _DATE_RE
    + r"""                                       # a calendar date
    | \b(?:tel|fax|phone)\b                      # phone / fax label
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Footer-metadata line shapes that carry fewer than two countable tokens yet are
# unambiguous on their own: a supporting-information note, a "DOI 10.…" line, a
# "Published online …" line, a "Volume N, … Pages N" journal citation, or an
# author-contribution footnote ("These authors contributed equally to this work").
# The OCR often splits a journal's page-bottom block into such one-line pieces, so
# none reaches the two-token bar alone.  The DOI, "Published online" and journal
# alternatives are anchored at the block *start* (and supporting-info is a fixed
# phrase), so body prose that merely mentions a volume/DOI/"published online"
# mid-sentence ("See volume 3, pages 45-67, …") does not match — only a line that
# opens as the citation/DOI/publication line does.  The author-contribution
# alternative is the one phrase not anchored at the block start — an author
# footnote reads "X and Y contributed equally [to this work]" mid-line — so it is
# instead anchored at its *end*: the clause must close the block (optionally with a
# trailing "to this/the work/study/…"), which a body sentence that merely runs
# "contributed equally to substrate binding …" past it does not.  The author-
# contribution clause is shared with the marker-restoration gate below
# (``_EQUAL_CONTRIBUTION_RE``), so both agree on what *is* such a footnote.
_EQUAL_CONTRIBUTION_PATTERN = (
    r"contributed\s+equally(?:\s+to\s+(?:this|the)\s+"
    r"(?:work|study|manuscript|article|paper|research|publication|project))?"
    r"\s*\.?\s*$"
)
_STRAY_METADATA_PHRASE_RE = re.compile(
    r"(?:additional\s+)?supporting information (?:may be found|is available)"
    r"|^\s*doi\b\s*:?\s*10\.\d"
    r"|^\s*published\s+online\b"
    r"|^\s*vol(?:\.|ume)?\s*\d+.*\bp(?:p\.?|ages?)\s*\d"
    # A submission-history line ("Received: May 25, 2019", "Accepted 12 July 2019",
    # "Received 19 June 2018; Revised 23 August 2018; …") — a publishing-process
    # label followed by a date that then *ends its entry*: the end of the line, a
    # newline (OCR merges the four ACS footer lines into one block, dates separated
    # by ``<br>``→newline), or a ";" (the separator in a one-line submission
    # history).  A body sentence opening with the keyword runs the date straight on
    # into lowercase prose ("Published May 25, 2019 in a leading journal, …"), which
    # none of those three terminators follows, so it stays in the body.
    r"|^\s*(?:received|revised|accepted|published|submitted)\b\s*:?\s*"
    + _DATE_RE
    + r"[ \t]*(?:\n|;|$)"
    + r"|"
    + _EQUAL_CONTRIBUTION_PATTERN,
    re.IGNORECASE,
)
# The footnote symbols a journal tags authors with (never numeric affiliation
# markers).  ``assemble._LEGEND_FOOTNOTE_MARKERS`` is the same set minus "*"
# (a "*"-led legend block is an italic organism name, not a footnote).
_FOOTNOTE_MARKER_CHARS = "*†‡§¶"
# A self-contained footer-metadata line (journal citation, correspondence, a
# "Received … DOI …" submission line, a supporting-information note) that OCR
# dropped into the body away from the leading front-matter run.  It is relocated
# on its own evidence, so the bar is high: a short block matching one of the fixed
# publication-line shapes above, or carrying two or more metadata tokens.  The
# token count rejects a body sentence that merely embeds one address/date; the
# length bound rejects a long prose run that happens to contain two.
_STRAY_METADATA_MAX_LEN = 400
_STRAY_METADATA_MIN_TOKENS = 2
# Publication-metadata headings that are bare banners, not label:value pairs:
# "OPEN ACCESS" carries no value paragraph (unlike "Citation"/"Editor"/…).  The
# value-capture clause in _extract_named_metadata_sections grabs the paragraph
# *directly* under any other label heading; for a banner it would swallow whatever
# body prose happens to follow, so a banner relocates on its own and never claims a
# trailing paragraph.  (Frontiers prints it atop its first-page sidebar.)
_PUBLICATION_BANNER_LABELS = frozenset({"open access"})
# A first-page metadata sidebar — the "Citation / Editor / Received / … / Competing
# interests" block PLOS and many open-access journals print beside the abstract —
# is OCR'd into the body away from the leading front-matter run, each entry a bold
# label-colon paragraph ("**Funding:** …").  A lone "**Word:**" line is too weak to
# relocate position-independently (a body paragraph may open "**Note:** …"), so
# these are recognised by a fixed vocabulary of publishing-process labels rather
# than by the bold-label shape alone.  The label is decisive evidence, so a matched
# line is relocated regardless of length (a Copyright/Funding statement runs long).
# Derived as the banner(s) | the label:value labels, so _PUBLICATION_BANNER_LABELS
# provably stays a subset; "specialty section" is Frontiers' sidebar routing line
# (among Edited by / Reviewed by / Correspondence / Citation), furniture not body.
_PUBLICATION_METADATA_LABELS = _PUBLICATION_BANNER_LABELS | frozenset(
    {
        "citation",
        "editor",
        "academic editor",
        "handling editor",
        "received",
        "accepted",
        "revised",
        "published",
        "copyright",
        "data availability",
        "data availability statement",
        "funding",
        "competing interests",
        "conflict of interest",
        "conflicts of interest",
        "provenance and peer review",
        "specialty section",
    }
)
# A publication banner the OCR bolds at the very start of a paragraph
# ("**OPEN ACCESS**" — sometimes its own block, sometimes glued onto the front of
# the abstract beneath it).  It carries no content (unlike the Frontiers heading
# form, which heads a sidebar of metadata), so it is stripped; any text after it on
# the same paragraph (the abstract) is kept.
_LEADING_BANNER_RE = re.compile(
    r"^<strong>\s*(?:"
    + "|".join(re.escape(b) for b in _PUBLICATION_BANNER_LABELS)
    + r")\s*</strong>\s*",
    re.IGNORECASE,
)
# Front matter is hidden in a collapsed panel, so misclassifying body prose as
# front matter makes it invisible.  A real prose paragraph under a metadata
# section is recognised by length + a sentence ending, and breaks the run.
_METADATA_PROSE_MIN_LEN = 80
# A metadata list (abbreviations, multi-affiliation / correspondence lines) packs
# several ";"-separated entries; body prose rarely uses more than one semicolon.
# This keeps such a list owned by its heading even when it is long enough to read
# like a sentence, and even when OCR splits the list across paragraphs.
_LIST_LIKE_MIN_SEMICOLONS = 2

_ABSTRACT_HEADING_RE = re.compile(r"^\s*abstract\b", re.IGNORECASE)
# Some journals (ACS) print the abstract with no heading, as a bold inline label
# on the paragraph itself ("**ABSTRACT**: …").  OCR places the colon inside or
# outside the bold (`<strong>ABSTRACT:</strong>` vs `<strong>ABSTRACT</strong>:`);
# match either and capture the label's end so it can be stripped.  The colon is
# *required* (in one position or the other) so a body paragraph merely opening with
# an emphasised word "Abstract" — no colon — is not mistaken for the abstract.
_INLINE_ABSTRACT_RE = re.compile(
    r"^<strong>\s*abstract\s*(?::\s*</strong>|</strong>\s*:)\s*", re.IGNORECASE
)
# Some journals run the article's copyright + journal citation onto the end of the
# abstract ("… University-Chico. © 2018 …, 47(2):124–132, 2019.").  That tail is front
# matter, not abstract prose; it is split off to the Metadata panel.  Anchored on the
# copyright *sign* followed by a year (a bare "(c)" is dropped — it false-matches
# chemistry/quantities like "(c) 2000 mg").  The prose group is *greedy* so the split
# lands on the **last** "© <year>" clause, not an earlier in-abstract "© 1998" mention.
_ABSTRACT_CITATION_TAIL_RE = re.compile(
    r"^(<p[^>]*>)(.*)(\s*©\s*\d{4}\b.*?)(</p>)\s*$",
    re.IGNORECASE | re.DOTALL,
)
# Affiliation / corresponding-author markers that accompany author names:
# superscript digits (`_SUP_DIGITS`, see text.py), a <sup>, or footnote symbols.
_AUTHOR_MARKER_RE = re.compile(rf"<sup>|[{_SUP_DIGITS}*†‡§]")
# A comma / "and" / ";"-separated author segment: a short, capitalized,
# digit-free name.
_NAME_SEGMENT_RE = re.compile(r"^[A-Z][^\d]*$")
# The delimiters a byline lists authors with — split a name list on any of them.
_NAME_LIST_SPLIT_RE = re.compile(r",|\s+and\s+|;")
# A lone-author byline ("Daniel D. Clark") has no comma to split on and no
# affiliation marker, so it is recognised as a "given-name … surname" frame: an
# alphabetic capitalised word at each end (``_NAME_ALPHA_WORD_RE``) with a *mid-name*
# initial between them (``_PERSONAL_NAME_INITIAL_RE``).  The mid-name initial is the
# positive author signal: a bare two-word name ("Jane Doe") is left ambiguous (it
# could be a subtitle) and stays in the body, and anchoring alphabetic words at both
# ends rejects an initialism-led phrase ("U.S. Army Corps").  Residual ambiguity
# remains for a Title-Case phrase that happens to share the exact frame ("Vitamin D.
# Levels") — only a lexicon could tell those apart — but such a block standing alone
# right after the title is rare.
_NAME_ALPHA_WORD_RE = re.compile(r"^[A-Z][a-z’'-]+$")
_PERSONAL_NAME_INITIAL_RE = re.compile(r"^(?:[A-Z]\.){1,3}$")
_PERSONAL_NAME_WORD_RE = re.compile(r"^(?:[A-Z]\.?|[A-Z][a-z’'-]+|(?:[A-Z]\.){2,})$")
_PERSONAL_NAME_MAX_WORDS = 5


def _is_title_heading(inner: str) -> bool:
    plain = _visible_text_folded(inner)
    return plain not in _DOCUMENT_TYPE_LABELS and not _ARTICLE_HEADING_RE.match(plain)


_MASTHEAD_MAX_WORDS = 4


def _is_masthead_heading(inner: str) -> bool:
    """A short all-caps multi-word heading — a journal masthead / running head
    ("PLOS ONE") the OCR emitted as a leading heading.  Multi-word so a minimal
    single-token title ("T") is never mistaken for one; the caller additionally
    requires a real title heading to follow before dropping it."""
    plain = _visible_text(inner).strip()
    return (
        2 <= len(plain.split()) <= _MASTHEAD_MAX_WORDS
        and plain == plain.upper()
        and plain.lower() != plain
    )


def _leading_title_follows(parts: list[str], idx: int) -> bool:
    """True when, skipping leading furniture headings (mastheads and document-type
    labels like "RESEARCH ARTICLE"), the next heading in the run is a real title —
    i.e. the heading at ``idx`` sits above the title, not at it.  A non-heading
    block before any title heading means there is none, so it returns False."""
    for k in range(idx + 1, len(parts)):
        heading = _heading_inner(parts[k])
        if heading is None:
            return False
        inner = heading[1]
        folded = _visible_text_folded(inner)
        if _is_masthead_heading(inner) or folded in _DOCUMENT_TYPE_LABELS:
            continue
        return _is_title_heading(inner)
    return False


def _looks_like_name_list(plain: str) -> bool:
    segments = [s.strip() for s in _NAME_LIST_SPLIT_RE.split(plain) if s.strip()]
    return len(segments) >= 2 and all(
        _NAME_SEGMENT_RE.match(s) and len(s.split()) <= _PERSONAL_NAME_MAX_WORDS
        for s in segments
    )


def _looks_like_personal_name(plain: str) -> bool:
    """A lone personal name ("Daniel D. Clark") — capitalised words framed by an
    alphabetic given name and surname with a mid-name initial between them."""
    words = plain.split()
    return (
        2 <= len(words) <= _PERSONAL_NAME_MAX_WORDS
        and all(_PERSONAL_NAME_WORD_RE.match(w) for w in words)
        and bool(_NAME_ALPHA_WORD_RE.match(words[0]))
        and bool(_NAME_ALPHA_WORD_RE.match(words[-1]))
        and any(_PERSONAL_NAME_INITIAL_RE.match(w) for w in words[1:-1])
    )


_BYLINE_EMPHASIS_RE = re.compile(r"</?(?:strong|em)>")
_BYLINE_MARKER_RE = re.compile(r"\*+")
_BR_TAG_RE = re.compile(r"<br\s*/?>")
# A trailing run of '*' is the unclosed-bold mis-pair leftover, collapsed to one marker.
_TRAILING_MARKER_RUN_RE = re.compile(r"\*+$")
# Past this many characters a "byline" is really a body paragraph, not an author line.
_BYLINE_MAX_CHARS = 400


def _byline_html(inner: str) -> str:
    """Authors as inline HTML, ``<br>``-separated affiliation runs joined with
    "; ".  Superscript/marker tags are kept (rendered as superscripts in the
    header), unlike ``_byline_text`` which flattens them for the byline predicate.

    A byline is bold/italic by *layout*, not intent, and its corresponding-author
    asterisks are footnote markers, not emphasis — but the OCR emits the line wrapped
    in ``**…**`` with the markers as bare ``*``, which markdown-it mis-pairs the
    opening ``**`` against, leaving a spurious ``<em>`` and a stray ``**``
    (``…ChangWoo Lee</em>**``).  So drop the layout emphasis (which also clears that
    mis-pairing) and re-cast each surviving literal ``*`` as a superscript marker.  A
    byline whose markers already arrived as ``<sup>`` (the ``$^{1,*}$`` LaTeX shape)
    carries no bare ``*`` and is left untouched."""
    inner = _BR_TAG_RE.sub("; ", inner).strip()
    inner = _BYLINE_EMPHASIS_RE.sub("", inner)
    if "<sup>" not in inner:
        # A *trailing* '*' run is the unclosed-bold mis-pair leftover (the real marker
        # was consumed into the emphasis pairing), so it stands for a single marker;
        # collapse it first.  An *inline* run is a genuine marker whose count can
        # distinguish authors ('*' vs '**'), so the wrap then keeps the run verbatim.
        inner = _TRAILING_MARKER_RUN_RE.sub("*", inner)
        inner = _BYLINE_MARKER_RE.sub(lambda m: f"<sup>{m.group()}</sup>", inner)
    return inner


def _byline_text(inner: str) -> str:
    return _visible_text(_byline_html(inner)).strip()


def _is_byline(inner: str) -> bool:
    """A block right after the title is the byline only when it positively looks
    like authors — it carries an affiliation/footnote marker or is a list of
    names.  Anything else (a date, DOI, journal line) falls through to the body
    rather than being silently moved into the header."""
    plain = _byline_text(inner)
    if not plain or len(plain) >= _BYLINE_MAX_CHARS or _SENTENCE_END_RE.search(plain):
        return False
    if _BOLD_LABEL_RE.match(inner):
        return False
    return (
        bool(_AUTHOR_MARKER_RE.search(inner))
        or _looks_like_name_list(plain)
        or _looks_like_personal_name(plain)
    )


def _is_article_page_md(md: str) -> bool:
    """A page is the article start if it carries an Abstract/Introduction
    heading (a cover ad / masthead has neither)."""
    for line in md.splitlines():
        m = _MD_HEADING_LINE_RE.match(line.strip())
        if m and _ARTICLE_HEADING_RE.match(_visible_text(m.group(1)).strip()):
            return True
    return False


def _leading_pages_to_skip_md(pages_md: list[str]) -> int:
    return next(
        (i for i, md in enumerate(pages_md) if _is_article_page_md(md)),
        0,
    )


@dataclass
class _Meta:
    title_html: str
    byline_html: str
    abstract: list[str]
    body: list[str]
    footnotes: list[str]


@dataclass
class _ClassifyState:
    """Mutable accumulator threaded through the single classify pass."""

    title_html: str = ""
    byline_html: str = ""
    abstract: list[str] = field(default_factory=list)
    body: list[str] = field(default_factory=list)
    footnotes: list[str] = field(default_factory=list)
    # The byline is only the block *immediately* after the title heading; this
    # window closes at the next block so a body sentence is never mistaken for it.
    expect_byline: bool = False
    in_abstract: bool = False
    # The article title is the only legitimate <h1>; a body section heading the
    # model mis-levelled as <h1> ("# Molecular Mass Determination of Lxmdh"
    # mid-Methods) is demoted to <h2> once the title is claimed, so the document
    # carries a single top-level heading.  ``seen_body_heading`` also gates the
    # unicode-superscript footnote sweep to the body proper.
    seen_body_heading: bool = False


def _is_body_footnote(inner_p: str, seen_body_heading: bool) -> bool:
    """A leading superscript-marker line that is an article footnote, not an
    affiliation: a "¹ Department of …, Country" affiliation shares the
    leading-marker shape but is front matter for the panel, never a footnote, even
    when OCR ordering drops it after the first body heading."""
    return bool(
        _SUP_MARKER_RE.match(inner_p)
        or (
            seen_body_heading
            and _UNICODE_SUP_MARKER_RE.match(inner_p)
            and not _is_affiliation_line(_visible_text(inner_p).strip())
        )
    )


def _classify_heading(
    state: _ClassifyState, level: int, inner: str, part: str, parts: list[str], idx: int
) -> None:
    if not state.title_html:
        # A masthead / running head ("PLOS ONE", which may sit above a "RESEARCH
        # ARTICLE" doc-type label) is dropped only when the real title heading
        # follows it in the leading heading run; a section heading like "Abstract"
        # after it means this heading IS the title.
        if _is_masthead_heading(inner) and _leading_title_follows(parts, idx):
            return
        if _is_title_heading(inner):
            state.title_html = inner
            state.expect_byline = True
            return
    state.expect_byline = False
    if _ABSTRACT_HEADING_RE.match(_visible_text(inner)):
        state.in_abstract = True
        return
    if _visible_text_folded(inner) in _DOCUMENT_TYPE_LABELS:
        return
    state.in_abstract = False
    if not (_is_named_metadata_heading(part) or _is_publication_metadata_heading(part)):
        state.seen_body_heading = True
    is_demoted_h1 = level == 1 and bool(state.title_html)
    state.body.append(f"<h2>{inner}</h2>" if is_demoted_h1 else part)


def _open_inline_abstract(state: _ClassifyState, remainder: str) -> None:
    # Open the abstract window even when the label stood alone (remainder empty): the
    # following paragraph is the abstract body.  Appending the empty remainder would
    # leak a stray "<p></p>" into the rendered abstract box (the sibling banner branch
    # in _classify_paragraph guards this the same way).
    state.in_abstract = True
    if remainder:
        state.abstract.append(f"<p>{remainder}</p>")


def _classify_paragraph(state: _ClassifyState, part: str) -> None:
    inner_p = _plain_p_text(part)
    # Strip a leading publication banner ("**OPEN ACCESS**") the OCR bolded onto
    # a paragraph: drop the block if that is all it was, else keep the remainder
    # (the abstract it was glued to) and reprocess it as a normal paragraph.
    if inner_p is not None and (m := _LEADING_BANNER_RE.match(inner_p)):
        inner_p = inner_p[m.end() :].lstrip()
        if not inner_p:
            return
        part = f"<p>{inner_p}</p>"
    if state.expect_byline:
        state.expect_byline = False
        if inner_p is not None and _is_byline(inner_p):
            state.byline_html = _byline_html(inner_p)
            return
    # A headingless abstract carried as an inline "ABSTRACT: …" bold label: strip
    # the label and route the paragraph to the abstract section, before it can be
    # mistaken for a bold-label front-matter line and swept into the panel.  Open
    # the abstract window (like the heading path) so a multi-paragraph abstract's
    # continuation is captured too, closing at the next bold label / heading.
    if inner_p is not None and (m := _INLINE_ABSTRACT_RE.match(inner_p)):
        _open_inline_abstract(state, inner_p[m.end() :].lstrip())
        return
    if state.in_abstract:
        if inner_p is not None and not _BOLD_LABEL_CAPTURE_RE.match(inner_p):
            state.abstract.append(part)
            return
        state.in_abstract = False
    if inner_p is not None and _is_body_footnote(inner_p, state.seen_body_heading):
        state.footnotes.append(f'<p class="footnote">{inner_p}</p>')
        return
    if inner_p is not None and _is_degenerate_repetition(inner_p):
        return
    state.body.append(part)


def _classify_parts(parts: list[str]) -> _Meta:
    """Single pass: pull the title, byline, abstract and footnotes out of the
    flat block list; everything else is body."""
    state = _ClassifyState()
    for idx, part in enumerate(parts):
        heading = _heading_inner(part)
        if heading is not None:
            _classify_heading(state, heading[0], heading[1], part, parts, idx)
        else:
            _classify_paragraph(state, part)
    return _Meta(
        state.title_html,
        state.byline_html,
        state.abstract,
        state.body,
        state.footnotes,
    )


def _heading_title(part: str) -> str | None:
    """A heading's section-number-stripped, case-folded title for label-membership
    tests, or ``None`` when the block isn't a heading.  Shared by the metadata-heading
    predicates so each tests the same normalized title against its own label set."""
    heading = _heading_inner(part)
    if heading is None:
        return None
    return _SECTION_NUMBER_RE.sub("", _visible_text_folded(heading[1]))


# Canonical top-level section names that are *never* subsections — used to anchor a
# mis-leveled body section heading back to <h2>.  Two deliberate exclusions keep the
# conservative pass from ever promoting a real subsection:
#   - ambiguous back-matter ("Author Contributions", "Notes", "ORCID", "Funding") that
#     appears as an <h3> under an "Author Information" section; and
#   - the bare single-word IMRaD names "Methods"/"Results"/"Discussion", which a paper
#     can nest as a subsection (e.g. "Results"/"Discussion" subsections under a combined
#     "Results and Discussion", or a "Methods" subsection under "Study Design") — only
#     the *compound* forms ("Materials and Methods", "Results and Discussion") are
#     unambiguously top-level.
# The article title is a unique string, never one of these, so this never demotes a
# title that classify left in body.
_TOP_LEVEL_SECTION_NAMES = frozenset(
    {
        "abstract",
        "introduction",
        "results and discussion",
        "conclusion",
        "conclusions",
        "materials and methods",
        "experimental section",
        "experimental procedures",
        "acknowledgment",
        "acknowledgments",
        "acknowledgement",
        "acknowledgements",
        "references",
    }
)
# A section-numbered heading ("2. Materials and Methods", "2.1. Plant materials"): the
# dotted-number depth sets the level deterministically (depth 1 -> <h2>, 2 -> <h3>, …).
# Two guards keep a leading *quantity* from reading as a section number — the
# conservative pass must never re-level a real heading off a false match:
#   - a *mandatory* trailing separator [.)] (a section number is "2." / "(2)" / "3.4.";
#     a quantity "0.5 M NaCl" / "5 mM Buffer" has none before the title word), and
#   - a non-zero leading component [1-9] (sections start at 1; "0.5 …"/"0.1% …" do not).
# Components are 1-2 digits so a bare year ("2019 …") or isotope ("3D …") can't match.
_HEADING_NUMBER_RE = re.compile(r"^\(?([1-9]\d?(?:\.\d{1,2})*)[.)]\s+\S")


def _normalized_heading_level(inner: str, level: int) -> int:
    """The level a body heading *should* render at, from high-confidence signals only;
    otherwise the OCR's own ``level`` (never a guess — see
    ``_normalize_heading_levels``)."""
    title = _visible_text(inner).strip()
    m = _HEADING_NUMBER_RE.match(title)
    if m is not None:
        depth = m.group(1).count(".") + 1
        return min(1 + depth, 4)
    folded = _SECTION_NUMBER_RE.sub("", _visible_text_folded(inner))
    if folded in _TOP_LEVEL_SECTION_NAMES:
        return 2
    return level


def _normalize_heading_levels(body: list[str]) -> list[str]:
    """Re-level body section headings the OCR leveled inconsistently, using only
    high-confidence signals so a real section is never demoted:

    - a dotted section number sets the level by its depth (``2.`` -> ``<h2>``,
      ``2.1.`` -> ``<h3>``, ``2.1.1`` -> ``<h4>``) — fixes a sibling subsection the
      model jittered up a level (31051047 ``3.4``/``3.5`` emitted as ``<h2>``);
    - a canonical top-level section name (Introduction, Results, References, …) is
      anchored to ``<h2>`` — fixes a section the model leveled as ``<h1>``/``<h3>``.

    Every other heading keeps the OCR's level (its reading order is always right; only
    the depth jitters), so an unrecognized real section — a journal-specific section
    name, an unnumbered subsection — is left alone rather than guessed at and possibly
    demoted.  Reading order is unchanged; only the ``<hN>`` tag of a heading block."""
    out: list[str] = []
    for part in body:
        heading = _heading_inner(part)
        if heading is None:
            out.append(part)
            continue
        level, inner = heading
        new_level = _normalized_heading_level(inner, level)
        out.append(
            part if new_level == level else f"<h{new_level}>{inner}</h{new_level}>"
        )
    return out


def _is_metadata_heading(part: str) -> bool:
    """A heading that is itself front matter ("Abbreviations", "Keywords") or a
    document-type label, as opposed to the heading that opens the body proper."""
    title = _heading_title(part)
    return title is not None and (
        title in _FRONTMATTER_HEADING_LABELS or title in _DOCUMENT_TYPE_LABELS
    )


def _is_named_metadata_heading(part: str) -> bool:
    """A heading naming a front-matter section ("Abbreviations", "Nomenclature",
    "Keywords") — the subset of metadata headings that can be located by name,
    excluding the document-type labels."""
    title = _heading_title(part)
    return title is not None and title in _FRONTMATTER_HEADING_LABELS


def _is_publication_metadata_heading(part: str) -> bool:
    """A heading whose text is a journal-sidebar label ("Citation", "Editor",
    "Received" …).  LightOnOCR renders the PLOS-style first-page metadata sidebar
    as a run of such label headings, each followed by its value paragraph(s),
    rather than the bold ``**Label:**`` lines other journals' OCR produces."""
    title = _heading_title(part)
    return title is not None and title in _PUBLICATION_METADATA_LABELS


def _is_publication_value_heading(part: str) -> bool:
    """A publication-metadata heading that takes a value paragraph ("Citation",
    "Editor", …), as opposed to a bare banner ("OPEN ACCESS").  Only such a heading
    owns the paragraph directly beneath it; a banner does not (see
    ``_PUBLICATION_BANNER_LABELS``)."""
    title = _heading_title(part)
    return (
        title is not None
        and title in _PUBLICATION_METADATA_LABELS
        and title not in _PUBLICATION_BANNER_LABELS
    )


def _is_frontmatter_text(part: str, *, strict: bool = True) -> bool:
    inner = _plain_p_text(part)
    if inner is None:
        return False
    if _BOLD_LABEL_RE.match(inner):
        return True
    plain = _visible_text(inner).lstrip()
    if _LEADING_SUP_RE.match(plain):
        return True
    if _is_affiliation_line(plain):
        return True
    if not _FRONTMATTER_TEXT_RE.match(plain):
        return False
    # _ends_sentence (not raw _SENTENCE_END_RE) so a body paragraph that merely opens
    # with a front-matter keyword but ends in a citation superscript still reads as a
    # finished sentence — otherwise the hidden period makes it look label-like and it
    # is swept into the metadata panel, defeating the _looks_like_body_prose guard.
    if not _ends_sentence(inner):
        return True
    # The line runs on like a sentence: either a body paragraph that merely opens
    # with the keyword ("Published reports indicate ….") or a genuine multi-clause
    # metadata line ("Received … DOI: …", "Address for correspondence: … e-mail:
    # …").  At the top level (strict) refuse it; inside an explicitly-headed
    # metadata section (strict=False) trust it only when it carries a metadata
    # token, which a body sentence does not.
    return not strict and bool(_METADATA_TOKEN_RE.search(plain))


def _looks_like_body_prose(part: str) -> bool:
    """A substantial plain-paragraph sentence — used to end a metadata section's
    sticky run so unheaded body prose isn't swallowed into the hidden panel.

    Front-matter exclusion is the caller's job: the one call site already gates
    this on ``not _is_frontmatter_text(part, strict=False)``, so re-testing it
    here would just strip tags and re-run the regexes a second time."""
    inner = _plain_p_text(part)
    if inner is None:
        return False
    # _ends_sentence (not a raw _SENTENCE_END_RE search) so a body sentence whose
    # terminal period sits behind a trailing citation superscript
    # ("…studied here.<sup>15</sup>") still ends the metadata run rather than being
    # swallowed into the hidden panel.
    text = _visible_text(inner)
    return len(text) > _METADATA_PROSE_MIN_LEN and _ends_sentence(inner)


def _is_list_like(part: str) -> bool:
    """A plain ``<p>`` of several ``;``-separated entries — an abbreviation list
    or a multi-affiliation line — as opposed to a body sentence."""
    inner = _plain_p_text(part)
    if inner is None:
        return False
    return _visible_text(inner).count(";") >= _LIST_LIKE_MIN_SEMICOLONS


def _front_matter_len(body: list[str]) -> int:
    """Length of the leading run of recognised front-matter blocks.

    Only positively-identified metadata is counted — affiliation/superscript
    paragraphs, labelled lines (keywords, correspondence, dates) and metadata
    headings with the content they introduce.  The run stops at the first block
    that is none of these, so a body opening with unlabelled prose (or one whose
    first section heading lacks a name we recognise) is never mistaken for front
    matter and relocated."""
    n = 0
    in_metadata_section = False
    for i, part in enumerate(body):
        if _heading_inner(part) is not None:
            if not _is_metadata_heading(part):
                break
            in_metadata_section = True
        elif in_metadata_section:
            # A metadata heading owns its content up to the next heading: a
            # keyword-led metadata line (correspondence, dates) or a ";"-separated
            # list (an abbreviation list, even one long enough to read like a
            # sentence or split across paragraphs).  Only a genuine unheaded body
            # paragraph — which is neither — ends the run.
            if (
                _looks_like_body_prose(part)
                and not _is_frontmatter_text(part, strict=False)
                and not _is_list_like(part)
            ):
                break
        elif not _is_frontmatter_text(part):
            break
        n = i + 1
    return n


def _extract_front_matter(body: list[str]) -> tuple[list[str], list[str]]:
    """Split the leading run of front-matter blocks off the body.

    Affiliations, keywords, abbreviations, the corresponding-author block and
    submission dates are OCR'd between the abstract and the body's first section,
    so the body opens with metadata rather than prose.  The leading run of
    recognised front-matter blocks is pulled out (the caller surfaces it in a
    collapsible "Metadata" panel after the abstract) so the body opens with
    prose.  Returns ``(front_matter, rest)``."""
    n = _front_matter_len(body)
    # Never hide the entire body: an all-front-matter classification signals a
    # detection failure, not a metadata-only document.
    if n >= len(body):
        return [], body
    # Beyond the leading run, a labelled front-matter line that survived into the
    # body — a keyword line stranded after a *headingless* abstract that stayed in
    # the body, or a publication-metadata label ("Citation:", "Editor:") the
    # pre-classify sweep missed because a stripped banner had hidden it behind the
    # leading <strong> anchor — is unambiguous front matter, so relocate it too.
    # Pulled here, after classify, not in the pre-classify stray sweep, because the
    # keyword label doubles as the abstract terminator there.  Scoped to the leading
    # run *before the first section heading*: past that we are in the body proper,
    # where a labelled line is a back-matter glossary ("Abbreviations" at the article
    # end) that belongs with its own heading, not yanked to the panel.
    trailing: list[str] = []
    kept: list[str] = []
    in_lead = True
    for part in body[n:]:
        if in_lead and _heading_inner(part) is not None:
            in_lead = False
        inner = _plain_p_text(part)
        if (
            in_lead
            and inner
            and (
                _is_inline_frontmatter_label(inner)
                or _is_publication_metadata_label(inner)
            )
        ):
            trailing.append(part)
        else:
            kept.append(part)
    return list(body[:n]) + trailing, kept


def _recover_headingless_abstract(body: list[str]) -> tuple[list[str], list[str]]:
    """Promote the leading prose run that precedes the first section heading to the
    abstract — returns ``(abstract, rest)``.

    Some journals (Frontiers, Bioscience Reports) print the abstract with neither an
    "Abstract" heading nor an inline bold ``**ABSTRACT:**`` label, so the classifier's
    cued paths miss it and it lands at the top of the body.  Run only as a fallback
    (when no abstract was otherwise found) and *after* front-matter extraction, so the
    affiliation/keyword furniture is already gone and the run opens on the abstract.

    Conservative against the never-hide-the-whole-body rule: only substantial
    body-prose paragraphs (no front-matter or list-like block), and only when a
    section heading immediately follows the run — a heading-less short note, which may
    carry no abstract at all, is left untouched."""
    n = 0
    for part in body:
        if _heading_inner(part) is not None:
            break
        if (
            _looks_like_body_prose(part)
            and not _is_frontmatter_text(part, strict=False)
            and not _is_list_like(part)
        ):
            n += 1
            continue
        break
    if n == 0 or n >= len(body) or _heading_inner(body[n]) is None:
        return [], body
    return list(body[:n]), list(body[n:])


def _split_abstract_citation(abstract: list[str]) -> tuple[list[str], list[str]]:
    """Split a trailing copyright/journal-citation clause off the abstract.

    Returns ``(abstract, tail)``: the abstract with the clause removed from its last
    paragraph, and ``tail`` a (possibly empty) list of blocks for the caller to put in
    the Metadata panel — the copyright/citation is front matter, not abstract prose."""
    if not abstract:
        return abstract, []
    m = _ABSTRACT_CITATION_TAIL_RE.match(abstract[-1])
    # Require real prose before the clause: a citation-*only* paragraph would otherwise
    # leave an empty ``<p></p>`` stub, and is too degenerate to be a real abstract.
    if m is None or not m.group(2).strip():
        return abstract, []
    prose = m.group(1) + m.group(2).rstrip() + m.group(4)
    return abstract[:-1] + [prose], [f"<p>{m.group(3).strip()}</p>"]


def _take_named_section(parts: list[str], i: int) -> tuple[list[str], int]:
    """The contiguous glossary-metadata run a named heading at ``parts[i]`` owns: the
    heading plus the following ";"-separated / labelled metadata blocks, stopping at
    the first heading, figure, list, table or body paragraph.  Returns
    ``(blocks, next_i)`` — or ``([], i)`` when ``parts[i]`` is not such a heading *or*
    the heading owns no metadata content (probably a real section title, left in the
    body so misfiled content isn't hidden)."""
    if not _is_named_metadata_heading(parts[i]):
        return [], i
    j = i + 1
    while j < len(parts) and (
        _is_frontmatter_text(parts[j], strict=False) or _is_list_like(parts[j])
    ):
        j += 1
    if j == i + 1:
        return [], i
    return parts[i:j], j


def _take_publication_sidebar(parts: list[str], i: int) -> tuple[list[str], int]:
    """The contiguous publication-sidebar run a label heading at ``parts[i]`` owns,
    returned as ``(blocks, next_i)`` — or ``([], i)`` when ``parts[i]`` is not a
    sidebar label heading.

    A sidebar label heading owns the ``<p>`` value(s) directly under it, and the
    sidebar is a contiguous run of such headings.  This pulls label headings and their
    values — but a ``<p>`` that neither directly follows a *value* label heading nor is
    itself recognised front matter ends the run, so trailing body prose is never swept
    into the hidden panel.  A bare banner ("OPEN ACCESS") owns no value paragraph, so it
    does not claim the ``<p>`` under it — only an independently-recognised front-matter
    ``<p>`` follows it into the panel.  (Unlike a named section, the heading itself is
    always taken, so this never returns an empty run for a matched heading.)"""
    if not _is_publication_metadata_heading(parts[i]):
        return [], i
    j = i + 1
    while j < len(parts):
        nxt = parts[j]
        if _is_publication_metadata_heading(nxt):
            j += 1
            continue
        if (
            _heading_inner(nxt) is None
            and _plain_p_text(nxt) is not None
            and (
                _is_publication_value_heading(parts[j - 1])
                or _is_frontmatter_text(nxt, strict=False)
            )
        ):
            j += 1
            continue
        break
    return parts[i:j], j


def _extract_named_metadata_sections(parts: list[str]) -> tuple[list[str], list[str]]:
    """Pull glossary-style metadata sections out of a single page's block stream,
    wherever they sit — returns ``(metadata, rest)``.

    OCR often places an "Abbreviations"/"Nomenclature" section *after* the
    article's opening prose rather than in the leading front matter, so
    ``_extract_front_matter`` (which only scans the leading run) never reaches it.
    The caller scopes this to the article's first page, so a same-named section
    deeper in the document (e.g. a back-matter "Nomenclature") stays in place.

    Each block is offered to the two section-takers in turn (a named glossary section,
    then a publication sidebar); the first that claims a run consumes it into the
    metadata, otherwise the block stays in the body.  A named heading that owns no
    metadata content falls through to the sidebar check (a heading can be both)."""
    metadata: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(parts):
        for take in (_take_named_section, _take_publication_sidebar):
            blocks, j = take(parts, i)
            if blocks:
                metadata.extend(blocks)
                i = j
                break
        else:
            rest.append(parts[i])
            i += 1
    return metadata, rest


def _bold_label_in(inner: str, labels: frozenset[str]) -> bool:
    """True when ``inner`` opens with a ``**Label:**`` bold label whose name is in
    ``labels`` — the shared shape of the metadata-label checks below."""
    m = _BOLD_LABEL_CAPTURE_RE.match(inner)
    if m is None:
        return False
    name = m.group(1) if m.group(1) is not None else m.group(2)
    return name.strip().lower() in labels


def _is_publication_metadata_label(inner: str) -> bool:
    """A journal-metadata bold label ("**Citation:**", "**Competing interests:**" …
    — see ``_PUBLICATION_METADATA_LABELS``).  (When the OCR emits the sidebar as label
    *headings* instead, ``_extract_named_metadata_sections`` relocates it.)"""
    return _bold_label_in(inner, _PUBLICATION_METADATA_LABELS)


def _is_glossary_metadata_label(inner: str) -> bool:
    """An inline glossary bold label ("**Abbreviations:**", "**Nomenclature:**" —
    see ``_GLOSSARY_METADATA_LABELS``).  Decisive regardless of the entry list's
    length, so it is matched before the stray-metadata length cap."""
    return _bold_label_in(inner, _GLOSSARY_METADATA_LABELS)


def _is_inline_frontmatter_label(inner: str) -> bool:
    """An inline bold label naming any front-matter section ("**Abbreviations:**",
    "**Keywords:**", "**Nomenclature:**") — the same labels recognised as headings
    (``_FRONTMATTER_HEADING_LABELS``).

    Broader than ``_is_glossary_metadata_label`` (it also matches keywords); used by
    ``_extract_front_matter`` *after* classify to relocate a labelled front-matter
    line that survived past the leading run, so it never displaces the keyword line's
    pre-classify role as the abstract terminator."""
    return _bold_label_in(inner, _FRONTMATTER_HEADING_LABELS)


def _is_stray_metadata(part: str) -> bool:
    """A self-contained footer-metadata line OCR'd into the body (see
    ``_STRAY_METADATA_MAX_LEN``).  Unlike ``_is_frontmatter_text`` it is judged
    position-independently, so it must stand on its own strong evidence: a
    recognised publishing-process label, a short block with two or more metadata
    tokens, or the fixed boilerplate phrase."""
    inner = _plain_p_text(part)
    if inner is None:
        return False
    if _is_publication_metadata_label(inner) or _is_glossary_metadata_label(inner):
        return True
    plain = _visible_text(inner)
    # An affiliation is front matter wherever it sits — relocate it even when the
    # leading run that _extract_front_matter scans was broken (e.g. a masthead
    # heading pushed the title, byline and affiliation down into the body).
    if _is_affiliation_line(plain):
        return True
    if len(plain) > _STRAY_METADATA_MAX_LEN:
        return False
    if _STRAY_METADATA_PHRASE_RE.search(plain):
        return True
    tokens = _METADATA_TOKEN_RE.findall(plain)
    if len(tokens) < _STRAY_METADATA_MIN_TOKENS:
        return False
    # A calendar date is weak evidence when judged position-independently: body
    # prose routinely cites date ranges ("between March 3, 2001 and December 12,
    # 2004"), which would otherwise reach the two-token bar on dates alone and be
    # hidden in the panel.  Require at least one non-date token; a genuine date-only
    # submission footer is caught by the keyword-anchored phrase above.
    return any(not _DATE_TOKEN_RE.fullmatch(tok) for tok in tokens)


# An author-contribution footnote ("These authors contributed equally …") refers
# to the authors the byline tags with a shared footnote symbol, but the OCR
# swallows that symbol by wrapping the note in "*…*" emphasis (rendered <em>…</em>,
# no visible marker).  When the note is relocated to the panel, restore its marker
# so it still references those authors — derived from the byline rather than
# assumed, since journals use any of these symbols, not always "*".  The gate is
# the *same* end-anchored clause the relocation predicate uses, so the marker is
# only prepended to a standalone note (not to an unrelated block that merely
# mentions "contributed equally" mid-text).  (When the byline carries two symbols
# each on ≥2 authors — e.g. "*" corresponding, "†" equal-contribution — the note's
# own marker is unrecoverable, so the choice is a best-effort, not exact.)
_EQUAL_CONTRIBUTION_RE = re.compile(_EQUAL_CONTRIBUTION_PATTERN, re.IGNORECASE)


def _byline_equal_contribution_marker(parts: list[str]) -> str | None:
    """The footnote symbol the byline puts on ≥2 authors — the one an equal-
    contribution note refers to (plural "these authors").  ``None`` when the byline
    carries no shared symbolic marker, so no marker is invented.  Numeric
    affiliation markers are ignored; only ``_FOOTNOTE_MARKER_CHARS`` count."""
    for part in parts:
        inner = _plain_p_text(part)
        if inner is None or not _is_byline(inner):
            continue
        text = _visible_text(inner)
        counts = Counter(ch for ch in text if ch in _FOOTNOTE_MARKER_CHARS)
        shared = [ch for ch, n in counts.items() if n >= 2]
        if not shared:
            return None
        # The marker on the most authors; ties broken by first appearance so the
        # choice is deterministic.
        return min(shared, key=lambda ch: (-counts[ch], text.index(ch)))
    return None


def _restore_equal_contribution_marker(part: str, marker: str | None) -> str:
    if marker is None:
        return part
    inner = _plain_p_text(part)
    if inner is None:
        return part
    plain = _visible_text(inner)
    if not _EQUAL_CONTRIBUTION_RE.search(plain) or (
        (h := plain.lstrip()) and h[0] in _FOOTNOTE_MARKER_CHARS
    ):
        return part
    return f"<p>{marker}{inner}</p>"


def _extract_stray_metadata(parts: list[str]) -> tuple[list[str], list[str]]:
    """Split self-contained stray metadata blocks (see ``_is_stray_metadata``) off
    a page's block stream — returns ``(metadata, rest)``.

    The caller scopes this to the first article page and runs it *before* the
    paragraph-merge, so a footer line ending in ")" (e.g. "… Published online …
    (wileyonlinelibrary.com)") is pulled out before the merge can glue the
    following body prose onto it."""
    marker = _byline_equal_contribution_marker(parts)
    metadata: list[str] = []
    rest: list[str] = []
    for part in parts:
        if _is_stray_metadata(part):
            metadata.append(_restore_equal_contribution_marker(part, marker))
        else:
            rest.append(part)
    return metadata, rest
