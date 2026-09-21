"""LaTeX and inline-markdown → HTML conversion.

Leaf module, no GPU or IO.  Covers the deterministic text transforms applied to
LightOnOCR's markdown: bold/italic, and inline ``$…$`` spans reduced to Unicode
super/subscripts and HTML ``<sup>``/``<sub>`` (full math is out of scope — a
later MathJax option).  A converted span also italicises its standalone
single-letter variables (math convention) while leaving numbers, operators, and
multi-letter identifiers upright.  Symbol-command translation is delegated to
pylatexenc's maintained macro table rather than a hand-curated map.
"""

from __future__ import annotations

import functools
import logging
import re

from pylatexenc.latex2text import LatexNodes2Text  # type: ignore[import-untyped]

from manuscribe.pipeline.text import _SUP_DIGITS

_log = logging.getLogger(__name__)

# Unicode superscript forms.  A LaTeX ``$^{…}$`` run is rendered with these
# glyphs when every character has one (so "NAD$^+$" → "NAD⁺"); otherwise it falls
# back to an HTML <sup> wrapper so nothing is lost.
_SUPERSCRIPT_MAP = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "−": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "°": "°",
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "g": "ᵍ",
    "h": "ʰ",
    "i": "ⁱ",
    "j": "ʲ",
    "k": "ᵏ",
    "l": "ˡ",
    "m": "ᵐ",
    "n": "ⁿ",
    "o": "ᵒ",
    "p": "ᵖ",
    "r": "ʳ",
    "s": "ˢ",
    "t": "ᵗ",
    "u": "ᵘ",
    "v": "ᵛ",
    "w": "ʷ",
    "x": "ˣ",
    "y": "ʸ",
    "z": "ᶻ",
}

# Symbol-command translation is delegated to pylatexenc's maintained macro
# table rather than a hand-curated map.  A command is matched as a maximal
# ``\name`` token and looked up whole, so a command can never eat the head of a
# longer one ("\to" vs "\top", "\sim" vs "\simeq").
_LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+")
# "\S" immediately before a digit is the model misreading the leading "S" of a
# supplementary-material label ("S4 Fig.", "S1 Raw images.") as the section command —
# it means the letter S, not "§".  A whole span the model wrapped in math mode
# ("$\S4$") is rewritten to the plain identifier ("S4") at the top level so it skips
# math-variable italicization; an embedded "\S<digit>" is still rewritten to "S" inside
# the span.  A standalone "\S" (a real footnote/section marker) still converts to "§".
_LATEX_S_LABEL_SPAN_RE = re.compile(r"(?<!\\)\$\\S ?(\d[\w.]*)\$")
_LATEX_S_LABEL_RE = re.compile(r"\\S ?(?=\d)")
# The same misread, but the model emitted the *resolved* section sign character ("§4
# Fig.") rather than the "\S" command.  Scoped to a line/block start (the label
# position) + optional leading bold markers, so a genuine mid-sentence "§3" reference
# is left intact while a supplementary label "§4 Fig." / "**§1 Raw images.**" is
# recovered to "S4"/"S1".
_LITERAL_S_LABEL_RE = re.compile(r"(?m)^(\*{0,2})§ ?(?=\d)")
_L2T = LatexNodes2Text()
# pylatexenc returns "" for a few common symbol macros (version-dependent), which
# would leak the raw "\ddagger"/"\S" into author/affiliation footnote markers.
# Supplied as a whole-token fallback so "\S" can never corrupt "\Section".  "\|"
# (non-alphabetic, so never matched by _LATEX_COMMAND_RE) is handled separately.
_LATEX_SYMBOL_FALLBACK = {
    r"\ddagger": "‡",
    r"\S": "§",
    r"\P": "¶",
}
_LATEX_VERT_RE = re.compile(r"\\\|")

# The degree idiom is the one place we override pylatexenc: "^\circ" means
# *degrees* ("°"), but \circ on its own is the ring operator ("∘"), which is
# what pylatexenc (correctly, for general LaTeX) returns.  \degree is unknown to
# pylatexenc, so it is handled here too.  This runs before script handling so the
# ``^`` is consumed and the result isn't wrapped in <sup>.
_LATEX_DEGREE_RE = re.compile(r"\^\s*\{?\s*\\circ\s*\}?|\\degree(?![a-zA-Z])")


@functools.cache
def _latex_command_to_unicode(command: str) -> str:
    """Translate a single no-arg ``\\name`` symbol command to its Unicode glyph.

    Only symbol macros are in scope.  Each command is matched and looked up in
    isolation, but pylatexenc parses full LaTeX *with* arguments — fed a bare
    arg-taking macro it either raises (``\\sqrt`` → KeyError) or returns its
    substitution template (``\\frac`` → ``"%s/%s"``).  Treat any such case, and
    an unknown macro (empty result), as untranslatable and keep the command
    literal so real math survives intact for a later MathJax pass rather than
    crashing the page or leaking ``%s`` garbage."""
    try:
        text = str(_L2T.latex_to_text(command)).strip()
    except Exception:
        # pylatexenc raises assorted errors on a malformed/unknown macro; any failure
        # means "untranslatable" — keep the command literal (per the docstring) so a
        # later MathJax pass can still render it, rather than crash the page.
        return command
    if not text or "%" in text:
        return _LATEX_SYMBOL_FALLBACK.get(command, command)
    return text


# A literal '*'/'_' surviving a converted math span (an author footnote marker
# like "1,*", or a multiplication "a*b") would be re-read as Markdown emphasis by
# the downstream inline parser, which pairs two such characters and italicises the
# text between them.  The span's emitted HTML (<sup>/<sub> tags, Unicode command
# glyphs) carries neither character, so escaping the whole converted span to HTML
# entities neutralises every residual marker without touching the structure.
_MD_EMPHASIS_ESCAPE = str.maketrans({"*": "&#42;", "_": "&#95;"})


def _to_superscript(core: str) -> str:
    if all(ch in _SUPERSCRIPT_MAP or ch.isspace() for ch in core):
        return "".join(_SUPERSCRIPT_MAP.get(ch, ch) for ch in core)
    return f"<sup>{core}</sup>"


# An inline math span: $…$ not preceded by a backslash, shortest match, on a
# single line (no DOTALL — a stray '$' must not swallow across paragraphs).  A
# single space *before* the span is captured so a span that opens with a script
# ("Sec $^{-1}$") can re-attach to the preceding token instead of stranding the
# superscript after a gap.
_LATEX_SPAN_RE = re.compile(r"( ?)(?<!\\)\$([^\n$]+)(?<!\\)\$")
# Only spans that actually contain TeX (a sub/superscript or a command) are
# converted; a paired '$' around plain text (e.g. currency "$5 … $10") is left
# untouched rather than stripped.
_LATEX_MATH_RE = re.compile(r"[_^\\]")
# A span whose content is a lone number ("$42.26$") is the model wrapping a plain
# value in math mode, so the delimiters are dropped and the number kept.  Currency
# is safe: the regex only pairs '$' spuriously across words ("$5 … $10"), whose
# content is never a lone number.
_BARE_NUMBER_RE = re.compile(r"[-+±]?\d[\d.,]*%?")
# A markup-free span can still be a plain inline equation/variable the model
# wrapped in math mode ("$x = 22$", "$a = 84.9$", "$P < 0.05$", "$a, b, c$"); its
# delimiters should drop like a bare number's.  See ``_is_inline_math_span``.
_MATH_RELATION_RE = re.compile(r"[=<>≤≥≈≠]")
_MATH_EXPR_RE = re.compile(r"[\w.,()\s=<>≤≥≈≠+\-−×⋅*/±°]+")
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]{2,}")
# _latex_to_html runs on the pre-markdown stream, which still carries raw HTML
# (<sup>, <sub>, <td>, <br>…).  The math-relation class above includes '<'/'>',
# which collide with a tag's angle brackets, so a span that straddles a tag would
# read as containing a relation; reject any span holding a complete HTML tag so a
# stray '$' pairing across markup isn't unwrapped (which drops a currency '$' and
# mangles the tag).  A bare inequality ("a < b") has no '>'-closed tag, so it is
# unaffected.
_HTML_TAG_RE = re.compile(r"</?\w[^>]*>")
# A parenthesized CIP stereodescriptor the model wraps in math mode ("$(R)$",
# "$(S)$", "$(R,S)$", "$(R/S)$"): one or more single descriptor letters (R/S
# absolute configuration, E/Z double-bond geometry), comma- or slash-separated.
# This is the only *bare* paren-led span unwrapped — a figure panel label "(A)" or
# a roman list item "(i)" has the same shape but must stay intact so its letter is
# not italicised as a variable.
_STEREODESCRIPTOR_RE = re.compile(r"\(\s*[RSEZ](?:\s*[,/]\s*[RSEZ])*\s*\)")


def _is_inline_math_span(content: str) -> bool:
    """True when a markup-free ``$…$`` span is plain inline math the model wrapped
    in math mode, so its delimiters should drop (like a bare number's).

    The whole span must be math-like (identifiers, numbers, operators), carry no
    HTML tag, and not open with a digit: currency is always digit-led ("$5", "$10"),
    so a spuriously paired span opens with a digit and is left intact — even one
    bracketing a relation ("$10 > $5") or a hyphen range ("$5 - $10").

    A paren lead is admitted only for a parenthesized CIP stereodescriptor
    ("(R)"/"(S)"/"(R,S)") or a genuine equation carrying a relation ("(n = 5)"); a
    bare parenthesized label — a figure panel "(A)" or roman list item "(i)" — has
    neither and is left intact, so its letter is not italicised.  Otherwise the lead
    must be a letter: a relational operator (``=``/``<``/``>``) then marks an equation
    ("x = 22", "pH = 7", "P < 0.05") and is decisive; failing that, only an
    operator/comma-separated list of single-letter variables ("a, b, c") qualifies,
    with no multi-letter English word — so a stray pairing over prose isn't
    swallowed."""
    stripped = content.strip()
    if not stripped or not _MATH_EXPR_RE.fullmatch(stripped):
        return False
    if _HTML_TAG_RE.search(stripped):
        return False
    head = stripped[:1]
    if head == "(":
        return bool(
            _STEREODESCRIPTOR_RE.fullmatch(stripped)
            or _MATH_RELATION_RE.search(stripped)
        )
    if not head.isalpha():
        return False
    if _MATH_RELATION_RE.search(stripped):
        return True
    return not _ENGLISH_WORD_RE.search(stripped)


# Sub/superscript inside a math span: ^{multi} / ^cmd / ^x and the _ forms.
# A bare script target may be a command pylatexenc left literal (``^\dagger``);
# capture the whole command, not just the leading backslash, so a stray "\<"
# can't reach markdown and get mangled into a broken tag.
_LATEX_SUP_RE = re.compile(r"\^\{([^{}]*)\}|\^(\\[a-zA-Z]+|\S)")
_LATEX_SUB_RE = re.compile(r"_\{([^{}]*)\}|_(\\[a-zA-Z]+|\S)")
# Font/style wrappers (\text{…}, \mathrm{…}) carry no semantics here — unwrap to
# their content so the inner sub/superscript handling sees plain text.
_LATEX_WRAP_RE = re.compile(
    r"\\(?:text|mathrm|mathit|mathbf|mathsf|operatorname)\{([^{}]*)\}"
)

# In math, a standalone single Latin letter is a variable and renders italic
# ("$x = 22$" → x italic, "$k_{cat}/K_m$" → k and K italic); numbers, operators,
# and multi-letter identifiers ("max", "cat") stay upright.  Italics are applied
# only at the span's *top level*: a single letter inside a generated <sub>/<sup>
# is left alone, so an affiliation/footnote marker ("$^{a}$" → "<sup>a</sup>") is
# never italicised.  The capturing group keeps the script spans in the split list
# (at the odd indices) so they pass through untouched.
_SCRIPT_SPAN_RE = re.compile(r"(<su[bp]>.*?</su[bp]>)", re.DOTALL)
_MATH_VARIABLE_RE = re.compile(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])")
# A single letter in *unit position* — trailing a numeric magnitude ("5 V",
# "10⁸ m/s", "9.8 m/s²", "25°C", "5 µg") — is a unit symbol, not a variable, and
# must stay upright.  The run starts at the first letter after a digit (ASCII or
# superscript, optionally one space) and extends across the unit cluster: more
# letters, the connectors '·'/'/'/'*' joining sub-units, and superscript exponents.
# A non-letter unit prefix may sit between the magnitude and the unit letter —
# a degree sign ("°C"/"°F") or a micro sign ("µg"/"μL") — so the letter after it is
# still read as a unit; ``_UNIT_PREFIX`` is the one place to extend that set.  A
# coefficient·variable like "2x" reads as unit position too, but such glued forms
# are vanishingly rare in the model's wrapped spans, whereas a trailing unit is
# common.
_UNIT_PREFIX = "°µμ"
_UNIT_RUN_RE = re.compile(
    rf"(?<=[0-9{_SUP_DIGITS}])\s?[{_UNIT_PREFIX}]?[A-Za-z]+(?:[·/*][A-Za-z]+|[{_SUP_DIGITS}])*"
)


def _italicize_math_variables(content: str) -> str:
    """Wrap each standalone single-letter variable in ``<em>``, top level only.

    ``<sub>``/``<sup>`` spans are split out first so an affiliation or footnote
    marker (``<sup>a</sup>``) is never italicised; a letter in unit position
    (``_UNIT_RUN_RE``) stays upright.
    """
    parts = _SCRIPT_SPAN_RE.split(content)
    parts[::2] = [_italicize_segment(segment) for segment in parts[::2]]
    return "".join(parts)


def _italicize_segment(segment: str) -> str:
    unit_spans = [m.span() for m in _UNIT_RUN_RE.finditer(segment)]

    def wrap(m: re.Match[str]) -> str:
        if any(lo <= m.start() < hi for lo, hi in unit_spans):
            return m.group(0)
        return f"<em>{m.group(0)}</em>"

    return _MATH_VARIABLE_RE.sub(wrap, segment)


def _latex_span_to_html(content: str) -> str:
    """Convert the inside of a ``$…$`` span: sub/superscripts to HTML, then drop
    residual TeX syntax.  Full math is out of scope (a later MathJax option)."""
    content = _LATEX_WRAP_RE.sub(r"\1", content)
    content = _LATEX_VERT_RE.sub("‖", content)
    content = _LATEX_DEGREE_RE.sub("°", content)
    content = _LATEX_S_LABEL_RE.sub("S", content)
    content = _LATEX_COMMAND_RE.sub(
        lambda m: _latex_command_to_unicode(m.group(0)), content
    )
    content = _LATEX_SUP_RE.sub(
        lambda m: _to_superscript(m.group(1) if m.group(1) is not None else m.group(2)),
        content,
    )
    content = _LATEX_SUB_RE.sub(
        lambda m: f"<sub>{m.group(1) if m.group(1) is not None else m.group(2)}</sub>",
        content,
    )
    content = content.replace("\\,", " ").replace("{", "").replace("}", "")
    content = _italicize_math_variables(content)
    return content.translate(_MD_EMPHASIS_ESCAPE)


# The model drops the "$…$" wrapper often enough that raw TeX reaches the reader as
# source text — the same document writes "$K_m$" in one table and a bare "V_{max}",
# "0.52 \pm 0.00 mM" or "\geq" in the prose beside it.  Only the two shapes that
# cannot be anything *but* TeX are converted without delimiters:
#
#   * a braced script ("V_{max}", "Author^{1}") — plain text never braces a script;
#   * a resolvable multi-letter macro ("\pm", "\geq", "\times").
#
# A single-letter macro is excluded: pylatexenc resolves "\r"/"\b"/"\c" to combining
# diacritics, and undelimited prose is far likelier to carry a stray backslash than a
# real accent.  The unbraced "K_m" shape is excluded too — identifiers in this corpus
# carry underscores (locus tags like "Xaut_4868", file names), so subscripting them
# would corrupt more than it fixes.  Everything excluded stays literal per the
# lossless-degradation rule and is reported by ``_convert_undelimited``.
_UNDELIMITED_SCRIPT_RE = re.compile(r"([_^])\{([^{}]*)\}")
# TeX-shaped residue left literal, for the declined-span record: an unbraced script.
# The class is deliberately loose — a plain snake_case identifier matches — because the
# record's claim is only "this reads as TeX and the rules could not classify it".
_UNBRACED_SCRIPT_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*[_^][A-Za-z0-9]+\b")
# _latex_to_html also runs over raw HTML in the pre-markdown stream (<table>, <sup>,
# <br>).  The undelimited pass keys on bare '_'/'^'/'\' rather than a paired delimiter,
# so tags are split out and left untouched rather than scanned for scripts.
_HTML_TAG_SPLIT_RE = re.compile(rf"({_HTML_TAG_RE.pattern})")


def _undelimited_script_html(m: re.Match[str]) -> str:
    # The content lands in the pre-markdown stream, so escape its markdown-active
    # characters exactly as _latex_span_to_html does for the delimited path.
    core = m.group(2).translate(_MD_EMPHASIS_ESCAPE)
    return _to_superscript(core) if m.group(1) == "^" else f"<sub>{core}</sub>"


def _convert_undelimited(text: str) -> str:
    """Convert the undelimited TeX shapes that are unambiguous, and log the rest.

    Runs after the ``$…$`` pass, on what it left.  A span the rules decline stays
    literal, which is silent corruption from the consumer's side — the output still
    holds TeX source and nothing else reports it — so every declined span is named in
    one debug record per call.  Debug, not warning: ``_UNBRACED_SCRIPT_RE`` also matches
    ordinary snake_case identifiers, so the record is a coverage note, not an alarm.
    """
    declined: list[str] = []

    def command(m: re.Match[str]) -> str:
        name = m.group(0)
        if len(name) > 2:
            glyph = _latex_command_to_unicode(name)
            if glyph != name:
                return glyph
        declined.append(name)
        return name

    parts = _HTML_TAG_SPLIT_RE.split(text)
    for i, part in enumerate(parts[::2]):
        part = _LATEX_COMMAND_RE.sub(command, part)
        part = _UNDELIMITED_SCRIPT_RE.sub(_undelimited_script_html, part)
        declined.extend(m.group(0) for m in _UNBRACED_SCRIPT_RE.finditer(part))
        parts[i * 2] = part
    if declined:
        _log.debug(
            "latex: left unconverted TeX-shaped spans: %s", sorted(set(declined))
        )
    return "".join(parts)


def _latex_to_html(text: str) -> str:
    """Replace each inline ``$…$`` *math* span with deterministic HTML, then reduce
    the unambiguous TeX the model emitted without delimiters
    (:func:`_convert_undelimited`).

    Runs on the markdown *before* parsing so the emitted ``<sub>``/``<sup>`` pass
    through as raw HTML and the ``_`` inside ``V_{max}`` isn't read as emphasis.
    Spans without any TeX markup are left verbatim.
    """

    def replace(m: re.Match[str]) -> str:
        space, content = m.group(1), m.group(2)
        if not _LATEX_MATH_RE.search(content):
            stripped = content.strip()
            if _BARE_NUMBER_RE.fullmatch(stripped):
                return space + stripped
            if _is_inline_math_span(content):
                return space + _latex_span_to_html(content)
            return m.group(0)
        html = _latex_span_to_html(content)
        if space and content.lstrip()[:1] in ("^", "_"):
            return html
        return space + html

    # Unwrap a supplementary-label span ("$\S4$" → "S4") before the generic span pass,
    # so the identifier stays plain rather than being math-variable italicised; and the
    # literal-section-sign form of the same misread ("§4 Fig." → "S4 Fig.").  Both run
    # here, before span processing, so they only see an OCR-literal "§" — never one the
    # span pass itself produces from a standalone "\S" footnote marker.
    # The captured label can carry an '_' ("$\S4_2$" → "S4_2"); it lands in the
    # pre-markdown stream, so escape it like every other literal emitted there
    # (_MD_EMPHASIS_ESCAPE), or the downstream inline parser re-reads it as emphasis.
    text = _LATEX_S_LABEL_SPAN_RE.sub(
        lambda m: "S" + m.group(1).translate(_MD_EMPHASIS_ESCAPE), text
    )
    text = _LITERAL_S_LABEL_RE.sub(r"\1S", text)
    return _convert_undelimited(_LATEX_SPAN_RE.sub(replace, text))
