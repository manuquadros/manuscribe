"""Best-effort DOI extraction for the structured document return.

The article DOI is metadata a consumer (the Annotation Hub Reference row) wants
without re-parsing the rendered HTML.  It is scanned from the article's first-page
**PDF text layer** first — deterministic and reliable on born-digital PDFs — falling
back to that page's OCR text for a scanned PDF whose text layer is empty; ``None``
when neither yields one.

Pure text in/out: the caller reads the page text off the shared, already-open
``_DocumentLayers`` (``layers.page_raw_text(start)``) so this module re-opens nothing.
"""

from __future__ import annotations

import re

# The standard DOI syntax: the ``10.`` registrant prefix, a 4–9 digit registrant
# code, then a suffix of the characters DOIs permit.  ``(?<!\d)`` keeps the match
# from starting inside a longer number, and the scan is case-insensitive because
# DOIs are case-insensitive and a suffix often carries mixed case.
_DOI_RE = re.compile(r"(?<!\d)10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)


def _clean_doi(doi: str) -> str:
    """Strip prose punctuation the greedy suffix absorbed from surrounding text:
    trailing sentence punctuation, and a final ``)`` that closes an enclosing
    ``(doi: …)`` rather than belonging to the DOI (more ``)`` than ``(``)."""
    doi = doi.rstrip(".,;:")
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1].rstrip(".,;:")
    return doi


def _find_doi(text: str) -> str | None:
    """The first DOI in ``text``, cleaned of trailing prose punctuation; ``None``
    when ``text`` carries none."""
    m = _DOI_RE.search(text)
    return _clean_doi(m.group(0)) if m else None


def _extract_doi(layer_text: str, ocr_text: str) -> str | None:
    """The article DOI from the text-layer string (deterministic), else the OCR
    text, else ``None``."""
    return _find_doi(layer_text) or _find_doi(ocr_text)
