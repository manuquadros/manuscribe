"""Best-effort DOI extraction (Tasks B of the annotation-hub integration).

Pure-string tests for the ``10.xxxx/suffix`` shape and trailing-punctuation cleanup,
plus a deterministic text-layer end-to-end over every fixture PDF — GPU-free, since
the DOI is read from the born-digital text layer, not the OCR.  The ad-prefixed
fixture (31051047) exercises the leading-skip: its article DOI sits on page 1, page 0
being an image advertisement with an empty text layer.
"""

from pathlib import Path

import pytest

from manuscribe.pipeline.doi import _clean_doi, _extract_doi, _find_doi
from manuscribe.pipeline.layers import _DocumentLayers

_FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestFindDoi:
    """The DOI scan over a text string."""

    def test_bare_doi(self) -> None:
        assert _find_doi("10.1002/bmb.21202") == "10.1002/bmb.21202"

    def test_doi_with_label_prefix(self) -> None:
        assert _find_doi("doi: 10.1042/BSR20190715") == "10.1042/BSR20190715"

    def test_doi_inside_url(self) -> None:
        assert (
            _find_doi("https://doi.org/10.1371/journal.pone.0235718")
            == "10.1371/journal.pone.0235718"
        )

    def test_mixed_case_suffix_preserved(self) -> None:
        # DOIs are case-insensitive but the suffix's printed case is kept verbatim.
        assert _find_doi("10.1042/BSR20190715") == "10.1042/BSR20190715"

    def test_trailing_sentence_period_stripped(self) -> None:
        assert (
            _find_doi("Published at 10.1021/acs.jafc.9b03262.")
            == "10.1021/acs.jafc.9b03262"
        )

    def test_unbalanced_trailing_paren_stripped(self) -> None:
        assert (
            _find_doi("(doi: 10.3389/fbioe.2020.00067)") == "10.3389/fbioe.2020.00067"
        )

    def test_balanced_paren_in_suffix_kept(self) -> None:
        assert _find_doi("10.1000/foo(bar)") == "10.1000/foo(bar)"

    def test_not_matched_inside_longer_number(self) -> None:
        assert _find_doi("build 110.1234/notadoi") is None

    def test_absent_returns_none(self) -> None:
        assert _find_doi("no identifier on this line") is None


class TestCleanDoi:
    """Trailing-punctuation cleanup the greedy suffix absorbs from prose."""

    def test_strips_trailing_sentence_punctuation(self) -> None:
        assert _clean_doi("10.1/x.") == "10.1/x"
        assert _clean_doi("10.1/x;") == "10.1/x"

    def test_strips_punctuation_then_unbalanced_paren(self) -> None:
        assert _clean_doi("10.1/x);") == "10.1/x"

    def test_keeps_clean_doi_unchanged(self) -> None:
        assert _clean_doi("10.1021/acs.jafc.9b03262") == "10.1021/acs.jafc.9b03262"


class TestExtractDoi:
    """Text-layer first, OCR fallback, then ``None``."""

    def test_layer_text_wins_over_ocr(self) -> None:
        assert _extract_doi("10.1234/layer", "10.5678/ocr") == "10.1234/layer"

    def test_falls_back_to_ocr_when_layer_empty(self) -> None:
        assert _extract_doi("scanned page, no text", "10.5678/ocr") == "10.5678/ocr"

    def test_none_when_neither_carries_one(self) -> None:
        assert _extract_doi("no doi", "no doi either") is None


# (stem, article-start page index, expected DOI).  31051047 is ad-prefixed, so its
# article — and DOI — start on page 1; the rest carry it on page 0.
_FIXTURE_DOIS = [
    ("30592559", 0, "10.1002/bmb.21202"),
    ("31051047", 1, "10.1002/bab.1760"),
    ("31123167", 0, "10.1042/BSR20190715"),
    ("31298526", 0, "10.1021/acs.jafc.9b03262"),
    ("32117944", 0, "10.3389/fbioe.2020.00067"),
    ("32639976", 0, "10.1371/journal.pone.0235718"),
]


@pytest.mark.parametrize("stem, start, expected", _FIXTURE_DOIS)
def test_doi_from_fixture_text_layer(stem: str, start: int, expected: str) -> None:
    pdf = _FIXTURE_DIR / f"{stem}.pdf"
    if not pdf.exists():
        pytest.skip(f"fixture PDF absent: {pdf}")
    with _DocumentLayers.open(pdf) as layers:
        assert _find_doi(layers.page_raw_text(start)) == expected


def test_page_raw_text_out_of_range_is_empty() -> None:
    pdf = _FIXTURE_DIR / "30592559.pdf"
    if not pdf.exists():
        pytest.skip(f"fixture PDF absent: {pdf}")
    with _DocumentLayers.open(pdf) as layers:
        assert layers.page_raw_text(9999) == ""
        assert layers.page_raw_text(-1) == ""
