"""Tests for pipeline.chandra — chandra-ocr-2's div-tree response parsed into
list[Block]. Synthetic div-tree strings only, no server/GPU — see the ingestion
adapter design ticket and spike_results/chandra_ocr2_phase1_fidelity.md."""

import base64
import io
import logging
import re
import sys
from pathlib import Path

import httpx
import pytest
from helpers import _abstract, _body, _header_h1
from PIL import Image

from manuscribe.pipeline.assemble import (
    _assemble_chandra_document,
    lightonocr_pdf_to_document,
)
from manuscribe.pipeline.block import BlockKind
from manuscribe.pipeline.chandra import (
    _FIGURE_LABELS,
    _iter_divs,
    parse_chandra_response,
)
from manuscribe.pipeline.classify import _leading_pages_to_skip_html
from manuscribe.pipeline.model import OcrEngine, OcrModel
from manuscribe.pipeline.text import _MAX_IDENTICAL_ELEMENT_RUN

_FIXTURES = Path(__file__).parent / "fixtures"


def _page() -> Image.Image:
    return Image.new("RGB", (1160, 1540), "white")


class TestIterDivs:
    def test_parses_data_bbox_attribute(self) -> None:
        raw = '<div data-bbox="10 20 30 40" data-label="Text"><p>Hi.</p></div>'
        divs = _iter_divs(raw)
        assert len(divs) == 1
        assert divs[0].label == "Text"
        assert divs[0].bbox == (10, 20, 30, 40)
        assert divs[0].inner_html == "<p>Hi.</p>"

    def test_parses_plain_data_attribute(self) -> None:
        # Confirmed fixed per page, not per document — a parser must accept
        # both attribute-name variants (spike_results/chandra_ocr2_phase1_fidelity.md).
        raw = '<div data="10 20 30 40" data-label="Text"><p>Hi.</p></div>'
        divs = _iter_divs(raw)
        assert len(divs) == 1
        assert divs[0].bbox == (10, 20, 30, 40)

    def test_multiple_divs_in_document_order(self) -> None:
        raw = (
            '<div data-bbox="0 0 10 10" data-label="Section-Header"><h2>A</h2></div>'
            '<div data-bbox="0 10 10 20" data-label="Text"><p>B.</p></div>'
        )
        divs = _iter_divs(raw)
        assert [d.label for d in divs] == ["Section-Header", "Text"]

    def test_wrong_number_of_coordinates_is_skipped_not_raised(self) -> None:
        # Letters never reach the guard — the bbox character class rejects them,
        # so the div simply doesn't match. A bbox with the right characters and
        # the wrong count is what the length check is there for.
        raw = '<div data-bbox="1 2 3" data-label="Text"><p>Hi.</p></div>'
        assert _iter_divs(raw) == []

    def test_malformed_number_in_bbox_is_skipped_not_raised(self) -> None:
        # "1.2.3" passes the character class and fails float() — a response is
        # untyped model output, so this must not abort the document.
        raw = '<div data-bbox="1.2.3 4 5 6" data-label="Text"><p>Hi.</p></div>'
        assert _iter_divs(raw) == []

    def test_skipped_bbox_is_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        raw = '<div data-bbox="1.2.3 4 5 6" data-label="Text"><p>Hi.</p></div>'
        with caplog.at_level(logging.WARNING):
            assert _iter_divs(raw) == []
        assert "1.2.3" in caplog.text

    def test_infinite_coordinate_is_reported_not_silently_dropped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A decode loop stuck inside the bbox attribute emits a digit run that
        # float() reads as inf, whose round() raises OverflowError rather than
        # ValueError. It must be skipped like any other unusable bbox, and said.
        bbox = f"{'9' * 309} 0 100 100"
        raw = f'<div data-bbox="{bbox}" data-label="Text"><p>Hi.</p></div>'
        with caplog.at_level(logging.WARNING):
            assert _iter_divs(raw) == []
        assert "did not parse as four coordinates" in caplog.text

    def test_response_matching_no_div_is_reported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The attribute name jitters per page, so a third spelling is the expected
        # failure — it must say so rather than silently render a blank page.
        raw = '<div data-box="0 0 10 10" data-label="Text"><p>Hi.</p></div>'
        with caplog.at_level(logging.WARNING):
            assert _iter_divs(raw) == []
        assert "data-box" in caplog.text

    def test_blank_response_is_not_reported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A blank page, and a page the pre-OCR ad-skip padded with "", are both
        # legitimately block-free.
        with caplog.at_level(logging.WARNING):
            assert _iter_divs("   \n") == []
        assert caplog.text == ""


class TestCollapseDecodeLoop:
    """The confirmed hallucination repeats one element *inside* a single div — a
    figure's ``<p>`` label emitted 342 times until the token budget ran out (the
    phase-1 spike dumps hold runs of 336 and 491) — so the run to collapse is
    between an element and its siblings, not between whole divs."""

    def _looped(self, label: str, element: str, n: int) -> str:
        return f'<div data-bbox="0 0 500 500" data-label="{label}">{element * n}</div>'

    def test_long_identical_element_run_inside_one_div_collapsed(self) -> None:
        raw = self._looped("Figure", "<p>His-tagged PtTRI</p>", 342)
        (div,) = _iter_divs(raw)
        assert div.inner_html.count("His-tagged PtTRI") == 1

    def test_short_run_inside_a_div_left_alone(self) -> None:
        # Three identical short labels ("kDa M" over two gel panels) can coincide.
        raw = self._looped("Figure", "<p>kDa M</p>", _MAX_IDENTICAL_ELEMENT_RUN)
        (div,) = _iter_divs(raw)
        assert div.inner_html.count("<p>kDa M</p>") == _MAX_IDENTICAL_ELEMENT_RUN

    def test_content_around_the_loop_survives(self) -> None:
        raw = (
            '<div data-bbox="0 0 500 500" data-label="Figure">'
            '<img alt="gel">' + "<p>Loop.</p>" * 9 + "<p>Figure 1. Real.</p></div>"
        )
        (div,) = _iter_divs(raw)
        assert div.inner_html.count("<p>Loop.</p>") == 1
        assert '<img alt="gel">' in div.inner_html
        assert "<p>Figure 1. Real.</p>" in div.inner_html

    def test_loop_does_not_reach_the_blocks(self) -> None:
        raw = self._looped("Text", "<p>His-tagged PtTRI</p>", 342)
        blocks = parse_chandra_response(raw, _page())
        assert sum(b.html.count("His-tagged PtTRI") for b in blocks) == 1


class TestUnclosedDiv:
    """A response truncated mid-div (what a decode loop exhausting the token
    budget produces) leaves that div unclosed. Its content must stop at the next
    div's opener: the block that follows is whole and separate, and on a
    figure-labeled div it would otherwise be discarded with the inner HTML."""

    def test_block_after_unclosed_div_survives(self) -> None:
        raw = (
            '<div data-bbox="0 0 500 500" data-label="Figure"><img alt="x"/>'
            + "<p>loop</p>" * 50
            + '<div data-bbox="0 600 500 700" data-label="Text">'
            "<p>Real prose.</p></div>"
        )
        blocks = parse_chandra_response(raw, _page())
        # The collapsed remnant of the loop is kept, like any other content a
        # figure div carries after its image — the truncation is what this pins.
        assert [b.kind for b in blocks] == [
            BlockKind.FIGURE,
            BlockKind.PARAGRAPH,
            BlockKind.PARAGRAPH,
        ]
        assert blocks[1].inner == "loop"
        assert blocks[2].inner == "Real prose."

    def test_unclosed_div_keeps_the_content_before_the_truncation(self) -> None:
        raw = (
            '<div data-bbox="0 0 500 500" data-label="Text"><p>Partial prose.</p>'
            '<div data-bbox="0 600 500 700" data-label="Text">'
            "<p>Next block.</p></div>"
        )
        first, second = _iter_divs(raw)
        assert first.inner_html == "<p>Partial prose.</p>"
        assert second.inner_html == "<p>Next block.</p>"

    def test_unclosed_div_at_end_of_response_kept(self) -> None:
        raw = '<div data-bbox="0 0 500 500" data-label="Text"><p>Cut off mid-'
        (div,) = _iter_divs(raw)
        assert div.inner_html == "<p>Cut off mid-"

    def test_truncated_loop_loses_neither_the_next_block_nor_the_loop(self) -> None:
        raw = (
            '<div data-bbox="0 0 500 500" data-label="Text">'
            + "<p>Loop.</p>" * 342
            + '<div data-bbox="0 600 500 700" data-label="Text">'
            "<p>Real prose.</p></div>"
        )
        blocks = parse_chandra_response(raw, _page())
        assert sum(b.html.count("<p>Loop.</p>") for b in blocks) == 1
        assert blocks[-1].inner == "Real prose."


class TestParseChandraResponse:
    def test_page_furniture_dropped_text_and_heading_kept(self) -> None:
        raw = (
            '<div data-bbox="0 0 100 5" data-label="Page-Header"><p>Journal</p></div>'
            '<div data-bbox="0 5 100 10" data-label="Section-Header">'
            "<h2>Intro</h2></div>"
            '<div data-bbox="0 10 100 20" data-label="Text"><p>Body text.</p></div>'
            '<div data-bbox="0 990 100 999" data-label="Page-Footer"><p>1</p></div>'
        )
        blocks = parse_chandra_response(raw, _page())
        assert [b.kind for b in blocks] == [BlockKind.HEADING, BlockKind.PARAGRAPH]
        assert blocks[1].inner == "Body text."

    def test_figure_div_crops_from_page_image(self) -> None:
        raw = '<div data-bbox="0 0 500 500" data-label="Figure"><img alt="x"></div>'
        blocks = parse_chandra_response(raw, _page())
        assert len(blocks) == 1
        assert blocks[0].kind is BlockKind.FIGURE
        assert blocks[0].html.startswith("<figure>")

    def test_fused_panel_label_split_out_of_figure_div(self) -> None:
        # The one exception found in the full spike run: a Diagram div bundling
        # a panel label and an image in one div.
        raw = (
            '<div data-bbox="0 0 500 500" data-label="Diagram">'
            '<p><b>(A)</b></p><img alt="x">'
            "</div>"
        )
        blocks = parse_chandra_response(raw, _page())
        assert len(blocks) == 2
        assert blocks[0].kind is BlockKind.PARAGRAPH
        assert blocks[1].kind is BlockKind.FIGURE

    def test_inverted_figure_bbox_declines_the_crop_and_keeps_the_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # x1 < x0 raises in Image.crop; the transcription is still good.
        raw = (
            '<div data-bbox="500 500 100 100" data-label="Figure">'
            '<img alt="x"/><p>Figure 1. A caption.</p></div>'
        )
        with caplog.at_level(logging.WARNING):
            blocks = parse_chandra_response(raw, _page())
        assert [b.kind for b in blocks] == [BlockKind.PARAGRAPH]
        assert blocks[0].inner == "Figure 1. A caption."
        assert "positive area" in caplog.text

    def test_out_of_range_figure_bbox_is_clamped_to_the_page(self) -> None:
        # A runaway coordinate is an unbounded crop — past Pillow's bomb threshold
        # it raises outside the ManuscribeError hierarchy, and just below it it
        # quietly embeds a multi-hundred-megabyte blank crop.
        page = _page()
        raw = (
            '<div data-bbox="0 0 9999999 9999999" data-label="Figure">'
            '<img alt="x"/><p>Figure 1. A caption.</p></div>'
        )
        blocks = parse_chandra_response(raw, page)
        assert [b.kind for b in blocks] == [BlockKind.FIGURE, BlockKind.PARAGRAPH]
        src = re.search(r'src="data:image/png;base64,([^"]+)"', blocks[0].html)
        assert src is not None
        crop = Image.open(io.BytesIO(base64.b64decode(src.group(1))))
        assert crop.size == page.size

    def test_source_page_threaded_through(self) -> None:
        raw = '<div data-bbox="0 0 10 10" data-label="Text"><p>x</p></div>'
        blocks = parse_chandra_response(raw, _page(), source_page=7)
        assert blocks[0].source_page == 7


# A coordinate near float's maximum that is still finite: it converts, so the
# parse keeps it, and the length of a token is therefore not what decides.
# Denormalizing it against the page *does* overflow, in figures._denormalize_bbox,
# outside the conversion guarded here — see the xfail below.
_FINITE_NEAR_MAX = f"0 0 500 {round(sys.float_info.max)}"

# (bbox attribute, whether the div survives the parse).  A bbox that is not four
# convertible tokens takes its div with it; every other box is kept however
# unusable it is as a crop, because the transcription still is.  Three separate
# escapes have come out of these three lines — a token float() rejects, a box
# Pillow refuses, a digit run that rounds to infinity — so the guard is tested as
# a corpus rather than as whichever case was found last.
_ADVERSARIAL_BBOXES: list[tuple[str, bool]] = [
    ("1.2.3 4 5 6", False),  # passes the character class, fails float()
    (f"{'9' * 309} 0 100 100", False),  # float inf; round(inf) is an OverflowError
    ("0 0 9999999 9999999", True),  # unclamped: past Pillow's bomb threshold
    ("0 0 10000 10000", True),  # unclamped: a ~180-megapixel crop, silently
    ("500 500 100 100", True),  # inverted
    ("100 100 100 100", True),  # zero area
    ("-500 -400 -100 -50", True),  # wholly off the page
    ("-100 -100 500 500", True),  # partly off the page
    ("0.5 0.5 999.5 999.5", True),  # non-integer coordinates
    (".", False),
    ("-", False),
    (". . . .", False),  # right count, right characters, none convertible
    ("1 2 3", False),  # too few
    ("1 2 3 4 5", False),  # too many
    ("  0 0 500 500  ", True),  # leading/trailing whitespace
    (_FINITE_NEAR_MAX, True),  # long, and convertible — unlike the inf row above
]


class TestAdversarialBbox:
    """No bbox a response can carry may abort the document."""

    @pytest.mark.parametrize("label", ["Text", "Figure"])
    @pytest.mark.parametrize(("bbox", "keeps_the_div"), _ADVERSARIAL_BBOXES)
    def test_no_bbox_raises_out_of_the_parse(
        self, bbox: str, keeps_the_div: bool, label: str
    ) -> None:
        # Both labels: the bbox parse is upstream of the figure branch, so an
        # escape there takes down any block, not only a figure.
        if bbox == _FINITE_NEAR_MAX and label == "Figure":
            pytest.xfail(
                "figures._denormalize_bbox scales the box before anything clamps"
                " it, so a finite near-max coordinate overflows to inf there and"
                " round() raises; the magnitude guard belongs in the denormalizer,"
                " which both engines reach"
            )
        img = '<img alt="x"/>' if label == "Figure" else ""
        inner = f"{img}<p>A caption.</p>"
        raw = f'<div data-bbox="{bbox}" data-label="{label}">{inner}</div>'
        blocks = parse_chandra_response(raw, _page())
        if keeps_the_div:
            assert "A caption." in "".join(b.html for b in blocks)
        else:
            assert blocks == []


class TestFigureDivSequence:
    """A figure div's content is a sequence, not a prefix: chandra fuses into one
    figure div whatever shares the region — a panel label before the image, and
    after it prose, further images with their labels, or a whole ``<table>`` it
    already transcribed. Over the twelve phase-1 spike dumps 58 figure divs carry
    content after their first ``<img>``, against 18 with leading content."""

    _TABLE_AFTER_IMG = (
        '<div data-bbox="0 0 500 500" data-label="Figure">'
        '<p>(a)</p><img alt="Bar chart"/>'
        '<table border="1"><tr><td>(A) 100% L</td><td>28.1</td></tr></table>'
        "</div>"
    )

    def test_table_after_the_image_survives_as_a_table(self) -> None:
        blocks = parse_chandra_response(self._TABLE_AFTER_IMG, _page())
        assert [b.kind for b in blocks] == [
            BlockKind.PARAGRAPH,
            BlockKind.FIGURE,
            BlockKind.TABLE,
        ]
        assert blocks[2].html.startswith("<table")
        assert "28.1" in blocks[2].html

    def test_trailing_table_reaches_the_assembled_body(self) -> None:
        page = (
            _div("Section-Header", "<h1>A Chandra Title</h1>", 30, 80)
            + _div("Section-Header", "<h2>Introduction</h2>", 200, 220)
            + _div("Text", "<p>Body prose here.</p>", 220, 400)
            + self._TABLE_AFTER_IMG
        )
        html, _, _ = _assemble_chandra_document([page], [_page()])
        body = _body(html)
        assert "<table" in body
        assert "28.1" in body
        assert '<figure><img src="data:image/png;base64,' in body

    def test_prose_after_the_image_kept_in_document_order(self) -> None:
        raw = (
            '<div data-bbox="0 0 500 500" data-label="Chemical-Block">'
            '<img alt="models"/><p>Both models feature a CoM binding pocket.</p>'
            "</div>"
        )
        blocks = parse_chandra_response(raw, _page())
        assert [b.kind for b in blocks] == [BlockKind.FIGURE, BlockKind.PARAGRAPH]
        assert blocks[1].inner == "Both models feature a CoM binding pocket."

    def test_labels_interleaved_with_images_survive_without_broken_imgs(self) -> None:
        # chandra's <img> carries no src, so one left in kept text would render
        # as a broken image; only the div's own bbox is croppable, so the whole
        # interleave becomes one crop plus the labels it also transcribed.
        raw = (
            '<div data-bbox="0 0 500 500" data-label="Chemical-Block">'
            '<p>R-HPC <img alt="s1"/> M-HPC <img alt="s2"/> S-HPC <img alt="s3"/></p>'
            "</div>"
        )
        blocks = parse_chandra_response(raw, _page())
        assert [b.kind for b in blocks] == [BlockKind.FIGURE, BlockKind.PARAGRAPH]
        assert "<img" not in blocks[1].html
        assert "M-HPC" in blocks[1].visible_text
        assert "S-HPC" in blocks[1].visible_text

    def test_heading_after_the_image_is_not_flattened_to_a_paragraph(self) -> None:
        raw = (
            '<div data-bbox="0 0 500 500" data-label="Figure">'
            '<img alt="x"/><h2>Results</h2>'
            "</div>"
        )
        blocks = parse_chandra_response(raw, _page())
        assert [b.kind for b in blocks] == [BlockKind.FIGURE, BlockKind.HEADING]

    def test_every_figure_label_crops_and_keeps_its_trailing_content(self) -> None:
        # Also the drift check on the label set's single home: a label missing
        # from it maps to PARAGRAPH here instead of cropping.
        assert {"Figure", "Image", "Chemical-Block", "Diagram"} == _FIGURE_LABELS
        for label in _FIGURE_LABELS:
            raw = f'<div data-bbox="0 0 500 500" data-label="{label}"><img alt="x"/>'
            raw += "<p>Trailing.</p></div>"
            blocks = parse_chandra_response(raw, _page())
            assert [b.kind for b in blocks] == [
                BlockKind.FIGURE,
                BlockKind.PARAGRAPH,
            ]


def _div(label: str, inner: str, y0: int, y1: int) -> str:
    return f'<div data-bbox="0 {y0} 1000 {y1}" data-label="{label}">{inner}</div>'


_ARTICLE_PAGE = (
    _div("Page-Header", "<p>Journal of X</p>", 0, 30)
    + _div("Section-Header", "<h1>A Chandra Title</h1>", 30, 80)
    + _div("Text", "<p>Ann Author<sup>1</sup>, Bob Writer<sup>2</sup></p>", 80, 100)
    + _div("Section-Header", "<h2>Abstract</h2>", 100, 120)
    + _div("Text", "<p>We study things.</p>", 120, 200)
    + _div("Section-Header", "<h2>Introduction</h2>", 200, 220)
    + _div("Text", "<p>Body prose here.</p>", 220, 400)
    + _div("Figure", '<img alt="x">', 400, 700)
    + _div("Caption", "<p>Figure 1. A caption.</p>", 700, 740)
    + _div("Page-Footer", "<p>1</p>", 960, 1000)
)
_AD_PAGE = _div("Text", "<p>Buy our product.</p>", 0, 1000)


def _assert_article_rendered(html: str) -> None:
    assert _header_h1(html) == "A Chandra Title"
    assert "We study things." in _abstract(html)
    body = _body(html)
    assert "<h2>Introduction</h2>" in body
    assert "Body prose here." in body
    assert '<figure><img src="data:image/png;base64,' in body
    assert "Figure 1. A caption." in body
    assert "Journal of X" not in html


class TestAssembleChandraDocument:
    """The chandra path enters the shared assembly tail (title/abstract/body
    classification, the document shell) with blocks parsed straight from the
    div-tree — no markdown stage, no server."""

    def test_synthetic_page_renders_title_abstract_body_figure(self) -> None:
        html, title, byline = _assemble_chandra_document([_ARTICLE_PAGE], [_page()])
        _assert_article_rendered(html)
        assert title == "A Chandra Title"
        assert byline == "Ann Author1, Bob Writer2"

    def test_leading_cover_page_without_article_heading_dropped(self) -> None:
        html, _, _ = _assemble_chandra_document(
            [_AD_PAGE, _ARTICLE_PAGE], [_page(), _page()]
        )
        assert "Buy our product." not in html
        _assert_article_rendered(html)

    @pytest.mark.parametrize("label", ["Text", "Figure"])
    @pytest.mark.parametrize("fragment", ["</", "<"])
    def test_truncated_tag_never_renders_as_text(
        self, label: str, fragment: str
    ) -> None:
        # The sanitizer closes the unbalanced element, but a half-written tag is
        # not markup to it — left in place it reaches the page escaped.
        page = _ARTICLE_PAGE + (
            f'<div data-bbox="0 800 1000 900" data-label="{label}">'
            f"<p>Kept prose.</p><p>Cut here{fragment}"
        )
        body = _body(_assemble_chandra_document([page], [_page()])[0])
        assert "Cut here" in body
        assert "&lt;" not in body

    def test_leading_pages_to_skip_reads_html_headings(self) -> None:
        assert _leading_pages_to_skip_html([_AD_PAGE, _ARTICLE_PAGE]) == 1
        assert _leading_pages_to_skip_html([_ARTICLE_PAGE]) == 0
        assert _leading_pages_to_skip_html([_AD_PAGE]) == 0


class TestEngineDispatch:
    """``lightonocr_pdf_to_document`` routes a chandra bundle's responses through
    the div-tree parser, not the markdown path, end to end (real render, real
    text layer, mocked server)."""

    def test_chandra_bundle_parses_div_tree(self) -> None:
        # The article on the first request only (serial, so that is page 0): the
        # same page returned for every request would (correctly) read as running
        # furniture and be stripped.
        served = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": []})
            served["n"] += 1
            content = _ARTICLE_PAGE if served["n"] == 1 else ""
            return httpx.Response(
                200, json={"choices": [{"message": {"content": content}}]}
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(
            client=client,
            base_url="http://srv/v1",
            model="lightonocr",
            concurrency=1,
            engine=OcrEngine.CHANDRA,
        )
        with ocr:
            doc = lightonocr_pdf_to_document(_FIXTURES / "30592559.pdf", ocr=ocr)
        _assert_article_rendered(doc.html)
        assert doc.title == "A Chandra Title"
        # The DOI scan is engine-agnostic: it reads the fixture's text layer.
        assert doc.doi is not None
