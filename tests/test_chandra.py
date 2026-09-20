"""Tests for pipeline.chandra — chandra-ocr-2's div-tree response parsed into
list[Block]. Synthetic div-tree strings only, no server/GPU — see the ingestion
adapter design ticket and spike_results/chandra_ocr2_phase1_fidelity.md."""

from pathlib import Path

import httpx
from helpers import _abstract, _body, _header_h1
from PIL import Image

from manuscribe.pipeline.assemble import (
    _assemble_chandra_document,
    lightonocr_pdf_to_document,
)
from manuscribe.pipeline.block import BlockKind
from manuscribe.pipeline.chandra import _iter_divs, parse_chandra_response
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

    def test_unparseable_bbox_is_skipped_not_raised(self) -> None:
        raw = '<div data-bbox="not a box" data-label="Text"><p>Hi.</p></div>'
        assert _iter_divs(raw) == []


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
        assert [b.kind for b in blocks] == [BlockKind.FIGURE, BlockKind.PARAGRAPH]
        assert blocks[1].inner == "Real prose."

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

    def test_source_page_threaded_through(self) -> None:
        raw = '<div data-bbox="0 0 10 10" data-label="Text"><p>x</p></div>'
        blocks = parse_chandra_response(raw, _page(), source_page=7)
        assert blocks[0].source_page == 7


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
