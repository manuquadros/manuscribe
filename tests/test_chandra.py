"""Tests for pipeline.chandra — chandra-ocr-2's div-tree response parsed into
list[Block]. Synthetic div-tree strings only, no server/GPU — see the ingestion
adapter design ticket and spike_results/chandra_ocr2_phase1_fidelity.md."""

from PIL import Image

from manuscribe.pipeline.block import BlockKind
from manuscribe.pipeline.chandra import (
    _MAX_IDENTICAL_BLOCK_RUN,
    _collapse_repeated_divs,
    _iter_divs,
    _RawDiv,
    parse_chandra_response,
)


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


class TestCollapseRepeatedDivs:
    def _run(self, n: int) -> list[_RawDiv]:
        return [_RawDiv(label="Text", bbox=(0, 0, 1, 1), inner_html="<p>Loop.</p>")] * n

    def test_short_run_left_alone(self) -> None:
        divs = self._run(_MAX_IDENTICAL_BLOCK_RUN)
        assert _collapse_repeated_divs(divs) == divs

    def test_long_run_collapsed_to_one(self) -> None:
        # Mirrors the real decode-loop hallucination found in the phase-1 spike
        # (a caption line repeated 342x), at a fraction of the size.
        divs = self._run(_MAX_IDENTICAL_BLOCK_RUN + 5)
        collapsed = _collapse_repeated_divs(divs)
        assert collapsed == [divs[0]]

    def test_run_broken_by_a_different_div_is_not_collapsed_across_it(self) -> None:
        loop = self._run(_MAX_IDENTICAL_BLOCK_RUN + 5)
        other = _RawDiv(label="Text", bbox=(0, 0, 1, 1), inner_html="<p>Other.</p>")
        divs = loop + [other] + loop
        collapsed = _collapse_repeated_divs(divs)
        assert collapsed == [loop[0], other, loop[0]]

    def test_different_labels_are_not_collapsed_together(self) -> None:
        divs = [
            _RawDiv(label="Text", bbox=(0, 0, 1, 1), inner_html="<p>x</p>"),
            _RawDiv(label="Caption", bbox=(0, 0, 1, 1), inner_html="<p>x</p>"),
        ]
        assert _collapse_repeated_divs(divs) == divs


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

    def test_decode_loop_collapsed_before_becoming_blocks(self) -> None:
        looped_div = (
            '<div data-bbox="0 0 10 10" data-label="Text"><p>Loop.</p></div>'
        ) * (_MAX_IDENTICAL_BLOCK_RUN + 5)
        blocks = parse_chandra_response(looped_div, _page())
        assert len(blocks) == 1

    def test_source_page_threaded_through(self) -> None:
        raw = '<div data-bbox="0 0 10 10" data-label="Text"><p>x</p></div>'
        blocks = parse_chandra_response(raw, _page(), source_page=7)
        assert blocks[0].source_page == 7
