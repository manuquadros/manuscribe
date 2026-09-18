"""Tests for the post-flatten typed block (pipeline.block.Block)."""

from pdfparser.pipeline.block import Block, BlockKind, CaptionLabel


class TestBlockKind:
    def test_heading(self) -> None:
        b = Block.of("<h2>Results</h2>")
        assert b.kind is BlockKind.HEADING
        assert b.heading_level == 2
        assert b.inner == "Results"

    def test_paragraph(self) -> None:
        b = Block.of("<p>Some prose.</p>")
        assert b.kind is BlockKind.PARAGRAPH
        assert b.heading_level is None
        assert b.inner == "Some prose."

    def test_table(self) -> None:
        b = Block.of("<table><tr><td>x</td></tr></table>")
        assert b.kind is BlockKind.TABLE
        assert b.inner is None

    def test_figure(self) -> None:
        b = Block.of('<figure><img src="a.png" alt=""></figure>')
        assert b.kind is BlockKind.FIGURE
        assert b.inner is None

    def test_other(self) -> None:
        b = Block.of("<ol><li>Ref 1</li></ol>")
        assert b.kind is BlockKind.OTHER
        assert b.inner is None

    def test_multi_paragraph_string_is_not_a_plain_paragraph(self) -> None:
        # _plain_p_text requires exactly one </p>; a reference-list-shaped
        # multi-<p> string must not be misclassified as a single paragraph.
        b = Block.of("<p>One.</p><p>Two.</p>")
        assert b.kind is BlockKind.OTHER


class TestDerivedFields:
    def test_visible_text_strips_tags(self) -> None:
        b = Block.of("<p>Hello <em>world</em>.</p>")
        assert b.visible_text == "Hello world."

    def test_ends_sentence(self) -> None:
        assert Block.of("<p>Finished.</p>").ends_sentence is True
        assert Block.of("<p>Not finished</p>").ends_sentence is False

    def test_ends_sentence_past_trailing_citation_superscript(self) -> None:
        assert Block.of("<p>Cited work.<sup>1,2</sup></p>").ends_sentence is True

    def test_bold_label_colon_inside(self) -> None:
        b = Block.of("<p><strong>Keywords:</strong> a, b, c</p>")
        assert b.bold_label == "Keywords"

    def test_bold_label_colon_outside(self) -> None:
        b = Block.of("<p><strong>Keywords</strong>: a, b, c</p>")
        assert b.bold_label == "Keywords"

    def test_bold_label_none_without_colon(self) -> None:
        b = Block.of("<p><strong>Important</strong> note</p>")
        assert b.bold_label is None

    def test_source_page_defaults_to_none_and_is_settable(self) -> None:
        assert Block.of("<p>x</p>").source_page is None
        assert Block.of("<p>x</p>", source_page=3).source_page == 3


class TestCaptionLabel:
    def test_figure_caption(self) -> None:
        b = Block.of("<p><strong>Figure 1.</strong> A description.</p>")
        assert b.caption_label is CaptionLabel.FIGURE

    def test_table_caption(self) -> None:
        b = Block.of("<p><strong>Table 1.</strong> A title.</p>")
        assert b.caption_label is CaptionLabel.TABLE

    def test_body_prose_has_no_caption_label(self) -> None:
        b = Block.of("<p>This figure shows something.</p>")
        assert b.caption_label is None
