"""Tests for the post-flatten typed block (pipeline.block.Block)."""

from manuscribe.pipeline.block import Block, BlockKind, CaptionLabel


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


class TestBlockFromChandraDiv:
    """``Block.from_chandra_div`` builds a Block from a chandra-ocr-2 response
    div's label + inner HTML directly, instead of ``Block.of``'s HTML-sniffing —
    see ``pipeline.chandra`` and ``spike_results/chandra_ocr2_phase1_fidelity.md``
    for the label vocabulary and shapes this is built from."""

    def test_page_header_and_footer_are_dropped(self) -> None:
        assert Block.from_chandra_div("Page-Header", "<p>Journal Name</p>") is None
        assert Block.from_chandra_div("Page-Footer", "<p>12</p>") is None

    def test_section_header_maps_to_heading(self) -> None:
        b = Block.from_chandra_div("Section-Header", "<h2>Results</h2>")
        assert b is not None
        assert b.kind is BlockKind.HEADING
        assert b.heading_level == 2
        assert b.inner == "Results"

    def test_section_header_without_heading_tag_falls_back_to_paragraph(self) -> None:
        # Observed in the full spike run: chandra doesn't always wrap a
        # Section-Header's content in <h*> — don't drop it, don't crash.
        b = Block.from_chandra_div("Section-Header", "Discussion")
        assert b is not None
        assert b.kind is BlockKind.PARAGRAPH
        assert b.inner == "Discussion"

    def test_table_maps_to_table_kind(self) -> None:
        b = Block.from_chandra_div("Table", "<table><tr><td>x</td></tr></table>")
        assert b is not None
        assert b.kind is BlockKind.TABLE

    def test_list_group_and_bibliography_map_to_other(self) -> None:
        for label in ("List-Group", "Bibliography"):
            b = Block.from_chandra_div(label, "<ol><li>Ref 1</li></ol>")
            assert b is not None
            assert b.kind is BlockKind.OTHER

    def test_figure_labels_map_to_figure_kind(self) -> None:
        for label in ("Figure", "Image", "Chemical-Block", "Diagram"):
            b = Block.from_chandra_div(
                label, '<figure><img src="a.png" alt=""></figure>'
            )
            assert b is not None
            assert b.kind is BlockKind.FIGURE

    def test_text_already_wrapped_in_p_passes_through(self) -> None:
        b = Block.from_chandra_div("Text", "<p>Some prose.</p>")
        assert b is not None
        assert b.kind is BlockKind.PARAGRAPH
        assert b.inner == "Some prose."
        assert b.html == "<p>Some prose.</p>"

    def test_bare_text_gets_wrapped_in_p(self) -> None:
        # Observed for a byline with an inline ORCID <img> and for a bare
        # caption label ("FIG 2") — chandra doesn't always wrap Text/Caption
        # content in <p>.
        b = Block.from_chandra_div("Text", 'Jane Doe <img alt="ORCID icon">')
        assert b is not None
        assert b.kind is BlockKind.PARAGRAPH
        assert b.html == '<p>Jane Doe <img alt="ORCID icon"></p>'

    def test_leading_and_trailing_whitespace_is_stripped(self) -> None:
        # chandra's Table divs observed with a leading/trailing newline around
        # the <table> content.
        b = Block.from_chandra_div("Table", "\n<table><tr><td>x</td></tr></table>\n")
        assert b is not None
        assert b.html.startswith("<table")
        assert b.html.endswith("</table>")

    def test_caption_and_footnote_map_to_paragraph(self) -> None:
        for label in ("Caption", "Footnote"):
            b = Block.from_chandra_div(label, "<p>Figure 1. A description.</p>")
            assert b is not None
            assert b.kind is BlockKind.PARAGRAPH

    def test_source_page_threaded_through(self) -> None:
        b = Block.from_chandra_div("Text", "<p>x</p>", source_page=4)
        assert b is not None
        assert b.source_page == 4

    def test_of_unaffected_by_the_shared_finish_refactor(self) -> None:
        # Guards against the _finish extraction changing Block.of's own
        # behavior — the existing TestBlockKind/TestDerivedFields classes above
        # already cover this in depth; this is a belt-and-suspenders spot check.
        b = Block.of("<p><strong>Keywords:</strong> a, b, c</p>")
        assert b.bold_label == "Keywords"
