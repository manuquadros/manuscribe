"""Tests for table caption/footnote colocation, localization, re-OCR gate."""

import re

from helpers import _as_blocks, _body, _run_lighton, _tables_text


class TestTableCaptionColocation:
    """A free-standing "Table N …" caption is folded into its <table> as a
    <caption> first child so it renders with the table, not adrift."""

    def test_caption_before_table_folded(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_captions

        parts = [
            "<p><strong>TABLE 1</strong> Enzyme kinetics of PtTRI and PtTRII</p>",
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
        ]
        assert _colocate_table_captions(_as_blocks(parts)) == [
            "<table><caption><strong>TABLE 1</strong> Enzyme kinetics of PtTRI "
            "and PtTRII</caption><tbody><tr><td>1</td></tr></tbody></table>"
        ]

    def test_caption_after_table_folded(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_captions

        parts = [
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
            "<p>Table 2. Results.</p>",
        ]
        out = _colocate_table_captions(_as_blocks(parts))
        assert out == [
            "<table><caption>Table 2. Results.</caption>"
            "<tbody><tr><td>1</td></tr></tbody></table>"
        ]

    def test_heading_form_caption_folded(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_captions

        # The model sometimes promotes a whole table caption to a section heading
        # ("## TABLE 2 …"); it must still fold into the table, not stay an <h2>.
        parts = [
            "<h2>TABLE 2 Comparison between various tropinone reductases</h2>",
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
        ]
        out = _colocate_table_captions(_as_blocks(parts))
        assert out == [
            "<table><caption>TABLE 2 Comparison between various tropinone "
            "reductases</caption><tbody><tr><td>1</td></tr></tbody></table>"
        ]

    def test_word_identifier_heading_not_promoted_to_caption(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_captions

        # A real section heading whose identifier is a word ("Table of Contents")
        # must not be folded into an adjacent table; only number-like identifiers
        # ("TABLE 2", "Table IV") are promoted from a heading.
        parts = [
            "<h2>Table of Contents</h2>",
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
        ]
        out = _colocate_table_captions(_as_blocks(parts))
        assert out == parts  # unchanged: heading stays, table stays captionless

    def test_caption_separated_by_figure_folded(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_captions

        # The reported bug: a figure floats between the caption and its table.
        parts = [
            "<p>Table 3 Kinetic constants</p>",
            "<figure><img src='x' alt=''></figure>",
            "<table><tr><td>a</td></tr></table>",
        ]
        out = _colocate_table_captions(_as_blocks(parts))
        assert out == [
            "<figure><img src='x' alt=''></figure>",
            "<table><caption>Table 3 Kinetic constants</caption>"
            "<tr><td>a</td></tr></table>",
        ]

    def test_caption_not_pulled_across_prose(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_captions

        # A real paragraph between caption and table breaks the association.
        parts = [
            "<p>Table 4 X</p>",
            "<p>Unrelated body sentence.</p>",
            "<table><tr><td>a</td></tr></table>",
        ]
        assert _colocate_table_captions(_as_blocks(parts)) == parts

    def test_orphan_caption_left_intact(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_captions

        parts = [
            "<p>Table 9 Orphan caption with no table near it.</p>",
            "<p>Prose.</p>",
        ]
        assert _colocate_table_captions(_as_blocks(parts)) == parts

    def test_two_tables_pair_with_own_captions(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_captions

        parts = [
            "<p>Table 1 A</p>",
            "<table><tr><td>1</td></tr></table>",
            "<p>Table 2 B</p>",
            "<table><tr><td>2</td></tr></table>",
        ]
        out = _colocate_table_captions(_as_blocks(parts))
        assert "<caption>Table 1 A</caption>" in out[0]
        assert "<caption>Table 2 B</caption>" in out[1]
        assert len(out) == 2

    def test_existing_caption_not_duplicated(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_captions

        parts = [
            "<p>Table 5 Duplicate guard</p>",
            "<table><caption>existing</caption><tr><td>1</td></tr></table>",
        ]
        out = _colocate_table_captions(_as_blocks(parts))
        # The table keeps its own caption; the stray block is left, not lost.
        assert out == parts

    def test_table_attributes_preserved(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_captions

        # The caption goes after the *whole* opening tag, not inside it.
        parts = [
            "<p>Table 1 Kinetics</p>",
            '<table class="data"><tbody><tr><td>1</td></tr></tbody></table>',
        ]
        out = _colocate_table_captions(_as_blocks(parts))
        assert out == [
            '<table class="data"><caption>Table 1 Kinetics</caption>'
            "<tbody><tr><td>1</td></tr></tbody></table>"
        ]

    def test_caption_between_tables_pairs_with_following(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_captions

        # A caption sat between two tables precedes the second, so it belongs to
        # it — the first (captionless) table must not forward-steal it.
        parts = [
            "<table><tr><td>1</td></tr></table>",
            "<p>Table 2 Results</p>",
            "<table><tr><td>2</td></tr></table>",
        ]
        out = _colocate_table_captions(_as_blocks(parts))
        assert out == [
            "<table><tr><td>1</td></tr></table>",
            "<table><caption>Table 2 Results</caption><tr><td>2</td></tr></table>",
        ]

    def test_reference_sentence_not_folded(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_captions

        # "Table N <lowercase verb> …" is a running reference, not a caption; it
        # must stay in the body, not be absorbed into the table.
        parts = [
            "<p>Table 1 summarizes the kinetic parameters of both enzymes.</p>",
            "<table><tr><td>1</td></tr></table>",
        ]
        assert _colocate_table_captions(_as_blocks(parts)) == parts

    def test_bare_label_rejoined_then_folded(self) -> None:
        from manuscribe.pipeline.merge import (
            _colocate_table_captions,
            _join_split_table_caption_labels,
        )

        # The reported bug: OCR split "TABLE I" from its title, stranding the
        # title between label and table so the caption never folds.
        parts = [
            "<p>TABLE I</p>",
            "<p>Selected substrates and inhibitors used to investigate.</p>",
            "<table><tbody><tr><td>a</td></tr></tbody></table>",
        ]
        out = _colocate_table_captions(
            _as_blocks(_join_split_table_caption_labels(parts))
        )
        assert out == [
            "<table><caption>TABLE I Selected substrates and inhibitors "
            "used to investigate.</caption><tbody><tr><td>a</td></tr></tbody></table>"
        ]

    def test_heading_label_rejoined_then_folded(self) -> None:
        from manuscribe.pipeline.merge import (
            _colocate_table_captions,
            _join_split_table_caption_labels,
        )

        # The OCR sometimes promotes the bare label to a heading
        # ("## TABLE IV" → <h2>), stranding the title below it just like the
        # <p>-form label; it must rejoin and fold the same way.
        parts = [
            "<h2>TABLE IV</h2>",
            "<p>Comparison of rR- and rS-HPCDH kinetic parameters.</p>",
            "<table><tbody><tr><td>a</td></tr></tbody></table>",
        ]
        out = _colocate_table_captions(
            _as_blocks(_join_split_table_caption_labels(parts))
        )
        assert out == [
            "<table><caption>TABLE IV Comparison of rR- and rS-HPCDH kinetic "
            "parameters.</caption><tbody><tr><td>a</td></tr></tbody></table>"
        ]

    def test_labelled_caption_not_rejoined(self) -> None:
        from manuscribe.pipeline.merge import _join_split_table_caption_labels

        # A label that already carries its title ("Table 4 X") is a complete
        # caption; the following block is unrelated prose and must stay separate.
        parts = [
            "<p>Table 4 X</p>",
            "<p>Unrelated body sentence.</p>",
        ]
        assert _join_split_table_caption_labels(parts) == parts

    def test_bare_label_before_table_not_rejoined_with_table(self) -> None:
        from manuscribe.pipeline.merge import _join_split_table_caption_labels

        # A bare label sitting directly on its table needs no rejoin (the next
        # block is the <table>, not a stray title paragraph).
        parts = [
            "<p>TABLE I</p>",
            "<table><tbody><tr><td>a</td></tr></tbody></table>",
        ]
        assert _join_split_table_caption_labels(parts) == parts

    def test_split_caption_lets_paragraph_merge_across_table(self) -> None:
        # End-to-end: a paragraph split across a captioned table rejoins once the
        # split caption is folded into the <table> (only the float remains between
        # the two halves).
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Results\n\n"
            "analyses of 2-butanol production revealed that,\n\n"
            "TABLE I\n\n"
            "Selected substrates and inhibitors used to investigate.\n\n"
            "<table><tbody><tr><td>Km</td></tr></tbody></table>\n\n"
            "with no additives present, all forms preferred a re-face addition."
        )
        body = _body(_run_lighton([md]))
        assert "revealed that, with no additives present, all forms preferred" in body
        assert "<caption>TABLE I Selected substrates" in body
        assert "revealed that,</p>" not in body

    def test_end_to_end_caption_heads_table(self) -> None:
        # Full assembly: the fragment must not absorb the caption, and the
        # caption must end up inside its table, not before the figures.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Results\n\n"
            "PtTRII catalyzed the reduction of tropinone to form\n\n"
            "![image](0,0,500,500)\n\n"
            "**TABLE 1** Enzyme kinetics of PtTRI and PtTRII\n\n"
            "<table><tbody><tr><td>Km</td></tr></tbody></table>"
        )
        body = _body(_run_lighton([md]))
        assert "<caption><strong>TABLE 1</strong> Enzyme kinetics" in body
        assert "to form <strong>TABLE 1</strong>" not in body
        # The caption no longer appears as a stand-alone paragraph.
        assert "<p><strong>TABLE 1</strong>" not in body

    def test_heading_label_title_not_absorbed_by_body_continuation(self) -> None:
        # 30592559 page 5→6: the OCR rendered "TABLE IV" as a heading with its
        # untitled-looking title stranded below.  Lacking terminal punctuation the
        # title was mistaken for a body fragment and the prose resuming after the
        # table ("reaction stereospecificity was not…") was glued onto the caption
        # instead of onto the page-5 paragraph it continues.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Implementation\n\n"
            "Although the general mechanism of a Zn-dependent alcohol "
            "dehydrogenase was covered, the\n\n"
            "## TABLE IV\n\n"
            "Comparison of rR- and rS-HPCDH kinetic parameters and stereoselectivity"
            "\n\n"
            "<table><tbody><tr><td>Km</td></tr></tbody></table>\n\n"
            "reaction stereospecificity was not. An additional prerequisite topic "
            "was prochirality."
        )
        body = _body(_run_lighton([md]))
        assert (
            "dehydrogenase was covered, the reaction stereospecificity was not." in body
        )
        assert "<caption>TABLE IV Comparison of rR- and rS-HPCDH" in body
        # The caption text must not have swallowed the body continuation.
        assert "stereoselectivity reaction stereospecificity" not in body


class TestTableFootnoteColocation:
    """A table's trailing footnote run — superscript-marker lines plus a note
    sentence wedged before them — is folded onto the table block, not left adrift
    or swept into the article footnote section."""

    def test_marker_footnotes_folded_onto_table(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_footnotes

        # The table carries the a/b markers its footnotes annotate.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td><td>E<sup>b</sup></td></tr>"
            "</tbody></table>",
            "<p><sup>a</sup>Apparent K values.</p>",
            "<p><sup>b</sup>ND = not determined.</p>",
        ]
        assert _colocate_table_footnotes(_as_blocks(parts)) == [
            "<table><tbody><tr><td>K<sup>a</sup></td><td>E<sup>b</sup></td></tr>"
            "</tbody></table>"
            '<p class="footnote"><sup>a</sup>Apparent K values.</p>'
            '<p class="footnote"><sup>b</sup>ND = not determined.</p>'
        ]

    def test_note_sentence_before_markers_folded(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_footnotes

        # The reported case: a note sentence sits between the table and its
        # superscript footnotes, so it rides along into the table block.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p>Molecule structures are shown in Fig. 3.</p>",
            "<p><sup>a</sup>Apparent K values.</p>",
        ]
        assert _colocate_table_footnotes(_as_blocks(parts)) == [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>"
            '<p class="footnote">Molecule structures are shown in Fig. 3.</p>'
            '<p class="footnote"><sup>a</sup>Apparent K values.</p>'
        ]

    def test_body_after_markers_stays_in_stream(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_footnotes

        # The body paragraph that resumes after the footnotes is not absorbed.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p><sup>a</sup>Apparent K values.</p>",
            "<p>with no additives present, all forms preferred re-face.</p>",
        ]
        out = _colocate_table_footnotes(_as_blocks(parts))
        assert out == [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>"
            '<p class="footnote"><sup>a</sup>Apparent K values.</p>',
            "<p>with no additives present, all forms preferred re-face.</p>",
        ]

    def test_article_footnote_marker_not_in_table_not_absorbed(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_footnotes

        # The hardening: a superscript line whose label the table does NOT carry
        # is an article footnote that merely follows the table, not a table
        # footnote, so it is left for the classifier to route before references.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p><sup>*</sup>Corresponding author: a@b.com.</p>",
        ]
        assert _colocate_table_footnotes(_as_blocks(parts)) == parts

    def test_numeric_marker_matching_table_exponent_not_absorbed(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_footnotes

        # A numbered article footnote whose digit collides with a table exponent
        # ("cm<sup>2</sup>") must not be folded — exponents are not footnote
        # referents, so the numeric label does not qualify as a table marker.
        parts = [
            "<table><tbody><tr><td>area cm<sup>2</sup></td></tr></tbody></table>",
            "<p><sup>2</sup>A numbered article footnote.</p>",
        ]
        assert _colocate_table_footnotes(_as_blocks(parts)) == parts

    def test_letter_marker_folded_despite_table_exponents(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_footnotes

        # A letter footnote still folds when the table mixes exponents and an
        # 'a' referent; the exponent does not interfere.
        parts = [
            "<table><tbody><tr><td>cm<sup>2</sup></td><td>K<sup>a</sup></td></tr>"
            "</tbody></table>",
            "<p><sup>a</sup>Apparent K values.</p>",
        ]
        assert _colocate_table_footnotes(_as_blocks(parts)) == [
            "<table><tbody><tr><td>cm<sup>2</sup></td><td>K<sup>a</sup></td></tr>"
            "</tbody></table>"
            '<p class="footnote"><sup>a</sup>Apparent K values.</p>'
        ]

    def test_note_before_unmatched_marker_not_absorbed(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_footnotes

        # A leading note rides along only when matching markers follow; an
        # unmatched marker abandons the run, so the note stays in the stream.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p>A note sentence.</p>",
            "<p><sup>*</sup>An article footnote.</p>",
        ]
        assert _colocate_table_footnotes(_as_blocks(parts)) == parts

    def test_note_without_markers_not_absorbed(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_footnotes

        # A plain paragraph after a table with no footnote markers is body, not a
        # note, and is left untouched.
        parts = [
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
            "<p>This paragraph continues the discussion.</p>",
        ]
        assert _colocate_table_footnotes(_as_blocks(parts)) == parts

    def test_runaway_leading_prose_not_swallowed(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_footnotes

        # More than a note's worth of prose before any marker is body, so the
        # run is abandoned and nothing is folded — even though the table carries
        # the late marker's label.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p>First body paragraph.</p>",
            "<p>Second body paragraph.</p>",
            "<p>Third body paragraph.</p>",
            "<p><sup>a</sup>A late marker.</p>",
        ]
        assert _colocate_table_footnotes(_as_blocks(parts)) == parts

    def test_second_leading_line_exceeds_note_cap(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_footnotes

        # A table note is a single line; two non-marker lines before the marker
        # exceed the cap, so the run is abandoned rather than swallowing a second
        # (possibly body) line.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p>First note line.</p>",
            "<p>Second note line.</p>",
            "<p><sup>a</sup>Apparent K values.</p>",
        ]
        assert _colocate_table_footnotes(_as_blocks(parts)) == parts

    def test_trailing_source_note_after_markers_folded(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_footnotes

        # The reported case: an attribution note with no superscript marker trails
        # the marker run, so the marker loop can't anchor it; its lexical shape
        # ("Data adapted from …") folds it onto the table.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p><sup>a</sup>Apparent K values.</p>",
            "<p>Data adapted from Clark et al. [7].</p>",
        ]
        assert _colocate_table_footnotes(_as_blocks(parts)) == [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>"
            '<p class="footnote"><sup>a</sup>Apparent K values.</p>'
            '<p class="footnote">Data adapted from Clark et al. [7].</p>'
        ]

    def test_standalone_source_note_without_markers_folded(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_footnotes

        # A source note can also be the table's only footnote, with no markers at
        # all; the lexical cue still folds it onto the table.
        parts = [
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
            "<p>Data adapted from Clark et al. [7].</p>",
            "<p>The discussion continues in ordinary prose here.</p>",
        ]
        assert _colocate_table_footnotes(_as_blocks(parts)) == [
            "<table><tbody><tr><td>1</td></tr></tbody></table>"
            '<p class="footnote">Data adapted from Clark et al. [7].</p>',
            "<p>The discussion continues in ordinary prose here.</p>",
        ]

    def test_body_line_before_standalone_source_note_not_swallowed(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_footnotes

        # A body line stranded between the table and a markerless source note must
        # not be dragged into the footnotes just because the source note folds: with
        # no anchoring markers, the ambiguous arrangement leaves both in the stream.
        parts = [
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
            "<p>This finding is discussed further in the next section.</p>",
            "<p>Data adapted from Clark et al. [7].</p>",
            "<p>More body prose follows here.</p>",
        ]
        assert _colocate_table_footnotes(_as_blocks(parts)) == parts

    def test_generic_verb_without_subject_not_a_source_note(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_footnotes

        # A body sentence opening with a generic participle ("Obtained from …")
        # is not an attribution note, so it stays in the body after the table.
        parts = [
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
            "<p>Obtained from a commercial supplier, the reagents were used.</p>",
        ]
        assert _colocate_table_footnotes(_as_blocks(parts)) == parts

    def test_plain_body_after_markers_not_mistaken_for_source_note(self) -> None:
        from manuscribe.pipeline.merge import _colocate_table_footnotes

        # The body resuming after the markers does not open with a source cue, so
        # the trailing-note sweep leaves it in the stream.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p><sup>a</sup>Apparent K values.</p>",
            "<p>Data presented here support the proposed mechanism in Fig. 2.</p>",
        ]
        assert _colocate_table_footnotes(_as_blocks(parts)) == [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>"
            '<p class="footnote"><sup>a</sup>Apparent K values.</p>',
            "<p>Data presented here support the proposed mechanism in Fig. 2.</p>",
        ]

    def test_end_to_end_table_footnotes_unblock_merge(self) -> None:
        # Full assembly: the table footnotes (a note sentence + marker lines) fold
        # into the table, so the paragraph split across the table rejoins and the
        # footnotes are not relocated to the article footnote section.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Results\n\n"
            "analyses of 2-butanol production revealed that,\n\n"
            "<table><tbody><tr><td>Km<sup>a</sup></td><td>E<sup>b</sup></td></tr>"
            "</tbody></table>\n\n"
            "Molecule structures are shown in Fig. 3.\n\n"
            "<sup>a</sup>Apparent K values.\n\n"
            "<sup>b</sup>ND = not determined.\n\n"
            "with no additives present, all forms preferred a re-face addition."
        )
        body = _body(_run_lighton([md]))
        assert "revealed that, with no additives present, all forms preferred" in body
        # The footnotes ride with the table inside the body, not before references.
        note = '<p class="footnote">Molecule structures are shown in Fig. 3.</p>'
        assert note in body
        assert '<p class="footnote"><sup>a</sup>Apparent K values.</p>' in body


class TestTableTextHelpers:
    """Pure markup helpers for table re-OCR substitution."""

    def test_cell_texts_and_count(self) -> None:
        from manuscribe.pipeline.tables import _cell_texts, _nonempty_cell_count

        table = (
            "<table><thead><tr><th>Metal ion</th><th></th></tr></thead>"
            "<tbody><tr><td>Mg<sup>2+</sup></td><td>3.0</td></tr></tbody></table>"
        )
        assert _cell_texts(table) == ["Metal ion", "Mg2+", "3.0"]
        # the empty <th> does not count
        assert _nonempty_cell_count(table) == 3

    def test_table_regions_group_consecutive_split_on_prose(self) -> None:
        from manuscribe.pipeline.tables import _table_regions

        md = (
            "intro\n\n"
            "<table><tr><td>a</td></tr></table>\n\n"
            "<table><tr><td>b</td></tr></table>\n\n"
            "some prose\n\n"
            "<table><tr><td>c</td></tr></table>"
        )
        regions = _table_regions(md)
        assert [len(tables) for _, _, tables in regions] == [2, 1]
        # the spans address only the <table> blocks, not the prose between regions
        start, end, _ = regions[1]
        assert md[start:end] == "<table><tr><td>c</td></tr></table>"

    def test_crop_trailing_returns_text_after_last_table(self) -> None:
        from manuscribe.pipeline.tables import _crop_trailing

        legend = "MW: molecular weight, NR: Not reported"
        md = f"<table><tr><td>x</td></tr></table>\n\n{legend}"
        assert _crop_trailing(md) == legend
        assert _crop_trailing("no table here") == ""

    def test_legend_footnote_preserves_sup_marker(self) -> None:
        from manuscribe.pipeline.tables import _legend_footnote_html

        # The recovered legend's <sup> marker and *emphasis* are OCR markup: render
        # them, don't HTML-escape (which would print a literal "<sup>a</sup>").
        legend = "<sup>a</sup>Each value represents the mean ± SD, *n* = 3."
        assert _legend_footnote_html(legend) == (
            '<p class="footnote"><sup>a</sup>Each value represents the mean ± SD, '
            "<em>n</em> = 3.</p>"
        )

    def test_legend_footnote_escapes_stray_markup(self) -> None:
        from manuscribe.pipeline.tables import _legend_footnote_html

        # A bare "<"/"&" in a legend ("n<5", "Tris & HCl") must be escaped, not left
        # to start a bogus tag — while a real <sup> marker still passes through.
        legend = "<sup>a</sup>Significant at n<5; Tris & HCl buffer"
        assert _legend_footnote_html(legend) == (
            '<p class="footnote"><sup>a</sup>Significant at n&lt;5; '
            "Tris &amp; HCl buffer</p>"
        )

    def test_collapse_repeated_rows_kills_decode_loop(self) -> None:
        from manuscribe.pipeline.tables import _collapse_repeated_rows

        # The crop re-OCR fell into a repetition loop, trailing a real table with
        # dozens of identical "RAMS Deviations" rows; collapse them to one.
        loop = "\n".join(
            "<tr><td>RAMS Deviations</td><td></td></tr>" for _ in range(55)
        )
        table = (
            "<table><tbody>"
            "<tr><td>bond lengths</td><td>0.011</td></tr>\n" + loop + "</tbody></table>"
        )
        collapsed = _collapse_repeated_rows(table)
        assert collapsed.count("RAMS Deviations") == 1
        assert "<td>bond lengths</td>" in collapsed

    def test_collapse_repeated_rows_keeps_short_repeats(self) -> None:
        from manuscribe.pipeline.tables import _collapse_repeated_rows

        # Genuine adjacent rows that happen to repeat a value a few times are not a
        # decode loop — a short run is left untouched.
        table = (
            "<table><tbody>"
            "<tr><td>x</td><td>1</td></tr>"
            "<tr><td>x</td><td>1</td></tr>"
            "<tr><td>x</td><td>1</td></tr>"
            "</tbody></table>"
        )
        assert _collapse_repeated_rows(table) == table

    def test_collapse_repeated_rows_preserves_repeated_cell_value(self) -> None:
        from manuscribe.pipeline.tables import _collapse_repeated_rows

        # A column repeating "NA" (or a number) across many rows is real data, not a
        # loop — the rows differ in their label cell, so they are not byte-identical
        # and must be left intact.
        table = (
            "<table><tbody>"
            + "".join(
                f"<tr><td>{label}</td><td>NA</td></tr>"
                for label in ("completeness", "redundancy", "resolution", "Rwork")
            )
            + "</tbody></table>"
        )
        assert _collapse_repeated_rows(table) == table

    def test_collapse_repeated_rows_preserves_sign_and_superscript_diffs(self) -> None:
        from manuscribe.pipeline.tables import _collapse_repeated_rows

        # The comparison is byte-exact, not normalized: rows sharing a label but
        # differing only by a sign or a superscript carry distinct data and must
        # survive (a normalized key would fold +1.5/-1.5 and R²/R2 together).
        table = (
            "<table><tbody>"
            "<tr><td>ΔG</td><td>+1.5</td></tr>"
            "<tr><td>ΔG</td><td>-1.5</td></tr>"
            "<tr><td>ΔG</td><td>+1.5</td></tr>"
            "<tr><td>ΔG</td><td>-1.5</td></tr>"
            "</tbody></table>"
        )
        assert _collapse_repeated_rows(table) == table

    def test_collapse_repeated_rows_md_collapses_page_table_loop(self) -> None:
        from manuscribe.pipeline.tables import _collapse_repeated_rows_md

        # The page-level re-OCR (not the crop path) can land a decode loop straight
        # in pages_md; the markdown-level pass collapses every table's loop in place
        # and leaves surrounding prose untouched.
        loop = "".join("<tr><td>RAMS Deviations</td><td></td></tr>" for _ in range(40))
        md = (
            "Some prose before.\n\n"
            '<table border="1" class="dataframe"><tbody>'
            "<tr><td>bond lengths</td><td>0.011</td></tr>" + loop + "</tbody></table>"
            "\n\nSome prose after."
        )
        collapsed = _collapse_repeated_rows_md(md)
        assert collapsed.count("RAMS Deviations") == 1
        assert "<td>bond lengths</td>" in collapsed
        assert "Some prose before." in collapsed
        assert "Some prose after." in collapsed

    def test_collapse_repeated_rows_md_is_idempotent_without_loop(self) -> None:
        from manuscribe.pipeline.tables import _collapse_repeated_rows_md

        # A page whose tables carry no degenerate run is returned byte-for-byte.
        md = (
            "Intro.\n\n<table><tbody>"
            "<tr><td>a</td><td>1</td></tr>"
            "<tr><td>b</td><td>2</td></tr>"
            "</tbody></table>\n\nOutro."
        )
        assert _collapse_repeated_rows_md(md) == md

    def test_extract_tables_strips_inner_caption(self) -> None:
        from manuscribe.pipeline.tables import _extract_tables

        # a level-1 heading is the overall caption (carried separately), so it is
        # not folded into the table
        md = (
            "# Table 2\n\n"
            "<table><caption>Table 2. X</caption><tr><td>x</td></tr></table>\n"
        )
        assert _extract_tables(md) == ["<table><tr><td>x</td></tr></table>"]

    def test_extract_tables_folds_subheading_as_spanning_row(self) -> None:
        from manuscribe.pipeline.tables import _extract_tables

        # the crop re-OCR lifts a sub-table label into a level-2 heading; it must
        # come back as a spanning header row spanning all the table's columns
        md = (
            "## A. Effect on activity\n\n"
            "<table><tbody><tr><td>None</td><td>100</td><td>99</td></tr>"
            "</tbody></table>"
        )
        [table] = _extract_tables(md)
        expected = '<thead><tr><th colspan="3">A. Effect on activity</th></tr></thead>'
        assert expected in table


class TestTableNormalization:
    """NFKD-plus-alnum folding closes the encoding gap between an OCR'd cell and
    the PDF text layer so the same content matches across both."""

    def test_superscript_and_micro_fold_to_text_layer_form(self) -> None:
        from manuscribe.pipeline.layers import _normalize

        assert _normalize("Mg²⁺") == _normalize("Mg2+") == "mg2"
        # micro sign vs Greek mu, and superscript ⁻¹ (a U+2212 minus) vs ASCII -1
        assert _normalize("µg L⁻¹") == _normalize("μg L-1")

    def test_index_map_recovers_source_range(self) -> None:
        from manuscribe.pipeline.layers import _normalize_with_map

        text = "A: Mg²⁺ ok"
        norm, idx_map = _normalize_with_map(text)
        assert len(norm) == len(idx_map)
        # the normalized "mg2" maps back onto the original "Mg²" span
        p = norm.index("mg2")
        assert text[idx_map[p] : idx_map[p + 2] + 1] == "Mg²"


class TestTableLocalization:
    """``_locate_bbox`` grows an anchor seed through the table's rows but halts at
    the wider margin to body prose, so a re-OCR crop stays tight on the table."""

    @staticmethod
    def _layout(
        lines: list[tuple[str, float | int, float | int]],
    ) -> tuple[
        str,
        list[tuple[float, float, float, float] | None],
        list[int | None],
    ]:
        # Lay each (text, y_top, x_left) line out as fixed 6×8 pt glyph boxes,
        # spaces and the inter-line newline carrying a degenerate (None) box —
        # mirroring how pdfium reports them.  Every glyph is upright (rotation 0);
        # the rotations list is index-aligned with the boxes.
        text = ""
        boxes: list[tuple[float, float, float, float] | None] = []
        rotations: list[int | None] = []
        for i, (s, y_top, x_left) in enumerate(lines):
            if i:
                text += "\n"
                boxes.append(None)
                rotations.append(None)
            x = float(x_left)
            top = float(y_top)
            for ch in s:
                text += ch
                boxes.append(None if ch == " " else (x, top - 8, x + 6, top))
                rotations.append(None if ch == " " else 0)
                x += 6
        return text, boxes, rotations

    def test_bbox_covers_table_rows_excludes_prose(self) -> None:
        from manuscribe.pipeline.layers import _normalize_with_map
        from manuscribe.pipeline.tables import _locate_bbox

        # A 5-row table at the top, then a wide gap, then dense prose.  Only the
        # heading row is a (unique) anchor; growth must still reach the trailing
        # rows yet stop before the prose.
        lines = [
            ("Effect of EDTA on activity", 700, 50),
            ("Relative activity percent", 688, 200),
            ("None 100 100", 676, 50),
            ("EDTA 100 99", 664, 50),
            ("12 34 56", 652, 50),
            ("Discussion text begins here now", 600, 50),
            ("and continues across the page", 588, 50),
        ]
        text, boxes, rotations = self._layout(lines)
        norm, idx_map = _normalize_with_map(text)
        located = _locate_bbox(
            ["effect of edta on activity"],
            norm,
            idx_map,
            boxes,
            rotations,
            (400.0, 800.0),
        )
        assert located is not None
        bbox, rot = located
        assert rot == 0
        left, bottom, right, top = bbox
        assert top >= 700 and bottom <= 644  # spans heading down to the "12 34 56" row
        assert bottom > 600  # the Discussion prose is excluded
        # the dropped-style interior row lies within the located region
        assert bottom <= 680 and top >= 688

    def test_returns_none_when_no_anchor_matches(self) -> None:
        from manuscribe.pipeline.layers import _normalize_with_map
        from manuscribe.pipeline.tables import _locate_bbox

        text, boxes, rotations = self._layout([("Effect of EDTA on activity", 700, 50)])
        norm, idx_map = _normalize_with_map(text)
        assert (
            _locate_bbox(["nowhere"], norm, idx_map, boxes, rotations, (400.0, 800.0))
            is None
        )

    def test_repeated_anchor_is_ambiguous_and_skipped(self) -> None:
        from manuscribe.pipeline.layers import _normalize_with_map
        from manuscribe.pipeline.tables import _locate_bbox

        # "alpha beta" occurs twice, so it cannot seed the box on its own.
        text, boxes, rotations = self._layout(
            [("alpha beta", 700, 50), ("filler here", 660, 50), ("alpha beta", 300, 50)]
        )
        norm, idx_map = _normalize_with_map(text)
        assert (
            _locate_bbox(
                ["alpha beta"], norm, idx_map, boxes, rotations, (400.0, 800.0)
            )
            is None
        )

    @staticmethod
    def _vlayout(
        strips: list[tuple[str, float | int, float | int]],
        rot: int,
    ) -> tuple[
        str,
        list[tuple[float, float, float, float] | None],
        list[int | None],
    ]:
        # A sideways table: each reading "line" is a vertical column-strip at a
        # fixed x whose glyphs advance *downward* in y, all rotated by ``rot``.
        text = ""
        boxes: list[tuple[float, float, float, float] | None] = []
        rotations: list[int | None] = []
        for i, (s, x_left, y_top) in enumerate(strips):
            if i:
                text += "\n"
                boxes.append(None)
                rotations.append(None)
            x = float(x_left)
            y = float(y_top)
            for ch in s:
                text += ch
                boxes.append(None if ch == " " else (x, y - 8, x + 8, y))
                rotations.append(None if ch == " " else rot)
                y -= 8
        return text, boxes, rotations

    def test_sideways_table_located_on_reading_axis_excludes_prose(self) -> None:
        from manuscribe.pipeline.layers import _normalize_with_map
        from manuscribe.pipeline.tables import _locate_bbox

        # A 270°-rotated table occupies three vertical column-strips on the right;
        # an upright body heading sits in the left column.  Localization must run
        # along the table's reading axis (left↔right over the strips) and exclude
        # the upright prose, rather than sweeping it in via the old y-axis growth.
        table_text, table_boxes, table_rot = self._vlayout(
            [
                ("methanol column", 200, 600),
                ("3231605 6521198", 220, 600),
                ("7778 5091 wt s1", 240, 600),
            ],
            rot=270,
        )
        body_text, body_boxes, body_rot = self._layout(
            [("CONCLUSION", 600, 40), ("Development of methylotrophy", 588, 40)]
        )
        text = table_text + "\n" + body_text
        boxes = table_boxes + [None] + body_boxes
        rotations = table_rot + [None] + body_rot
        norm, idx_map = _normalize_with_map(text)
        located = _locate_bbox(
            ["methanol column"], norm, idx_map, boxes, rotations, (400.0, 800.0)
        )
        assert located is not None
        bbox, rot = located
        assert rot == 270
        left, bottom, right, top = bbox
        assert left > 180  # the upright body column (x≈40) is excluded
        assert left <= 200 and right >= 248  # spans the three table strips


class TestCoverageGate:
    """``_region_fully_captured`` skips a table's crop re-OCR only when the page
    already captured every distinctive text-layer token inside the located bbox."""

    @staticmethod
    def _centers(
        lines: list[tuple[str, float | int, float | int]],
    ) -> tuple[str, list[tuple[float, float] | None]]:
        from manuscribe.pipeline.layers import _normalize_with_map
        from manuscribe.pipeline.tables import _glyph_centers

        text, boxes, _ = TestTableLocalization._layout(lines)
        norm, idx_map = _normalize_with_map(text)
        return norm, _glyph_centers(norm, idx_map, boxes)

    def test_in_bbox_tokens_keeps_only_in_box_words(self) -> None:
        from manuscribe.pipeline.tables import _in_bbox_tokens

        norm, centers = self._centers(
            [("alpha beta", 700, 50), ("gamma delta", 600, 50)]
        )
        toks = _in_bbox_tokens((0.0, 690.0, 400.0, 710.0), norm, centers)
        assert toks == ["alpha", "beta"]  # top line only, bottom line excluded

    def test_in_bbox_tokens_does_not_glue_clipped_word(self) -> None:
        from manuscribe.pipeline.tables import _in_bbox_tokens

        # The right edge cuts off "beta"; "alpha" must survive as its own token —
        # the box-less space still has to break the words, not weld "alphabeta".
        norm, centers = self._centers([("alpha beta", 700, 50)])
        toks = _in_bbox_tokens((0.0, 690.0, 82.0, 710.0), norm, centers)
        assert toks == ["alpha"]

    def test_fully_captured_when_all_distinctive_tokens_present(self) -> None:
        from manuscribe.pipeline.tables import _region_fully_captured

        norm, centers = self._centers([("alpha beta", 700, 50)])
        assert _region_fully_captured(
            (0.0, 690.0, 400.0, 710.0), norm, centers, {"alpha", "beta"}
        )

    def test_not_captured_when_a_token_is_missing(self) -> None:
        from manuscribe.pipeline.tables import _region_fully_captured

        # "beta" is in the text layer but not the captured cells — content the page
        # pass dropped — so the gate must not fire and the region is re-OCR'd.
        norm, centers = self._centers([("alpha beta", 700, 50)])
        assert not _region_fully_captured(
            (0.0, 690.0, 400.0, 710.0), norm, centers, {"alpha"}
        )

    def test_numeric_drop_is_not_skipped(self) -> None:
        from manuscribe.pipeline.tables import _region_fully_captured

        # All words captured but a multi-digit data value (26621) missing — a dropped
        # data cell. Numbers count as distinctive evidence, so the gate must not fire.
        norm, centers = self._centers([("alpha 26621", 700, 50)])
        assert not _region_fully_captured(
            (0.0, 690.0, 400.0, 710.0), norm, centers, {"alpha"}
        )

    def test_empty_text_layer_forces_reocr(self) -> None:
        from manuscribe.pipeline.tables import _region_fully_captured

        # A bbox over a region with no glyphs (a scanned page) gives no evidence, so
        # the gate stays off — it never opts out on absence of evidence.
        norm, centers = self._centers([("alpha beta", 700, 50)])
        assert not _region_fully_captured(
            (0.0, 100.0, 400.0, 200.0), norm, centers, {"alpha", "beta"}
        )

    def test_short_non_numeric_tokens_are_not_evidence(self) -> None:
        from manuscribe.pipeline.tables import _region_fully_captured

        # Only short non-numeric tokens in the bbox — no distinctive token to judge
        # completeness, so the gate does not fire even with nothing captured.
        norm, centers = self._centers([("ab cd ef", 700, 50)])
        assert not _region_fully_captured(
            (0.0, 690.0, 400.0, 710.0), norm, centers, set()
        )

    def test_adjacent_para_tokens_pick_caption_and_legend(self) -> None:
        from manuscribe.pipeline.tables import _adjacent_para_tokens

        md = (
            "Table 1. Effect of metals on activity\n\n"
            "<table><tr><td>x</td></tr></table>\n\n"
            "MW molecular weight reported"
        )
        start = md.index("<table>")
        end = md.index("</table>") + len("</table>")
        toks = _adjacent_para_tokens(md, start, end)
        assert {"effect", "metals"} <= toks  # caption above the table
        assert {"molecular", "reported"} <= toks  # legend below the table

    def test_adjacent_para_ignores_long_prose_block(self) -> None:
        from manuscribe.pipeline.tables import _adjacent_para_tokens

        # A long block flanking the table (e.g. body prose the bbox overran, or the
        # whole pre-table content when no blank line separates it) is NOT folded into
        # 'captured' — otherwise its tokens could mask a genuinely dropped table cell.
        long_prose = "discussion " * 60  # well past _ADJACENT_PARA_MAX_LEN
        md = f"{long_prose}<table><tr><td>x</td></tr></table>"
        start = md.index("<table>")
        end = md.index("</table>") + len("</table>")
        assert "discussion" not in _adjacent_para_tokens(md, start, end)


class TestUnclosedTableClosing:
    """A table the OCR leaves open at a page bottom must be closed before assembly,
    or it swallows whatever follows (most visibly the next page's prose)."""

    def test_close_unclosed_tables_balances_and_is_idempotent(self) -> None:
        from manuscribe.pipeline.tables import _close_unclosed_tables

        assert (
            _close_unclosed_tables("<table><tr><td>a</td><td>0")
            == "<table><tr><td>a</td><td>0</table>"
        )
        balanced = "<table><tr><td>a</td></tr></table>"
        assert _close_unclosed_tables(balanced) == balanced

    def test_overrun_table_does_not_swallow_next_page_prose(self) -> None:
        # page 1's table runs off the bottom with no </table>; page 2 opens with
        # prose that must render as body text, not inside the table
        page1 = "<table>\n<tr><td>Organism</td><td>M.W.</td></tr>\n<tr><td>X</td><td>0"
        page2 = "Downstream prose that follows the table on the next page."
        html = _run_lighton([page1, page2])
        assert "Downstream prose that follows" in _body(html)
        assert "Downstream prose that follows" not in _tables_text(html)


class TestTextLayerTableRepair:
    """A two-column stats table the OCR mangles (drops the empty header cell, shifting
    the first rows off by one and losing a value; truncates the tail) is rebuilt from
    the deterministic PDF text layer, keeping the OCR's cell formatting by match
    (the 31298526 Table 2; see the codebase-review Done log)."""

    @staticmethod
    def _glyphs(text: str, x0: float, y: float, cw: float = 5.0, gh: float = 8.0):
        # build (char, x0, x1, y-center, y-bottom, y-top) glyphs around centre y with
        # glyph height gh; a space advances x by a wide gap, no glyph
        out, x = [], float(x0)
        for ch in text:
            if ch == " ":
                x += cw
                continue
            out.append((ch, x, x + cw, float(y), y - gh / 2, y + gh / 2))
            x += cw + 1.0
        return out

    def test_glyph_cell_text_spaces_words_keeps_numbers(self) -> None:
        from manuscribe.pipeline.tables import _glyph_cell_text

        assert _glyph_cell_text(self._glyphs("PDB code", 0, 100)) == "PDB code"
        assert _glyph_cell_text(self._glyphs("37991", 0, 100)) == "37991"
        # a digit run the gap heuristic over-split (a glyph placed past the gap) rejoins
        glyphs = self._glyphs("0.08", 0, 100) + [("1", 60.0, 63.0, 100.0, 96.0, 104.0)]
        assert _glyph_cell_text(glyphs) == "0.081"

    def test_rows_to_cells_header_divider_and_footnote_stop(self) -> None:
        from manuscribe.pipeline.tables import _group_glyph_rows, _rows_to_cells

        glyphs = (
            self._glyphs("CgKARI", 80, 200)  # value-only row (empty label) at the top
            + self._glyphs("PDB code", 10, 188)
            + self._glyphs("6JX2", 80, 188)  # a label|value data row, value col
            + self._glyphs("wavelength", 10, 176)
            + self._glyphs("0.97934", 80, 176)
            + self._glyphs("space group", 10, 164)
            + self._glyphs("P21", 80, 164)
            + self._glyphs("a footnote sentence well over the length limit", 10, 140)
        )
        cells = _rows_to_cells(_group_glyph_rows(glyphs))
        assert cells[0] == ("", "CgKARI")  # empty label, value column
        assert ("PDB code", "6JX2") in cells
        assert ("wavelength", "0.97934") in cells
        # the long value-less footnote (below the data) stops the table
        assert not any("footnote" in lab for lab, _ in cells)

    def test_trim_rows_below_table_drops_body_prose(self) -> None:
        # The column glyph filter has only a top bound, so on a 2-column page it sweeps
        # body prose below the table into the rows; the gap-to-prose trim must cut at
        # the wider margin so the prose can't inflate the rebuild past the OCR table
        # and win the substitution gate with a wrong table.
        from manuscribe.pipeline.tables import (
            _group_glyph_rows,
            _trim_rows_below_table,
        )

        glyphs = (
            self._glyphs("PDB code", 10, 200)
            + self._glyphs("6JX2", 80, 200)
            + self._glyphs("wavelength", 10, 188)
            + self._glyphs("0.97934", 80, 188)
            + self._glyphs("space group", 10, 176)
            + self._glyphs("P212121", 80, 176)
            + self._glyphs("redundancy", 10, 164)
            + self._glyphs("4.2", 80, 164)
            # body prose far below the table: edge whitespace 36pt ≫ 1.5 × the 4pt
            # inter-row whitespace (12-pt pitch, 8-pt glyphs)
            + self._glyphs("The following paragraph discusses the structure", 10, 120)
            + self._glyphs("and continues with more body prose here", 10, 108)
        )
        rows = _group_glyph_rows(glyphs)
        seeds = [(10.0, 164.0, 200.0, 200.0)]  # spans the four data rows
        trimmed = _trim_rows_below_table(rows, seeds)
        assert len(trimmed) == 4
        flat = " ".join(g[0] for row in trimmed for g in row)
        assert "paragraph" not in flat and "prose" not in flat

    def test_trim_rows_below_table_drops_short_prose_at_modest_margin(self) -> None:
        # Two regressions in one shape.  (1) Metric: a prose row a modest margin
        # below a tight table reads as "within 1.5 × pitch" under the old
        # centre-to-centre metric and is wrongly kept; the edge-to-edge whitespace
        # metric (matching _locate_bbox) drops it — here the row pitch is 12 (4-pt
        # whitespace, 8-pt glyphs) and the prose sits at pitch 16 (kept by 1.5×12=18 ≥
        # 16, dropped by 1.5×4=6 < edge gap 8).  (2) Distinct contribution: the
        # below-table line is short and value-less ("see main text", <
        # _RECON_FOOTNOTE_LEN), so _rows_to_cells' footnote-stop would *not* remove it —
        # only the gap-trim does.
        from manuscribe.pipeline.tables import (
            _group_glyph_rows,
            _trim_rows_below_table,
        )

        glyphs = (
            self._glyphs("PDB code", 10, 200)
            + self._glyphs("6JX2", 80, 200)
            + self._glyphs("wavelength", 10, 188)
            + self._glyphs("0.97934", 80, 188)
            + self._glyphs("redundancy", 10, 176)
            + self._glyphs("4.2", 80, 176)
            + self._glyphs("see main text", 10, 160)  # short, value-less, pitch 16
        )
        rows = _group_glyph_rows(glyphs)
        seeds = [(10.0, 176.0, 200.0, 200.0)]  # spans the three data rows
        trimmed = _trim_rows_below_table(rows, seeds)
        assert len(trimmed) == 3
        assert "main" not in " ".join(g[0] for row in trimmed for g in row)

    def test_trim_rows_below_table_keeps_contiguous_rows(self) -> None:
        # No wide gap → nothing is body prose → every row is kept (a trailing data row
        # the anchors did not cover must still survive the trim).  The seed box overlaps
        # the top two rows, so the seeded-region spacing median (gaps[lo:hi], not the
        # whole-table fallback) drives the grow-down through the un-seeded third row.
        from manuscribe.pipeline.tables import (
            _group_glyph_rows,
            _trim_rows_below_table,
        )

        glyphs = (
            self._glyphs("PDB code", 10, 200)
            + self._glyphs("6JX2", 80, 200)
            + self._glyphs("wavelength", 10, 188)
            + self._glyphs("0.97934", 80, 188)
            + self._glyphs("bond lengths", 10, 176)  # trailing row, no seed
            + self._glyphs("0.011", 80, 176)
        )
        rows = _group_glyph_rows(glyphs)
        seeds = [(10.0, 188.0, 200.0, 200.0)]  # only the top two rows are seeded
        assert len(_trim_rows_below_table(rows, seeds)) == 3

    def test_format_cell_recovers_ocr_subscripts(self) -> None:
        from manuscribe.pipeline.tables import _cell_format_map, _format_cell

        ocr = (
            "<table><tr><td><em>R</em><sub>sym</sub> or <em>R</em><sub>merge</sub></td>"
            "<td>9.4</td></tr></table>"
        )
        fmt = _cell_format_map(ocr)
        # the plain reconstructed label matches the OCR cell (spaces ignored)
        assert _format_cell("Rsym or Rmerge", fmt) == (
            "<em>R</em><sub>sym</sub> or <em>R</em><sub>merge</sub>"
        )
        assert _format_cell("not in the table", fmt) == "not in the table"

    def test_format_cell_escapes_text_layer_fallback(self) -> None:
        from manuscribe.pipeline.tables import _format_cell

        # A rebuilt value the OCR never produced (no format-map entry) is raw
        # text-layer text; "<"/">"/"&" must be escaped or it injects markup / breaks
        # the <td> (a "<0.001" statistic, an "a & b").
        assert _format_cell("<0.001", {}) == "&lt;0.001"
        assert _format_cell("a & b", {}) == "a &amp; b"
        assert _format_cell("< 2.0", {}) == "&lt; 2.0"

    def test_leading_caption_rows_extracted(self) -> None:
        from manuscribe.pipeline.tables import _leading_caption_rows

        table = (
            "<table>"
            '<tr><th colspan="2">Table 2. Data Collection Statistics</th></tr>'
            "<tr><td>PDB code</td><td>6JX2</td></tr></table>"
        )
        rows, text = _leading_caption_rows(table)
        assert rows == [
            '<tr><th colspan="2">Table 2. Data Collection Statistics</th></tr>'
        ]
        assert "data collection statistics" in text

    def test_rows_to_cells_keeps_mid_table_divider_in_caption(self) -> None:
        # A mid-table value-less divider whose label is a substring of the caption
        # ("Refinement" ⊂ "… Structural Refinement Statistics") must NOT be dropped as a
        # caption fragment — only the wrapped fragments *above* the first data row are.
        # This exercises the column filter via the public rows_to_cells output shape.
        from manuscribe.pipeline.tables import _group_glyph_rows, _rows_to_cells

        glyphs = (
            self._glyphs("Refinement Statistics", 60, 220)  # caption fragment (top)
            + self._glyphs("resolution", 10, 200)
            + self._glyphs("2.6", 80, 200)  # first data row
            + self._glyphs("Refinement", 10, 188)  # a mid-table divider (value-less)
            + self._glyphs("R factor", 10, 176)
            + self._glyphs("0.17", 80, 176)
        )
        cells = _rows_to_cells(_group_glyph_rows(glyphs))
        # the divider survives in the raw cells; the caption-fragment filter (scoped to
        # above the body) is what would otherwise endanger it — see
        # _reconstruct_table_from_text_layer.
        assert ("Refinement", "") in cells

    def test_repair_fixes_31298526_table_2_deterministically(self) -> None:
        # End-to-end on the real PDF text layer (no GPU): a broken off-by-one Table 2
        # (the OCR shape) is rebuilt correctly and deterministically.  Skips if the
        # fixture PDF is absent.
        import pathlib

        from manuscribe.pipeline.tables import _repair_tables_from_text_layer

        pdf = pathlib.Path(__file__).parent / "fixtures" / "31298526.pdf"
        if not pdf.exists():
            import pytest

            pytest.skip(f"fixture PDF not found: {pdf}")
        # a condensed broken Table 2: off-by-one header (PDB code|CgKARI),
        # and a handful of anchor labels — enough to localize the column.
        broken = (
            "<table>"
            '<tr><th colspan="2">Table 2. Data Collection and Structural Refinement '
            "Statistics</th></tr>"
            "<tr><th>PDB code</th><th>CgKARI_NADP⁺</th></tr>"
            "<tr><td>wavelength (Å)</td><td>6JX2</td></tr>"
            "<tr><td>space group</td>"
            "<td>P2<sub>1</sub>2<sub>1</sub>2<sub>1</sub></td></tr>"
            "<tr><td>resolution range (Å)</td><td>78.9–2.6</td></tr>"
            "<tr><td>unique reflections</td><td>37991</td></tr>"
            "<tr><td><em>R</em><sub>sym</sub> or <em>R</em><sub>merge</sub></td>"
            "<td>9.4 (30.5)</td></tr>"
            "<tr><td>redundancy</td><td>4.2 (3.4)</td></tr>"
            "<tr><td>no. reflections</td><td>35327</td></tr>"
            "<tr><td>bond lengths (Å)</td><td>0.011</td></tr>"
            "</table>"
        )
        from manuscribe.pipeline.layers import _DocumentLayers

        with _DocumentLayers.open(str(pdf)) as layers:
            pages = ["" for _ in range(len(layers))]
            pages[2] = broken
            out = _repair_tables_from_text_layer(layers, pages)[2]
        run = lambda a, b: f"<td>{a}</td><td>{b}</td>"  # noqa: E731
        # the empty header cell is restored: CgKARI_NADP⁺ alone in column 2
        assert "<tr><td></td><td>CgKARI_NADP⁺</td></tr>" in out
        # the off-by-one is fixed: 6JX2 is the PDB code, wavelength gets its own value
        assert run("PDB code", "6JX2") in out
        assert re.search(r"wavelength \(Å\)</td><td>0\.97934", out)
        assert "6JX2" not in out.split("wavelength")[1][:40]
        # the dropped tail is recovered, and OCR sub/sup formatting is reused on a match
        assert all(w in out for w in ("favored", "allowed", "outliers"))
        assert "<sub>sym</sub>" in out  # R_sym formatting from a matched cell
        # the title row is preserved
        assert "Data Collection and Structural Refinement Statistics" in out

    def test_repair_fires_despite_decode_loop_inflation(self) -> None:
        # The page-level OCR can decode-loop on this dense table, inflating its raw <tr>
        # count past the reconstruction's (collapsed only later in _assemble_html).  The
        # substitution gate must compare against the *collapsed* OCR rows, or the loop
        # masks the off-by-one table and the repair never fires (the live-only bug).
        import pathlib

        from manuscribe.pipeline.tables import _repair_tables_from_text_layer

        pdf = pathlib.Path(__file__).parent / "fixtures" / "31298526.pdf"
        if not pdf.exists():
            import pytest

            pytest.skip(f"fixture PDF not found: {pdf}")
        loop = "".join("<tr><td>RAMS Deviations</td><td></td></tr>" for _ in range(50))
        broken = (
            "<table>"
            "<tr><td>PDB code</td><td>CgKARI_NADP⁺</td></tr>"
            "<tr><td>wavelength (Å)</td><td>6JX2</td></tr>"
            "<tr><td>space group</td><td>P21 21 21</td></tr>"
            "<tr><td>resolution range (Å)</td><td>78.9–2.6</td></tr>"
            "<tr><td>unique reflections</td><td>37991</td></tr>"
            "<tr><td>redundancy</td><td>4.2 (3.4)</td></tr>" + loop + "</table>"
        )
        from manuscribe.pipeline.layers import _DocumentLayers

        with _DocumentLayers.open(str(pdf)) as layers:
            pages = ["" for _ in range(len(layers))]
            pages[2] = broken
            out = _repair_tables_from_text_layer(layers, pages)[2]
        # the loop-inflated table is still replaced by the correct reconstruction
        assert "<tr><td></td><td>CgKARI_NADP⁺</td></tr>" in out
        assert re.search(r"wavelength \(Å\)</td><td>0\.97934", out)
        assert out.count("RAMS Deviations") <= 1


class TestRepairLazyExtraction:
    """_repair_page_tables extracts the page text layer lazily — at most once, and not
    at all when no complete 2-column table is present — so a page with only
    multi-column or unclosed tables pays nothing (the _page_layer call is thousands of
    native pdfium calls)."""

    def test_no_extraction_when_no_two_column_table(self) -> None:
        from unittest.mock import MagicMock

        import pypdfium2 as pdfium
        import pytest

        from manuscribe.pipeline import tables
        from manuscribe.pipeline.layers import _DocumentLayers, _PageLayer

        calls: list[int] = []
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "manuscribe.pipeline.layers._page_layer",
                lambda page: calls.append(1) or _PageLayer("", [], [], [], "", []),
            )
            # a 3-column table: the repair only understands label|value (2-col) tables
            md = "<table><tr><td>a</td><td>b</td><td>c</td></tr></table>"
            layers = _DocumentLayers(MagicMock(spec=pdfium.PdfDocument))
            assert tables._repair_page_tables(md, layers, 0) == md
        assert calls == []

    def test_extraction_runs_once_across_two_column_tables(self) -> None:
        from unittest.mock import MagicMock

        import pypdfium2 as pdfium
        import pytest

        from manuscribe.pipeline import tables
        from manuscribe.pipeline.layers import _DocumentLayers, _PageLayer

        calls: list[int] = []
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "manuscribe.pipeline.layers._page_layer",
                lambda page: calls.append(1) or _PageLayer("", [], [], [], "", []),
            )
            # _repair_page_tables resolves _reconstruct_table_from_text_layer in the
            # rebuild submodule, so patch it there (not the tables package re-export).
            mp.setattr(
                "manuscribe.pipeline.tables.rebuild._reconstruct_table_from_text_layer",
                lambda layer, table: None,
            )
            tbl = "<table><tr><td>label</td><td>value</td></tr></table>"
            md = tbl + "\n\n" + tbl  # two complete 2-column tables on one page
            layers = _DocumentLayers(MagicMock(spec=pdfium.PdfDocument))
            tables._repair_page_tables(md, layers, 0)
        assert len(calls) == 1


class TestDocumentLayersCache:
    """_DocumentLayers shares one _PageLayer per page across the post-OCR passes: the
    char-by-char extraction runs once per page index however many passes localize
    against it, and pages no pass touches are never extracted."""

    def test_page_layer_extracted_once_across_passes(self) -> None:
        from unittest.mock import MagicMock

        import pypdfium2 as pdfium
        import pytest

        from manuscribe.pipeline.layers import _DocumentLayers, _PageLayer

        calls: list[int] = []
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "manuscribe.pipeline.layers._page_layer",
                lambda page: calls.append(1) or _PageLayer("", [], [], [], "", []),
            )
            layers = _DocumentLayers(MagicMock(spec=pdfium.PdfDocument))
            # the table pass and the figure pass both localize against page 0
            first = layers.page_layer(0)
            second = layers.page_layer(0)
            other = layers.page_layer(1)
        assert first is second  # memoized, same instance handed back
        assert other is not first
        assert calls == [1, 1]  # once for page 0, once for page 1 — never re-extracted


class TestTableCellBold:
    """Re-applying <strong> to bold table cells from the PDF text layer: the bold
    threshold, the contextual (document-order) cell matcher, the majority vote, and the
    th/already-bold/miss guards — plus an end-to-end run on the real fixture layer."""

    def test_is_bold_glyph_threshold(self) -> None:
        from manuscribe.pipeline.tables import _is_bold_glyph

        assert not _is_bold_glyph(None)
        assert not _is_bold_glyph(315)
        assert not _is_bold_glyph(599)
        assert _is_bold_glyph(600)
        assert _is_bold_glyph(708)

    def test_majority_vote_bolds_cell(self) -> None:
        from manuscribe.pipeline.tables import _rewrap_bold_cells

        # "Total" is bold (708×5), "x" is normal (315) — only the bold cell is wrapped.
        table = "<table><tr><td>Total</td><td>x</td></tr></table>"
        out = _rewrap_bold_cells(table, "total x", [708] * 5 + [None, 315])
        assert "<td><strong>Total</strong></td>" in out
        assert "<td>x</td>" in out

    def test_one_thin_glyph_still_bold_majority(self) -> None:
        from manuscribe.pipeline.tables import _rewrap_bold_cells

        # 4/5 glyphs bold (0.8 >= 0.6) — one antialiased-thin glyph doesn't un-bold it.
        table = "<table><tr><td>Total</td></tr></table>"
        out = _rewrap_bold_cells(table, "total", [708, 708, 315, 708, 708])
        assert "<td><strong>Total</strong></td>" in out

    def test_lone_bold_glyph_does_not_bold_cell(self) -> None:
        from manuscribe.pipeline.tables import _rewrap_bold_cells

        # 1/5 glyphs bold (0.2 < 0.6) — a stray bold glyph leaves the cell plain.
        table = "<table><tr><td>Total</td></tr></table>"
        out = _rewrap_bold_cells(table, "total", [708, 315, 315, 315, 315])
        assert "<td>Total</td>" in out

    def test_repeated_cell_binds_in_document_order(self) -> None:
        from manuscribe.pipeline.tables import _rewrap_bold_cells

        # "Resolution" twice: the first occurrence is bold, the second normal — the
        # monotonic cursor binds each cell to its own occurrence, not always the first.
        table = (
            "<table><tr><td>Resolution</td></tr><tr><td>Resolution</td></tr></table>"
        )
        weights = [708] * 10 + [None] + [315] * 10
        out = _rewrap_bold_cells(table, "resolution resolution", weights)
        assert out.count("<strong>") == 1
        assert out.index("<strong>") < out.index("</tr><tr>")

    def test_miss_leaves_cell_unchanged(self) -> None:
        from manuscribe.pipeline.tables import _rewrap_bold_cells

        # The OCR text isn't in the layer here → no guess, cell unchanged.
        table = "<table><tr><td>xyz</td></tr></table>"
        out = _rewrap_bold_cells(table, "abc", [708, 708, 708])
        assert out == table

    def test_header_th_not_bolded(self) -> None:
        from manuscribe.pipeline.tables import _rewrap_bold_cells

        # A <th> matches and advances the cursor (so later cells stay aligned) but is
        # never wrapped — the render's CSS already differentiates headers.
        table = "<table><tr><th>Header</th></tr></table>"
        out = _rewrap_bold_cells(table, "header", [708] * 6)
        assert out == table

    def test_already_bold_cell_not_double_wrapped(self) -> None:
        from manuscribe.pipeline.tables import _rewrap_bold_cells

        table = "<table><tr><td><strong>foo</strong></td></tr></table>"
        out = _rewrap_bold_cells(table, "foo", [708, 708, 708])
        assert out.count("<strong>") == 1

    def test_empty_cell_unchanged_and_keeps_cursor(self) -> None:
        from manuscribe.pipeline.tables import _rewrap_bold_cells

        # An empty cell is skipped without consuming the cursor, so the following bold
        # cell still binds to its glyph run.
        table = "<table><tr><td></td><td>Total</td></tr></table>"
        out = _rewrap_bold_cells(table, "total", [708] * 5)
        assert "<td></td>" in out
        assert "<td><strong>Total</strong></td>" in out

    def test_inline_markup_cell_rewrapped_whole(self) -> None:
        from manuscribe.pipeline.tables import _rewrap_bold_cells

        # A bold cell carrying inline markup is wrapped around its whole inner HTML,
        # preserving the sub-tag (a blind string replace would corrupt it).
        table = "<table><tr><td>R<sub>merge</sub></td></tr></table>"
        out = _rewrap_bold_cells(table, "rmerge", [708] * 6)
        assert "<td><strong>R<sub>merge</sub></strong></td>" in out

    def test_apply_table_bold_no_text_layer_is_noop(self) -> None:
        from unittest.mock import MagicMock

        import pypdfium2 as pdfium
        import pytest

        from manuscribe.pipeline.layers import _DocumentLayers, _PageLayer
        from manuscribe.pipeline.tables import _apply_table_bold

        # An image-only / scanned page: the text layer carries no font-weight metadata
        # (every weight None) → every table is returned unchanged.
        md = "<table><tr><td>Total</td></tr></table>"
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "manuscribe.pipeline.layers._page_layer",
                lambda page: _PageLayer("", [], [], [], "", []),
            )
            layers = _DocumentLayers(MagicMock(spec=pdfium.PdfDocument))
            assert _apply_table_bold(md, layers, 0) == md

    def test_recover_table_cell_bold_31123167_deterministically(self) -> None:
        # End-to-end on the real PDF text layer (no GPU): Table 1's boldfaced section
        # rows come back wrapped in <strong>, a plain data value does not, and the
        # twice-occurring normal-weight "Resolution" stays plain.  Skips if absent.
        import pathlib
        import re as _re

        import pytest

        from manuscribe.pipeline.layers import _DocumentLayers
        from manuscribe.pipeline.tables import _apply_table_bold

        pdf = pathlib.Path(__file__).parent / "fixtures" / "31123167.pdf"
        dump = (
            pathlib.Path(__file__).parent
            / "data"
            / "dumps"
            / "31123167_raw_markdown.md"
        )
        if not pdf.exists() or not dump.exists():
            pytest.skip(f"fixture not found: {pdf} / {dump}")
        pages = [
            p.strip("\n")
            for p in _re.split(
                r"^===== PAGE \d+ =====$", dump.read_text("utf-8"), flags=_re.M
            )[1:]
        ]
        with _DocumentLayers.open(str(pdf)) as layers:
            out = _apply_table_bold(pages[4], layers, 4)
        for row in (
            "Data collection",
            "Refinement",
            "Number of atoms",
            "Ramachandran plot (%)",
        ):
            assert f"<strong>{row}</strong>" in out
        assert "<strong>0.9795" not in out  # a plain data value is not bolded
        assert "<strong>Resolution" not in out  # normal-weight, appears twice


class TestTableReocrApply:
    """``_apply_page_results`` splices a region's batched re-OCR back into the page,
    but only when the crop came back at least as rich — the substitution gate that
    stops a worse (or empty) re-OCR from degrading a table the full-page pass got
    right.  Pure markdown work: a ``_RegionPlan`` is built directly with a blank crop,
    so no PDF or localization is needed."""

    _ORIG = (
        "<table><tr><td>Sample</td><td>Value</td></tr>"
        "<tr><td>A</td><td>1</td></tr></table>"  # 4 non-empty cells
    )
    _RICHER = (
        "<table><tr><th>Sample</th><th>Low</th><th>High</th></tr>"
        "<tr><td>A</td><td>1</td><td>2</td></tr>"
        "<tr><td>B</td><td>3</td><td>4</td></tr></table>"  # 9 cells
    )
    _SPARSER = "<table><tr><td>A</td></tr></table>"  # 1 cell

    def _plan(self, md: str):
        from PIL import Image

        from manuscribe.pipeline.tables.recover import _RegionPlan

        start = md.index(self._ORIG)
        return _RegionPlan(
            start, start + len(self._ORIG), [self._ORIG], Image.new("RGB", (10, 10))
        )

    def test_richer_reocr_replaces_original(self) -> None:
        from manuscribe.pipeline.tables.recover import _apply_page_results

        md = f"Intro paragraph.\n\n{self._ORIG}\n\nTrailing body paragraph."
        out = _apply_page_results(md, [(self._plan(md), self._RICHER)])
        assert "<th>High</th>" in out  # the richer crop's structure replaced the orig
        assert self._ORIG not in out
        assert "Intro paragraph." in out and "Trailing body paragraph." in out

    def test_sparser_reocr_keeps_original(self) -> None:
        from manuscribe.pipeline.tables.recover import _apply_page_results

        md = f"Intro paragraph.\n\n{self._ORIG}\n\nTrailing body paragraph."
        # Fewer non-empty cells than the original -> the full-page table is kept.
        assert _apply_page_results(md, [(self._plan(md), self._SPARSER)]) == md

    def test_reocr_without_a_table_keeps_original(self) -> None:
        from manuscribe.pipeline.tables.recover import _apply_page_results

        md = f"Intro paragraph.\n\n{self._ORIG}\n\nTrailing body paragraph."
        assert _apply_page_results(md, [(self._plan(md), "prose, no table")]) == md

    def test_recovered_legend_folded_as_table_footnote(self) -> None:
        from manuscribe.pipeline.tables.recover import _apply_page_results

        md = f"Intro paragraph.\n\n{self._ORIG}\n\nTrailing body paragraph."
        crop = self._RICHER + "\n\naValues are means of triplicate assays."
        out = _apply_page_results(md, [(self._plan(md), crop)])
        assert '<p class="footnote">' in out
        assert "means of triplicate assays" in out

    def test_legend_already_in_page_not_duplicated(self) -> None:
        from manuscribe.pipeline.tables.recover import _apply_page_results

        legend = "aValues are means of triplicate assays."
        md = f"Intro paragraph.\n\n{self._ORIG}\n\n{legend}"
        crop = self._RICHER + "\n\n" + legend
        out = _apply_page_results(md, [(self._plan(md), crop)])
        # The crop re-OCR caught the legend the page already has -> not folded again.
        assert '<p class="footnote">' not in out

    def test_multiple_regions_spliced_back_to_front(self) -> None:
        from PIL import Image

        from manuscribe.pipeline.tables.recover import _apply_page_results, _RegionPlan

        t2 = "<table><tr><td>X</td><td>Y</td></tr><tr><td>9</td><td>8</td></tr></table>"
        r2 = (
            "<table><tr><th>X</th><th>Y</th><th>Z</th></tr>"
            "<tr><td>9</td><td>8</td><td>7</td></tr></table>"
        )
        md = f"Lead.\n\n{self._ORIG}\n\nMiddle.\n\n{t2}\n\nTail."
        blank = Image.new("RGB", (10, 10))
        p1 = _RegionPlan(
            md.index(self._ORIG),
            md.index(self._ORIG) + len(self._ORIG),
            [self._ORIG],
            blank,
        )
        p2 = _RegionPlan(md.index(t2), md.index(t2) + len(t2), [t2], blank)
        out = _apply_page_results(md, [(p1, self._RICHER), (p2, r2)])
        # Both regions replaced and the interleaved prose intact: the back-to-front
        # splice order kept the earlier offsets valid after the later splice.
        assert "<th>High</th>" in out and "<th>Z</th>" in out
        assert "Lead." in out and "Middle." in out and "Tail." in out


class TestTableReocrEndToEnd:
    """``_recover_dropped_tables`` plans each page's crops from the text layer, OCRs
    them in **one batched call**, and splices the results.  Driven against a real
    fixture's text layer with a fake ``ocr_regions`` so no GPU/server is needed (only
    the injected re-OCR is faked); the localization and crop render are pure CPU."""

    @staticmethod
    def _dump_pages(stem: str):
        import pathlib
        import re as _re

        import pytest

        pdf = pathlib.Path(__file__).parent / "fixtures" / f"{stem}.pdf"
        dump = (
            pathlib.Path(__file__).parent / "data" / "dumps" / f"{stem}_raw_markdown.md"
        )
        if not pdf.exists() or not dump.exists():
            pytest.skip(f"fixture not found: {pdf} / {dump}")
        pages = [
            p.strip("\n")
            for p in _re.split(
                r"^===== PAGE \d+ =====$", dump.read_text("utf-8"), flags=_re.M
            )[1:]
        ]
        return pdf, pages

    def test_crops_are_batched_in_one_call_and_richer_reocr_spliced(self) -> None:
        from manuscribe.pipeline.layers import _DocumentLayers
        from manuscribe.pipeline.tables.recover import _recover_dropped_tables

        pdf, pages = self._dump_pages("30592559")
        # 240 distinct cells out-cells the fixture's biggest planned region (101), so a
        # planned region always clears the cell-count gate and splices.  Rows must be
        # *distinct* — byte-identical rows would be folded by the decode-loop collapse
        # before the gate, masking the richer cell count.
        rich = (
            "<table>"
            + "".join(
                "<tr>"
                + "".join(f"<td>RECOVEREDCELL{r}_{c}</td>" for c in range(8))
                + "</tr>"
                for r in range(30)
            )
            + "</table>"
        )
        calls: list[int] = []

        def fake_ocr_regions(crops):
            calls.append(len(crops))
            return [rich] * len(crops)

        with _DocumentLayers.open(str(pdf)) as layers:
            out = _recover_dropped_tables(layers, pages, fake_ocr_regions)

        assert len(calls) == 1  # every crop OCR'd in a single batched request
        assert calls[0] >= 1  # at least one table region localized + planned
        assert "RECOVEREDCELL0_0" in "\n".join(out)  # the richer re-OCR was spliced in
        assert len(out) == len(pages)

    def test_pages_without_a_table_incur_no_ocr(self) -> None:
        from unittest.mock import MagicMock

        import pypdfium2 as pdfium

        from manuscribe.pipeline.layers import _DocumentLayers
        from manuscribe.pipeline.tables.recover import _recover_dropped_tables

        called: list[int] = []

        def fake_ocr_regions(crops):
            called.append(len(crops))
            return [""] * len(crops)

        # No "<table" on any page -> never touches the (mocked) document or the OCR.
        layers = _DocumentLayers(MagicMock(spec=pdfium.PdfDocument))
        pages = ["# Intro\n\nProse, no tables here.", "More prose, still no table."]
        out = _recover_dropped_tables(layers, pages, fake_ocr_regions)
        assert out == pages
        assert called == []

    def test_sideways_table_localizes_to_a_planned_crop(self) -> None:
        # 32117944 Table 2 is printed rotated 270°: localization records the glyph
        # rotation and the planner rotates the crop upright before re-OCR.  Declining
        # every crop here (empty re-OCR) keeps the originals, but the rotation-aware
        # localize→batch path still runs.
        from manuscribe.pipeline.layers import _DocumentLayers
        from manuscribe.pipeline.tables.recover import _recover_dropped_tables

        pdf, pages = self._dump_pages("32117944")
        seen: list[int] = []

        def fake_ocr_regions(crops):
            seen.append(len(crops))
            return [""] * len(crops)

        with _DocumentLayers.open(str(pdf)) as layers:
            out = _recover_dropped_tables(layers, pages, fake_ocr_regions)

        assert seen and seen[0] >= 1  # the rotated table localized to a planned crop
        assert len(out) == len(pages)

    def test_unlocalizable_table_is_left_untouched(self) -> None:
        from manuscribe.pipeline.layers import _DocumentLayers
        from manuscribe.pipeline.tables.recover import _recover_dropped_tables

        pdf, _ = self._dump_pages("30592559")
        called: list[int] = []

        def fake_ocr_regions(crops):
            called.append(len(crops))
            return ["<table><tr><td>x</td></tr></table>"] * len(crops)

        # Cells that appear nowhere in the page's text layer can't be localized, so
        # the region is skipped — no crop, no OCR, the original table left as-is.
        nonsense = "<table><tr><td>NOMATCHAA</td><td>NOMATCHBB</td></tr></table>"
        with _DocumentLayers.open(str(pdf)) as layers:
            out = _recover_dropped_tables(layers, [nonsense], fake_ocr_regions)

        assert out == [nonsense]  # unchanged
        assert called == []  # localization failed first -> no crop was ever OCR'd
