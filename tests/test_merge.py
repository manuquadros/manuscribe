"""Tests for paragraph/reference stitching and dehyphenation."""

import re

from helpers import (
    _as_blocks,
    _body,
    _run_lighton,
)


class TestCrossPageMerge:
    """A paragraph split across a page break is rejoined."""

    def test_cross_page_paragraph_merge(self) -> None:
        page1 = "# T\n\n## Abstract\n\nA.\n\n## Body\n\nThis suggests that TRI and"
        page2 = "TRII compete for the same substrate tropinone."
        html = _run_lighton([page1, page2])
        assert "This suggests that TRI and TRII compete for the same substrate" in html
        assert "This suggests that TRI and</p>" not in html

    def test_isotope_led_continuation_still_merges(self) -> None:
        # The footnote-marker continuation guard must match a footnote shape, not any
        # leading superscript: an isotope-led continuation ("³⁵S methionine …") is
        # prose and must still rejoin its fragment.
        from pdfparser.pipeline.merge import _merge_split_paragraphs_stable

        out = _merge_split_paragraphs_stable(
            [
                "<p>The radiolabel was incorporated using</p>",
                "<p>³⁵S methionine in all growth media.</p>",
            ]
        )
        assert len(out) == 1
        assert "using ³⁵S methionine in all growth media." in out[0]

    def test_mixed_case_identifier_continuation_after_the(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # the fragment ends in "The"; the continuation opens with a mixed-case
        # identifier ("SpRDH"), which is mid-sentence, not a new-sentence capital —
        # so the merge fires across the intervening table float
        parts = [
            "<p>enter the pentose phosphate pathway for carbon metabolism. The</p>",
            "<table><tr><td>x</td></tr></table>",
            "<p>SpRDH operon of the genome contains a transporter.</p>",
        ]
        merged = _merge_split_paragraphs(parts)
        assert any("metabolism. The SpRDH operon of the genome" in p for p in merged)

    def test_stranded_table_legend_not_spliced_into_cross_table_sentence(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs_stable

        # A markerless table's abbreviation legend ("MW: …, NR: …") the OCR strands
        # between the table and the prose resuming after it must be stepped over, not
        # taken as the sentence continuation: the two sentence halves rejoin
        # contiguously and the legend survives elsewhere in the output.
        parts = [
            "<p>enter the pentose phosphate pathway for carbon metabolism. The</p>",
            "<table><caption>Table 2. Metal ion analysis.</caption>"
            "<tr><td>Cu2+</td></tr></table>",
            "<table><caption>Table 3. Biochemical properties.</caption>"
            "<tr><td>MW</td></tr></table>",
            "<p>MW: molecular weight, NR: Not reported</p>",
            "<p>SpRDH operon of the genome contains a transporter (S4 Fig).</p>",
        ]
        out = _merge_split_paragraphs_stable(parts)
        text = "".join(out)
        assert "metabolism. The SpRDH operon of the genome" in re.sub(
            r"<[^>]+>", "", text
        )
        assert "Not reported SpRDH" not in re.sub(r"<[^>]+>", "", text)
        assert any("MW: molecular weight, NR: Not reported" in p for p in out)

    def test_function_word_guard_still_refuses_new_sentence(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # a plain capitalized word after "The" (no internal capital) signals a
        # dropped continuation — the guard must still refuse the merge
        parts = [
            "<p>they use the Entner-Doudoroff pathway for glucose metabolism. The</p>",
            "<p>Many sugar alcohols enter the pentose phosphate pathway.</p>",
        ]
        assert _merge_split_paragraphs(parts) == parts

    def test_citation_superscript_does_not_hide_sentence_end(self) -> None:
        # A paragraph ending with a citation superscript ("…humans.<sup>15–18</sup>",
        # "…software.³²") is a finished sentence; the trailing citation must not hide
        # the period and let the next paragraph be glued on as a continuation.
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        parts = [
            "<p>this bacterium is harmless to humans.<sup>15–18</sup></p>",
            "<p>In <em>C. glutamicum</em>, L-valine is synthesized by the pathway.</p>",
            "<p>checked using PyMOL software.³²</p>",
            "<p>The metal activity was measured at maximum.</p>",
        ]
        assert _merge_split_paragraphs(parts) == parts

    def test_isotope_superscript_end_still_merges(self) -> None:
        # The citation look-past is digit-anchored, so a trailing charge/isotope
        # superscript ("…the cofactor NADP⁺") is not mistaken for a terminal-period
        # citation: a genuinely unterminated fragment still rejoins its continuation.
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        parts = [
            "<p>the enzyme binds the cofactor NADP⁺</p>",
            "<p>and two magnesium ions in the active site.</p>",
        ]
        out = _merge_split_paragraphs(parts)
        assert len(out) == 1
        assert "binds the cofactor NADP⁺ and two magnesium ions" in out[0]


class TestReferenceListMerge:
    """Inside the references section each entry is its own block: a DOI-terminated
    entry (no terminal punctuation) must not absorb the next entry, but a genuinely
    wrapped entry (lowercase continuation) still rejoins."""

    def test_doi_terminated_entries_stay_separate(self) -> None:
        # Each reference trails off in a DOI with no terminal punctuation; the
        # next entry opens with a capitalised surname, so it stays its own block
        # instead of being glued into one giant paragraph.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\nBody text.\n\n"
            "## References\n\n"
            "Arfman, N. (1997). Properties of a dehydrogenase. "
            "doi: 10.1111/j.1432-1033.1997.00426.x\n\n"
            "Bradford, M. M. (1976). A rapid method for protein. "
            "doi: 10.1016/0003-2697(76)90527-3\n\n"
            "Cahn, J. K. (2016). Mutations in adenine pockets. "
            "doi: 10.1093/protein/gzv057"
        )
        body = _body(_run_lighton([md]))
        for entry in ("Arfman, N.", "Bradford, M. M.", "Cahn, J. K."):
            assert f"<p>{entry}" in body
        assert "00426.x Bradford" not in body
        assert "90527-3 Cahn" not in body

    def test_wrapped_reference_entry_still_rejoins(self) -> None:
        # A single entry split mid-sentence by a column/page break resumes
        # lowercase, so the two halves still merge into one reference.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\nBody text.\n\n"
            "## References\n\n"
            "Marcal, D. (2009). 1,3-Propanediol dehydrogenase: decameric "
            "quaternary structure\n\n"
            "and possible subunit cooperativity. doi: 10.1128/JB.01077-08"
        )
        body = _body(_run_lighton([md]))
        assert "quaternary structure and possible subunit cooperativity" in body

    def test_mixed_case_surname_entry_stays_separate(self) -> None:
        # A new entry whose surname carries an internal capital ("McKenzie") must
        # not be glued onto the prior DOI-terminated entry: capital-led = new entry,
        # with no mid-sentence-acronym exception inside the references section.
        from pdfparser.pipeline.merge import _merge_split_paragraphs_stable

        parts = [
            "<h2>References</h2>",
            "<p>Smith J, Doe A (2015). A title. doi:10.1000/abc.00426.x</p>",
            "<p>McKenzie EF, Jones AB (2019). Another title. doi:10.1000/xyz</p>",
        ]
        out = _merge_split_paragraphs_stable(parts)
        assert len(out) == 3
        assert not any("00426.x McKenzie" in p for p in out)

    def test_body_bracket_one_does_not_trigger_references_guard(self) -> None:
        # The references guard keys on a References *heading*, not a "[1]"-led block:
        # a numbered list item in the body must not switch it on and suppress a
        # legitimate capital-led body-prose merge that follows.
        from pdfparser.pipeline.merge import _merge_split_paragraphs_stable

        parts = [
            "<p>[1] to define the pathway in the organism studied here</p>",
            "<p>We characterized the recombinant enzyme isolate designated</p>",
            "<p>Sphingomonas cells grown on ribitol were the source.</p>",
        ]
        out = _merge_split_paragraphs_stable(parts)
        assert any("designated Sphingomonas cells grown" in p for p in out)


class TestNumberedReferenceConsolidation:
    """Period-less numbered bibliography entries the OCR emits as plain <p> blocks
    (because it dropped the markdown list period: "9 Peck …" not "9. Peck") are
    folded into one <ol> so a reference list split across pages renders uniformly."""

    def test_period_less_entries_extend_preceding_ol(self) -> None:
        from pdfparser.pipeline.assemble import _consolidate_numbered_references

        parts = [
            "<h2>References</h2>",
            "<ol>\n<li>\n<p>Fellman, J.H. (1980) A. doi:10.1/a</p>\n</li>\n</ol>",
            "<p>9 Peck, S.C. (2019) B. doi:10.2/b</p>",
            "<p>10 Xing, M. (2019) C. doi:10.3/c</p>",
        ]
        out = _consolidate_numbered_references(_as_blocks(parts))
        # one list, the loose entries appended as <li> with their leading number
        # (which the <ol> renders itself) dropped
        assert len(out) == 2
        ol = out[1]
        assert ol.count("<li>") == 3
        assert "<p>Peck, S.C. (2019) B." in ol
        assert "<p>9 Peck" not in ol
        assert "<p>10 Xing" not in ol

    def test_free_standing_run_wrapped_with_start(self) -> None:
        from pdfparser.pipeline.assemble import _consolidate_numbered_references

        # No preceding <ol> (the perioded entries were on an earlier page now gone):
        # the run is wrapped in a new <ol start=N> so it renders from its real number.
        parts = [
            "<h2>References</h2>",
            "<p>9 Peck, S.C. (2019) B. doi:10.2/b</p>",
            "<p>10 Xing, M. (2019) C. doi:10.3/c</p>",
        ]
        out = _consolidate_numbered_references(_as_blocks(parts))
        assert len(out) == 2
        assert out[1].startswith('<ol start="9">')
        assert out[1].count("<li>") == 2

    def test_numbered_paragraph_before_references_untouched(self) -> None:
        from pdfparser.pipeline.assemble import _consolidate_numbered_references

        # A numbered <p> in the body (before the References heading) is not a
        # bibliography entry and must be left alone.
        parts = [
            "<p>9 Samples were collected from Site A and analyzed.</p>",
            "<h2>References</h2>",
            "<p>9 Peck, S.C. (2019) B. doi:10.2/b</p>",
        ]
        out = _consolidate_numbered_references(_as_blocks(parts))
        assert out[0] == "<p>9 Samples were collected from Site A and analyzed.</p>"

    def test_stranded_entry_tail_folded_into_last_li(self) -> None:
        from pdfparser.pipeline.assemble import _consolidate_numbered_references

        # A page break dropped an entry's head/number, stranding its tail as a loose
        # lowercase <p> after the <ol>; it is folded into the last <li> so it reads
        # inside the bibliography, not as a stray paragraph.
        parts = [
            "<h2>References</h2>",
            "<ol>\n<li>\n<p>Sukpipat W. xylitol dehydrogenase. J Biosci Bioeng.</p>"
            "\n</li>\n</ol>",
            "<p>dehydrogenase is independent of the oxidative fermentation. "
            "Biosci Biotechnol Biochem. 2001; 65(1):115.</p>",
        ]
        out = _consolidate_numbered_references(_as_blocks(parts))
        assert len(out) == 2  # the stray <p> is gone, merged into the <ol>
        assert "J Biosci Bioeng. dehydrogenase is independent" in out[1]
        assert out[1].count("<li>") == 1

    def test_capital_led_entry_after_ol_not_folded(self) -> None:
        from pdfparser.pipeline.assemble import _consolidate_numbered_references

        # A capital-led <p> after the <ol> is a new entry (author surname), not a
        # continuation — it must not be swallowed into the previous list item.
        parts = [
            "<h2>References</h2>",
            "<ol>\n<li>\n<p>Sukpipat W. Some title. J Biosci Bioeng.</p>\n</li>\n</ol>",
            "<p>Onuț-Brännström I. Sharing of photobionts. Sci Rep. 2018.</p>",
        ]
        out = _consolidate_numbered_references(_as_blocks(parts))
        assert len(out) == 3  # left as its own block
        assert out[2].startswith("<p>Onuț-Brännström")


class TestCaptionMergeBarrier:
    """A figure/table caption is never absorbed as a paragraph continuation,
    even across intervening floats and even when wrapped in <strong>."""

    def test_table_caption_after_floats_not_glued_to_fragment(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        parts = [
            "<p>PtTRII catalyzed the reduction of tropinone to form</p>",
            "<figure><img src='a' alt=''></figure>",
            "<figure><img src='b' alt=''></figure>",
            "<p><strong>TABLE 1</strong> Enzyme kinetics of PtTRI and PtTRII</p>",
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
        ]
        out = _merge_split_paragraphs(parts)
        # Caption stays its own block, immediately before its table; the
        # fragment and the floats keep their order.
        assert out == parts
        assert "to form <strong>TABLE 1</strong>" not in "".join(out)

    def test_real_continuation_still_merges(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        parts = [
            "<p>This suggests that TRI and</p>",
            "<p>TRII compete for the substrate.</p>",
        ]
        out = _merge_split_paragraphs(parts)
        assert out == [
            "<p>This suggests that TRI and TRII compete for the substrate.</p>"
        ]

    def test_function_word_with_trailing_comma_blocks_capital_continuation(
        self,
    ) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # "revealed that," is grammatically incomplete; a capitalised new
        # sentence is not its continuation (here an OCR-misplaced figure caption),
        # so the trailing comma must not disarm the capital-letter guard.
        parts = [
            "<p>analyses of 2-butanol production revealed that,</p>",
            "<p>Molecule structures are shown in Fig. 3.</p>",
        ]
        assert _merge_split_paragraphs(parts) == parts

    def test_function_word_with_trailing_comma_still_merges_lowercase(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # The genuine lowercase continuation of the same clause still joins.
        parts = [
            "<p>analyses of 2-butanol production revealed that,</p>",
            "<p>with no additives present, all forms preferred re-face addition.</p>",
        ]
        out = _merge_split_paragraphs(parts)
        assert out == [
            "<p>analyses of 2-butanol production revealed that, with no additives "
            "present, all forms preferred re-face addition.</p>"
        ]

    def test_preposition_comma_does_not_block_proper_noun_continuation(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # The comma allowance is only for clause-introducers; after a preposition
        # a trailing comma before a capitalised proper noun is a genuine
        # continuation, so the merge must still join across the break.
        parts = [
            "<p>the epoxide-metabolising strains studied here consist of,</p>",
            "<p>Xanthobacter autotrophicus and related species.</p>",
        ]
        out = _merge_split_paragraphs(parts)
        assert out == [
            "<p>the epoxide-metabolising strains studied here consist of, "
            "Xanthobacter autotrophicus and related species.</p>"
        ]

    def test_capital_continuation_behind_leading_tag_still_blocked(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # A new sentence opening with an italicised token (a math variable or a
        # species name) leads with "<em>", not its capital.  The function-word
        # guard must judge the continuation by its *visible* head, not the raw
        # "<", or the leading tag hides the capital and the sentences are wrongly
        # glued into one paragraph.
        parts = [
            "<p>the kinetics were analysed assuming that</p>",
            "<p><em>P</em> denotes the partial pressure.</p>",
        ]
        assert _merge_split_paragraphs(parts) == parts

    def test_metadata_line_not_merged_into_following_prose(self) -> None:
        # A self-contained footer-metadata line that ends without terminal
        # punctuation (here a ")") must not be treated as an incomplete paragraph
        # and glued to the body prose the OCR placed after it.
        from pdfparser.pipeline.merge import _merge_split_paragraphs_stable

        parts = [
            "<p>Published online 28 December 2018 in Wiley Online Library "
            "(wileyonlinelibrary.com)</p>",
            "<p>Herein, I propose that the method generalizes to other enzymes.</p>",
        ]
        assert _merge_split_paragraphs_stable(parts) == parts

    def test_continuation_after_two_figures_and_a_table_merges(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # A column break stranded the continuation behind a figure+figure+table
        # cluster (the table's caption already folded in by colocation).
        parts = [
            "<p>PtTRII catalyzed the reduction of tropinone to form</p>",
            "<figure><img src='a' alt=''></figure>",
            "<figure><img src='b' alt=''></figure>",
            "<table><caption>TABLE 1</caption><tbody><tr><td>1</td></tr></tbody>"
            "</table>",
            "<p>pseudotropine with higher affinity to tropinone.</p>",
        ]
        out = _merge_split_paragraphs(parts)
        assert out[0] == (
            "<p>PtTRII catalyzed the reduction of tropinone to form "
            "pseudotropine with higher affinity to tropinone.</p>"
        )
        # The floats are relocated after the joined paragraph, order preserved.
        assert out[1:] == parts[1:4]


class TestInlineTableTitleHoist:
    """A table title the model bakes into the first row as a lone spanning
    ``<th colspan=N>Table N …</th>`` cell is hoisted into a ``<caption>``."""

    def test_thead_title_row_hoisted_to_caption(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        parts = [
            "<table>\n  <thead>\n    <tr>\n"
            '      <th colspan="5">Table 1. Kinetic Analysis of CgKARI</th>\n'
            "    </tr>\n    <tr>\n      <th></th>\n      <th>Vmax</th>\n"
            "      <th>Km</th>\n      <th>kcat</th>\n      <th>kcat/Km</th>\n"
            "    </tr>\n  </thead>\n  <tbody>\n"
            "    <tr><td>NADPH</td><td>0.4</td><td>0.02</td>"
            "<td>13</td><td>578</td></tr>\n"
            "  </tbody>\n</table>"
        ]
        (out,) = _colocate_table_captions(_as_blocks(parts))
        assert "<caption>Table 1. Kinetic Analysis of CgKARI</caption>" in out
        # The title row is gone, the real header survives, no empty <thead> left.
        assert 'colspan="5">Table 1' not in out
        assert "<th>Vmax</th>" in out
        assert "<thead></thead>" not in out

    def test_caption_first_thead_title_row_hoisted_without_thead(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        # The text-layer-rebuilt shape: no <thead>, the title is the first <tr>.
        parts = [
            "<table>\n<tr>\n"
            '      <th colspan="2">Table 2. Data Collection Statistics</th>\n'
            "    </tr>\n"
            "<tr><td></td><td>CgKARI_NADP⁺</td></tr>\n"
            "<tr><td>PDB code</td><td>6JX2</td></tr>\n</table>"
        ]
        (out,) = _colocate_table_captions(_as_blocks(parts))
        assert "<caption>Table 2. Data Collection Statistics</caption>" in out
        assert 'colspan="2">Table 2' not in out
        assert "<td>CgKARI_NADP⁺</td>" in out

    def test_spanning_section_header_not_hoisted(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        # A genuine full-width *section* header ("A. Effect of EDTA …") is not a
        # "Table N" caption, so it stays a row and no <caption> is fabricated.
        parts = [
            "<table>\n  <thead>\n"
            '    <tr><th colspan="3">A. Effect of EDTA on SpRDH activity</th></tr>\n'
            "    <tr><th></th><th>1 mM</th><th>10 mM</th></tr>\n"
            "  </thead>\n  <tbody>\n"
            "    <tr><td>None</td><td>100</td><td>100</td></tr>\n  </tbody>\n</table>"
        ]
        (out,) = _colocate_table_captions(_as_blocks(parts))
        assert "<caption>" not in out
        assert 'colspan="3">A. Effect of EDTA' in out

    def test_ordinary_multicell_header_row_untouched(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        # A normal multi-cell first row (no lone spanning title cell) is left alone.
        parts = [
            "<table><thead><tr><th>Gene</th><th>Length</th></tr></thead>"
            "<tbody><tr><td>xecD1</td><td>800</td></tr></tbody></table>"
        ]
        assert _colocate_table_captions(_as_blocks(parts)) == parts


class TestMergeSplitPanelTables:
    """A composite multi-panel table the model split into per-panel <table>s (on the
    blank line between panels) is re-fused when the panels carry sequential A./B./…
    header labels and the later panels are captionless."""

    _A = (
        "<table>\n  <thead>\n"
        "    <tr><th>A. Effect of EDTA on SpRDH activity</th><th></th></tr>\n"
        "  </thead>\n  <tbody>\n    <tr><td>None</td><td>100</td></tr>\n"
        "  </tbody>\n</table>"
    )
    _B = (
        "<table>\n  <thead>\n"
        "    <tr><th>B. ICP-MS analysis</th><th></th></tr>\n"
        "  </thead>\n  <tbody>\n    <tr><td>Zn²⁺</td><td>&lt; 1.0</td></tr>\n"
        "  </tbody>\n</table>"
    )

    def test_sequential_panel_tables_merged(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_panel_tables

        out = _merge_split_panel_tables(_as_blocks([self._A, self._B]))
        assert len(out) == 1
        # one <table> wrapper, both panels' content inside it
        assert out[0].count("<table") == 1
        assert out[0].count("</table>") == 1
        assert "A. Effect of EDTA" in out[0]
        assert "B. ICP-MS analysis" in out[0]
        assert "Zn²⁺" in out[0]  # panel B's data survives

    def test_caption_then_merged_table_keeps_one_caption(self) -> None:
        from pdfparser.pipeline.merge import (
            _colocate_table_captions,
            _merge_split_panel_tables,
        )

        # the real pipeline order: a free-standing caption, then the two panels
        parts = ["<p>Table 2. Metal ion analysis.</p>", self._A, self._B]
        merged = _merge_split_panel_tables(_as_blocks(parts))
        out = _colocate_table_captions(_as_blocks(merged))
        joined = "".join(out)
        assert joined.count("<caption>") == 1
        assert "<caption>Table 2. Metal ion analysis.</caption>" in joined
        assert joined.count("<table") == 1

    def test_distinct_adjacent_tables_not_fused(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_panel_tables

        # ordinary column headers (no A./B. panel labels) → left as two tables
        t1 = "<table><thead><tr><th>Gene</th><th>Length</th></tr></thead></table>"
        t2 = "<table><thead><tr><th>Metal</th><th>Conc.</th></tr></thead></table>"
        assert _merge_split_panel_tables(_as_blocks([t1, t2])) == [t1, t2]

    def test_non_sequential_panel_letters_not_fused(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_panel_tables

        # A then C (a gap) is not a contiguous panel run → not merged
        c = self._B.replace("B. ICP-MS analysis", "C. Something else")
        assert _merge_split_panel_tables(_as_blocks([self._A, c])) == [self._A, c]

    def test_captioned_second_table_not_absorbed(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_panel_tables

        # if the second panel-letter table already carries its own caption it is a
        # distinct table, not a panel to fuse
        b = self._B.replace("<table>", "<table><caption>Table 3. Other</caption>")
        assert _merge_split_panel_tables(_as_blocks([self._A, b])) == [self._A, b]


class TestReflowWrappedParagraph:
    """When LightOnOCR preserves a column's visual line wrapping, a paragraph
    arrives as soft-wrapped lines inside one <p>: words break across the wrap
    with a soft hyphen, and a dropped paragraph break lands on a line boundary.
    """

    def test_soft_hyphen_rejoined_without_space(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        # 31051047: "Unfortu-\nnately" must read "Unfortunately", not "Unfortu- nately".
        (block,) = _md_to_html_blocks(
            "interesting plant species for studying TA biosynthesis. Unfortu-\n"
            "nately, biosynthesis and regulation of TAs are unknown at the\n"
            "molecular, biochemical, and biotechnological level."
        )
        assert "Unfortunately, biosynthesis" in block
        assert "Unfortu" not in block.replace("Unfortunately", "")

    def test_soft_hyphen_rejoined_inside_emphasis(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        # 31051047: "*At-\nropa belladonna*" must render "<em>Atropa belladonna</em>".
        (block,) = _md_to_html_blocks(
            "species, such as *Hyoscyamus niger* [2], *Datura species* [3], *At-\n"
            "ropa belladonna* [4], *P. tangutica* [5], and so on, the highest."
        )
        assert "<em>Atropa belladonna</em>" in block

    def test_dropped_paragraph_break_recovered(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        # 31051047: a sentence ending at a line boundary where the next line's
        # first word would have fit marks a paragraph the model failed to break.
        # Verbatim wrapped lines from the fixture, so the widest line (the
        # self-calibrating fill width) matches the real column.
        blocks = _md_to_html_blocks(
            "gives the highest yields of TAs [1]. It is not only a valuable plant\n"
            "source for commercially producing hyoscyamine but also an\n"
            "interesting plant species for studying TA biosynthesis. Unfortu-\n"
            "nately, biosynthesis and regulation of TAs are unknown at the\n"
            "level. Therefore, it is necessary to develop novel methods to\n"
            "increase the yield of TA using metabolic engineering [6].\n"
            "Although the precise biosynthetic pathway of TAs is still\n"
            "unclear, several enzymes and their corresponding genes have"
        )
        assert len(blocks) == 2
        assert blocks[0].endswith("metabolic engineering [6].</p>")
        assert "Unfortunately, biosynthesis" in blocks[0]
        assert blocks[1].startswith("<p>Although the precise")

    def test_mid_paragraph_sentence_end_not_split(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        # A sentence ending at the widest line is a wrap, not a paragraph break:
        # the next word could not have fit, so the lines stay one paragraph.
        (block,) = _md_to_html_blocks(
            "the reduction of the 3-carbonyl group of tropinone yields one.\n"
            "Tropinone reductase reduces the ketone to the tropine alcohol."
        )
        assert block.count("<p>") == 1
        assert "yields one. Tropinone reductase" in block

    def test_hard_break_block_left_untouched(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        # Explicit <br> hard breaks (affiliation lists) are not soft wrap: reflow
        # must not join or split across them.
        (block,) = _md_to_html_blocks(
            "First affiliation line.  \nSecond affiliation line."
        )
        assert "<br" in block
        assert block.count("<p>") == 1


class TestDehyphenateJoin:
    """A line/block break drops a soft hyphen only for a syllabic word split;
    a genuine compound, range, or acronym hyphen is kept.  Shared by the
    markdown reflow and the block-merge stitcher."""

    def test_syllabic_split_merging_to_real_word_is_joined(self) -> None:
        from pdfparser.pipeline.dehyphenate import _dehyphenate_join

        # Merged form is a dictionary word -> drop the hyphen.
        assert _dehyphenate_join("biosynthesis. Unfortu-", "nately, regulation") == (
            "biosynthesis. Unfortunately, regulation"
        )
        assert _dehyphenate_join("co-", "operate fully") == "cooperate fully"
        # "fore" is a word, but "therefore" is too, so the merged form wins.
        assert _dehyphenate_join("there-", "fore we") == "therefore we"

    def test_lowercase_nonword_split_is_joined(self) -> None:
        from pdfparser.pipeline.dehyphenate import _dehyphenate_join

        # Neither the merge nor the halves are words; a lowercase-to-lowercase
        # boundary is a syllabic split (a genus name here), so drop the hyphen.
        assert _dehyphenate_join("such as *At-", "ropa belladonna*") == (
            "such as *Atropa belladonna*"
        )

    def test_compound_hyphen_is_kept(self) -> None:
        from pdfparser.pipeline.dehyphenate import _dehyphenate_join

        # Both halves are words but the merge is not -> a real compound.
        assert _dehyphenate_join("a well-", "known result") == "a well-known result"
        assert _dehyphenate_join("high-", "density lipoprotein") == (
            "high-density lipoprotein"
        )
        # An acronym half ("TA") is a known word, so the compound hyphen survives.
        assert _dehyphenate_join("TA-", "producing plants") == "TA-producing plants"
        # "self-"/"cross-" are compound-formers, not solid prefixes -> keep.
        assert _dehyphenate_join("self-", "assembly of") == "self-assembly of"
        assert _dehyphenate_join("cross-", "section view") == "cross-section view"

    def test_solid_prefix_fuses_even_when_solid_form_absent(self) -> None:
        from pdfparser.pipeline.dehyphenate import _dehyphenate_join

        # Productive prefixes attach without a hyphen; their solid form is absent
        # from a general dictionary, so the prefix list (not the dict) must fuse
        # them — the exact scientific vocabulary this pipeline processes.
        assert _dehyphenate_join("over-", "expression of") == "overexpression of"
        assert _dehyphenate_join("co-", "expression levels") == "coexpression levels"
        assert _dehyphenate_join("up-", "regulation of") == "upregulation of"
        assert _dehyphenate_join("pseudo-", "tropine reductase") == (
            "pseudotropine reductase"
        )

    def test_solid_prefix_keeps_hyphen_before_capital_or_number(self) -> None:
        from pdfparser.pipeline.dehyphenate import _dehyphenate_join

        # A prefix before a capital or digit is a real hyphenated coinage.
        assert _dehyphenate_join("anti-", "CRISPR system") == "anti-CRISPR system"
        assert _dehyphenate_join("pre-", "2020 data") == "pre-2020 data"

    def test_non_alphabetic_boundary_keeps_hyphen(self) -> None:
        from pdfparser.pipeline.dehyphenate import _dehyphenate_join

        # A numeric range is not a word split; the hyphen must survive.
        assert _dehyphenate_join("pages 2-", "3 here") == "pages 2-3 here"

    def test_hyphen_across_inline_tag_keeps_compound(self) -> None:
        from pdfparser.pipeline.dehyphenate import _dehyphenate_join

        # The continuation opens with a tag, so the right-hand word is opaque;
        # default to keeping the hyphen ("multi-faceted") rather than fusing.
        assert _dehyphenate_join("the multi-", "<em>faceted</em> view") == (
            "the multi-<em>faceted</em> view"
        )

    def test_break_without_hyphen_joins_with_space(self) -> None:
        from pdfparser.pipeline.dehyphenate import _dehyphenate_join

        assert _dehyphenate_join("normal word", "continues here") == (
            "normal word continues here"
        )


class TestMergePredicates:
    """The named guards extracted out of ``_merge_split_paragraphs`` keep their
    individual contracts (the full-function behaviour is covered above; these pin
    each predicate so a future edit to one can't silently invert it)."""

    def test_open_fragment_true_for_unterminated_prose(self) -> None:
        from pdfparser.pipeline.merge import _is_open_fragment

        inner = "This suggests that TRI and"
        assert _is_open_fragment(f"<p>{inner}</p>", inner, inner, inner) is True

    def test_open_fragment_false_for_terminated_and_for_caption_label(self) -> None:
        from pdfparser.pipeline.merge import _is_open_fragment

        done = "A complete sentence."
        assert _is_open_fragment(f"<p>{done}</p>", done, done, done) is False
        # a caption label is self-contained even without terminal punctuation
        cap = "Figure 1"
        assert _is_open_fragment(f"<p>{cap}</p>", cap, cap, cap) is False

    def test_valid_continuation_true_for_lowercase_prose(self) -> None:
        from pdfparser.pipeline.merge import _is_valid_continuation

        cont = "TRII compete for the same substrate."
        result = _is_valid_continuation(cont, cont, "suggests that TRI and", False)
        assert result is True

    def test_valid_continuation_rejects_capital_after_function_word(self) -> None:
        from pdfparser.pipeline.merge import _is_valid_continuation

        # a function-word tail + an ordinary (non-identifier) capital head is a
        # dropped continuation, not the real one — refuse the merge
        cont = "However the assay failed."
        assert _is_valid_continuation(cont, cont, "we measured the", False) is False

    def test_valid_continuation_in_refs_rejects_capital_head(self) -> None:
        from pdfparser.pipeline.merge import _is_valid_continuation

        cont = "Peck, S.C., et al. 2016."
        # outside refs a capital head with no function-word tail merges…
        assert _is_valid_continuation(cont, cont, "trailing doi 10.1", False) is True
        # …inside refs the same head is the next bibliography entry, left alone
        assert _is_valid_continuation(cont, cont, "trailing doi 10.1", True) is False

    def test_skip_floats_collects_floats_and_legend_to_continuation(self) -> None:
        from pdfparser.pipeline.merge import _skip_floats

        parts = [
            "<p>fragment</p>",
            "<table><tr><td>x</td></tr></table>",
            "<figure></figure>",
            "<p>MW: molecular weight, NR: Not reported</p>",  # legend, not a float
            "<p>continuation</p>",
        ]
        j, floats = _skip_floats(parts, 1)
        assert j == 4
        assert floats == parts[1:4]

    def test_skip_floats_stops_at_float_budget(self) -> None:
        from pdfparser.pipeline.merge import _MAX_FLOATS_TO_SKIP, _skip_floats

        parts = ["<figure></figure>"] * (_MAX_FLOATS_TO_SKIP + 2)
        j, floats = _skip_floats(parts, 0)
        # only the budgeted floats are stepped over; the next is left as a barrier
        assert j == _MAX_FLOATS_TO_SKIP
        assert len(floats) == _MAX_FLOATS_TO_SKIP
