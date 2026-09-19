"""Tests for PDF-text-layer reconciliation (truncated-tail recovery)."""


class TestTextLayerReconciliation:
    """A short tail the OCR truncated is recovered from the PDF text layer; every
    other shape — a reordered block, furniture, a weak bound, a broken glyph, a
    missing layer — is declined, so the pass never splices wrong text."""

    def test_truncated_tail_recovered_faithfully(self) -> None:
        from manuscribe.pipeline.reconcile import _reconcile_page

        page = "The absorbance was monitored with a Tecan reader at 15 s intervals"
        layer = (
            "The absorbance was monitored with a Tecan reader at 15 s intervals "
            "for 2–3 min at RT.\n© 2019 The Authors"
        )
        out = _reconcile_page(page, layer, set())
        # the raw layer slice is spliced, so the en-dash survives (not a hyphen)
        assert "for 2–3 min at RT." in out
        assert "©" not in out  # the footer past the recovered clause is not pulled

    def test_large_gap_declined(self) -> None:
        # A gap past the tail-length ceiling is a block the OCR reordered, not a
        # tail — splicing it would duplicate correctly-transcribed prose.
        from manuscribe.pipeline.reconcile import _reconcile_page

        page = "This paragraph ends on some distinctive trailing words here"
        layer = (
            "This paragraph ends on some distinctive trailing words here "
            "followed by a long continuation that clearly runs past the sixty "
            "character ceiling a real tail would respect"
        )
        assert _reconcile_page(page, layer, set()) == page

    def test_recurring_footer_declined(self) -> None:
        from manuscribe.pipeline.reconcile import _reconcile_page, _recurring_furniture

        footer = "Journal of Important Studies www.example.org 8 Volume 12 Article"
        furniture = _recurring_furniture([footer, footer, footer])
        page = "The discussion paragraph ends just before the running footer line"
        layer = (
            "The discussion paragraph ends just before the running footer line "
            + footer
        )
        assert _reconcile_page(page, layer, furniture) == page

    def test_weak_heading_bound_does_not_truncate(self) -> None:
        # A one-word next block ("FUNDING") whose token also occurs inside the span
        # must not bound the gap early and splice a partial ("…and") tail.
        from manuscribe.pipeline.reconcile import _reconcile_page

        page = "The author roles were as follows for supervision, project\n\nFUNDING"
        layer = (
            "The author roles were as follows for supervision, project "
            "administration, and funding acquisition. FUNDING We acknowledge the "
            "support from the national grant program."
        )
        # ambiguous weak bound -> page-end gap over the ceiling -> declined
        assert _reconcile_page(page, layer, set()) == page

    def test_broken_layer_glyph_declined(self) -> None:
        from manuscribe.pipeline.reconcile import _reconcile_page

        page = "This sentence has a distinctive and recognizable ending"
        layer = (
            "This sentence has a distinctive and recognizable ending "
            "with modifi￾cation noted."
        )
        assert _reconcile_page(page, layer, set()) == page

    def test_missing_layer_is_noop(self) -> None:
        from manuscribe.pipeline.reconcile import _reconcile_page

        page = "Body text that the scanned PDF has no text layer for"
        assert _reconcile_page(page, "", set()) == page

    def test_tail_absent_from_layer_is_noop(self) -> None:
        # OCR paraphrased the tail, so its anchor is not in the layer -> no anchor,
        # no recovery, rather than a guess.
        from manuscribe.pipeline.reconcile import _reconcile_page

        page = "Completely paraphrased wording not matching the layer verbatim"
        layer = "The original layer states something entirely unrelated to it."
        assert _reconcile_page(page, layer, set()) == page

    def test_recovered_text_is_markdown_escaped(self) -> None:
        # Raw layer text enters the pre-markdown stream, so markdown-active chars
        # (_ * [ ]) must be entity-escaped or markdown-it re-reads them as markup.
        from manuscribe.pipeline.reconcile import _reconcile_page

        page = "This block has a clearly distinctive trailing phrase here"
        layer = (
            "This block has a clearly distinctive trailing phrase here "
            "with K_m and V_max [3] noted."
        )
        out = _reconcile_page(page, layer, set())
        assert "&#95;" in out and "_max" not in out  # underscores neutralized
        assert "&#91;3&#93;" in out  # [3] would otherwise be link syntax

    def test_repeated_anchor_declined(self) -> None:
        # The tail token-run occurs twice on the page, so which occurrence is this
        # block is ambiguous — decline rather than splice the text after the wrong
        # one (rfind would have taken the later occurrence).
        from manuscribe.pipeline.reconcile import _reconcile_page

        page = "the reaction was initiated by the addition of"
        layer = (
            "the reaction was initiated by the addition of 0.5 ug enzyme A. "
            "Separately the reaction was initiated by the addition of 2 ug enzyme B."
        )
        assert _reconcile_page(page, layer, set()) == page

    def test_table_continuation_fragment_not_recovered(self) -> None:
        # A <table> split by an internal blank line yields a continuation fragment
        # lacking '<table'; it is still table content and must not receive a tail.
        from manuscribe.pipeline.reconcile import _reconcile_page

        page = (
            "<table><tr><td>x</td></tr>\n\n"
            "<tr><td>alpha beta gamma delta epsilon zeta eta</td></tr></table>"
        )
        layer = "alpha beta gamma delta epsilon zeta eta then a spurious tail here."
        assert _reconcile_page(page, layer, set()) == page

    def test_figure_label_gap_declined(self) -> None:
        # The recovered run is a figure caption, not a prose tail -> declined.
        from manuscribe.pipeline.reconcile import _reconcile_page

        page = "The samples were prepared according to standard protocol"
        layer = (
            "The samples were prepared according to standard protocol "
            "Figure 3. Effect of pH on activity over time."
        )
        assert _reconcile_page(page, layer, set()) == page

    def test_caption_block_not_extended(self) -> None:
        # A figure-caption block is not a recovery target (appending would corrupt
        # the label the figure pass matches on).
        from manuscribe.pipeline.reconcile import _reconcile_page

        page = "Figure 2. Overall structure of the enzyme complex shown"
        layer = "Figure 2. Overall structure of the enzyme complex shown in ribbons."
        assert _reconcile_page(page, layer, set()) == page
