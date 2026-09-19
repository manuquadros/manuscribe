"""Tests for figure geometry, crop recovery, and <figure> assembly."""

import base64
import io
import logging
import re
from pathlib import Path

import numpy as np
import pytest
from helpers import (
    _body,
    _fake_image,
    _figure_sizes,
    _run_lighton,
)
from PIL import Image


class TestSplitFigureCaption:
    """A bare ``FIG. N`` label and its descriptive sentence the model emitted as
    two blocks are rejoined into one figcaption; stray panel labels are dropped."""

    def test_bare_label_rejoins_following_caption_block(self) -> None:
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\n"
            "![image](i.png)100,100,900,600\n\n"
            "FIG. 2\n\n"
            "Protein alignments of TRI and TRII. (A) panel one. (B) panel two."
        )
        html = _run_lighton([md], image=img)
        assert (
            "<figcaption>FIG. 2 Protein alignments of TRI and TRII."
            " (A) panel one. (B) panel two.</figcaption>" in html
        )
        # The descriptive caption is owned by the figure, not stranded as a body
        # paragraph (it appears only inside the figcaption, never in a <p>).
        assert "<p>Protein alignments of TRI and TRII" not in html

    def test_following_heading_echoed_onto_caption_tail_stripped(self) -> None:
        # The model sometimes echoes the next section heading onto the tail of a
        # figure's last panel description (no '#', no terminal punctuation), then emits
        # the real heading too.  The echo must not be baked into the figcaption — the
        # heading text belongs only to the <h2> that follows.
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\nIntro prose.\n\n"
            "![image](i.png)100,100,900,600\n\n"
            "Figure 4. Crystal structures of *BkTauF*\n\n"
            "(A) Quaternary structure of *BkTauF*.\n\n"
            "(B) Subunit structure shown in ribbon. helices in orange, blue, and "
            "green, respectively. Crystal structure of BkTauF\n\n"
            "## Crystal structures of *BkTauF*\n\n"
            "Crystal structures were solved at high resolution."
        )
        html = _run_lighton([md], image=img)
        figcap = html[html.find("<figcaption>") : html.find("</figcaption>") + 13]
        # the echo is gone from the caption (it ends on the closed legend sentence)…
        assert "respectively." in figcap
        assert "Crystal structure of BkTauF" not in figcap
        # …while the real heading still renders as its own <h2>
        assert "<h2>Crystal structures of <em>BkTauF</em></h2>" in html

    def test_panel_labels_between_split_boxes_dropped(self) -> None:
        # The motivating Fig 2 case: the model split the figure into two panel
        # boxes and emitted the bare "A"/"B" panel labels as their own blocks; the
        # stray "B" must not survive to glue onto the caption ("B Protein …").
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\n"
            "A\n\n"
            "![image](a.png)100,100,900,500\n\n"
            "B\n\n"
            "![image](b.png)100,500,900,600\n\n"
            "FIG. 2\n\n"
            "Protein alignments of TRI and TRII."
        )
        body = _body(_run_lighton([md], image=img))
        assert "<p>A</p>" not in body
        assert "<p>B</p>" not in body
        assert "B Protein alignments" not in body

    def test_bolditalic_caption_nests_balanced(self) -> None:
        # The model bolds the whole caption and italicises a species name at its end
        # ("**Figure 2. … of *BkTauF***").  The inline renderer must close the tags in
        # order (<strong>…<em>BkTauF</em></strong>), not the mis-ordered
        # "<em>BkTauF</strong></em>" the old hand-rolled regex produced.
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\n"
            "![image](i.png)100,100,900,600\n\n"
            "**Figure 2. SDS/PAGE analyses of *BkTauF***"
        )
        html = _run_lighton([md], image=img)
        assert (
            "<figcaption><strong>Figure 2. SDS/PAGE analyses of "
            "<em>BkTauF</em></strong></figcaption>" in html
        )
        # no mis-ordered / stray-asterisk artefacts
        assert "</strong></em>" not in html
        assert "BkTauF***" not in html

    def test_full_label_caption_does_not_swallow_next_block(self) -> None:
        # A caption already complete in one block must not pull the next paragraph.
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\n"
            "![image](i.png)100,100,900,600\n\n"
            "FIG. 3 Phylogenetic tree analysis.\n\n"
            "This is the next body paragraph, not part of the caption."
        )
        html = _run_lighton([md], image=img)
        assert "<figcaption>FIG. 3 Phylogenetic tree analysis.</figcaption>" in html
        assert "next body paragraph, not part of the caption" in _body(html)

    def test_split_panel_descriptions_folded_into_caption(self) -> None:
        # The model splits a full caption header and its "(A) … (B) … (C) …" panel
        # descriptions into separate paragraphs; the panel block belongs to the
        # figcaption, not the body.
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\n"
            "![image](i.png)100,100,900,600\n\n"
            "**Figure 1. Gene clusters and metabolic pathways**\n\n"
            "(A) Gene clusters containing IsfD. (B) Pathways relying on the"
            " isozymes. (C) The dissimilation pathway.\n\n"
            "In this pathway, taurine is imported by a transporter."
        )
        html = _run_lighton([md], image=img)
        cap = re.search(r"<figcaption>(.*?)</figcaption>", html, re.DOTALL).group(1)
        assert "(A) Gene clusters containing IsfD." in cap
        assert "(C) The dissimilation pathway." in cap
        assert "<p>(A) Gene clusters containing IsfD" not in _body(html)
        # the genuine body sentence after the panels stays in the body
        assert "taurine is imported by a transporter" in _body(html)

    def test_lowercase_roman_enumeration_not_folded(self) -> None:
        # A body paragraph after a caption that opens with a lowercase roman
        # enumeration "(i) …" is not a panel block (capital-only) and stays in body.
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\n"
            "![image](i.png)100,100,900,600\n\n"
            "FIG. 4 Reaction scheme.\n\n"
            "(i) first the substrate binds, then (ii) the product is released."
        )
        body = _body(_run_lighton([md], image=img))
        assert "(i) first the substrate binds" in body

    def test_headerless_figure_does_not_absorb_panel_enumeration(self) -> None:
        # A figure with no caption header must not absorb a following "(A) …" block
        # (a body enumeration, or the next figure's caption): the fold only runs
        # when a header was actually found.
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\n"
            "![image](i.png)100,100,900,600\n\n"
            "(A) We first cloned the gene; (B) then expressed it."
        )
        html = _run_lighton([md], image=img)
        assert "<figcaption>" not in html
        assert "(A) We first cloned the gene" in _body(html)

    def test_panel_block_merged_with_heading_not_folded_whole(self) -> None:
        # When the OCR merges a "(A) …" panel line with a following heading into one
        # block (no blank separator), folding it whole would swallow the section, so
        # the block is left in the body instead.
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\n"
            "![image](i.png)100,100,900,600\n\n"
            "FIG. 5 Overview.\n\n"
            "(A) Panel description line.\n## Results\n\nKey findings below."
        )
        html = _run_lighton([md], image=img)
        assert "<figcaption>FIG. 5 Overview.</figcaption>" in html
        figcap = html[html.find("<figcaption>") : html.find("</figcaption>")]
        assert "Results" not in figcap


class TestSplitTableFragmentClassification:
    """``_split_md_blocks`` breaks a ``<table>`` on an internal blank line into a tail
    fragment opening ``<tr>``/``</table>`` with no ``<table``; the caption/table
    predicates must recognise it as a table, not fold it into a figcaption."""

    def test_caption_continuation_rejects_split_table_fragment(self) -> None:
        from manuscribe.pipeline.assemble import _is_caption_continuation

        assert not _is_caption_continuation("<tr><td>1</td></tr>\n</table>")
        assert not _is_caption_continuation("</table>")
        assert not _is_caption_continuation("<tbody>\n<tr><td>x</td></tr>")
        # genuine continuation prose is still a continuation
        assert _is_caption_continuation("Continued legend describing panel B.")

    def test_is_table_md_recognises_split_table_fragment(self) -> None:
        from manuscribe.pipeline.assemble import _is_table_md, _MdBlock

        assert _is_table_md(_MdBlock("<table>\n<tr><td>a</td></tr>"))
        assert _is_table_md(_MdBlock("<tr><td>b</td></tr>\n</table>"))
        assert not _is_table_md(_MdBlock("Just a paragraph."))
        assert not _is_table_md(None)


class TestFigureLabelPredicates:
    def test_bare_figure_label(self) -> None:
        from manuscribe.pipeline.figures import _is_bare_figure_label

        assert _is_bare_figure_label("FIG. 2")
        assert _is_bare_figure_label("Figure 3.")
        assert _is_bare_figure_label("**Fig 4**")
        assert not _is_bare_figure_label("FIG. 2 Protein alignments of TRI and TRII.")
        assert not _is_bare_figure_label("Figures are shown below.")

    def test_panel_label(self) -> None:
        from manuscribe.pipeline.figures import _is_panel_label

        assert _is_panel_label("A")
        assert _is_panel_label("(B)")
        assert _is_panel_label("C.")
        assert not _is_panel_label("AB")
        assert not _is_panel_label("A nice sentence.")


class TestRecoverDroppedFigures:
    """Pure helpers for recovering a figure LightOnOCR drops whole from a page.

    The model occasionally emits neither the ``![image]`` placeholder nor the
    "Figure N" caption for a figure (the BSR 31123167 Figure 4 case); the gap in
    the caption numbering is detected, the figure band re-OCR'd, and the recovered
    placeholder remapped to page coordinates and spliced back in."""

    def test_caption_labels_detected_references_ignored(self) -> None:
        from manuscribe.pipeline.recover_figures import _emitted_figure_numbers

        md = (
            "Figure 1. Gene clusters and pathways\n\n"
            "FIG 3\n\n"
            "FIGURE 5 | Sequence alignment of the enzyme\n\n"
            "**Figure 6.** Active site residues\n\n"
            "As shown in Fig. 2, it was predicted that the rate\n\n"
            "The interface resembles FucO (Figure 4A) and is stable."
        )
        # captions counted; in-prose references ("Fig. 2,", "(Figure 4A)") are not
        assert _emitted_figure_numbers([md]) == {1, 3, 5, 6}

    def test_gap_in_emitted_numbering(self) -> None:
        from manuscribe.pipeline.recover_figures import _emitted_figure_numbers

        pages = ["Figure 1. A", "Figure 2. B", "Figure 3. C", "Figure 5. E"]
        assert _emitted_figure_numbers(pages) == {1, 2, 3, 5}

    def test_crop_box_not_collapsed_by_ghost_caption_line(self) -> None:
        from manuscribe.pipeline.recover_figures import _figure_crop_box

        # A faux-bold caption double-renders its first line with a sub-point
        # vertical offset, so a ghost copy sits a fraction above the "FIG. 6" label.
        # The crop must still span the figure above the caption (region top reaching
        # the page top here), not collapse onto the ghost line and clip the figure.
        label = "FIG. 6 Title"
        ghost = "Title"
        text = f"{label}\n{ghost}\n"
        boxes: list[tuple[float, float, float, float] | None] = []
        for ch in label:  # label line: y in [366, 376]
            boxes.append(None if ch == " " else (40.0, 366.0, 50.0, 376.0))
        boxes.append(None)  # newline
        for _ in ghost:  # ghost line: bottom (376.3) barely above the label's top
            boxes.append((95.0, 376.3, 105.0, 383.6))
        boxes.append(None)  # newline
        rotations = [0 if b is not None else None for b in boxes]

        located = _figure_crop_box(text, boxes, rotations, 6, (612.0, 792.0))
        assert located is not None
        (_, _, _, region_top), cap_top = located
        assert region_top == 792.0
        assert region_top - cap_top > 400.0  # the figure band, not the caption row

    def test_extract_recovered_figure_folds_caption_stops_at_body(self) -> None:
        from manuscribe.pipeline.recover_figures import _extract_recovered_figure

        crop_md = (
            "![image](image_1.png)210,50,865,440\n\n"
            "Figure 4. Crystal structures of *BkTauF*\n\n"
            "(A) Quaternary structure. (B) Subunit structure.\n\n"
            "## Crystal structure of *BkTauF*\n\n"
            "Crystal structures of *BkTauF* were solved at 1.9 Å."
        )
        result = _extract_recovered_figure(crop_md, 4)
        assert result is not None
        bbox, caption = result
        assert bbox == (210, 50, 865, 440)
        # the caption header and its panel description are folded; the body heading
        # and prose the generous crop also captured are left out
        assert caption.startswith("Figure 4. Crystal structures")
        assert "(A) Quaternary structure" in caption
        assert "Crystal structure of" not in caption.replace("Crystal structures", "")
        assert "were solved" not in caption

    def test_extract_recovered_figure_none_without_placeholder(self) -> None:
        from manuscribe.pipeline.recover_figures import _extract_recovered_figure

        # a crop that re-OCR'd to no figure box recovers nothing (fail-safe)
        no_box = "Figure 4. Crystal structures\n\nprose"
        assert _extract_recovered_figure(no_box, 4) is None

    def test_extract_recovered_figure_caption_before_placeholder(self) -> None:
        from manuscribe.pipeline.recover_figures import _extract_recovered_figure

        # The crop re-OCR sometimes emits the caption *before* the ![image]
        # placeholder (observed for FIG. 6 of 31051047); the caption — split into a
        # bare "FIG. N" label and its title block — must still be captured, or the
        # recovered figure renders with no <figcaption> and reads as "missing".
        crop_md = (
            "FIG. 6\n\n"
            "The optimum pH points and enzymatic activities. "
            "(A) Reduction reaction activities of PtTRI. "
            "(B) Reduction reaction activities of PtTRII.\n\n"
            "![image](image_1.png)123,147,865,775"
        )
        result = _extract_recovered_figure(crop_md, 6)
        assert result is not None
        bbox, caption = result
        assert bbox == (123, 147, 865, 775)
        assert caption.startswith("FIG. 6")
        assert "The optimum pH points and enzymatic activities" in caption
        assert "(A) Reduction reaction activities of PtTRI" in caption

    def test_extract_recovered_figure_bare_label_title_after_placeholder(self) -> None:
        from manuscribe.pipeline.recover_figures import _extract_recovered_figure

        # The same split (bare label, then title) after the placeholder: the title
        # block that follows a bare "FIG. N" label is its caption, not body prose.
        crop_md = (
            "![image](image_1.png)10,20,800,600\n\n"
            "Figure 3\n\n"
            "Phylogenetic tree analysis. (A) Tree. (B) Tissue profile.\n\n"
            "## Results\n\nThe tree shows three clades."
        )
        result = _extract_recovered_figure(crop_md, 3)
        assert result is not None
        _, caption = result
        assert caption.startswith("Figure 3")
        assert "Phylogenetic tree analysis" in caption
        assert "(A) Tree" in caption
        # body heading/prose the crop also captured is still excluded
        assert "Results" not in caption
        assert "three clades" not in caption

    def test_extract_recovered_figure_ignores_neighbouring_figure_caption(self) -> None:
        from manuscribe.pipeline.recover_figures import _extract_recovered_figure

        # The tight crop can catch a neighbouring figure's caption tail above the
        # image; it must not be taken as THIS figure's caption — match the numbered
        # label, not any figure-like block.
        crop_md = (
            "Figure 5. Previous figure caption tail.\n\n"
            "![image](image_1.png)5,6,7,8\n\n"
            "Figure 6. The real caption for this figure."
        )
        result = _extract_recovered_figure(crop_md, 6)
        assert result is not None
        bbox, caption = result
        assert bbox == (5, 6, 7, 8)
        assert caption.startswith("Figure 6. The real caption")
        assert "Figure 5" not in caption

    def test_extract_recovered_figure_no_matching_label_yields_empty(self) -> None:
        from manuscribe.pipeline.recover_figures import _extract_recovered_figure

        # Only a *different* figure/scheme caption is in the crop (no caption for the
        # requested number): recover the image with an empty caption rather than
        # mislabeling it with the neighbour's caption.
        crop_md = (
            "Scheme 2. Synthetic route.\n\n"
            "![image](image_1.png)1,2,3,4\n\n"
            "resumed body text unrelated to the figure."
        )
        result = _extract_recovered_figure(crop_md, 5)
        assert result == ((1, 2, 3, 4), "")

    def test_extract_recovered_figure_bare_label_then_heading(self) -> None:
        from manuscribe.pipeline.recover_figures import _extract_recovered_figure

        # A bare "FIG. N" label followed by a heading (the crop reached into the body
        # below): the heading must NOT be claimed as the title — caption is the label.
        crop_md = (
            "![image](image_1.png)10,20,30,40\n\n"
            "FIG. 6\n\n## Results\n\nThe assay showed activity."
        )
        result = _extract_recovered_figure(crop_md, 6)
        assert result is not None
        _, caption = result
        assert caption == "FIG. 6"

    def test_extract_recovered_figure_bare_label_last_block(self) -> None:
        from manuscribe.pipeline.recover_figures import _extract_recovered_figure

        # A bare label with no following block (no title arrives) yields a label-only
        # caption, not a crash.
        crop_md = "![image](image_1.png)1,2,3,4\n\nFIG. 6"
        result = _extract_recovered_figure(crop_md, 6)
        assert result == ((1, 2, 3, 4), "FIG. 6")

    def test_caption_already_present_matches_split_label_title(self) -> None:
        from manuscribe.pipeline.recover_figures import _caption_already_present

        # The recovered caption is a label/title split; the dedup must flatten it so a
        # caption already on the page (in an em-dash form the label regex missed) is
        # recognised and the figure is spliced image-only, not as a visible duplicate.
        caption = "FIG. 6\n\nThe optimum pH points and enzymatic activities."
        page_md = (
            "Prose. FIG. 6 — The optimum pH points and enzymatic activities. More."
        )
        assert _caption_already_present(caption, page_md) is True
        # A bare label with no title still folds too short to match and splices.
        assert _caption_already_present("FIG. 6", page_md) is False

    def test_remap_full_width_region_keeps_x_offsets_y_into_band(self) -> None:
        from manuscribe.pipeline.recover_figures import _remap_bbox_to_page

        # full-width crop spanning the page's top band [y 600..800] of an 800-pt page;
        # x stays as-is (full width), y maps into the band measured from the page top.
        page_size = (600.0, 800.0)
        region = (0.0, 600.0, 600.0, 800.0)  # left, bottom, right, top (PDF points)
        # crop-relative box: top-left quarter of the crop
        bbox = _remap_bbox_to_page((0, 0, 500, 500), region, page_size)
        # x unchanged (0..500); y: crop top is page-top (0 from top) down half the
        # 200-pt band → 100 pt from top → 125/1000
        assert bbox == (0, 0, 500, 125)

    def test_splice_top_figure_prepended(self) -> None:
        from manuscribe.pipeline.recover_figures import _splice_figures_into_page

        # caption near the top of an 800-pt page (cap_top 750) → figure prepended
        out = _splice_figures_into_page("body prose", [(750.0, "FIGBLOCK")], 800.0)
        assert out == "FIGBLOCK\n\nbody prose"

    def test_splice_bottom_figure_appended(self) -> None:
        from manuscribe.pipeline.recover_figures import _splice_figures_into_page

        # caption low on the page (cap_top 200) → figure appended after the prose
        out = _splice_figures_into_page("body prose", [(200.0, "FIGBLOCK")], 800.0)
        assert out == "body prose\n\nFIGBLOCK"

    def test_two_top_figures_keep_on_page_order(self) -> None:
        from manuscribe.pipeline.recover_figures import _splice_figures_into_page

        # both captions in the top half; the higher one (cap_top 760) must precede
        # the lower (cap_top 700), not be reversed by sequential prepends
        out = _splice_figures_into_page(
            "body", [(700.0, "LOWER"), (760.0, "HIGHER")], 800.0
        )
        assert out == "HIGHER\n\nLOWER\n\nbody"

    def test_top_and_bottom_figures_bracket_the_page(self) -> None:
        from manuscribe.pipeline.recover_figures import _splice_figures_into_page

        # one figure high (prepended), one low (appended) — body stays between them
        out = _splice_figures_into_page(
            "body", [(720.0, "TOP"), (120.0, "BOTTOM")], 800.0
        )
        assert out == "TOP\n\nbody\n\nBOTTOM"

    def test_caption_already_present_detected_through_separator_variation(self) -> None:
        from manuscribe.pipeline.recover_figures import _caption_already_present

        recovered = "Figure 4. Crystal structures of BkTauF"
        # the page emitted the same caption with an em-dash the label regex misses;
        # NFKD folding collapses the separator difference so it's recognised
        page = "<p>Figure 4 — Crystal structures of BkTauF (A) Quaternary…</p>"
        assert _caption_already_present(recovered, page)
        # absent from the page → not a duplicate, splice the caption normally
        assert not _caption_already_present(recovered, "<p>unrelated prose</p>")

    def test_caption_present_check_ignores_bare_label(self) -> None:
        from manuscribe.pipeline.recover_figures import _caption_already_present

        # a bare "Figure 4" folds too short to match, so an in-text reference like
        # "(Figure 4A)" in the body never suppresses the recovered caption
        assert not _caption_already_present("Figure 4.", "<p>see Figure 4A here</p>")

    def test_column_bounds_full_width_for_single_column(self) -> None:
        from manuscribe.pipeline.recover_figures import _column_bounds

        # a body line spanning the page centre → single column → full width
        cap_box = (40.0, 100.0, 300.0, 110.0)
        lines = [(40.0, 50.0, 560.0, 60.0)]  # crosses mid (300) of a 600-pt page
        assert _column_bounds(lines, cap_box, 110.0, 600.0) == (0.0, 600.0)

    def test_column_bounds_clamps_to_caption_half_for_two_columns(self) -> None:
        from manuscribe.pipeline.recover_figures import _column_bounds

        # no body line crosses the centre → two columns; caption on the right half
        cap_box = (320.0, 100.0, 560.0, 110.0)
        lines = [(40.0, 50.0, 280.0, 60.0), (320.0, 50.0, 560.0, 60.0)]
        assert _column_bounds(lines, cap_box, 110.0, 600.0) == (300.0, 600.0)

    def test_attempt_page_figure_declines_on_crop_ocr_failure(self) -> None:
        from manuscribe.pipeline.layers import _DocumentLayers
        from manuscribe.pipeline.recover_figures import _attempt_page_figure

        # A transient GPU OOM in the crop re-OCR must decline that figure (return None),
        # not abort the document. Figure 1's caption localizes deterministically on
        # page 1 of this fixture (text layer + geometry, no GPU), so the crop is built
        # and the injected re-OCR is actually reached and raised.
        reached: list[int] = []

        def boom(_image: Image.Image) -> str:
            reached.append(1)
            raise RuntimeError("CUDA out of memory")

        with _DocumentLayers.open(Path("tests/fixtures/30592559.pdf")) as layers:
            result = _attempt_page_figure(layers, 1, 1, boom, "")
        assert reached == [1]  # the crop re-OCR was reached (localization succeeded)
        assert result is None  # …and the failure was caught, not propagated


class TestSafeOcrRegion:
    """The shared best-effort re-OCR guard returns the OCR output on success and
    degrades to ``None`` (logged) on a transient failure, so one crop OOM declines a
    single refinement instead of aborting the whole conversion."""

    def test_returns_ocr_output_on_success(self) -> None:
        from manuscribe.pipeline.figures import _safe_ocr_region

        assert _safe_ocr_region(lambda _i: "markdown", Image.new("RGB", (8, 8))) == (
            "markdown"
        )

    def test_returns_none_and_logs_on_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from manuscribe.pipeline.figures import _safe_ocr_region

        def boom(_image: Image.Image) -> str:
            raise RuntimeError("CUDA out of memory")

        with caplog.at_level(logging.WARNING):
            assert _safe_ocr_region(boom, Image.new("RGB", (8, 8))) is None
        assert "re-OCR" in caplog.text


class TestTableFigureDedup:
    """A table the model boxed as a figure (placeholder + "TABLE N" caption) right
    before the ``<table>`` it also transcribed is the table duplicated as an image:
    the figure is dropped and its caption folded into the real table."""

    def test_table_figure_is_dropped_and_caption_folded(self) -> None:
        md = (
            "![image](image_1.png)540,77,810,887\n"
            "TABLE 2 | Kinetic parameters for Lxmdh mutants.\n\n"
            "<table>\n<thead><tr><th>Substrate</th><th>Vmax</th></tr></thead>\n"
            "<tbody><tr><td>WT</td><td>302.7</td></tr></tbody>\n</table>"
        )
        html = _run_lighton([md])
        assert "<figure" not in html
        assert "data:image" not in html
        assert "<table><caption>TABLE 2 | Kinetic parameters for Lxmdh mutants." in html

    def test_table_figure_with_standalone_caption_block_is_dropped(self) -> None:
        # The model often emits the "TABLE N" caption as its own block (not baked
        # onto the placeholder line); the figure still gets caption=None, so the
        # dedup must look past a standalone caption block to the <table> and drop
        # the duplicate image, leaving the caption block to fold into the table.
        md = (
            "![image](image_1.png)540,77,810,887\n\n"
            "TABLE 2 | Kinetic parameters for Lxmdh mutants.\n\n"
            "<table>\n<thead><tr><th>Substrate</th><th>Vmax</th></tr></thead>\n"
            "<tbody><tr><td>WT</td><td>302.7</td></tr></tbody>\n</table>"
        )
        html = _run_lighton([md])
        assert "<figure" not in html
        assert "data:image" not in html
        assert "<table><caption>TABLE 2 | Kinetic parameters for Lxmdh mutants." in html

    def test_pipe_caption_requires_title_not_bare_pipe(self) -> None:
        # A stray pipe-delimited prose line ("Table 1 | 2 | 3") is not a caption and
        # must not trigger the dedup drop of a genuinely adjacent figure.
        from manuscribe.pipeline.text import _opens_with_table_label

        assert _opens_with_table_label("TABLE 2 | Kinetic parameters for mutants.")
        assert not _opens_with_table_label("Table 1 | 2 | 3")
        assert not _opens_with_table_label("Table 2 | see column three for details")

    def test_real_figure_before_table_is_kept(self) -> None:
        # A genuine figure (a "FIGURE N" caption) that merely precedes a table must
        # not be mistaken for a boxed table — its crop is still emitted.
        md = (
            "![image](image_1.png)100,100,400,400\n"
            "FIGURE 3 | Activity assay results.\n\n"
            "<table>\n<thead><tr><th>Substrate</th><th>Vmax</th></tr></thead>\n"
            "<tbody><tr><td>WT</td><td>302.7</td></tr></tbody>\n</table>"
        )
        html = _run_lighton([md])
        assert "<figure" in html
        assert "FIGURE 3 | Activity assay results." in html

    def test_captioned_figure_before_table_caption_and_table_is_kept(self) -> None:
        # A genuine figure with its own "FIG N" caption, immediately followed by a
        # *separate* table (its "TABLE N" caption then the <table>) with no "---"
        # between, must not be deduped as a boxed table — only a caption-less figure
        # is.  (Real case: FIG. 6 + TABLE 1 on page 9 of 31051047 when the OCR drops
        # the separator.)  Both the figure and the table survive.
        md = (
            "![image](image_1.png)113,89,865,514\n"
            "FIG. 6 The optimum pH points and enzymatic activities. "
            "(A) Reduction reaction activities of PtTRI.\n\n"
            "TABLE 1 Enzyme kinetics of PtTRI and PtTRII\n\n"
            "<table>\n<thead><tr><th>Reductase</th><th>Km</th></tr></thead>\n"
            "<tbody><tr><td>PtTRI</td><td>0.52</td></tr></tbody>\n</table>"
        )
        html = _run_lighton([md])
        assert "<figure" in html
        assert "FIG. 6 The optimum pH points" in html
        assert "<table><caption>TABLE 1 Enzyme kinetics of PtTRI and PtTRII" in html

    def test_figure_with_non_fig_caption_before_table_is_kept(self) -> None:
        # The dedup keeps a figure that carries ANY caption of its own, not only a
        # "FIG N"-labelled one — per the figures-over-include trade, an ambiguous
        # captioned figure before a table is kept rather than risk dropping a real
        # figure whose caption the model didn't prefix with "FIG N".  (A boxed table
        # that the model captions descriptively would double-render here; that stray
        # image is the accepted lesser evil versus losing a genuine figure.)
        md = (
            "![image](image_1.png)100,100,400,400\n"
            "Schematic of the reaction apparatus and flow path.\n\n"
            "TABLE 1 Enzyme kinetics of PtTRI and PtTRII\n\n"
            "<table>\n<thead><tr><th>Reductase</th><th>Km</th></tr></thead>\n"
            "<tbody><tr><td>PtTRI</td><td>0.52</td></tr></tbody>\n</table>"
        )
        html = _run_lighton([md])
        assert "<figure" in html
        assert "Schematic of the reaction apparatus" in html
        assert "<table><caption>TABLE 1 Enzyme kinetics of PtTRI and PtTRII" in html


class TestFigureFileOutput:
    """With an image directory, crops are written as sidecar PNGs and referenced
    by a relative path instead of inlined as base64."""

    def test_image_dir_writes_sidecar_png(self, tmp_path: Path) -> None:
        from manuscribe.pipeline.assemble import _assemble_html
        from manuscribe.pipeline.figures import _file_image_writer

        img = _fake_image(1190, 1540)
        md = "# T\n\n## Abstract\n\nA.\n\n## Body\n\n![image](i.png)0,0,1000,1000"
        image_dir = tmp_path / "doc_files"
        html = _assemble_html([md], [img], None, _file_image_writer(image_dir))
        assert "data:image/png;base64," not in html
        assert 'src="doc_files/fig_001.png"' in html
        assert Image.open(image_dir / "fig_001.png").size == (1190, 1540)


class TestImageSink:
    """The injectable ``encode_image`` seam (Tasks B): a sink receives each crop's
    PNG bytes + MIME and its returned value becomes the figure's ``<img src>``, so a
    caller can store the bytes elsewhere (e.g. served assets) instead of inlining."""

    _MD = "# T\n\n## Abstract\n\nA.\n\n## Body\n\n![image](i.png)0,0,1000,1000"

    def test_sink_receives_png_bytes_and_src_used(self) -> None:
        from manuscribe.pipeline.assemble import _assemble_html

        received: list[tuple[bytes, str]] = []

        def sink(image_bytes: bytes, mime: str) -> str:
            received.append((image_bytes, mime))
            return f"https://assets.example/fig{len(received)}.png"

        html = _assemble_html([self._MD], [_fake_image(1190, 1540)], None, sink)
        assert len(received) == 1
        png, mime = received[0]
        assert mime == "image/png"
        assert png.startswith(b"\x89PNG\r\n\x1a\n")  # PNG signature
        assert Image.open(io.BytesIO(png)).size == (1190, 1540)
        assert 'src="https://assets.example/fig1.png"' in html
        assert "data:image/png;base64," not in html

    def test_base64_default_inlines_data_uri(self) -> None:
        from manuscribe.pipeline.figures import _base64_src

        out = _base64_src(b"\x89PNG\r\n\x1a\nfake", "image/png")
        assert out.startswith("data:image/png;base64,")
        assert base64.b64decode(out.split(",", 1)[1]) == b"\x89PNG\r\n\x1a\nfake"


class TestDenormalizeBbox:
    """[0,1000]-normalized model boxes scale to the image's pixel size."""

    def test_full_box_maps_to_full_image(self) -> None:
        from manuscribe.pipeline.figures import _denormalize_bbox

        assert _denormalize_bbox((0, 0, 1000, 1000), _fake_image(1190, 1540)) == (
            0,
            0,
            1190,
            1540,
        )

    def test_half_box(self) -> None:
        from manuscribe.pipeline.figures import _denormalize_bbox

        assert _denormalize_bbox((0, 0, 500, 500), _fake_image(1000, 2000)) == (
            0,
            0,
            500,
            1000,
        )


class TestFigureBoxMerge:
    """A figure the model over-segments into stacked boxes is unioned into one
    crop; genuinely separate figures stay separate."""

    def test_same_column_adjacent_boxes_merge(self) -> None:
        from manuscribe.pipeline.figures import _figures_same

        assert _figures_same((100, 100, 900, 500), (110, 500, 890, 560), 50.0) is True

    def test_vertically_separated_boxes_do_not_merge(self) -> None:
        from manuscribe.pipeline.figures import _figures_same

        assert _figures_same((100, 100, 900, 300), (100, 800, 900, 950), 50.0) is False

    def test_side_by_side_boxes_do_not_merge(self) -> None:
        from manuscribe.pipeline.figures import _figures_same

        assert _figures_same((0, 0, 100, 500), (200, 0, 300, 500), 50.0) is False

    def test_union_box(self) -> None:
        from manuscribe.pipeline.figures import _union_box

        assert _union_box([(100, 100, 900, 500), (120, 480, 880, 560)]) == (
            100,
            100,
            900,
            560,
        )

    def test_split_figure_emits_single_crop(self) -> None:
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\n"
            "![image](a.png)100,100,900,500\n\n"
            "![image](b.png)100,500,900,560\n\n"
            "FIG. 1 One caption."
        )
        sizes = _figure_sizes(_run_lighton([md], image=img))
        assert len(sizes) == 1
        # The union spans both boxes (down to y≈862 px), not just the first.
        assert sizes[0][1] > 700

    def test_two_separated_figures_stay_separate(self) -> None:
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\n"
            "![image](a.png)100,100,900,300\n\n"
            "Some intervening prose between the two figures.\n\n"
            "![image](b.png)100,800,900,950"
        )
        assert len(_figure_sizes(_run_lighton([md], image=img))) == 2


class TestFigureBottomGrowth:
    """The crop grows down over contiguous ink to the figure's true bottom and
    stops at the whitespace before the caption; a box already ending in
    whitespace grows nothing, so caption text is never pulled in."""

    @staticmethod
    def _image() -> Image.Image:
        # White page: figure block y[100,300), caption block y[360,380),
        # separated by a 60 px whitespace gap.
        img = Image.new("RGB", (400, 800), "white")
        img.paste(Image.new("RGB", (300, 200), "black"), (50, 100))
        img.paste(Image.new("RGB", (300, 20), "black"), (50, 360))
        return img

    def test_tight_box_grows_to_figure_bottom(self) -> None:
        from manuscribe.pipeline.figures import _extend_edge

        assert _extend_edge(self._image(), (50, 100, 350, 250), "bottom") == 300

    def test_box_at_bottom_does_not_grow(self) -> None:
        from manuscribe.pipeline.figures import _extend_edge

        assert _extend_edge(self._image(), (50, 100, 350, 300), "bottom") == 300

    def test_no_growth_when_ink_runs_without_gap(self) -> None:
        # Ink continues past the search window with no whitespace gap (caption /
        # body text below a correct box) → ambiguous → leave the box unchanged.
        from manuscribe.pipeline.figures import _extend_edge

        img = Image.new("RGB", (400, 800), "white")
        img.paste(Image.new("RGB", (300, 300), "black"), (50, 100))
        assert _extend_edge(img, (50, 100, 350, 250), "bottom") == 250

    def test_narrow_content_below_box_is_not_read_as_gap(self) -> None:
        # A figure tail narrower than the box (here 3 px of a 300 px-wide box,
        # ~1% ink) must count as content, not be mistaken for the whitespace gap
        # — otherwise the clipped bottom is dropped.
        from manuscribe.pipeline.figures import _extend_edge

        img = Image.new("RGB", (400, 800), "white")
        img.paste(Image.new("RGB", (300, 150), "black"), (50, 100))  # y[100,250)
        img.paste(Image.new("RGB", (3, 40), "black"), (198, 250))  # narrow tail
        assert _extend_edge(img, (50, 100, 350, 270), "bottom") == 290

    def test_growth_stops_before_caption(self) -> None:
        from manuscribe.pipeline.figures import _extend_edge

        assert _extend_edge(self._image(), (50, 100, 350, 250), "bottom") < 360

    def test_safe_crop_excludes_caption(self) -> None:
        from manuscribe.pipeline.figures import _safe_crop

        crop = _safe_crop(self._image(), (50, 100, 350, 250))
        assert crop is not None and crop.size == (300, 200)


class TestFigureRightGrowth:
    """The crop grows right over contiguous ink to the figure's true right edge
    and stops at the whitespace before the page margin or inter-column gutter; a
    box already ending in whitespace grows nothing, so a neighbouring column is
    never pulled in."""

    @staticmethod
    def _image() -> Image.Image:
        # White page: figure block x[100,300), a strip at x[360,380) (page-margin
        # neighbour), separated by a 60 px whitespace gap.
        img = Image.new("RGB", (800, 400), "white")
        img.paste(Image.new("RGB", (200, 300), "black"), (100, 50))
        img.paste(Image.new("RGB", (20, 300), "black"), (360, 50))
        return img

    def test_tight_box_grows_to_figure_right(self) -> None:
        from manuscribe.pipeline.figures import _extend_edge

        assert _extend_edge(self._image(), (100, 50, 250, 350), "right") == 300

    def test_box_at_right_does_not_grow(self) -> None:
        from manuscribe.pipeline.figures import _extend_edge

        assert _extend_edge(self._image(), (100, 50, 300, 350), "right") == 300

    def test_no_growth_when_ink_runs_without_gap(self) -> None:
        # Ink continues past the search window with no whitespace gap (a column
        # abutting a correct box) → ambiguous → leave the box unchanged.
        from manuscribe.pipeline.figures import _extend_edge

        img = Image.new("RGB", (800, 400), "white")
        img.paste(Image.new("RGB", (300, 300), "black"), (100, 50))
        assert _extend_edge(img, (100, 50, 250, 350), "right") == 250

    def test_narrow_content_right_of_box_is_not_read_as_gap(self) -> None:
        # A figure tail narrower than the box (here 3 px of a 300 px-tall box,
        # ~1% ink) must count as content, not be mistaken for the whitespace gap.
        from manuscribe.pipeline.figures import _extend_edge

        img = Image.new("RGB", (800, 400), "white")
        img.paste(Image.new("RGB", (150, 300), "black"), (100, 50))  # x[100,250)
        img.paste(Image.new("RGB", (40, 3), "black"), (250, 198))  # narrow tail
        assert _extend_edge(img, (100, 50, 270, 350), "right") == 290

    def test_gutter_stops_growth_before_next_column(self) -> None:
        # Left-column figure, a whitespace gutter, then right-column text: growth
        # recovers the clipped figure edge but stops at the gutter, never reaching
        # the next column.
        from manuscribe.pipeline.figures import _extend_edge

        img = Image.new("RGB", (800, 400), "white")
        img.paste(Image.new("RGB", (200, 300), "black"), (50, 50))  # x[50,250)
        img.paste(Image.new("RGB", (200, 300), "black"), (310, 50))  # right column
        assert _extend_edge(img, (50, 50, 200, 350), "right") == 250

    def test_safe_crop_excludes_neighbour_column(self) -> None:
        from manuscribe.pipeline.figures import _safe_crop

        crop = _safe_crop(self._image(), (100, 50, 250, 350))
        assert crop is not None and crop.size == (200, 300)


class TestFigureLeftGrowth:
    """The crop grows left over contiguous ink to the figure's true left edge and
    stops at the whitespace before the page margin or inter-column gutter; a box
    already ending in whitespace grows nothing, so a neighbouring column is never
    pulled in.  Mirror of :class:`TestFigureRightGrowth`."""

    @staticmethod
    def _image() -> Image.Image:
        # White page: a strip at x[20,40) (page-margin neighbour), a 60 px gap, then
        # the figure block x[100,300).
        img = Image.new("RGB", (800, 400), "white")
        img.paste(Image.new("RGB", (20, 300), "black"), (20, 50))
        img.paste(Image.new("RGB", (200, 300), "black"), (100, 50))
        return img

    def test_tight_box_grows_to_figure_left(self) -> None:
        from manuscribe.pipeline.figures import _extend_edge

        # box left clipped 50 px into the figure → grows back out to x=100
        assert _extend_edge(self._image(), (150, 50, 300, 350), "left") == 100

    def test_box_at_left_does_not_grow(self) -> None:
        from manuscribe.pipeline.figures import _extend_edge

        assert _extend_edge(self._image(), (100, 50, 300, 350), "left") == 100

    def test_no_growth_when_ink_runs_without_gap(self) -> None:
        # Ink continues left past the search window with no gap (a column abutting a
        # correct box) → ambiguous → leave the box unchanged.
        from manuscribe.pipeline.figures import _extend_edge

        img = Image.new("RGB", (800, 400), "white")
        img.paste(Image.new("RGB", (300, 300), "black"), (100, 50))  # x[100,400)
        assert _extend_edge(img, (350, 50, 400, 350), "left") == 350

    def test_gutter_stops_growth_before_previous_column(self) -> None:
        # Right-column figure, a whitespace gutter, then left-column text: growth
        # recovers the clipped figure edge but stops at the gutter, never reaching
        # the previous column.
        from manuscribe.pipeline.figures import _extend_edge

        img = Image.new("RGB", (800, 400), "white")
        img.paste(Image.new("RGB", (200, 300), "black"), (50, 50))  # left column
        img.paste(Image.new("RGB", (200, 300), "black"), (360, 50))  # x[360,560)
        assert _extend_edge(img, (410, 50, 560, 350), "left") == 360

    def test_safe_crop_recovers_clipped_left_edge(self) -> None:
        from manuscribe.pipeline.figures import _safe_crop

        # box left clipped to x=150; the crop grows back out to the figure's x=100.
        crop = _safe_crop(self._image(), (150, 50, 300, 350))
        assert crop is not None and crop.size == (200, 300)


class TestFigureTopGrowth:
    """The crop grows up over contiguous ink to the figure's true top edge and
    stops at the whitespace before the preceding paragraph or a caption; a box
    already ending in whitespace grows nothing, so text above is never pulled in.
    Vertical mirror of :class:`TestFigureLeftGrowth` — motivated by Figure 5 of the
    32117944 fixture, whose box clipped the top panel labels (A/B) and frame line."""

    @staticmethod
    def _image() -> Image.Image:
        # White page: a text strip at y[20,40) (the preceding paragraph), a 60 px
        # gap, then the figure block y[100,300).
        img = Image.new("RGB", (400, 800), "white")
        img.paste(Image.new("RGB", (300, 20), "black"), (50, 20))
        img.paste(Image.new("RGB", (300, 200), "black"), (50, 100))
        return img

    def test_tight_box_grows_to_figure_top(self) -> None:
        from manuscribe.pipeline.figures import _extend_edge

        # box top clipped 50 px into the figure → grows back out to y=100
        assert _extend_edge(self._image(), (50, 150, 350, 300), "top") == 100

    def test_box_at_top_does_not_grow(self) -> None:
        from manuscribe.pipeline.figures import _extend_edge

        assert _extend_edge(self._image(), (50, 100, 350, 300), "top") == 100

    def test_no_growth_when_ink_runs_without_gap(self) -> None:
        # Ink continues up past the search window with no gap (content abutting a
        # correct box) → ambiguous → leave the box unchanged.
        from manuscribe.pipeline.figures import _extend_edge

        img = Image.new("RGB", (400, 800), "white")
        img.paste(Image.new("RGB", (300, 300), "black"), (50, 100))  # y[100,400)
        assert _extend_edge(img, (50, 350, 350, 400), "top") == 350

    def test_gap_above_box_stops_growth_before_paragraph(self) -> None:
        # A correct box already ending in whitespace: the paragraph above is
        # separated by a leading gap, so growth is declined, not pulled in.
        from manuscribe.pipeline.figures import _extend_edge

        assert _extend_edge(self._image(), (50, 100, 350, 300), "top") == 100

    def test_safe_crop_recovers_clipped_top_edge(self) -> None:
        from manuscribe.pipeline.figures import _safe_crop

        # box top clipped to y=150; the crop grows back out to the figure's y=100.
        crop = _safe_crop(self._image(), (50, 150, 350, 300))
        assert crop is not None and crop.size == (300, 200)


class TestSwallowedCaptionTrim:
    """A bottom band that growth recovers is trimmed when it reads as the figure's
    caption (short prose ink-runs) but kept when it is figure content (long shaded
    runs); trimming happens only when a caption is known to follow the figure."""

    @staticmethod
    def _image(ink_run: int, gap_run: int) -> Image.Image:
        # 800x400 white; a solid figure body at y[50,200) and, contiguous below it,
        # a 30 px band at y[200,230) whose horizontal ink-run length is set by
        # (ink_run, gap_run) — short runs read as prose, long runs as figure.
        a = np.full((400, 800), 255, np.uint8)
        a[50:200, 50:750] = 0
        row = np.full(700, 255, np.uint8)
        for x in range(0, 700, ink_run + gap_run):
            row[x : x + ink_run] = 0
        a[200:230, 50:750] = np.tile(row, (30, 1))
        return Image.fromarray(a, "L").convert("RGB")

    def test_prose_band_trimmed_when_caption_present(self) -> None:
        from manuscribe.pipeline.figures import _safe_crop

        crop = _safe_crop(self._image(3, 6), (50, 50, 750, 200), caption_text="cap")
        assert crop is not None and crop.size == (700, 150)  # band dropped

    def test_prose_band_kept_without_caption(self) -> None:
        from manuscribe.pipeline.figures import _safe_crop

        crop = _safe_crop(self._image(3, 6), (50, 50, 750, 200), caption_text=None)
        assert crop is not None and crop.size == (700, 180)  # band recovered

    def test_figure_band_kept_even_with_caption(self) -> None:
        from manuscribe.pipeline.figures import _safe_crop

        crop = _safe_crop(self._image(600, 10), (50, 50, 750, 200), caption_text="cap")
        assert crop is not None and crop.size == (700, 180)  # dense band kept

    def test_prose_scores_below_figure_run_length(self) -> None:
        from manuscribe.pipeline.figures import _mean_norm_run_length

        width = 700
        prose = np.zeros((10, width), bool)
        prose[:, ::9] = True  # 1-px runs, 8-px gaps → letterform-like
        figure = np.zeros((10, width), bool)
        figure[:, :600] = True  # one long shaded run
        assert _mean_norm_run_length(prose) < 0.07 <= _mean_norm_run_length(figure)


class TestBakedCaptionTrim:
    """When the figure is itself text (an alignment), its caption is pixel-identical
    to it and the model can box it *inside* the figure.  The trailing text bands are
    re-OCRed and dropped when they reproduce the caption — guarded against a figure
    row the caption merely names, and against a repeated-token OCR wall."""

    _CAPTION = (
        "Fig 9. Multiple sequence alignments of widget and gadget proteins. "
        "Catalytic residues are marked in cyan and the binding site in orange."
    )

    def test_band_is_caption_matches_caption_words(self) -> None:
        from manuscribe.pipeline.figures import _WORD_RE, _band_is_caption

        words = set(_WORD_RE.findall(self._CAPTION.lower()))
        assert _band_is_caption(self._CAPTION, words)

    def test_band_is_caption_rejects_low_overlap(self) -> None:
        from manuscribe.pipeline.figures import _WORD_RE, _band_is_caption

        words = set(_WORD_RE.findall(self._CAPTION.lower()))
        # a figure row mentioning a few caption words amid mostly non-caption data:
        # enough distinct caption words to clear the wall guard, but the matched
        # fraction stays under the bar
        assert not _band_is_caption(
            "catalytic residues marked QWERTY ZXCVB ASDFG HJKL", words
        )

    def test_band_is_caption_rejects_repeated_token_wall(self) -> None:
        from manuscribe.pipeline.figures import _WORD_RE, _band_is_caption

        # a row the model fails to read collapses to one caption word repeated —
        # ~1.0 word-overlap but no diversity, so it must be rejected as degenerate
        words = set(_WORD_RE.findall((self._CAPTION + " bmsdh").lower()))
        assert not _band_is_caption("bmsdh " * 200, words)

    def test_band_is_caption_rejects_short_repeated_wall(self) -> None:
        from manuscribe.pipeline.figures import _WORD_RE, _band_is_caption

        # the wall need not be long: three identical caption words must still fail
        # (the old type-ratio guard let this through; the distinct-word floor stops it)
        words = set(_WORD_RE.findall((self._CAPTION + " panel").lower()))
        assert not _band_is_caption("panel panel panel", words)

    def test_band_is_caption_rejects_too_few_words(self) -> None:
        from manuscribe.pipeline.figures import _WORD_RE, _band_is_caption

        words = set(_WORD_RE.findall(self._CAPTION.lower()))
        assert not _band_is_caption("Fig 9", words)

    def test_ink_bands_split_on_gaps(self) -> None:
        from manuscribe.pipeline.figures import _ink_bands

        mask = np.zeros((200, 50), bool)
        mask[10:40] = True  # band 1
        mask[120:160] = True  # band 2, separated by a 80-row gap
        assert _ink_bands(mask, gap=12) == [(10, 40), (120, 160)]

    def test_ocr_band_pads_thin_band(self) -> None:
        from manuscribe.pipeline.figures import _FIGURE_OCR_MIN_BAND_PX, _ocr_band

        seen: list[tuple[int, int]] = []
        _ocr_band(
            Image.new("RGB", (700, 600), "white"),
            (0, 100, 700, 110),  # a 10-px band
            lambda im: seen.append(im.size) or "",
        )
        assert seen and seen[0][1] >= _FIGURE_OCR_MIN_BAND_PX

    @staticmethod
    def _striped(a: np.ndarray, y0: int, y1: int) -> None:
        a[y0:y1, 50:750:9] = 0  # text-like: thin ink columns (short runs)

    def _text_image(self) -> Image.Image:
        # figure (top), then a caption band and a note band low in the crop
        a = np.full((600, 800), 255, np.uint8)
        for y0, y1 in ((40, 300), (400, 470), (500, 530)):
            self._striped(a, y0, y1)
        return Image.fromarray(a, "L").convert("RGB")

    def test_trim_baked_caption_drops_caption_and_note(self) -> None:
        from manuscribe.pipeline.figures import _trim_baked_caption

        # scan runs bottom→top: note (DOI) → caption → figure band (re-OCRed once to
        # confirm the boundary, then the scan stops as it isn't caption)
        replies = iter(
            ["see https://doi.org/10.1/x", self._CAPTION, "unrelated figure axis tick"]
        )
        y1 = _trim_baked_caption(
            self._text_image(), 0, 800, 0, 530, self._CAPTION, lambda im: next(replies)
        )
        assert y1 == 400  # trimmed to the caption's top, note swept with it

    def test_safe_crop_without_ocr_region_keeps_baked_caption(self) -> None:
        from manuscribe.pipeline.figures import _safe_crop

        # no ocr_region → the OCR trim never runs, so a text-bodied baked caption
        # stays (the crop reaches the note band's bottom)
        crop = _safe_crop(
            self._text_image(), (0, 0, 800, 530), caption_text=self._CAPTION
        )
        assert crop is not None and crop.size[1] == 530


class TestParseFigurePlaceholder:
    """LightOnOCR-bbox emits figures as `![image](...)x0,y0,x1,y1`; the parser
    must recover the crop box, recognise a bbox-less placeholder, and reject
    ordinary prose."""

    def test_box_extracted(self) -> None:
        from manuscribe.pipeline.figures import _parse_figure_placeholder

        result = _parse_figure_placeholder("![image](image_1.png)122,89,877,614")
        assert result.is_placeholder
        assert result.bbox_norm == (122, 89, 877, 614)

    def test_box_with_surrounding_whitespace(self) -> None:
        from manuscribe.pipeline.figures import _parse_figure_placeholder

        result = _parse_figure_placeholder("  ![image](img.png) 10, 20, 30, 40 ")
        assert result.is_placeholder
        assert result.bbox_norm == (10, 20, 30, 40)

    def test_bboxless_placeholder_is_placeholder_without_bbox(self) -> None:
        from manuscribe.pipeline.figures import _parse_figure_placeholder

        result = _parse_figure_placeholder("![image](image_1.png)")
        assert result.is_placeholder
        assert result.bbox_norm is None

    def test_caption_line_is_not_a_placeholder(self) -> None:
        from manuscribe.pipeline.figures import _parse_figure_placeholder

        result = _parse_figure_placeholder("FIG. 2 Protein alignments of TRI.")
        assert not result.is_placeholder
        assert result.bbox_norm is None

    def test_inline_image_in_prose_is_not_a_placeholder(self) -> None:
        from manuscribe.pipeline.figures import _parse_figure_placeholder

        line = "Some prose with ![inline](x.png) embedded mid-sentence."
        assert not _parse_figure_placeholder(line).is_placeholder


class TestFigureRecoveryOrchestration:
    """``_recover_dropped_figures`` enumerates the text-layer figure numbers the OCR
    never emitted, localizes and re-OCRs each on its page, and splices the recovered
    placeholder back.  Driven against a real fixture's text layer with a fake
    ``ocr_region`` so no GPU/server is needed — only the injected re-OCR is faked; the
    caption localization and crop render are pure CPU."""

    _FIXTURE = Path(__file__).parent / "fixtures" / "30592559.pdf"  # figs 1–4, no ad

    def test_textlayer_caption_pages_maps_numbers_to_pages(self) -> None:
        from manuscribe.pipeline.layers import _DocumentLayers
        from manuscribe.pipeline.recover_figures import _textlayer_caption_pages

        with _DocumentLayers.open(self._FIXTURE) as layers:
            truth = _textlayer_caption_pages(layers.pdf)
        assert set(truth) == {1, 2, 3, 4}
        assert all(ps and ps == sorted(set(ps)) for ps in truth.values())

    def test_missing_figures_are_localized_and_spliced(self) -> None:
        from manuscribe.pipeline.layers import _DocumentLayers
        from manuscribe.pipeline.recover_figures import _recover_dropped_figures

        calls: list[int] = []

        def fake_ocr_region(image: Image.Image) -> str:
            calls.append(1)
            return (
                "![image](image_1.png)120,120,880,880"
                "\n\nFigure 1. Recovered legend title."
            )

        with _DocumentLayers.open(self._FIXTURE) as layers:
            # Nothing emitted -> every text-layer figure (1–4) is "missing".
            pages = [""] * len(layers.pdf)
            out = _recover_dropped_figures(layers, pages, fake_ocr_region)

        joined = "\n".join(out)
        assert len(calls) == 4  # one re-OCR per missing figure (not batched)
        assert joined.count("![image]") == 4  # all four recovered and spliced back
        assert "Recovered legend title" in joined  # figure 1's caption matched + kept

    def test_no_ocr_when_every_figure_already_emitted(self) -> None:
        from manuscribe.pipeline.layers import _DocumentLayers
        from manuscribe.pipeline.recover_figures import _recover_dropped_figures

        calls: list[int] = []

        def fake_ocr_region(image: Image.Image) -> str:
            calls.append(1)
            return ""

        emitted = ["Figure 1. A\n\nFigure 2. B\n\nFigure 3. C\n\nFigure 4. D"]
        with _DocumentLayers.open(self._FIXTURE) as layers:
            out = _recover_dropped_figures(layers, emitted, fake_ocr_region)
        assert out == emitted  # no gap between text-layer and emitted -> untouched
        assert calls == []  # ...and not a single re-OCR

    def test_attempt_declines_before_ocr_when_caption_not_on_page(self) -> None:
        from manuscribe.pipeline.layers import _DocumentLayers
        from manuscribe.pipeline.recover_figures import _attempt_page_figure

        calls: list[int] = []

        def fake_ocr_region(image: Image.Image) -> str:
            calls.append(1)
            return "![image](image_1.png)0,0,1000,1000"

        with _DocumentLayers.open(self._FIXTURE) as layers:
            # Figure 999 has no caption label anywhere -> no crop box -> declined.
            result = _attempt_page_figure(layers, 0, 999, fake_ocr_region, "")
        assert result is None
        assert calls == []  # localization fails first, so the re-OCR is never reached

    def test_candidate_reocring_to_no_placeholder_injects_no_figure(self) -> None:
        from manuscribe.pipeline.layers import _DocumentLayers
        from manuscribe.pipeline.recover_figures import _recover_dropped_figures

        calls: list[int] = []

        def fake_ocr_region(image: Image.Image) -> str:
            calls.append(1)
            return "Prose the crop caught, but no image placeholder came back."

        with _DocumentLayers.open(self._FIXTURE) as layers:
            pages = [""] * len(layers.pdf)
            out = _recover_dropped_figures(layers, pages, fake_ocr_region)

        # Candidates were localized and re-OCR'd, but none returned a placeholder — so
        # the module's safety contract holds: a spurious candidate injects no figure.
        assert calls  # the crops were attempted
        assert "![image]" not in "\n".join(out)
