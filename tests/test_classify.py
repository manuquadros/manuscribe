"""Tests for document-structure classification and HTML assembly."""

from helpers import (
    _as_blocks,
    _body,
    _fake_image,
    _figure_sizes,
    _header_h1,
    _metadata,
    _run_lighton,
)
from PIL import Image


class TestArticlePageDetection:
    """Cover ads / mastheads carry no Abstract or Introduction heading, so the
    article start is the first page that does."""

    def test_ad_page_is_not_article(self) -> None:
        from manuscribe.pipeline.classify import _is_article_page_md

        ad = "# Virtual Conference\n\n## Data integrity seminar\n\nRegister here."
        assert _is_article_page_md(ad) is False

    def test_abstract_page_is_article(self) -> None:
        from manuscribe.pipeline.classify import _is_article_page_md

        assert _is_article_page_md("# Title\n\n## Abstract\n\nWe did things.") is True

    def test_introduction_page_is_article(self) -> None:
        from manuscribe.pipeline.classify import _is_article_page_md

        assert _is_article_page_md("## 1. Introduction\n\nText.") is True

    def test_summary_page_is_article(self) -> None:
        """Cell Press labels the abstract section "Summary", not "Abstract"."""
        from manuscribe.pipeline.classify import _is_article_page_md

        assert _is_article_page_md("## SUMMARY\n\nWe describe a new enzyme.") is True

    def test_wiley_numbered_pipe_introduction_is_article(self) -> None:
        from manuscribe.pipeline.classify import _is_article_page_md

        assert _is_article_page_md("## 1 | INTRODUCTION\n\nText.") is True

    def test_ieee_roman_introduction_is_article(self) -> None:
        from manuscribe.pipeline.classify import _is_article_page_md

        assert _is_article_page_md("## I. INTRODUCTION\n\nText.") is True

    def test_late_introduction_subsection_is_not_article(self) -> None:
        """A body subsection merely starting with "Introduction" is not the
        article's own Introduction heading and must not match."""
        from manuscribe.pipeline.classify import _is_article_page_md

        md = "### Introduction of point mutations\n\nWe mutated residue 42."
        assert _is_article_page_md(md) is False

    def test_leading_ad_page_skipped(self) -> None:
        from manuscribe.pipeline.classify import _leading_pages_to_skip_md

        ad = "# Conference\n\nRegister here."
        article = "# Real Title\n\n## Abstract\n\nBody."
        assert _leading_pages_to_skip_md([ad, article]) == 1
        assert _leading_pages_to_skip_md([article]) == 0

    def test_late_introduction_subsection_does_not_widen_skip(self) -> None:
        """Reviewer probe: a Wiley-style "1 | INTRODUCTION" on the real title page
        must be recognised as the article start so a later body subsection that
        merely starts with "Introduction" can't push the skip past it and drop
        the title/byline/abstract."""
        from manuscribe.pipeline.classify import _leading_pages_to_skip_md

        title_page = "# Real Title\n\n## 1 | INTRODUCTION\n\nText."
        results_page = "## 2 | RESULTS\n\nText."
        subsection_page = (
            "### Introduction of point mutations\n\nWe mutated residue 42."
        )
        pages = [title_page, results_page, subsection_page]
        assert _leading_pages_to_skip_md(pages) == 0


class TestRunningFurniture:
    """Short header/footer lines that recur across pages — even with differing
    page numbers — are dropped; real repeated sentences are kept."""

    def test_page_numbered_footer_removed(self) -> None:
        from manuscribe.pipeline.furniture import _strip_running_furniture

        parts = [
            "<p>Biotechnology and Applied Biochemistry 601</p>",
            "<p>Real body sentence one.</p>",
            "<p>Biotechnology and Applied Biochemistry 602</p>",
        ]
        out = _strip_running_furniture(_as_blocks(parts))
        assert out == ["<p>Real body sentence one.</p>"]

    def test_repeated_real_sentence_kept(self) -> None:
        from manuscribe.pipeline.furniture import _strip_running_furniture

        parts = ["<p>This is a sentence.</p>", "<p>This is a sentence.</p>"]
        assert _strip_running_furniture(_as_blocks(parts)) == parts

    def test_short_enumerated_labels_kept(self) -> None:
        # "Fig 1"/"Fig 2" share a digit-stripped key but must not be removed —
        # only substantial recurring text (a journal footer) is furniture.
        from manuscribe.pipeline.furniture import _strip_running_furniture

        parts = ["<p>Fig 1</p>", "<p>body</p>", "<p>Fig 2</p>"]
        assert _strip_running_furniture(_as_blocks(parts)) == parts

    def test_short_digit_free_footer_removed(self) -> None:
        # A bare author-surname running foot ("Clark" on alternating pages) is
        # short but digit-free, so the digit-strip collision the length floor
        # guards against can't happen — it must still be recognised as furniture.
        from manuscribe.pipeline.furniture import _strip_running_furniture

        parts = [
            "<p>Clark</p>",
            "<p>Real body sentence one.</p>",
            "<p>Clark</p>",
        ]
        assert _strip_running_furniture(_as_blocks(parts)) == [
            "<p>Real body sentence one.</p>"
        ]

    def test_heading_form_footer_removed(self) -> None:
        # OCR transcribes the running journal line as a heading on a sparse page
        # (last page / after references); it must still count as furniture and be
        # stripped, not survive as an <h1>.
        from manuscribe.pipeline.furniture import _strip_running_furniture

        parts = [
            "<p>Biotechnology and Applied Biochemistry 601</p>",
            "<p>Real body sentence one.</p>",
            "<h1>Biotechnology and Applied Biochemistry</h1>",
        ]
        out = _strip_running_furniture(_as_blocks(parts))
        assert out == ["<p>Real body sentence one.</p>"]

    def test_abbreviation_terminated_running_head_removed(self) -> None:
        # A running head ending in an abbreviation ("… Sphingomonas sp.") only
        # looks like a finished sentence; recurring on 3+ pages, it is furniture
        # and must be stripped — otherwise it interleaves between a paragraph's
        # halves and blocks their cross-page merge.
        from manuscribe.pipeline.furniture import _strip_running_furniture

        head = "<p>A ribitol dehydrogenase from <em>Sphingomonas</em> sp.</p>"
        parts = [head, "<p>Body one.</p>", head, "<p>Body two.</p>", head]
        body = ["<p>Body one.</p>", "<p>Body two.</p>"]
        assert _strip_running_furniture(_as_blocks(parts)) == body

    def test_twice_repeated_sentence_like_line_kept(self) -> None:
        # The same abbreviation-terminated line appearing only twice stays: two
        # occurrences are too few to outweigh its sentence-like shape, matching
        # the repeated-real-sentence guard.
        from manuscribe.pipeline.furniture import _strip_running_furniture

        head = "<p>A ribitol dehydrogenase from <em>Sphingomonas</em> sp.</p>"
        parts = [head, "<p>Body one.</p>", head]
        assert _strip_running_furniture(_as_blocks(parts)) == parts

    def test_heading_repeated_only_as_heading_kept(self) -> None:
        # A section heading the article legitimately repeats ("Purification of
        # SpRDH" under both Methods and Results) recurs but never appears as a
        # plain paragraph, so it is structure, not a running header, and must
        # survive in both places.
        from manuscribe.pipeline.furniture import _strip_running_furniture

        parts = [
            "<h3>Purification of SpRDH</h3>",
            "<p>Real body sentence one.</p>",
            "<h2>Purification of SpRDH</h2>",
        ]
        assert _strip_running_furniture(_as_blocks(parts)) == parts

    def test_verbatim_digit_citation_heading_removed(self) -> None:
        # A journal-citation running head ("… (2019) … BSR20190715") the OCR emits
        # only as a heading on several pages — its paragraph form differs (it also
        # carries a DOI line), so keys never match — is stripped because its
        # *verbatim* text recurs and carries digits.
        from manuscribe.pipeline.furniture import _strip_running_furniture

        cit = "<h2>Bioscience Reports (2019) 39 BSR20190715</h2>"
        parts = [cit, "<p>Body one.</p>", cit, "<p>Body two.</p>", cit]
        assert _strip_running_furniture(_as_blocks(parts)) == [
            "<p>Body one.</p>",
            "<p>Body two.</p>",
        ]

    def test_distinct_numbered_headings_kept(self) -> None:
        # Two distinct numbered headings ("Step 1: …" / "Step 2: …") collapse to one
        # digit-stripped key but their verbatim texts differ, so neither is a running
        # head; both must survive (they only appear as headings, never paragraphs).
        from manuscribe.pipeline.furniture import _strip_running_furniture

        parts = [
            "<h2>Step 1: Purification of Xylanase</h2>",
            "<p>Body one.</p>",
            "<h2>Step 2: Purification of Xylanase</h2>",
            "<p>Body two.</p>",
        ]
        assert _strip_running_furniture(_as_blocks(parts)) == parts

    def test_standalone_page_number_removed(self) -> None:
        # OCR sometimes isolates the folio into its own block, away from the
        # journal line, so digit-stripped recurrence can't catch it; a number-only
        # block is the page number itself and must be dropped.
        from manuscribe.pipeline.furniture import _strip_running_furniture

        parts = ["<p>601</p>", "<p>Real body sentence one.</p>", "<h2>602</h2>"]
        assert _strip_running_furniture(_as_blocks(parts)) == [
            "<p>Real body sentence one.</p>"
        ]

    def test_section_number_kept(self) -> None:
        # A numbered section heading ("3.4 …") is not a bare folio and stays.
        from manuscribe.pipeline.furniture import _strip_running_furniture

        parts = ["<h2>3.4 Enzymatic activities</h2>", "<p>4</p>"]
        assert _strip_running_furniture(_as_blocks(parts)) == [
            "<h2>3.4 Enzymatic activities</h2>"
        ]


class TestCaptureLicenseFooter:
    """A recurring copyright/open-access license footer the furniture strip drops is
    captured once for the Metadata panel; a one-off copyright is left alone."""

    _CC = (
        "<p>© 2019 The Author(s). This is an open access article published by "
        "Portland Press Limited and distributed under the Creative Commons "
        "Attribution License 4.0 (CC BY).</p>"
    )

    def test_recurring_license_captured_once(self) -> None:
        from manuscribe.pipeline.furniture import _capture_license_footer

        parts = [self._CC, "<p>Body prose.</p>", self._CC, self._CC]
        assert _capture_license_footer(_as_blocks(parts)) == self._CC

    def test_single_occurrence_not_captured(self) -> None:
        from manuscribe.pipeline.furniture import _capture_license_footer

        # one copy is not running furniture (the strip leaves it in the body), so it
        # must not be pulled — that would duplicate it into the panel
        assert _capture_license_footer(_as_blocks([self._CC, "<p>Body.</p>"])) is None

    def test_non_license_prose_ignored(self) -> None:
        from manuscribe.pipeline.furniture import _capture_license_footer

        # a "© 2019" mention without a license phrase is not a license footer
        parts = ["<p>© 2019 someone, all rights here.</p>"] * 2
        assert _capture_license_footer(_as_blocks(parts)) is None

    def test_recurring_license_relocated_to_panel_end_to_end(self) -> None:
        # the furniture strip drops the per-page repeats from the body (≥3 for a
        # sentence-like line); the captured copy must surface in the Metadata panel,
        # exactly once, not vanish entirely
        cc = (
            "© 2019 The Author(s). This is an open access article distributed under "
            "the Creative Commons Attribution License (CC BY)."
        )
        pages = [
            f"# T\n\n## Abstract\n\nA.\n\n## Body\n\nProse one.\n\n{cc}",
            f"Prose two continues here.\n\n{cc}",
            f"Prose three continues here as well.\n\n{cc}",
        ]
        html = _run_lighton(pages)
        assert "Creative Commons Attribution License" in _metadata(html)
        assert "Creative Commons Attribution License" not in _body(html)
        assert html.count("Creative Commons Attribution License") == 1


class TestByline:
    """The block after the title becomes the header byline only when it
    positively looks like authors; otherwise it stays in the body."""

    def test_marker_line_is_byline(self) -> None:
        from manuscribe.pipeline.classify import _is_byline

        assert _is_byline("Nianyang Wu¹") is True
        assert _is_byline("Daniel D. Clark <sup>*</sup>") is True

    def test_name_list_is_byline(self) -> None:
        from manuscribe.pipeline.classify import _is_byline

        assert _is_byline("Jane Doe and John Smith") is True

    def test_metadata_lines_are_not_byline(self) -> None:
        from manuscribe.pipeline.classify import _is_byline

        assert _is_byline("Received 26 March 2019") is False
        assert _is_byline("DOI: 10.1002/bab.1760") is False
        assert _is_byline("This is a complete sentence.") is False

    def test_unmarked_single_name_is_not_byline(self) -> None:
        # No marker and not a list → ambiguous → not promoted (stays in body).
        from manuscribe.pipeline.classify import _is_byline

        assert _is_byline("Jane Doe") is False

    def test_single_author_with_initial_is_byline(self) -> None:
        # A lone author with a mid-name initial ("Daniel D. Clark") carries a
        # positive name signal a subtitle never has, so it is promoted.
        from manuscribe.pipeline.classify import _is_byline

        assert _is_byline("Daniel D. Clark") is True
        # A title fragment / subtitle of capitalised words is still refused.
        assert _is_byline("A Case Study in Kinetics") is False
        assert _is_byline("Enzyme Kinetics") is False
        # An initialism-led phrase and an edge-only initial are not the given-name +
        # surname frame, so they stay in the body.
        assert _is_byline("U.S. Army Corps") is False
        assert _is_byline("D. Clark") is False

    def test_single_author_byline_promoted_end_to_end(self) -> None:
        md = (
            "# Article\n\n## The Real Title Here\n\nDaniel D. Clark\n\n"
            "### Abstract\n\nThe abstract paragraph.\n\n## Introduction\n\nProse."
        )
        html = _run_lighton([md])
        header = html[html.find("<header>") : html.find("</header>")]
        assert "Daniel D. Clark" in header
        assert "Daniel D. Clark" not in _body(html)

    def test_metadata_after_title_goes_to_metadata_panel(self) -> None:
        # A date line under the title is metadata: it must not be promoted into
        # the header (and must not be lost) — it belongs in the Metadata panel.
        md = (
            "# T\n\nReceived 26 March 2019\n\n## Abstract\n\nThe abstract.\n\n"
            "## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        header = html[html.find("<header>") : html.find("</header>")]
        assert "Received 26 March 2019" in _metadata(html)
        assert "Received 26 March 2019" not in header
        assert "Received 26 March 2019" not in _body(html)

    def test_marked_authors_after_title_are_promoted(self) -> None:
        md = "# T\n\nNianyang Wu¹, Xiaoqiang Liu¹*\n\n## Abstract\n\nA."
        html = _run_lighton([md])
        header = html[html.find("<header>") : html.find("</header>")]
        assert "Nianyang Wu" in header
        assert "Nianyang Wu" not in _body(html)

    def test_byline_superscript_markers_rendered_not_flattened(self) -> None:
        md = (
            "# T\n\nYan Zhou$^{1,*}$, Yifeng Wei$^{2,*}$, Huimin Zhao$^{2,3}$\n\n"
            "## Abstract\n\nA."
        )
        html = _run_lighton([md])
        header = html[html.find("<header>") : html.find("</header>")]
        # the '*' footnote markers survive (not eaten by the markdown emphasis
        # parser pairing the two adjacent asterisks) and stay superscripts
        assert "<sup>1,*</sup>" in header
        assert "<sup>2,*</sup>" in header
        # a multi-digit affiliation marker renders as a superscript, not flat text
        assert "<sup>2,3</sup>" in header
        # no stray emphasis injected, and no double-comma from a suppressed marker
        assert "<em>" not in header
        assert ",," not in header

    def test_byline_has_no_stray_or_misnested_markup(self) -> None:
        # PLOS prints the byline bold and tags corresponding authors with '*'; the
        # OCR emits the line bold-wrapped with bare-'*' markers, which markdown-it
        # mis-pairs into a stray '**' + spurious <em> (…ChangWoo Lee</em>**).  The
        # byline render strips the layout bold and re-casts the markers as <sup>.
        from manuscribe.pipeline.classify import _byline_html

        for inner in (
            # the clean OCR shape: bold-wrapped, markers escaped to literal '*'
            "<strong>Kiet N. Tran, Sei-Heon Jang*, ChangWoo Lee*</strong>",
            # the mis-paired OCR shape markdown-it produced from bare-'*' markers
            "<em><em>Kiet N. Tran, Sei-Heon Jang</em>, ChangWoo Lee</em>**",
        ):
            out = _byline_html(inner)
            assert "<em>" not in out
            assert "<strong>" not in out
            assert "**" not in out
            assert "ChangWoo Lee<sup>*</sup>" in out

    def test_byline_latex_superscript_markers_left_intact(self) -> None:
        # a byline whose markers already arrived as <sup> (the $^{1,*}$ LaTeX shape)
        # carries no bare '*', so the marker re-cast must not touch it
        from manuscribe.pipeline.classify import _byline_html

        inner = "Yan Zhou<sup>1,*</sup>, Yifeng Wei<sup>2,*</sup>"
        assert _byline_html(inner) == inner

    def test_byline_inline_double_marker_count_preserved(self) -> None:
        # An *inline* '**' is a genuine distinct marker (e.g. '*' vs '**' on different
        # authors) and its count is kept; only a *trailing* '**' is the unclosed-bold
        # mis-pair artifact and collapses to a single marker.
        from manuscribe.pipeline.classify import _byline_html

        assert _byline_html("Author A**, Author B*") == (
            "Author A<sup>**</sup>, Author B<sup>*</sup>"
        )
        assert _byline_html("Author A*, Author B**") == (
            "Author A<sup>*</sup>, Author B<sup>*</sup>"  # trailing ** = artifact
        )


class TestHeadingLevelNormalization:
    """Body section headings the OCR leveled inconsistently are re-leveled from
    high-confidence signals only (section numbering, canonical section names); every
    other heading keeps the OCR's level so a real section is never demoted."""

    def test_section_number_depth_sets_level(self) -> None:
        from manuscribe.pipeline.classify import _normalize_heading_levels

        # depth 1 -> h2, depth 2 -> h3, depth 3 -> h4
        body = [
            "<h3>2. Materials and Methods</h3>",  # over-nested by the OCR
            "<h2>2.1. Plant materials</h2>",  # under-nested by the OCR
            "<h2>2.1.1. Sampling</h2>",
        ]
        out = _normalize_heading_levels(body)
        assert out[0] == "<h2>2. Materials and Methods</h2>"
        assert out[1] == "<h3>2.1. Plant materials</h3>"
        assert out[2] == "<h4>2.1.1. Sampling</h4>"

    def test_sibling_subsection_jitter_fixed(self) -> None:
        # the 31051047 motivating bug: 3.4/3.5 emitted as <h2> beside 3.1-3.3 <h3>
        from manuscribe.pipeline.classify import _normalize_heading_levels

        body = ["<h3>3.3. Foo</h3>", "<h2>3.4. Bar</h2>", "<h2>3.5. Baz</h2>"]
        assert _normalize_heading_levels(body) == [
            "<h3>3.3. Foo</h3>",
            "<h3>3.4. Bar</h3>",
            "<h3>3.5. Baz</h3>",
        ]

    def test_canonical_section_name_anchored_to_h2(self) -> None:
        from manuscribe.pipeline.classify import _normalize_heading_levels

        body = [
            "<h3>Introduction</h3>",
            "<h1>References</h1>",
            "<h3>Materials and Methods</h3>",
        ]
        assert _normalize_heading_levels(body) == [
            "<h2>Introduction</h2>",
            "<h2>References</h2>",
            "<h2>Materials and Methods</h2>",
        ]

    def test_bare_imrad_names_not_promoted(self) -> None:
        # "Methods"/"Results"/"Discussion" can be real subsections (e.g. under a
        # combined "Results and Discussion", or a "Methods" subsection of "Study
        # Design"), so the bare single-word forms are NOT anchored to <h2> — only the
        # unambiguous compound forms are.  Guards against promoting a real subsection.
        from manuscribe.pipeline.classify import _normalize_heading_levels

        body = ["<h3>Methods</h3>", "<h3>Results</h3>", "<h3>Discussion</h3>"]
        assert _normalize_heading_levels(body) == body

    def test_unknown_heading_keeps_ocr_level(self) -> None:
        # an unnumbered, non-canonical heading (a journal-specific section, an
        # ambiguous back-matter name) is left at the OCR's level, never guessed at
        from manuscribe.pipeline.classify import _normalize_heading_levels

        body = [
            "<h2>Metal Binding Mode of CgKARI</h2>",
            "<h3>Author Contributions</h3>",  # subsection under "Author Information"
            "<h2>Case Study Description</h2>",
        ]
        assert _normalize_heading_levels(body) == body

    def test_year_like_number_not_read_as_section(self) -> None:
        from manuscribe.pipeline.classify import _normalize_heading_levels

        # "2019 …" must not read as section number 2019 and force <h2>
        assert _normalize_heading_levels(["<h3>2019 in Review</h3>"]) == [
            "<h3>2019 in Review</h3>"
        ]

    def test_leading_quantity_not_read_as_section_number(self) -> None:
        # A heading opening with a measurement ("0.5 M NaCl Wash", "5 mM Buffer") must
        # not read as a dotted section number and get re-leveled: the number lacks a
        # trailing separator and (for "0.5") starts with zero, so neither matches.
        from manuscribe.pipeline.classify import _normalize_heading_levels

        body = ["<h2>0.5 M NaCl Wash</h2>", "<h3>5 mM Sodium Phosphate Buffer</h3>"]
        assert _normalize_heading_levels(body) == body
        # a genuine dotted/paren section number (always with a trailing separator) is
        # still re-leveled by depth
        assert _normalize_heading_levels(["<h2>2.1. Plant materials</h2>"]) == [
            "<h3>2.1. Plant materials</h3>"
        ]
        assert _normalize_heading_levels(["<h3>(2) Methods</h3>"]) == [
            "<h2>(2) Methods</h2>"
        ]

    def test_non_heading_blocks_untouched(self) -> None:
        from manuscribe.pipeline.classify import _normalize_heading_levels

        body = ["<p>1. A numbered list item, not a heading.</p>", "<table></table>"]
        assert _normalize_heading_levels(body) == body


class TestDegenerateRepetition:
    """A figure the model fails to box can be OCRed into a repeated-token wall;
    such a paragraph is dropped from the body, real prose is kept."""

    def test_token_wall_flagged(self) -> None:
        from manuscribe.pipeline.furniture import _is_degenerate_repetition

        assert _is_degenerate_repetition("AaTRI, " * 40) is True

    def test_real_prose_not_flagged(self) -> None:
        from manuscribe.pipeline.furniture import _is_degenerate_repetition

        prose = (
            "The enzyme catalyzes the stereospecific oxidation of the substrate"
            " to the corresponding ketone under physiological conditions."
        )
        assert _is_degenerate_repetition(prose) is False

    def test_token_wall_dropped_from_body(self) -> None:
        wall = "AaTRI, " * 50
        md = f"# T\n\n## Abstract\n\nA.\n\n## Body\n\n{wall}\n\nReal sentence here."
        body = _body(_run_lighton([md]))
        assert "AaTRI" not in body
        assert "Real sentence here." in body


class TestLightonAssembly:
    """End-to-end markdown → HTML assembly (the new model-free seam)."""

    def test_title_skips_document_type_label(self) -> None:
        # "# Article" is a document-type label; the real title is the next heading.
        md = "# Article\n\n## The Real Title\n\nA. Author\n\n### Abstract\n\nText."
        html = _run_lighton([md])
        assert "<h1>The Real Title</h1>" in html
        assert "Article" not in _header_h1(html)

    def test_byline_extracted_and_dropped_from_body(self) -> None:
        md = "# A Study of Things\n\nJane Doe¹\n\n## Abstract\n\nThe abstract body."
        html = _run_lighton([md])
        header = html[html.find("<header>") : html.find("</header>")]
        assert "Jane Doe" in header
        assert "Jane Doe" not in _body(html)

    def test_abstract_wrapped_in_section(self) -> None:
        md = (
            "# T\n\nA. U.\n\n## Abstract\n\nThe abstract paragraph here.\n\n"
            "## Body\n\nProse."
        )
        html = _run_lighton([md])
        start = html.find("<section class='abstract'>")
        end = html.find("</section>", start)
        assert "The abstract paragraph here." in html[start:end]
        assert "Prose." not in html[start:end]

    def test_abstract_citation_tail_relocated_to_panel(self) -> None:
        # A copyright/journal-citation clause the OCR ran onto the abstract's end is
        # front matter: split off and surfaced in the Metadata panel, not the abstract.
        md = (
            "# T\n\n## Abstract\n\nThe abstract prose ends here. "
            "© 2018 International Union of Biochemistry, 47(2):124–132, 2019.\n\n"
            "## Introduction\n\nBody."
        )
        html = _run_lighton([md])
        start = html.find("<section class='abstract'>")
        abstract = html[start : html.find("</section>", start)]
        assert "The abstract prose ends here." in abstract
        assert "International Union" not in abstract
        assert "© 2018 International Union" in _metadata(html)

    def test_split_abstract_citation_pure(self) -> None:
        from manuscribe.pipeline.classify import _split_abstract_citation

        abstract = ["<p>Prose ends here. © 2019 A Publisher, 66(4):597–606, 2019</p>"]
        kept, tail = _split_abstract_citation(abstract)
        assert kept == ["<p>Prose ends here.</p>"]
        assert tail == ["<p>© 2019 A Publisher, 66(4):597–606, 2019</p>"]
        # an abstract without a copyright tail is returned unchanged
        plain = ["<p>Just abstract prose, no citation, mentioning 2019 in passing.</p>"]
        assert _split_abstract_citation(plain) == (plain, [])
        assert _split_abstract_citation([]) == ([], [])

    def test_split_abstract_citation_anchors_on_last_clause(self) -> None:
        from manuscribe.pipeline.classify import _split_abstract_citation

        # An in-abstract "© <year>" mention must be kept; only the trailing journal
        # citation is split off (the prose group is greedy, anchoring on the last ©).
        abstract = [
            "<p>An older work © 1998 noted X. The abstract continues here. "
            "© 2019 Publisher, 1(1):1–2, 2019</p>"
        ]
        kept, tail = _split_abstract_citation(abstract)
        assert "© 1998 noted X. The abstract continues here." in kept[0]
        assert tail == ["<p>© 2019 Publisher, 1(1):1–2, 2019</p>"]

    def test_split_abstract_citation_ignores_parenthetical_c_and_citation_only(
        self,
    ) -> None:
        from manuscribe.pipeline.classify import _split_abstract_citation

        # A bare "(c) <number>" is not a copyright sign — a quantity like "(c) 2000 mg"
        # must not be mistaken for a citation tail.
        quantity = ["<p>The molecule (c) 2000 mg was tested for activity.</p>"]
        assert _split_abstract_citation(quantity) == (quantity, [])
        # A citation-only paragraph is left intact (no empty <p></p> stub is produced).
        citation_only = ["<p>© 2018 Publisher, 1(1):1–2, 2018</p>"]
        assert _split_abstract_citation(citation_only) == (citation_only, [])

    def test_headingless_abstract_recovered_to_section(self) -> None:
        # A journal that prints the abstract with neither an "Abstract" heading nor
        # an inline bold label (Frontiers, Bioscience Reports): the leading prose run
        # before the first section heading is promoted to the abstract section.
        md = (
            "# A Study\n\nJane Doe¹\n\n"
            "This study reports a finding presented in a headingless abstract "
            "paragraph that precedes the first section heading.\n\n"
            "## Introduction\n\nThe body begins here."
        )
        html = _run_lighton([md])
        start = html.find("<section class='abstract'>")
        end = html.find("</section>", start)
        assert "This study reports a finding" in html[start:end]
        assert "This study reports a finding" not in _body(html)
        assert "<h2>Introduction</h2>" in _body(html)
        assert "The body begins here." in _body(html)

    def test_headingless_recovery_skipped_without_following_heading(self) -> None:
        # Never hide the whole body: with no section heading after the leading prose
        # (a short note that may carry no abstract) the recovery leaves it in place.
        from manuscribe.pipeline.classify import _recover_headingless_abstract

        body = [
            "<p>A lone substantial prose paragraph with no section heading after "
            "it, so it cannot be assumed to be an abstract.</p>",
            "<p>Another body paragraph, still no heading.</p>",
        ]
        abstract, rest = _recover_headingless_abstract(body)
        assert abstract == []
        assert rest == body

    def test_headingless_recovery_skips_existing_abstract_cue(self) -> None:
        # The recovery is a fallback: a cued abstract (heading or inline label) is
        # already classified, so the leading body run is left untouched.
        md = (
            "# T\n\n## Abstract\n\nThe cued abstract paragraph.\n\n"
            "## Introduction\n\nThe body begins here."
        )
        html = _run_lighton([md])
        start = html.find("<section class='abstract'>")
        end = html.find("</section>", start)
        assert "The cued abstract paragraph." in html[start:end]
        assert "The body begins here." in _body(html)

    def test_keywords_relocated_to_panel_colon_outside_bold(self) -> None:
        # OCR emits the keyword label with the colon *outside* the bold
        # ("**Keywords**:" → "<strong>Keywords</strong>:").  It terminates the
        # abstract, then is relocated from the body head to the Metadata panel — the
        # same destination the colon-inside shape already reaches.
        md = (
            "# T\n\n## Abstract\n\nThe abstract prose here.\n\n"
            "**Keywords**: alpha, beta, gamma\n\n"
            "## Introduction\n\nThe body begins here."
        )
        html = _run_lighton([md])
        start = html.find("<section class='abstract'>")
        abstract = html[start : html.find("</section>", start)]
        assert "alpha, beta, gamma" in _metadata(html)
        assert "alpha, beta, gamma" not in _body(html)
        assert "Keywords" not in abstract
        assert "The body begins here." in _body(html)

    def test_keywords_relocated_to_panel_colon_inside_bold(self) -> None:
        # The colon-inside shape ("**Keywords:**" → "<strong>Keywords:</strong>")
        # reaches the panel as before — the broadened capture must not regress it.
        md = (
            "# T\n\n## Abstract\n\nThe abstract prose here.\n\n"
            "**Keywords:** alpha, beta, gamma\n\n"
            "## Introduction\n\nThe body begins here."
        )
        html = _run_lighton([md])
        assert "alpha, beta, gamma" in _metadata(html)
        assert "alpha, beta, gamma" not in _body(html)

    def test_non_frontmatter_bold_label_kept_in_body(self) -> None:
        # The relocation is keyed on the label *name* (keywords/abbreviations/…), so
        # a body paragraph merely opening with an unrelated bold label stays in place.
        md = (
            "# T\n\n## Abstract\n\nThe abstract prose here.\n\n"
            "**Summary**: this leading clause is genuine body prose, not metadata.\n\n"
            "## Introduction\n\nThe body begins here."
        )
        html = _run_lighton([md])
        assert "this leading clause is genuine body prose" in _body(html)
        # Nothing was relocated, so no Metadata panel is created at all.
        assert "<details class='metadata'>" not in html

    def test_structured_abstract_labels_kept_in_abstract_with_heading(self) -> None:
        # A structured abstract's own section labels (Background/Methods/Results/
        # Conclusions) must not be mistaken for a front-matter label ("Keywords:")
        # that ends the abstract — all four paragraphs stay in the abstract.
        md = (
            "# T\n\n## Abstract\n\n"
            "**Background:** The background paragraph of the structured abstract.\n\n"
            "**Methods:** The methods paragraph of the structured abstract.\n\n"
            "**Results:** The results paragraph of the structured abstract.\n\n"
            "**Conclusions:** The conclusions paragraph of the structured abstract.\n\n"
            "## Introduction\n\nThe body begins here."
        )
        html = _run_lighton([md])
        start = html.find("<section class='abstract'>")
        abstract = html[start : html.find("</section>", start)]
        assert "The background paragraph of the structured abstract" in abstract
        assert "The conclusions paragraph of the structured abstract" in abstract
        assert "<details class='metadata'>" not in html
        assert "The body begins here." in _body(html)

    def test_structured_abstract_labels_kept_in_abstract_without_heading(self) -> None:
        # The same structured labels, with neither an "Abstract" heading nor an
        # inline "ABSTRACT:" label: the classifier leaves them atop the body, and
        # the headingless-abstract fallback must still recover them as the abstract
        # rather than the front-matter sweep hiding them in the Metadata panel.
        md = (
            "# A Study\n\n"
            "**Background:** This structured abstract's background paragraph is "
            "long enough to read as genuine prose, easily.\n\n"
            "**Methods:** This methods paragraph is also long enough to describe "
            "what was done in some detail.\n\n"
            "**Results:** This results paragraph reports the findings at "
            "sufficient length to count as real prose.\n\n"
            "**Conclusions:** This concluding paragraph draws out the "
            "implications at length, as real prose would.\n\n"
            "## Introduction\n\nThe body begins here."
        )
        html = _run_lighton([md])
        start = html.find("<section class='abstract'>")
        abstract = html[start : html.find("</section>", start)]
        assert "This structured abstract's background paragraph" in abstract
        assert "This concluding paragraph draws out the implications" in abstract
        assert "<details class='metadata'>" not in html
        assert "The body begins here." in _body(html)

    def test_structured_abstract_labels_kept_in_abstract_colon_outside_bold(
        self,
    ) -> None:
        # The colon-outside-bold shape ("**Background**:") must keep the abstract
        # open the same way the colon-inside shape does, rather than closing it
        # after the first label and leaving an empty abstract with loose paragraphs.
        md = (
            "# T\n\n## Abstract\n\n"
            "**Background**: The background paragraph, colon outside the bold.\n\n"
            "**Methods**: The methods paragraph, colon outside the bold.\n\n"
            "**Results**: The results paragraph, colon outside the bold.\n\n"
            "**Conclusions**: The conclusions paragraph, colon outside the bold.\n\n"
            "## Introduction\n\nThe body begins here."
        )
        html = _run_lighton([md])
        start = html.find("<section class='abstract'>")
        abstract = html[start : html.find("</section>", start)]
        assert "The background paragraph, colon outside the bold" in abstract
        assert "The conclusions paragraph, colon outside the bold" in abstract
        assert "<details class='metadata'>" not in html
        assert "The body begins here." in _body(html)

    def test_leading_superscript_routed_to_footnote_before_refs(self) -> None:
        md = (
            "# T\n\n## Abstract\n\nAbstract.\n\n## Body\n\n"
            "<sup>*</sup>To whom correspondence should be addressed.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        fn = html.find("To whom correspondence")
        ref = html.find("[1] A reference")
        assert 0 < fn < ref
        assert 'class="footnote"' in html

    def test_footnotes_anchor_on_references_heading_not_stray_bracket_one(self) -> None:
        # A "[1]"-led body paragraph (an inline numbered note) precedes the real
        # bibliography; footnotes must anchor on the References *heading*, not strand
        # mid-body before the stray "[1]".
        md = (
            "# T\n\n## Abstract\n\nAbstract.\n\n## Body\n\n"
            "[1] An inline numbered note in the body, not the bibliography.\n\n"
            "<sup>*</sup>To whom correspondence should be addressed.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        stray = html.find("An inline numbered note")
        fn = html.find("To whom correspondence")
        ref = html.find("[1] A reference")
        assert 0 < stray < fn < ref

    def test_footnotes_fall_back_to_bracket_one_when_headingless(self) -> None:
        # A heading-less numbered bibliography (no "References" heading, just "[1] …")
        # still anchors the footnotes before the bibliography via the looser fallback.
        md = (
            "# T\n\n## Abstract\n\nAbstract.\n\n## Body\n\nBody prose here.\n\n"
            "<sup>*</sup>To whom correspondence should be addressed.\n\n"
            "[1] A reference with no References heading."
        )
        html = _run_lighton([md])
        fn = html.find("To whom correspondence")
        ref = html.find("[1] A reference with no")
        assert 0 < fn < ref
        assert 'class="footnote"' in html

    def test_body_h1_demoted_to_h2(self) -> None:
        # Only the title is a legitimate <h1>; a body section heading the model
        # mis-levelled as <h1> is demoted so the document has one top-level head.
        md = (
            "# The Real Title\n\nA. U.\n\n## Abstract\n\nAbstract.\n\n"
            "## Introduction\n\nProse.\n\n"
            "# Molecular Mass Determination\n\nMore prose."
        )
        html = _run_lighton([md])
        assert "<h1>The Real Title</h1>" in html
        assert "<h2>Molecular Mass Determination</h2>" in _body(html)
        assert "<h1>Molecular Mass Determination</h1>" not in html

    def test_unicode_superscript_footnote_routed_and_not_glued(self) -> None:
        # A page footnote the model emits as a raw unicode superscript ("¹http://…")
        # mid-body is routed to the footnote run, not glued into the split sentence
        # it interrupts; the two prose halves merge cleanly.
        md = (
            "# T\n\n## Abstract\n\nAbstract.\n\n## Introduction\n\n"
            "The native enzyme was determined by gel filtration\n\n"
            "¹http://example.org/tool/home.htm\n\n"
            "chromatography using a Sephacryl column.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        assert "gel filtration chromatography using a Sephacryl column." in _body(html)
        fn = html.find("example.org/tool")
        ref = html.find("[1] A reference")
        assert 0 < fn < ref
        assert '<p class="footnote">¹http://example.org/tool/home.htm</p>' in html

    def test_unicode_superscript_table_note_stays_in_body(self) -> None:
        # An asterisk-marked table note is not a numbered page footnote; it must
        # stay with its table, not get pulled into the article footnote run.
        md = (
            "# T\n\n## Abstract\n\nAbstract.\n\n## Results\n\n"
            "<table><tbody><tr><td>Mg</td><td>3.0</td></tr></tbody></table>\n\n"
            "*Each value represents the mean of three measurements.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        assert "Each value represents the mean" in _body(html)
        assert 'class="footnote"' not in html

    def test_isotope_led_body_paragraph_not_routed_to_footnotes(self) -> None:
        # A body paragraph opening with an isotope/mass-number superscript ("²H NMR",
        # "³⁵S-labeled") is prose, not a footnote marker, so it stays in the body.
        for opener in (
            "²H NMR spectroscopy revealed a singlet at 4 ppm.",
            "³⁵S-labeled methionine was added to the medium.",
        ):
            md = (
                "# T\n\n## Abstract\n\nAbstract.\n\n## Introduction\n\n"
                f"{opener}\n\n## References\n\n[1] A reference."
            )
            html = _run_lighton([md])
            assert opener in _body(html), opener
            assert f'<p class="footnote">{opener}' not in html

    def test_superscript_affiliation_after_body_heading_not_a_footnote(self) -> None:
        # A "¹ Department of …, Country" affiliation shares the leading-marker shape
        # but is front matter, not an article footnote — even when OCR ordering drops
        # it after the first body heading (so seen_body_heading is already set).
        from manuscribe.pipeline.classify import _classify_parts

        meta = _classify_parts(
            _as_blocks(
                [
                    "<h1>T</h1>",
                    "<p>A. Author</p>",
                    "<h2>Introduction</h2>",
                    "<p>¹ Department of Chemistry, Example University, Daejeon, "
                    "South Korea</p>",
                ]
            )
        )
        assert any("Department of Chemistry" in b for b in meta.body)
        assert not any("Department of Chemistry" in f for f in meta.footnotes)

    def test_frontmatter_moved_to_metadata_panel_after_abstract(self) -> None:
        # Affiliations, keywords and abbreviations are OCR'd between the abstract
        # and the body's first section; they are pulled into the collapsible
        # Metadata panel (after the abstract) so the body opens with prose.
        md = (
            "# A Study\n\nJane Doe¹\n\n"
            "¹Department of Examples, Example University\n\n"
            "## Abstract\n\nThe abstract.\n\n"
            "**Keywords:** alpha, beta, gamma\n\n"
            "## Abbreviations\n\nTRI, tropine reductase.\n\n"
            "## 1. Introduction\n\nThe study begins here.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        # The front matter is in the Metadata panel, not the body.
        assert "<summary>Metadata</summary>" in meta
        for fragment in (
            "Department of Examples",
            "Keywords:",
            "TRI, tropine reductase.",
        ):
            assert fragment in meta
            assert fragment not in body
        # The panel sits after the abstract and before the body.
        assert (
            html.find("</section>")
            < html.find("<details class='metadata'>")
            < html.find('<div class="body">')
        )
        # The body opens with prose (the Introduction), not metadata.
        assert "<h2>1. Introduction</h2>" in body

    def test_frontmatter_boundary_is_name_agnostic(self) -> None:
        # The boundary is the first non-metadata heading, not a literal
        # "Introduction" — a body opening with "Background" works the same.
        md = (
            "# A Study\n\n"
            "**Keywords:** alpha, beta\n\n"
            "## Background\n\nThe study begins here.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        assert "Keywords:" in _metadata(html)
        assert "Keywords:" not in _body(html)
        assert "<h2>Background</h2>" in _body(html)

    def test_keywords_after_headingless_abstract_relocated(self) -> None:
        # An abstract with no "Abstract" heading is recovered into the abstract
        # section (see test_headingless_abstract_recovered_to_section); the keyword
        # line right after it is still relocated to the panel (post-classify, before
        # the recovery), and never leaks into the body.
        md = (
            "# T\n\nWe report the discovery of an enzyme, described in this "
            "abstract which carries no heading and so remains in the body.\n\n"
            "**Keywords:** alpha, beta, gamma\n\n"
            "## Introduction\n\nThe body begins here."
        )
        html = _run_lighton([md])
        assert "Keywords:" in _metadata(html)
        assert "Keywords:" not in _body(html)
        start = html.find("<section class='abstract'>")
        end = html.find("</section>", start)
        assert "We report the discovery of an enzyme" in html[start:end]
        assert "We report the discovery of an enzyme" not in _body(html)

    def test_extract_front_matter_relocates_trailing_label(self) -> None:
        from manuscribe.pipeline.classify import _extract_front_matter

        body = [
            "<p>A long abstract prose paragraph that stays in the body proper.</p>",
            "<p><strong>Keywords:</strong> alpha, beta</p>",
            "<h2>Introduction</h2>",
            "<p>Body prose.</p>",
        ]
        front, rest = _extract_front_matter(body)
        assert front == ["<p><strong>Keywords:</strong> alpha, beta</p>"]
        assert rest[0].startswith("<p>A long abstract")
        assert "<h2>Introduction</h2>" in rest

    def test_back_matter_glossary_label_not_relocated(self) -> None:
        # The trailing relocation is scoped to the leading region before the first
        # section heading; a back-matter "**Abbreviations:**" glossary stays in the
        # body with its own heading rather than being yanked to the front panel.
        from manuscribe.pipeline.classify import _extract_front_matter

        body = [
            "<h2>Introduction</h2>",
            "<p>Real body prose paragraph.</p>",
            "<h2>Abbreviations</h2>",
            "<p><strong>Abbreviations:</strong> ACT, x; PQQ, y</p>",
        ]
        front, rest = _extract_front_matter(body)
        assert front == []
        assert any("<strong>Abbreviations:</strong>" in r for r in rest)

    def test_banner_hidden_publication_label_relocated(self) -> None:
        # A "**Citation:**" line the pre-classify sweep missed because a banner hid it
        # behind the leading <strong> anchor is relocated to the panel post-classify
        # (here it follows a headingless abstract, so it is not in the leading run).
        from manuscribe.pipeline.classify import _extract_front_matter

        body = [
            "<p>A long headingless abstract paragraph that stays in the body here.</p>",
            "<p><strong>Citation:</strong> Doe J (2020) Title. Journal 1:1</p>",
            "<h2>Introduction</h2>",
            "<p>Body prose.</p>",
        ]
        front, rest = _extract_front_matter(body)
        assert any("<strong>Citation:</strong>" in f for f in front)
        assert not any("<strong>Citation:</strong>" in r for r in rest)

    def test_inline_abstract_requires_colon(self) -> None:
        # A body paragraph merely opening with a bold word "Abstract" (no colon) must
        # not be captured as the abstract; both colon forms of a real label are.
        from manuscribe.pipeline.classify import _INLINE_ABSTRACT_RE

        assert not _INLINE_ABSTRACT_RE.match("<strong>Abstract</strong> reasoning here")
        assert _INLINE_ABSTRACT_RE.match("<strong>ABSTRACT:</strong> text")
        assert _INLINE_ABSTRACT_RE.match("<strong>ABSTRACT</strong>: text")

    def test_inline_abstract_captures_multiple_paragraphs(self) -> None:
        # An inline-labelled abstract spanning two paragraphs is fully captured; a
        # following bold label (colon inside or outside) ends it rather than being
        # absorbed as abstract prose.
        from manuscribe.pipeline.classify import _classify_parts

        meta = _classify_parts(
            _as_blocks(
                [
                    "<h1>T</h1>",
                    "<h2>X</h2>",
                    "<p><strong>ABSTRACT</strong>: First abstract paragraph.</p>",
                    "<p>Second abstract paragraph continues here.</p>",
                    "<p><strong>KEYWORDS</strong>: alpha, beta</p>",
                    "<h2>Introduction</h2>",
                    "<p>Body.</p>",
                ]
            )
        )
        assert len(meta.abstract) == 2
        assert not any("KEYWORDS" in a for a in meta.abstract)
        assert any("KEYWORDS" in b for b in meta.body)

    def test_inline_abstract_label_alone_emits_no_empty_paragraph(self) -> None:
        # The OCR sometimes emits the inline abstract label as its own block; the
        # remainder is empty.  The window must still open (the next block is the
        # abstract body) but no stray "<p></p>" leaks into the abstract section.
        from manuscribe.pipeline.classify import _classify_parts

        meta = _classify_parts(
            _as_blocks(
                [
                    "<h1>T</h1>",
                    "<h2>X</h2>",
                    "<p><strong>ABSTRACT:</strong></p>",
                    "<p>The abstract body follows on the next block.</p>",
                    "<h2>Introduction</h2>",
                    "<p>Body.</p>",
                ]
            )
        )
        assert "<p></p>" not in meta.abstract
        assert meta.abstract == ["<p>The abstract body follows on the next block.</p>"]
        assert any("Body." in b for b in meta.body)

    def test_frontmatter_unchanged_when_body_opens_with_section(self) -> None:
        # First block is already a body section heading → nothing precedes it →
        # order left intact.
        md = "# T\n\n## Abstract\n\nA.\n\n## Methods\n\nFirst.\n\nSecond."
        body = _body(_run_lighton([md]))
        assert body.find("First.") < body.find("Second.")

    def test_unlabeled_body_prose_not_relocated(self) -> None:
        # An article whose body opens with unlabelled prose (no "Introduction"
        # heading, first heading is "Methods") must not have that prose moved to
        # the end: only positively-recognised metadata is relocated.
        md = (
            "# T\n\nUnlabeled opening prose paragraph here.\n\n"
            "More opening prose follows.\n\n"
            "## Methods\n\nMethod text.\n\n## References\n\n[1] A reference."
        )
        body = _body(_run_lighton([md]))
        assert body.find("Unlabeled opening prose paragraph here.") < body.find(
            "<h2>Methods</h2>"
        )

    def test_prose_starting_with_metadata_keyword_not_hidden(self) -> None:
        # A leading body paragraph that merely begins with a metadata keyword
        # ("Published…") is a sentence, not a front-matter label — it must stay
        # visible in the body, not be hidden in the Metadata panel.  (A keywords
        # label closes the abstract so the prose is the next, leading body block.)
        md = (
            "# T\n\n## Abstract\n\nThe abstract.\n\n**Keywords:** alpha, beta\n\n"
            "Published studies have shown that the enzyme is active.\n\n"
            "## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        assert "Published studies have shown" in _body(html)
        assert "Published studies have shown" not in _metadata(html)

    def test_unheaded_prose_after_metadata_heading_not_hidden(self) -> None:
        # Sticky metadata-section capture must stop at a real prose paragraph, so
        # an unheaded opening section after "## Keywords" is not swallowed.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Keywords\n\nalpha, beta, gamma.\n\n"
            "The introduction begins here without its own heading and reads as"
            " ordinary prose.\n\n## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        assert "The introduction begins here" in _body(html)
        assert "The introduction begins here" not in _metadata(html)
        assert "alpha, beta, gamma." in _metadata(html)  # short keywords stay metadata

    def test_long_abbreviation_list_stays_in_panel(self) -> None:
        # A semicolon-separated abbreviation list ending in a period reads like a
        # sentence, but as the block directly under "## Abbreviations" it is the
        # heading's own content and belongs in the panel, not the body.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Abbreviations\n\n"
            "TAs, tropane alkaloids; TRI, tropine-forming reductase; "
            "GC-MS, gas chromatography-mass spectrometer.\n\n"
            "## 1. Introduction\n\nThe study begins here without abbreviation.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        assert "TAs, tropane alkaloids" in _metadata(html)
        assert "TAs, tropane alkaloids" not in _body(html)
        assert "<h2>1. Introduction</h2>" in _body(html)

    def test_multi_block_abbreviation_list_fully_in_panel(self) -> None:
        # A long abbreviation list OCR-split across two paragraphs: both halves
        # are ";"-separated lists, so the heading owns the whole run and neither
        # half leaks into the body (the second block is not heading-adjacent).
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Abbreviations\n\n"
            "TAs, tropane alkaloids; TRI, tropine-forming reductase; "
            "TRII, pseudotropine-forming reductase; qPCR, quantitative PCR.\n\n"
            "PGK, protein kinase; SDRs, short-chain dehydrogenases; "
            "GC-MS, gas chromatography-mass spectrometer; HPLC, chromatography.\n\n"
            "## 1. Introduction\n\nThe study begins here without abbreviation."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        for fragment in ("TAs, tropane alkaloids", "PGK, protein kinase"):
            assert fragment in meta
            assert fragment not in body
        assert "<h2>1. Introduction</h2>" in body

    def test_keyword_led_metadata_after_section_pulled_into_panel(self) -> None:
        # Correspondence / received / DOI lines OCR'd after the abbreviation list
        # are keyword-led metadata; inside the headed front-matter region they are
        # pulled into the panel even though each ends like a sentence.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Abbreviations\n\n"
            "TAs, tropane alkaloids; GC-MS, gas chromatography-mass spectrometer.\n\n"
            "Address for correspondence: Jane Doe, Example University, City. "
            "Tel. 123; e-mail: jane@example.edu.\n\n"
            "Received 26 March 2019; accepted 29 April 2019 DOI: 10.1002/bab.1760.\n\n"
            "## 1. Introduction\n\nThe study begins here without abbreviation."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        for fragment in ("Address for correspondence", "Received 26 March 2019"):
            assert fragment in meta
            assert fragment not in body
        assert "<h2>1. Introduction</h2>" in body

    def test_keyword_led_body_prose_in_section_not_hidden(self) -> None:
        # A body paragraph that merely opens with a front-matter keyword
        # ("Published …"), appearing inside a headed metadata section, must stay
        # in the body — it carries no metadata token (e-mail/DOI/date), unlike a
        # genuine "Received … DOI: …" line, so it is not hidden in the panel.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Abbreviations\n\n"
            "TAs, tropane alkaloids; GC-MS, gas chromatography-mass spectrometer.\n\n"
            "Published reports indicate that the enzyme is active across many "
            "plant species and remains the focus of ongoing research.\n\n"
            "## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        assert "Published reports indicate" in _body(html)
        assert "Published reports indicate" not in _metadata(html)

    def test_midbody_abbreviations_section_pulled_into_panel(self) -> None:
        # OCR places the Abbreviations section *after* the Introduction prose on
        # the first page, out of the leading front-matter run.  Its heading and
        # ";"-separated glossary are pulled into the panel; the misfiled body
        # content that follows (a learning-objectives list, the next section) is
        # left visible.
        md = (
            "# A Study\n\n## Abstract\n\nThe abstract.\n\n"
            "## Introduction\n\nThe study begins here with ordinary prose.\n\n"
            "## Abbreviations\n\n"
            "2-HEC, 2-(2-hydroxyethylthio)ethanesulfonate; CoM, coenzyme M; "
            "EE, enantiomeric excess; HPC, hydroxypropyl thioether.\n\n"
            "1. Students improve their appreciation for kinetic data.\n\n"
            "## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        assert "<h2>Abbreviations</h2>" in meta
        assert "2-HEC, 2-(2-hydroxyethylthio)ethanesulfonate" in meta
        assert "2-HEC, 2-(2-hydroxyethylthio)ethanesulfonate" not in body
        # The Introduction prose and the misfiled learning-objective stay visible.
        assert "The study begins here" in body
        assert "Students improve their appreciation" in body
        assert "<h2>Methods</h2>" in body

    def test_midbody_named_metadata_scoped_to_first_page(self) -> None:
        # A same-named section deeper in the document (here on a later page, e.g. a
        # back-matter glossary) is left in the body — only the first article page
        # is scanned for misplaced metadata sections.
        page1 = "# A Study\n\n## Abstract\n\nThe abstract.\n\n## Introduction\n\nProse."
        page2 = (
            "## Nomenclature\n\n"
            "F, force in newtons; m, mass in kilograms; a, acceleration.\n\n"
            "## References\n\n[1] A reference."
        )
        body = _body(_run_lighton([page1, page2]))
        assert "<h2>Nomenclature</h2>" in body
        assert "F, force in newtons" in body

    def test_midbody_bare_named_heading_stays_in_body(self) -> None:
        # A "Nomenclature" heading that opens a real prose section (no glossary
        # content under it) is a section title, not front matter: leave it visible.
        md = (
            "# A Study\n\n## Abstract\n\nThe abstract.\n\n"
            "## Introduction\n\nProse opening the article body.\n\n"
            "## Nomenclature\n\n"
            "This section explains the naming conventions used throughout the "
            "paper in ordinary prose that reads as a real section.\n\n"
            "## Methods\n\nMethod text."
        )
        body = _body(_run_lighton([md]))
        assert "<h2>Nomenclature</h2>" in body
        assert "This section explains the naming conventions" in body

    def test_stray_footer_metadata_pulled_into_panel(self) -> None:
        # The first page's bottom-of-page footer (journal citation + correspondence,
        # a supporting-info note, a submission/DOI line) is OCR'd into the body
        # after the Introduction.  Each self-contained metadata line is relocated to
        # the panel; the body prose that follows stays visible.
        md = (
            "# A Study\n\n## Abstract\n\nThe abstract.\n\n"
            "## Introduction\n\nIntro paragraph one is reasonably long prose here.\n\n"
            "Volume 47, Number 2, March/April 2019, Pages 124-132 *To whom "
            "correspondence should be addressed. Daniel D. Clark, Tel.: "
            "(530)-898-5251. E-mail: ddclark@csuchico.edu.\n\n"
            "Additional Supporting Information may be found in the online version "
            "of this article.\n\n"
            "Received 19 June 2018; Revised 23 August 2018; Accepted 6 December "
            "2018 DOI 10.1002/bmb.21202 Published online 28 December 2018 in Wiley "
            "Online Library (wileyonlinelibrary.com)\n\n"
            "Herein, I propose that data from the characterization can augment the "
            "teaching of enzyme kinetics. The case study had five goals in mind.\n\n"
            "## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        for fragment in (
            "Volume 47",
            "Additional Supporting Information",
            "Received 19 June",
        ):
            assert fragment in meta
            assert fragment not in body
        # Running before the merge keeps the ")" -terminated footer line from
        # absorbing the body prose that follows it.
        assert "Herein, I propose" in body
        assert "Herein, I propose" not in meta

    def test_interleaved_publication_sidebar_pulled_into_panel(self) -> None:
        # 32639976 (PLOS ONE) page 0: the left-column metadata sidebar — a
        # postal-code-less affiliation plus a run of bold label-colon lines
        # (Citation/Editor/.../Competing interests) — is OCR'd into the body, the
        # affiliation as the body's first block and the labelled run stranded after
        # the Introduction heading and a column-break rule.  Every piece belongs in
        # the panel; the Introduction prose stays in the body.
        md = (
            "# Purification of a novel ribitol dehydrogenase\n\n"
            "**Kiet N. Tran, Nhung Pham, Sei-Heon Jang\\*, ChangWoo Lee\\***\n\n"
            "*Department of Biomedical Science and Center for Bio-Nanomaterials, "
            "Daegu University, Gyeongsan, South Korea*\n\n"
            "## Abstract\n\nThe abstract sentence is here.\n\n"
            "## Introduction\n\n"
            "Lichens have traditionally been considered a symbiotic association.\n\n"
            "---\n\n"
            "**Citation:** Tran KN, Lee C (2020) Purification. PLoS ONE 15(7): "
            "e0235718. https://doi.org/10.1371/journal.pone.0235718\n\n"
            "**Editor:** Leonidas Matsakas, Luleå University of Technology, SWEDEN\n\n"
            "**Received:** April 23, 2020\n\n"
            "**Copyright:** © 2020 Tran et al. This is an open access article.\n\n"
            "**Funding:** This work was supported by the NRF.\n\n"
            "**Competing interests:** The authors have declared that no competing "
            "interests exist.\n\n"
            "Polyols have a role in carbohydrate storage and stress protection."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        for fragment in (
            "Department of Biomedical Science",
            "<strong>Citation:</strong>",
            "<strong>Editor:</strong>",
            "<strong>Received:</strong>",
            "<strong>Copyright:</strong>",
            "<strong>Funding:</strong>",
            "<strong>Competing interests:</strong>",
        ):
            assert fragment in meta, fragment
            assert fragment not in body, fragment
        # The body keeps its prose and the section heading, with no stray rule.
        assert "Lichens have traditionally been considered" in body
        assert "Polyols have a role" in body
        assert "<hr" not in body

    def test_open_access_banner_does_not_swallow_following_body_prose(self) -> None:
        # "OPEN ACCESS" is a bare banner heading, not a label:value pair like
        # "Citation"/"Editor": it must relocate on its own and must NOT claim the
        # paragraph directly under it.  A body paragraph stranded right after a
        # mislaid banner stays visible in the body, never hidden in the panel.
        md = (
            "# A Real Article Title\n\n"
            "**Jane Doe¹**\n\n"
            "¹ Department of Biology, Some University, Seoul, South Korea\n\n"
            "## Abstract\n\nThe abstract sentence is here.\n\n"
            "## Introduction\n\n"
            "The first introduction paragraph establishes the study's background.\n\n"
            "## OPEN ACCESS\n\n"
            "This is genuine body prose that follows the stranded banner and must "
            "remain visible in the body, not vanish into the collapsed panel.\n\n"
            "## Methods\n\nMethod text."
        )
        meta, body = _metadata(_run_lighton([md])), _body(_run_lighton([md]))
        assert "genuine body prose" in body
        assert "genuine body prose" not in meta
        # The banner itself is still relocated, not left as a body heading.
        assert "OPEN ACCESS" in meta
        assert "OPEN ACCESS" not in body

    def test_frontiers_open_access_sidebar_pulled_into_panel(self) -> None:
        # 32117944 (Frontiers) page 0: the first-page sidebar opens with an
        # "OPEN ACCESS" banner heading and carries a "Specialty section:" routing
        # line after the Edited by / Reviewed by / Correspondence / Citation run.
        # The banner heading broke the leading front-matter run, stranding it and
        # the specialty line in the body; both belong in the panel.
        # The abstract carries no "Abstract" heading and directly follows a
        # multi-superscript affiliation run that ends "…South Korea" with no
        # terminal punctuation: the merge must not glue the abstract onto the
        # affiliation (which, opening with "¹", would then hide both in the panel).
        md = (
            "# Discovery of a Methanol Dehydrogenase\n\n"
            "**Jin-Young Lee¹ and Seung-Goo Lee\\***\n\n"
            "¹ Synthetic Biology Research Center, KRIBB, Daejeon, South Korea,"
            "² Department of Biosystems and Bioengineering, University of Science "
            "and Technology, Daejeon, South Korea,³ School of Biological Sciences "
            "and Technology, Chonnam National University, Gwangju, South Korea\n\n"
            "Bioconversion of C1 chemicals such as methane and methanol into higher "
            "carbon-chain chemicals has been widely studied in recent years.\n\n"
            "**Keywords:** methanol dehydrogenase, methylotrophy\n\n"
            "**Edited by:**\nDong-Yup Lee, Sungkyunkwan University, South Korea\n\n"
            "**Citation:**\nLee J-Y (2020) Discovery. Front. Bioeng. Biotechnol. "
            "8:67. doi: 10.3389/fbioe.2020.00067\n\n"
            "## OPEN ACCESS\n\n"
            "**Specialty section:**\nThis article was submitted to Synthetic "
            "Biology, a section of the journal Frontiers in Bioengineering and "
            "Biotechnology\n\n"
            "## INTRODUCTION\n\n"
            "In this regard, Mdh is a crucial enzyme for\n\n"
            "**Abbreviations:** ACT, endogenous activator protein; Mdh, methanol "
            "dehydrogenase; PQQ, pyrroloquinoline quinone.\n\n"
            "bioconversion of valuable multi-carbon chemicals from C1 chemicals."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        for fragment in (
            "Synthetic Biology Research Center",
            "OPEN ACCESS",
            "Specialty section",
            "submitted to Synthetic",
            "endogenous activator protein",
        ):
            assert fragment in meta, fragment
            assert fragment not in body, fragment
        # The abstract is recovered into the abstract section, not glued onto the
        # affiliation and hidden in the panel.
        start = html.find("<section class='abstract'>")
        end = html.find("</section>", start)
        assert "Bioconversion of C1 chemicals" in html[start:end]
        assert "Bioconversion of C1 chemicals" not in meta
        assert "Bioconversion of C1 chemicals" not in body
        # The glossary footnote is pulled out before the merge, so the paragraph it
        # split rejoins as one block in the body.
        assert (
            "crucial enzyme for bioconversion of valuable multi-carbon chemicals"
            in body
        )

    def test_body_sentence_with_one_email_not_hidden(self) -> None:
        # The stray-metadata sweep must not hide a body sentence that merely embeds
        # a single address: one token is below the threshold.
        md = (
            "# A Study\n\n## Abstract\n\nThe abstract.\n\n"
            "## Introduction\n\n"
            "Raw data are available on request from the author at "
            "ddclark@csuchico.edu.\n\n"
            "## Methods\n\nMethod text."
        )
        body = _body(_run_lighton([md]))
        assert "Raw data are available on request" in body

    def test_metadata_run_ends_at_body_prose_with_trailing_citation(self) -> None:
        from manuscribe.pipeline.classify import (
            _front_matter_len,
            _looks_like_body_prose,
        )

        # A body paragraph whose terminal period sits behind a trailing citation
        # superscript still ends a metadata section's sticky run — the period must
        # not be hidden by the <sup>, or the paragraph (and the body after it) is
        # swept into the hidden Metadata panel (the never-hide-body invariant).
        prose = (
            "<p>This substantial body sentence describes the experimental method "
            "and its results in considerable detail here.<sup>15</sup></p>"
        )
        assert _looks_like_body_prose(prose) is True
        body = [
            "<h2>Keywords</h2>",
            "<p>enzyme; catalysis; crystal structure; cofactor</p>",
            prose,
            "<p>Another body paragraph follows here.</p>",
        ]
        # Only the heading + the keyword list are front matter; the citation-ended
        # body prose ends the run rather than being counted into it.
        assert _front_matter_len(body) == 2

    def test_keyword_led_body_with_citation_not_hidden_as_frontmatter(self) -> None:
        from manuscribe.pipeline.classify import _front_matter_len, _is_frontmatter_text

        # A body paragraph that merely opens with a front-matter keyword ("Published
        # …") but ends in a citation superscript must not be mistaken for a metadata
        # label: _is_frontmatter_text reads the period past the <sup>, so the body
        # run still breaks instead of the paragraph (and the body after it) being
        # swept into the hidden panel — otherwise it defeats the body-prose guard.
        prose = (
            "<p>Published reports indicate that this compound is harmless to all "
            "humans studied to date.<sup>15</sup></p>"
        )
        assert _is_frontmatter_text(prose, strict=False) is False
        body = [
            "<h2>Keywords</h2>",
            "<p>enzyme; catalysis; crystal structure; cofactor</p>",
            prose,
            "<p>More body text here.</p>",
        ]
        assert _front_matter_len(body) == 2

    def test_ends_like_sentence_sees_past_trailing_citation(self) -> None:
        from manuscribe.pipeline.furniture import _ends_like_sentence

        # _ends_like_sentence (used for running-furniture detection) must also look
        # past a trailing citation superscript, or a recurring line ending in one is
        # judged non-sentence-like and dropped at a lower repeat threshold.
        (block,) = _as_blocks(["<p>A recurring header line here.<sup>3</sup></p>"])
        assert _ends_like_sentence(block)

    def test_equal_contribution_marker_derived_from_byline(self) -> None:
        from manuscribe.pipeline.classify import _byline_equal_contribution_marker

        # The marker is the footnote symbol the byline puts on ≥2 authors.
        byline = (
            "<p>Yan Zhou<sup>1,*</sup>, Yifeng Wei<sup>2,*</sup>, Lin<sup>1</sup></p>"
        )
        assert _byline_equal_contribution_marker([byline]) == "*"
        # A different symbol is derived, not a hard-coded "*".
        dagger = "<p>A. One<sup>1,†</sup>, B. Two<sup>2,†</sup></p>"
        assert _byline_equal_contribution_marker([dagger]) == "†"
        # A symbol on only one author (e.g. a lone corresponding-author mark) is not
        # an equal-contribution marker, and numeric affiliation markers never count.
        single = "<p>A. One<sup>1,*</sup>, B. Two<sup>2</sup>, C. Three<sup>3</sup></p>"
        assert _byline_equal_contribution_marker([single]) is None

    def test_equal_contribution_footnote_marker_restored(self) -> None:
        from manuscribe.pipeline.classify import _restore_equal_contribution_marker

        # The OCR swallows the footnote's marker into emphasis ("*…*" -> <em>);
        # restore the byline-derived marker so the note still references its authors.
        assert (
            _restore_equal_contribution_marker(
                "<p><em>These authors contributed equally to this work.</em></p>", "*"
            )
            == "<p>*<em>These authors contributed equally to this work.</em></p>"
        )
        # A note that already carries a footnote marker is left untouched.
        assert (
            _restore_equal_contribution_marker(
                "<p>† These authors contributed equally to this work.</p>", "*"
            )
            == "<p>† These authors contributed equally to this work.</p>"
        )
        # A block that only *mentions* the phrase mid-text (relocated for another
        # reason) is not a standalone note → no marker is prepended at its start.
        merged = (
            "<p>Correspondence: a@b.edu. These authors contributed equally. Tel: 1.</p>"
        )
        assert _restore_equal_contribution_marker(merged, "*") == merged
        # No derivable marker → nothing invented.
        assert (
            _restore_equal_contribution_marker(
                "<p><em>These authors contributed equally.</em></p>", None
            )
            == "<p><em>These authors contributed equally.</em></p>"
        )
        # A non-footnote metadata block is never given a spurious marker.
        assert (
            _restore_equal_contribution_marker("<p>DOI 10.1042/BSR20190715</p>", "*")
            == "<p>DOI 10.1042/BSR20190715</p>"
        )

    def test_stray_metadata_predicate(self) -> None:
        from manuscribe.pipeline.classify import _is_stray_metadata

        # Two tokens (tel + e-mail) → relocated.
        assert _is_stray_metadata(
            "<p>*To whom correspondence should be addressed. Tel.: (530)-898-5251. "
            "E-mail: ddclark@csuchico.edu.</p>"
        )
        # Boilerplate phrase, no token → relocated.
        assert _is_stray_metadata(
            "<p>Additional Supporting Information may be found in the online "
            "version of this article.</p>"
        )
        # Single-token publication lines the OCR splits off the page-bottom block,
        # each unambiguous on its own shape → relocated.
        assert _is_stray_metadata(
            "<p>Volume 47, Number 2, March/April 2019, Pages 124-132</p>"
        )
        assert _is_stray_metadata("<p>DOI 10.1002/bmb.21202</p>")
        assert _is_stray_metadata(
            "<p>Published online 28 December 2018 in Wiley Online Library "
            "(wileyonlinelibrary.com)</p>"
        )
        # A submission-history footer — a publishing-process label immediately
        # followed by a date — is relocated even when unbolded, in US "Month DD,
        # YYYY" order (ACS) or day-first order, whether OCR keeps the four lines
        # glued into one block (markdown-it emits each hard break as "<br />\n", so
        # the dates end on a newline), splits each onto its own line, or runs the
        # submission history on one ";"-separated line.
        assert _is_stray_metadata(
            "<p>Received: May 25, 2019<br />\nRevised: July 10, 2019<br />\n"
            "Accepted: July 12, 2019<br />\nPublished: July 12, 2019</p>"
        )
        assert _is_stray_metadata("<p>Received: May 25, 2019</p>")
        assert _is_stray_metadata("<p>Accepted 12 July 2019</p>")
        assert _is_stray_metadata(
            "<p>Received 19 June 2018; Revised 23 August 2018; "
            "Accepted 6 December 2018</p>"
        )
        # A body sentence merely opening with the keyword and running the date on
        # into prose (no line/entry break after it) is not a footer and stays.
        assert not _is_stray_metadata(
            "<p>Published reports indicate the enzyme is highly conserved.</p>"
        )
        assert not _is_stray_metadata(
            "<p>Published May 25, 2019 in a leading journal, the work spread.</p>"
        )
        # Body prose citing a date range must not be swept into the panel on its
        # dates alone — a metadata block needs a non-date token (e-mail/DOI/phone).
        assert not _is_stray_metadata(
            "<p>Patients enrolled between March 3, 2001 and December 12, 2004 "
            "were followed for five years.</p>"
        )
        # A word that merely opens with a month prefix ("decided", "Mayor") is not a
        # month, so it cannot form a spurious date token.
        assert not _is_stray_metadata(
            "<p>The board decided 5 2019 was the augment 3 2020 target year.</p>"
        )
        # An author-contribution footnote ("These authors contributed equally …")
        # the OCR stranded among body paragraphs is relocated — the clause closing
        # the block (bare, or trailing "to this work/study/…") is the discriminator.
        assert _is_stray_metadata(
            "<p><em>These authors contributed equally to this work.</em></p>"
        )
        assert _is_stray_metadata("<p>D.L. and J.H. contributed equally.</p>")
        assert _is_stray_metadata("<p>All authors contributed equally to the study</p>")
        # Body prose that runs the phrase mid-sentence onto a non-publication object
        # is not a footnote and stays in the body.
        assert not _is_stray_metadata(
            "<p>The two catalytic domains contributed equally to substrate "
            "binding across the assayed pH range.</p>"
        )
        assert not _is_stray_metadata(
            "<p>Both pathways contributed equally to this increase in metabolic "
            "flux under anaerobic conditions.</p>"
        )
        # A recognised journal-metadata bold label is relocated on the label alone,
        # even with no metadata token and even when the value runs long.
        assert _is_stray_metadata(
            "<p><strong>Competing interests:</strong> The authors have declared "
            "that no competing interests exist.</p>"
        )
        assert _is_stray_metadata(
            "<p><strong>Editor:</strong> Leonidas Matsakas, Luleå University of "
            "Technology, SWEDEN</p>"
        )
        assert _is_stray_metadata(
            "<p><strong>Funding:</strong> "
            + "This work was generously supported by many sources. " * 12
            + "</p>"
        )
        # An inline glossary footnote ("Abbreviations:"/"Nomenclature:") the OCR
        # drops mid-section is relocated on the label alone, regardless of how long
        # the entry list runs, so the paragraph it split rejoins in the body.
        assert _is_stray_metadata(
            "<p><strong>Abbreviations:</strong> ACT, endogenous activator protein; "
            "IMAC, immobilized metal affinity chromatography; Mdh, methanol "
            "dehydrogenase; PQQ, pyrroloquinoline quinone.</p>"
        )
        assert _is_stray_metadata(
            "<p><strong>Nomenclature:</strong> k, rate constant; T, temperature.</p>"
        )
        # "Keywords:" is excluded — it sits after the abstract and does not split
        # prose, so it stays where it is rather than being swept into the panel.
        assert not _is_stray_metadata(
            "<p><strong>Keywords:</strong> methanol dehydrogenase, methylotrophy</p>"
        )
        # A bold label-colon that is *not* a publishing-process label is ordinary
        # body emphasis, not metadata, and must not be relocated on the shape alone.
        assert not _is_stray_metadata(
            "<p><strong>Note:</strong> the reaction was repeated three times.</p>"
        )
        assert not _is_stray_metadata(
            "<p><strong>Results:</strong> the enzyme retained full activity.</p>"
        )
        # A single embedded e-mail is below the two-token bar → stays in body.
        assert not _is_stray_metadata(
            "<p>Raw data are available from the author at jane@example.edu.</p>"
        )
        # Prose that merely uses the words "volume"/"pages" or "published online"
        # — not the citation/publication shapes — stays in body.
        assert not _is_stray_metadata(
            "<p>The dataset volume reached 47 GB after we processed pages from "
            "124 to 132 of the log.</p>"
        )
        assert not _is_stray_metadata(
            "<p>These results were later published online for peer review.</p>"
        )
        # A body sentence that inline-cites a volume/pages reference is prose, not
        # a citation block: the journal shape is anchored at the block start, so
        # "Volume 47 … Pages 124" matches but a mid-sentence citation does not.
        assert not _is_stray_metadata(
            "<p>See volume 3, pages 45-67, for the original derivation.</p>"
        )
        assert not _is_stray_metadata(
            "<p>In volume 12, pages 8-9 of the proceedings, the method appeared.</p>"
        )
        assert not _is_stray_metadata(
            "<p>As shown in Vol. 5 pp. 30 onward, the trend continues.</p>"
        )
        # A long prose run with two tokens is not a footer line → stays in body.
        assert not _is_stray_metadata(
            "<p>" + "Lorem ipsum dolor sit amet. " * 20 + "Contact a@b.edu or "
            "c@d.edu.</p>"
        )

    def test_stray_metadata_position_independent_false_positives(self) -> None:
        # Three shapes a page-0, position-independent sweep can mistake for
        # footer/affiliation/footnote metadata, each a normal Methods/Results
        # sentence that must stay in the body.
        from manuscribe.pipeline.classify import _is_stray_metadata

        # A URL and a date both clear the two-token bar, but neither is the
        # e-mail/DOI/phone shape a genuine footer carries — a Methods sentence
        # citing a database URL and a download date is not metadata.
        assert not _is_stray_metadata(
            "<p>Genome sequences were downloaded from "
            "https://www.ncbi.nlm.nih.gov/genome on March 3, 2019.</p>"
        )
        # A column-split fragment ending "…, City, City2, Country" has no
        # institution word, but it is still a finite-verb clause ("were collected
        # from farms near"), not an address noun phrase.
        assert not _is_stray_metadata(
            "<p>Soil samples were collected from farms near Lahore, Faisalabad, "
            "Multan, Pakistan</p>"
        )
        # The clause closes the block exactly like a genuine equal-contribution
        # footnote, but its subject ("both substitutions") does not identify the
        # authors.
        assert not _is_stray_metadata(
            "<p>In the double mutant, both substitutions contributed equally.</p>"
        )
        # The two known-correct equal-contribution notes (unmarked, author-
        # identifying subject) must still be relocated — not weakened by the fix
        # above.
        assert _is_stray_metadata("<p>D.L. and J.H. contributed equally.</p>")
        assert _is_stray_metadata("<p>All authors contributed equally to the study</p>")

    def test_bare_affiliation_line_pulled_into_panel(self) -> None:
        # An author+affiliation line OCR'd between the title and the abstract,
        # without its author's superscript marker ("Name From the Department …,
        # City, Region, postcode"), is recognised structurally and pulled into the
        # panel, so it no longer breaks the leading run and strands the keywords.
        md = (
            "# A Study\n\n"
            "Daniel D. Clark From the Department of Chemistry and Biochemistry, "
            "California State University-Chico, Chico, California, 95929\n\n"
            "## Abstract\n\nThe abstract.\n\n"
            "**Keywords:** enzymology; enzyme kinetics; dehydrogenase\n\n"
            "## Introduction\n\nThe study begins here with ordinary prose.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        for fragment in ("From the Department of Chemistry", "Keywords:"):
            assert fragment in meta
            assert fragment not in body
        assert "The study begins here" in body
        assert "<h2>Introduction</h2>" in body

    def test_body_sentence_mentioning_university_not_hidden(self) -> None:
        # The affiliation detector must not hide a body sentence that merely names
        # an institution: a terminal period marks it as prose, not an address.
        # (Placed after a real abstract+heading so the headingless-abstract recovery
        # doesn't claim it — the recovery only promotes the leading pre-heading run.)
        md = (
            "# A Study\n\n## Abstract\n\nThe abstract.\n\n## Methods\n\n"
            "The work was carried out with the University of Example, the "
            "Department of Chemistry, and several partners."
        )
        body = _body(_run_lighton([md]))
        assert "The work was carried out with the University of Example" in body

    def test_truncated_body_fragment_with_institution_not_hidden(self) -> None:
        # An OCR-truncated prose clause that names institutions and lacks terminal
        # punctuation must stay visible: without a postal-code tail it is not an
        # address.  (Guards against the false positive the no-punctuation rule
        # alone allowed.)
        md = (
            "# A Study\n\n"
            "In this work, conducted jointly with the Department of Biology, the "
            "School of Medicine, and several partner hospitals across the region\n\n"
            "## Methods\n\nMethod text."
        )
        body = _body(_run_lighton([md]))
        assert "In this work, conducted jointly" in body

    def test_affiliation_line_predicate(self) -> None:
        from manuscribe.pipeline.affiliations import _is_affiliation_line

        assert _is_affiliation_line(
            "Daniel D. Clark From the Department of Chemistry and Biochemistry, "
            "California State University-Chico, Chico, California, 95929"
        )
        # A terminal period marks prose, not an address.
        assert not _is_affiliation_line(
            "The work was done at the University of Example, City, Region."
        )
        # No institution keyword.
        assert not _is_affiliation_line("Jane Doe, John Smith, and Mary Major")
        # Too few comma-separated segments to be an address layout.
        assert not _is_affiliation_line("Department of Chemistry")
        # An institution-naming prose clause with no postal-code tail is not an
        # address, even without terminal punctuation — it does not *open* with the
        # institution name, so the head-anchored affiliation cue does not fire.
        assert not _is_affiliation_line(
            "In this work, conducted jointly with the Department of Biology, the "
            "School of Medicine, and several partner hospitals across the region"
        )
        # A number earlier in the line cannot stand in for the address tail.
        assert not _is_affiliation_line(
            "enrolled 250 patients from the Department of Cardiology, the ICU, "
            "and two partner clinics"
        )
        # An address that *opens* with the institution name and *closes* on a place
        # name is an affiliation even with no postal code — international addresses
        # often end on a country, not a code.  Both ends are load-bearing: the prose
        # clauses above open with the keyword too but run on into lowercase words
        # instead of closing on a city/country, so they stay in the body.
        assert _is_affiliation_line(
            "Department of Chemistry, University of Oxford, Oxford, United Kingdom"
        )
        assert _is_affiliation_line(
            "Department of Biomedical Science and Center for Bio-Nanomaterials, "
            "Daegu University, Gyeongsan, South Korea"
        )
        # International laboratory spellings (laboratoire/laboratorio) share the
        # stem, so the head cue fires for them as well.
        assert _is_affiliation_line(
            "Laboratoire de Biologie Moléculaire, Université de Paris, Paris, France"
        )
        assert _is_affiliation_line(
            "Laboratorio de Química, Universidad de Madrid, Madrid, Spain"
        )
        # A lowercase connector inside the closing place name ("Republic of Korea")
        # is allowed; the tail must only *open* on a capital.
        assert _is_affiliation_line(
            "Institute of Microbiology, Korea University, Seoul, Republic of Korea"
        )
        # Opening with the keyword is not enough when the line runs on into a
        # lowercase clause instead of closing on a place — kept visible in the body.
        assert not _is_affiliation_line(
            "University researchers, clinicians, and administrators worked across "
            "the region and beyond"
        )
        # The country/region tail is the language-independent signal: it fires even
        # when the institution word is in a language no stem covers (German "Labor"/
        # "Fakultät", Czech "Ústav"/"Katedra"), so these are recognised too.
        assert _is_affiliation_line(
            "Labor für Mikrobiologie, Klinikum der Universität, Berlin, Germany"
        )
        assert _is_affiliation_line("Fakultät für Chemie, Wien, Österreich")
        assert _is_affiliation_line(
            "Ústav organické chemie, Univerzita Karlova, Praha, Czech Republic"
        )
        assert _is_affiliation_line("Katedra biologie, Praha, Česko")
        # An uppercase country tail ("…, SWEDEN") still matches (OCR casing varies).
        assert _is_affiliation_line(
            "Department of Chemistry, Luleå University of Technology, SWEDEN"
        )
        # A prose clause that merely *mentions* countries but does not *close* on a
        # bare country name (the final segment is "and Spain", not "Spain") stays
        # in the body.
        assert not _is_affiliation_line("We ran trials in France, Germany, and Spain")

    def test_all_frontmatter_body_kept_visible(self) -> None:
        # If every body block looks like front matter, that signals misdetection,
        # not a metadata-only doc: keep it visible rather than emptying the body.
        # The submission-date footer is unambiguous metadata and is relocated to the
        # panel, but the ambiguous affiliation-looking line stays in the body.
        md = (
            "# T\n\n## Abstract\n\nThe abstract.\n\n**Keywords:** alpha, beta\n\n"
            "¹Affiliation One, City\n\nReceived 26 March 2019"
        )
        html = _run_lighton([md])
        assert "Affiliation One" in _body(html)
        assert "Received 26 March 2019" not in _body(html)
        assert "Received 26 March 2019" in _metadata(html)

    def test_figure_placeholder_becomes_cropped_figure(self) -> None:
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\nBefore.\n\n"
            "![image](image_1.png)100,100,900,900\n\n"
            "FIG. 1 A nice figure caption.\n\nAfter."
        )
        html = _run_lighton([md])
        assert "data:image/png;base64," in html
        assert "<figcaption>FIG. 1 A nice figure caption.</figcaption>" in html
        assert "100,100,900,900" not in html

    def test_figure_bbox_is_denormalized_to_full_extent(self) -> None:
        # A full box in [0,1000] space must crop the whole page, not a top-left
        # sliver (the truncation bug: coords are normalized, not pixels).
        img = _fake_image(1190, 1540)
        md = "# T\n\n## Abstract\n\nA.\n\n## Body\n\n![image](i.png)0,0,1000,1000"
        html = _run_lighton([md], image=img)
        assert _figure_sizes(html) == [(1190, 1540)]


class TestDegenerateInputs:
    """The OCR is non-deterministic and can emit malformed shapes (out-of-bounds /
    inverted / zero-area figure boxes, empty pages); the render-free core must
    degrade gracefully — clamp or drop the crop, never crash."""

    def _assemble(self, md: str, w: int = 1190, h: int = 1540) -> str:
        from manuscribe.pipeline.assemble import _assemble_html

        return _assemble_html([md], [Image.new("RGB", (w, h))])

    def test_empty_page_list_yields_shell(self) -> None:
        from manuscribe.pipeline.assemble import _assemble_html

        html = _assemble_html([], [])
        assert html.startswith("<!DOCTYPE html>")
        assert "<body>" in html

    def test_blank_and_whitespace_pages_do_not_crash(self) -> None:
        for md in ("", "   \n\n  \t"):
            assert "<body>" in self._assemble(md)

    def test_out_of_bounds_figure_box_is_clamped(self) -> None:
        from manuscribe.pipeline.figures import _safe_crop

        crop = _safe_crop(_fake_image(100, 100), (0, 0, 5000, 5000))
        assert crop is not None and crop.size == (100, 100)

    def test_inverted_figure_box_is_dropped(self) -> None:
        from manuscribe.pipeline.figures import _safe_crop

        # x1 < x0 / y1 < y0 -> negative area < the minimum, so no crop is produced.
        assert _safe_crop(_fake_image(100, 100), (90, 90, 10, 10)) is None

    def test_zero_area_figure_box_is_dropped(self) -> None:
        from manuscribe.pipeline.figures import _safe_crop

        assert _safe_crop(_fake_image(100, 100), (50, 50, 50, 50)) is None

    def test_negative_coords_not_parsed_as_placeholder(self) -> None:
        from manuscribe.pipeline.figures import _parse_figure_placeholder

        # the bbox grammar is unsigned, so a negative coordinate is simply "not a
        # figure placeholder" rather than a crash or a bogus negative box
        assert not _parse_figure_placeholder(
            "![image](i.png)-5,0,100,100"
        ).is_placeholder

    def test_out_of_bounds_placeholder_assembles_without_crash(self) -> None:
        md = "# T\n\n![image](i.png)0,0,1500,1500\n\n## Abstract\n\nThe abstract."
        html = self._assemble(md)
        # the figure crop is produced (clamped) and no prose is lost
        assert "<figure>" in html
        assert "The abstract." in html

    def test_figure_box_larger_than_tiny_image_does_not_crash(self) -> None:
        from manuscribe.pipeline.assemble import _assemble_html

        html = _assemble_html(
            ["![image](i.png)0,0,1000,1000"], [Image.new("RGB", (3, 3))]
        )
        assert "<body>" in html

    def test_unterminated_latex_span_passes_through(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # an unmatched "$" is left verbatim (no span conversion, no exception)
        assert _latex_to_html("mass $^{1,*") == "mass $^{1,*"

    def test_unclosed_table_html_does_not_crash(self) -> None:
        from manuscribe.pipeline.markdown import _md_to_html_blocks

        assert _md_to_html_blocks("<table><tr><td>x") == ["<table><tr><td>x"]


class TestSuperscriptMarkerCharClass:
    """The leading/author superscript-marker char-classes are the canonical
    [¹²³⁰⁴-⁹] (superscript digits) plus footnote symbols — NOT [⁰-ⁿ], which over-
    matches the non-digit superscripts ⁱⁿ⁺⁻⁼⁽⁾ (the CLAUDE.md digit-class gotcha)."""

    def test_leading_sup_matches_real_markers(self) -> None:
        from manuscribe.pipeline.classify import _LEADING_SUP_RE

        # a superscript-digit affiliation marker and a footnote symbol
        assert _LEADING_SUP_RE.match("¹ Department of Chemistry")
        assert _LEADING_SUP_RE.match("⁵ Second affiliation")
        assert _LEADING_SUP_RE.match("*Corresponding author")
        assert _LEADING_SUP_RE.match("† Equal contribution")

    def test_leading_sup_rejects_non_digit_superscripts(self) -> None:
        from manuscribe.pipeline.classify import _LEADING_SUP_RE

        # ⁿ (U+207F), ⁺ (U+207A) and ⁻ (U+207B) are not affiliation markers; the old
        # [⁰-ⁿ] range matched them, the canonical [⁰⁴-⁹] does not
        assert _LEADING_SUP_RE.match("ⁿ-hexane as solvent") is None
        assert _LEADING_SUP_RE.match("⁺ charged residue") is None
        assert _LEADING_SUP_RE.match("⁻ control lane") is None

    def test_author_marker_ignores_superscript_exponent(self) -> None:
        from manuscribe.pipeline.classify import _AUTHOR_MARKER_RE

        # a real author marker (digit or <sup>) is found…
        assert _AUTHOR_MARKER_RE.search("Jane Smith¹")
        assert _AUTHOR_MARKER_RE.search("Jane Smith<sup>a</sup>")
        # …but a superscript "ⁿ" exponent in a math expression is not a marker
        assert _AUTHOR_MARKER_RE.search("the xⁿ term") is None


class TestLeadingBoldLabelVocabulary:
    """The leading-bold-label vocabulary sets keep their subset invariants by
    *construction* (the superset is derived as ``subset | {rest}``), so a banner can
    never drift out of the publication-metadata set nor a glossary label out of the
    front-matter set.  These lock that against a revert to independent literals."""

    def test_banner_is_subset_of_publication_metadata(self) -> None:
        from manuscribe.pipeline.classify import (
            _PUBLICATION_BANNER_LABELS,
            _PUBLICATION_METADATA_LABELS,
        )

        assert _PUBLICATION_BANNER_LABELS <= _PUBLICATION_METADATA_LABELS
        assert "open access" in _PUBLICATION_METADATA_LABELS

    def test_glossary_is_subset_of_frontmatter(self) -> None:
        from manuscribe.pipeline.classify import (
            _FRONTMATTER_HEADING_LABELS,
            _GLOSSARY_METADATA_LABELS,
        )

        assert _GLOSSARY_METADATA_LABELS < _FRONTMATTER_HEADING_LABELS  # strict subset
        # the keyword labels are the front-matter-only delta over the glossary
        delta = _FRONTMATTER_HEADING_LABELS - _GLOSSARY_METADATA_LABELS
        assert delta == {"keywords", "key words"}

    def test_single_either_colon_matcher_handles_both_shapes(self) -> None:
        from manuscribe.pipeline.classify import _BOLD_LABEL_CAPTURE_RE

        # the one either-colon matcher captures the label name for colon-inside…
        m1 = _BOLD_LABEL_CAPTURE_RE.match("<strong>Keywords:</strong> a, b")
        assert m1 is not None and (m1.group(1) or m1.group(2)) == "Keywords"
        # …and colon-outside the bold
        m2 = _BOLD_LABEL_CAPTURE_RE.match("<strong>Keywords</strong>: a, b")
        assert m2 is not None and (m2.group(1) or m2.group(2)) == "Keywords"
        # a bold run with no colon is not a label (left in the body)
        assert _BOLD_LABEL_CAPTURE_RE.match("<strong>Important</strong> note") is None
