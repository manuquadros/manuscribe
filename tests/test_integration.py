"""Full-pipeline integration tests (run the real vLLM server)."""

import re
from pathlib import Path

import pytest
from helpers import (
    _abstract,
    _body,
    _figure_size_by_caption,
    _figure_sizes,
    _header_h1,
    _metadata,
    _tables_text,
)

_FIXTURE_PDF = Path(__file__).parent / "fixtures" / "30592559.pdf"
_AD_PREFIX_PDF = Path(__file__).parent / "fixtures" / "31051047.pdf"
_PLOS_PDF = Path(__file__).parent / "fixtures" / "32639976.pdf"
_FRONTIERS_PDF = Path(__file__).parent / "fixtures" / "32117944.pdf"
_BSR_PDF = Path(__file__).parent / "fixtures" / "31123167.pdf"
_JAFC_PDF = Path(__file__).parent / "fixtures" / "31298526.pdf"
_OUTPUT_DIR = Path(__file__).parent / "fixtures"


def _run_document_to_file(pdf: Path, ocr: object) -> object:
    """Run the full pipeline on ``pdf`` and save the HTML for visual inspection.

    Returns the :class:`ParsedDocument` (HTML + structured title/byline/DOI).  The
    HTML lands at ``tests/fixtures/<pdf-stem>.html`` so every integration run leaves
    an on-disk copy of each fixture's rendering to open in a browser.
    """
    from manuscribe.pipeline import OcrModel, lightonocr_pdf_to_document

    assert isinstance(ocr, OcrModel)
    _OUTPUT_DIR.mkdir(exist_ok=True)
    doc = lightonocr_pdf_to_document(
        pdf, ocr=ocr, image_dir=_OUTPUT_DIR / f"{pdf.stem}_files"
    )
    (_OUTPUT_DIR / f"{pdf.stem}.html").write_text(doc.html, encoding="utf-8")
    return doc


def _run_pipeline_to_file(pdf: Path, ocr: object) -> str:
    """The HTML of :func:`_run_document_to_file` — for the fixtures asserting on the
    rendered string only."""
    return _run_document_to_file(pdf, ocr).html


@pytest.fixture(scope="session")
def ocr_model() -> object:
    """Load the LightOnOCR model bundle once per session; skip if unavailable."""
    try:
        from manuscribe.pipeline import load_ocr_model

        return load_ocr_model()
    except Exception as e:
        pytest.skip(f"LightOnOCR model not available: {e}")


@pytest.fixture(scope="session")
def parsed_document(ocr_model: object) -> object:
    """Parse the no-ad fixture once into a ParsedDocument; skip if the model is absent.

    Writes the HTML to tests/fixtures/30592559.html so the file stays current after
    each integration run; ``article_html`` is its ``.html`` (so the string-asserting
    tests and the structured-return tests share one pipeline run).
    """
    if not _FIXTURE_PDF.exists():
        pytest.skip(f"Fixture PDF not found: {_FIXTURE_PDF}")
    return _run_document_to_file(_FIXTURE_PDF, ocr_model)


@pytest.fixture(scope="session")
def article_html(parsed_document: object) -> str:
    from manuscribe.pipeline import ParsedDocument

    assert isinstance(parsed_document, ParsedDocument)
    return parsed_document.html


@pytest.mark.integration
class TestPipeline:
    """Integration tests: run the full LightOnOCR pipeline on the fixture PDF.

    Skipped when the model is not available (no GPU, weights not downloaded).
    Each run also refreshes tests/fixtures/30592559.html.
    """

    def test_abstract_no_column_break(self, article_html: str) -> None:
        abstract_start = article_html.find("<section class='abstract'>")
        abstract_end = article_html.find("</section>", abstract_start)
        abstract_block = article_html[abstract_start:abstract_end]
        # Both halves must be collected: if either is absent the pipeline
        # dropped a fragment it should have kept.
        assert "classical and contemporary" in abstract_block
        assert "experimental biochemistry." in abstract_block
        # And they must appear in the same paragraph — no split <p>.
        assert "classical and contemporary</p>" not in abstract_block

    def test_abstract_citation_tail_in_panel_not_abstract(
        self, article_html: str
    ) -> None:
        # The OCR runs the article's copyright + journal citation
        # ("© 2018 International Union …, 47(2):124–132, 2019.") onto the abstract's
        # end; it is front matter and must be relocated to the Metadata panel.
        abstract_start = article_html.find("<section class='abstract'>")
        abstract = article_html[
            abstract_start : article_html.find("</section>", abstract_start)
        ]
        assert "University-Chico." in abstract  # the real abstract still ends here
        assert "International Union of Biochemistry" not in abstract
        assert "International Union of Biochemistry" in _metadata(article_html)

    def test_nad_plus_superscript_rendered_in_body(self, article_html: str) -> None:
        # NAD$^+$ -> NAD⁺: the end-to-end LaTeX-superscript path (OCR -> span
        # conversion -> render), only otherwise covered as a unit test.
        assert "NAD⁺" in _body(article_html)

    def test_stereodescriptor_math_span_unwrapped(self, article_html: str) -> None:
        # The model wraps CIP stereodescriptors in math mode ("$(R)$-2-alkanols");
        # the delimiters must drop (no literal '$') and the letter italicise.
        body = _body(article_html)
        assert "$(R)$" not in body and "$(S)$" not in body
        assert "(<em>R</em>)-2-alkanols" in body

    def test_which_is_not_merged_with_as_a_testament(self, article_html: str) -> None:
        assert "which is As a testament to the utility" not in article_html

    def test_ternary_complex_followed_by_clearly_showed(
        self, article_html: str
    ) -> None:
        expected = (
            "The 1.8 Å ternary complex (enzyme + 2-KPC + NAD⁺)"
            " clearly showed interaction of the R152"
        )
        assert expected in article_html

    def test_first_page_footer_metadata_in_panel(self, article_html: str) -> None:
        # The first page's bottom-of-page footer — the journal-citation /
        # correspondence line and the supporting-information note — is front matter
        # the OCR drops into the body.  It is relocated to the collapsed Metadata
        # panel (which renders before the body), not left inline.
        panel, body = _metadata(article_html), _body(article_html)
        # The OCR splits the page-bottom block into one-line pieces; every one is
        # relocated, including the journal-citation / DOI / "Published online" lines
        # that carry only a single metadata token.
        for fragment in (
            "To whom correspondence should be addressed",
            "Additional Supporting Information",
            "Received 19 June 2018",
            "Volume 47",
            "DOI 10.1002/bmb.21202",
            "Published online 28 December 2018",
        ):
            assert fragment in panel
            assert fragment not in body
        # Pulling the orphan lines out before the (cross-page) paragraph-merge stops
        # them from chaining into the page-2 prose the OCR placed after them.
        assert "Herein, I propose" in body

    def test_figure_caption_not_glued_to_following_paragraph(
        self, article_html: str
    ) -> None:
        # The Fig. 1 caption is emitted as its own <figcaption>; the body
        # paragraph that follows ("Herein, I propose …") must stay a separate
        # block, not be absorbed onto the caption.
        body = _body(article_html)
        assert "carboxylase.</figcaption>" in body
        assert "<p>Herein, I propose" in body
        assert "carboxylase. Herein, I propose" not in body
        assert "carboxylase.</em> Herein, I propose" not in body

    def test_paragraph_rejoined_across_table_i(self, article_html: str) -> None:
        # The clause "…revealed that," / "with no additives present…" is split by
        # TABLE I (its caption and footnotes folded in); it must read as one
        # paragraph, not be left split or glued to the stray Fig. 3 note sentence.
        body = _body(article_html)
        assert "revealed that, with no additives present, all forms of rR-HPCDH" in body
        assert "revealed that,</p>" not in body
        assert "revealed that, Molecule structures" not in body

    def test_table_footnotes_ride_with_their_table(self, article_html: str) -> None:
        # TABLE I's footnote run — the "Molecule structures … Fig. 3." note plus
        # the <sup>-marked footnotes — is folded onto the table block, so it
        # renders under the table rather than being swept into the article
        # footnote section before the references.
        body = _body(article_html)
        note = '<p class="footnote">Molecule structures are shown in Fig. 3.</p>'
        assert note in body
        # The note sits immediately after a </table>, not adrift in the prose.
        assert '</table><p class="footnote">Molecule structures' in body

    def test_table_iv_heading_label_does_not_absorb_body_continuation(
        self, article_html: str
    ) -> None:
        # Page 5→6: "TABLE IV" is OCR'd as a heading with its title stranded below.
        # The page-5 paragraph ("…alcohol dehydrogenase was covered, the") continues
        # as "reaction stereospecificity was not. …prochirality." after the table; it
        # must rejoin that paragraph, not be glued onto the TABLE IV caption.
        body = _body(article_html)
        assert (
            "alcohol dehydrogenase was covered, the reaction stereospecificity"
            " was not. An additional prerequisite topic, covered on the same day"
            " of the case study, was prochirality." in body
        )
        # The TABLE IV title is folded into the table as a caption, not left as a
        # body paragraph that swallowed the continuation.
        assert (
            "<caption>TABLE IV Comparison of rR- and rS-HPCDH kinetic parameters"
            " and stereoselectivity</caption>" in body
        )
        assert "stereoselectivity reaction stereospecificity was not" not in body


@pytest.fixture(scope="session")
def ad_prefix_run(ocr_model: object) -> tuple[str, list[int]]:
    """Run the ad-prefixed 31051047.pdf fixture once, spying on the full-page OCR batch.

    Returns the pipeline HTML plus the image counts each ``_ocr_pages`` call received.
    The first call is the full-page pass (``_ocr_document_pages``); the later calls are
    the table/figure crop re-OCR batches — so ``sizes[0]`` proves how many pages were
    actually sent to the model, i.e. that the image-only ad page was skipped before OCR.
    Writes the result to tests/fixtures/31051047.html for visual inspection.
    """
    if not _AD_PREFIX_PDF.exists():
        pytest.skip(f"Fixture PDF not found: {_AD_PREFIX_PDF}")
    from manuscribe.pipeline import assemble

    sizes: list[int] = []
    real_ocr_pages = assemble._ocr_pages

    def _spy(
        images: list[object], ocr: object, *args: object, **kwargs: object
    ) -> list[str]:
        sizes.append(len(images))
        return real_ocr_pages(images, ocr, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(assemble, "_ocr_pages", _spy)
        html = _run_pipeline_to_file(_AD_PREFIX_PDF, ocr_model)
    return html, sizes


@pytest.fixture(scope="session")
def ad_prefix_html(ad_prefix_run: tuple[str, list[int]]) -> str:
    return ad_prefix_run[0]


@pytest.mark.integration
class TestAdPageExclusion:
    """The 31051047.pdf fixture has an advertisement as its first page; the
    pipeline must drop it and start the document at the real article title."""

    def test_ad_page_skipped_before_ocr(
        self, ad_prefix_run: tuple[str, list[int]]
    ) -> None:
        # The pre-OCR text-layer skip drops the image-only ad (page 0) before OCR, so
        # the full-page pass — the first _ocr_pages call — sees only the 10 article
        # pages, never the ad.  That is 1–2 fewer ~10-24 s round trips per ad-prefixed
        # document, the perf win, and it costs the model no OCR of a page whose output
        # the leading-page skip discards anyway.
        _, sizes = ad_prefix_run
        assert sizes[0] == 10  # 11 pages total, ad page 0 excluded

    def test_title_starts_with_article_title(self, ad_prefix_html: str) -> None:
        # The title carries the PDF's intra-title line breaks; normalize runs of
        # whitespace before matching the prefix.
        title = re.sub(r"\s+", " ", _header_h1(ad_prefix_html)).strip()
        assert title.startswith(
            "Biochemical characterization reveals the functional divergence"
        )

    def test_species_name_italicized(self, ad_prefix_html: str) -> None:
        assert "<em>Przewalskia tangutica</em>" in ad_prefix_html

    def test_temperature_unit_stays_upright(self, ad_prefix_html: str) -> None:
        # "$25 \\pm 1^\\circ \\text{C}$" -> "25 ± 1°C": the degree sign bridges the
        # magnitude and the unit letter, so C is a unit symbol, not an italic
        # variable ("25 ± 1°<em>C</em>").
        body = _body(ad_prefix_html)
        assert "25 ± 1°C" in body
        assert "1°<em>C</em>" not in body

    def test_gene_names_not_collapsed_by_ocr(self, ad_prefix_html: str) -> None:
        # The motivating bug: the old OCR misread both PtTRI and PtTRII as the
        # single string "PITRI", collapsing two distinct genes.  LightOnOCR reads
        # them correctly; tables are excluded so the check targets the prose.
        prose = re.sub(r"<table.*?</table>", "", ad_prefix_html, flags=re.DOTALL)
        assert "PtTRI" in prose
        assert "PtTRII" in prose
        assert "PITRI" not in prose

    def test_numbered_subsections_normalized_to_consistent_level(
        self, ad_prefix_html: str
    ) -> None:
        # The OCR jitters ##/### levels: section 3's subsections 3.1-3.3 came as
        # <h3> but 3.4/3.5 as <h2>.  Heading-level normalization re-levels every
        # dotted-number subsection by its depth, so all of 3.1-3.5 render as <h3>
        # and the top-level numbered sections (3. Results, 4. Discussion) as <h2>.
        body = _body(ad_prefix_html)
        for n in range(1, 6):
            assert f"<h3>3.{n}." in body
        assert "<h2>3. Results</h2>" in body
        assert "<h2>4. Discussion</h2>" in body
        # no numbered subsection leaked to <h2> (the pre-fix jitter)
        assert "<h2>3.4." not in body and "<h2>3.5." not in body

    def test_figures_embedded_by_detector(self, ad_prefix_html: str) -> None:
        # LightOnOCR-bbox emits a crop box per figure (incl. the Fig 2 alignment
        # the old engine OCRed into a token wall); each is cropped and embedded.
        assert len(_figure_sizes(ad_prefix_html, _OUTPUT_DIR)) >= 4

    @pytest.mark.parametrize("num", [1, 2, 3, 4, 5, 6])
    def test_every_known_figure_present(self, ad_prefix_html: str, num: int) -> None:
        # The article has six figures; each must survive to the output as a cropped
        # <figure> carrying its "FIG. N" caption.  FIG. 6 in particular was dropped
        # whole on some OCR runs and the whole-figure recovery declined it: a ghost
        # faux-bold copy of the caption's first line, a fraction above the "FIG. 6"
        # label, collapsed the recovery crop onto the caption and clipped the figure.
        # The label is anchored with a non-digit lookahead so "FIG. 1" can't be
        # satisfied by a surviving "FIG. 10" when a low-numbered figure is dropped.
        label = re.compile(rf"FIG\.\s*{num}(?!\d)")
        captions = re.findall(
            r"<figcaption>(.*?)</figcaption>", ad_prefix_html, re.DOTALL
        )
        assert any(label.search(c) for c in captions), f"FIG. {num} missing from output"

    def test_figures_not_truncated(self, ad_prefix_html: str) -> None:
        # The model emits boxes normalized to [0, 1000]; cropping them as raw
        # pixels truncated every figure.  A page-spanning figure (the Fig 2
        # alignment) only exceeds 1000 px wide once the box is denormalized to
        # the ~1190 px render width — impossible if coords are read as pixels.
        widest = max(w for w, _ in _figure_sizes(ad_prefix_html, _OUTPUT_DIR))
        assert widest > 1000, f"widest figure is only {widest}px — boxes not scaled"

    def test_cross_page_paragraph_not_split(self, ad_prefix_html: str) -> None:
        # The clause "…TRI and" / "TRII compete…" spans a page break; it must
        # be a single paragraph, not split at the page boundary.
        assert (
            "This suggests that TRI and TRII compete for the same substrate"
            in ad_prefix_html
        )
        assert "This suggests that TRI and</p>" not in ad_prefix_html

    def test_fig2_caption_rejoined_and_panel_label_not_stray(
        self, ad_prefix_html: str
    ) -> None:
        # The model splits Fig 2 into two panel boxes and emits bare "A"/"B" panel
        # labels as their own blocks.  The descriptive caption ("FIG. 2" + "Protein
        # alignments …", two OCR blocks) must be rejoined onto the figure, and the
        # stray "B" must not survive to prefix the caption ("B Protein alignments").
        fig2 = next(
            (
                f
                for f in re.findall(r"<figure>.*?</figure>", ad_prefix_html, re.DOTALL)
                if "Protein alignments of TRI and TRII" in f
            ),
            None,
        )
        assert fig2 is not None, "Fig 2 caption not attached to its figure"
        assert "B Protein alignments" not in _body(ad_prefix_html)
        assert "<p>B</p>" not in _body(ad_prefix_html)

    def test_keywords_relocated_to_panel_not_body(self, ad_prefix_html: str) -> None:
        # The keyword label terminates the abstract, then is relocated from the body
        # head to the Metadata panel — even when OCR emits the colon outside the bold
        # ("<strong>Keywords</strong>:"), the shape that previously stranded it in body.
        kw = "enzyme kinetics, functional identification"
        assert kw in _metadata(ad_prefix_html)
        assert kw not in _body(ad_prefix_html)


@pytest.fixture(scope="session")
def plos_run(ocr_model: object) -> object:
    """Run the PLOS ONE 32639976.pdf pipeline once, capturing the table re-OCR
    batching and gate decisions so the gate test needs no second OCR pass.

    Wraps ``_recover_dropped_tables`` to record the size of each batched
    ``ocr_regions`` call, and ``_region_fully_captured`` to count how many regions
    the coverage gate actually skipped.  Returns the HTML plus that spy data;
    ``plos_html`` derives from it, so the whole fixture costs a single pipeline run.
    Writes tests/fixtures/32639976.html too.
    """
    if not _PLOS_PDF.exists():
        pytest.skip(f"Fixture PDF not found: {_PLOS_PDF}")
    from types import SimpleNamespace

    from manuscribe.pipeline import OcrModel, assemble, lightonocr_pdf_to_html, tables

    assert isinstance(ocr_model, OcrModel)
    real_recover = assemble._recover_dropped_tables
    real_gate = tables._region_fully_captured
    spy = SimpleNamespace(batches=[], gate_skips=0, html="")

    def recover_with_spy(layers, pages_md, ocr_regions):  # type: ignore[no-untyped-def]
        def counting(regions):  # type: ignore[no-untyped-def]
            spy.batches.append(len(regions))
            return ocr_regions(regions)

        return real_recover(layers, pages_md, counting)

    def gate_with_spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        skipped = real_gate(*args, **kwargs)
        spy.gate_skips += int(skipped)
        return skipped

    mp = pytest.MonkeyPatch()
    mp.setattr(assemble, "_recover_dropped_tables", recover_with_spy)
    # Count actual gate decisions, so the gate test can't be satisfied by a region
    # that merely failed to localize (which also lowers the batch size).  Patch it in
    # tables.recover, where _plan_page_tables resolves _region_fully_captured.
    mp.setattr(
        "manuscribe.pipeline.tables.recover._region_fully_captured", gate_with_spy
    )
    _OUTPUT_DIR.mkdir(exist_ok=True)
    try:
        spy.html = lightonocr_pdf_to_html(
            _PLOS_PDF, ocr=ocr_model, image_dir=_OUTPUT_DIR / f"{_PLOS_PDF.stem}_files"
        )
    finally:
        mp.undo()
    (_OUTPUT_DIR / f"{_PLOS_PDF.stem}.html").write_text(spy.html, encoding="utf-8")
    return spy


@pytest.fixture(scope="session")
def plos_html(plos_run: object) -> str:
    """Full pipeline HTML for the PLOS fixture (the single run in ``plos_run``)."""
    return plos_run.html


@pytest.mark.integration
class TestPlosSidebarMetadata:
    """32639976.pdf (PLOS ONE) prints a left-column metadata sidebar beside the
    abstract; the OCR interleaves it into the body.  Every piece — the affiliation
    and the Citation/Editor/dates/Copyright/Data-Availability/Funding/
    Competing-interests run, which LightOnOCR emits as ``<h3>`` label headings —
    must land in the collapsed Metadata panel, not the body."""

    def test_sidebar_metadata_in_panel_not_body(self, plos_html: str) -> None:
        panel, body = _metadata(plos_html), _body(plos_html)
        # The labels relocate as their own headings (not bold "**Label:**" lines),
        # so match the heading form — and use it for "Funding" specifically, whose
        # bare word also appears in the body's "Funding acquisition:" CRediT role.
        for fragment in (
            "Department of Biomedical Science and Center for Bio-Nanomaterials",
            "<h3>Citation</h3>",
            "<h3>Editor</h3>",
            "<h3>Received</h3>",
            "<h3>Accepted</h3>",
            "<h3>Published</h3>",
            "<h3>Copyright</h3>",
            "<h3>Data Availability Statement</h3>",
            "<h3>Funding</h3>",
            "<h3>Competing interests</h3>",
        ):
            assert fragment in panel, f"{fragment!r} missing from metadata panel"
            assert fragment not in body, f"{fragment!r} leaked into body"

    def test_title_in_header_not_masthead(self, plos_html: str) -> None:
        # The journal masthead "PLOS ONE" must not be taken as the article title;
        # the real title (with its italicised species) belongs in the header.
        header = plos_html[: plos_html.find("</header>")]
        title = (
            "Purification and characterization of a novel medium-chain ribitol "
            "dehydrogenase from a lichen-associated bacterium "
            "<em>Sphingomonas</em> sp."
        )
        assert title in header, "article title missing from header"
        assert "<h1>PLOS ONE</h1>" not in header

    def test_byline_has_no_stray_or_misnested_markup(self, plos_html: str) -> None:
        # The author line is bolded with '*' corresponding-author markers
        # ("**… Jang*, ChangWoo Lee***").  The byline render strips the layout bold
        # and re-casts the markers as superscripts — no leftover '**', no mis-ordered
        # "<em>…</strong></em>" / redundant "<em><em>", and no bold wrapper.
        header = plos_html[: plos_html.find("</header>")]
        assert "ChangWoo Lee" in header
        assert "**" not in header
        assert "</strong></em>" not in header
        assert "<em><em>" not in header
        # the byline paragraph (the <p> after the <h1> title) carries no layout
        # emphasis and renders the surviving marker as a superscript.  Scoped to the
        # byline because the title legitimately italicises a species name.
        after_title = header[header.find("</h1>") :]
        byline = after_title[after_title.find("<p>") : after_title.find("</p>") + 4]
        assert "ChangWoo Lee<sup>*</sup>" in byline
        assert "<em>" not in byline
        assert "<strong>" not in byline

    def test_body_opens_with_introduction_prose(self, plos_html: str) -> None:
        # With the sidebar relocated, the body proper opens at the Introduction —
        # its prose must remain visible, not be swept into the panel.
        body = _body(plos_html)
        assert "Lichens have traditionally been considered" in body
        assert "Polyols have a role in carbohydrate storage" in body

    def test_running_head_not_in_body(self, plos_html: str) -> None:
        # The running head ends in "Sphingomonas sp." — an abbreviation that only
        # looks like a sentence end — so it must still be stripped as furniture,
        # not leak onto every page of the body.
        body = _body(plos_html)
        assert "lichen-associated <em>Sphingomonas</em> sp.</p>" not in body

    def test_supporting_info_labels_render_s_not_section_sign(
        self, plos_html: str
    ) -> None:
        # The Supporting Information labels ("S1 Fig.", "S4 Fig.", "S1 Raw images.")
        # carry a leading "S"; the model sometimes misreads it as the section sign,
        # emitting "§4 Fig." either as "\S4" (math) or the literal "§4".  Whichever
        # shape, no supplementary label may render as a section sign before a digit.
        assert "Supporting information" in plos_html
        assert re.search(r"S\d+\w*\s+(?:Fig|Raw images)", plos_html)
        assert not re.search(r"§\s?\d", plos_html)

    def test_growth_paragraph_merged_across_interleaved_figure(
        self, plos_html: str
    ) -> None:
        # The Results paragraph breaks across a page, with Fig 1 (and the stripped
        # running head) interleaved between its halves; once the running head is
        # gone the merge bridges the figure, joining the two fragments into one.
        body = _body(plos_html)
        assert "or D-mannitol) at 15°C and its growth measured for 16 days." in body

    def test_repeated_section_heading_kept(self, plos_html: str) -> None:
        # "Purification of SpRDH" titles both a Methods subsection and a Results
        # section; recurring as a heading is not running furniture, so both stay.
        body = _body(plos_html)
        assert "<h3>Purification of SpRDH</h3>" in body
        assert "<h2>Purification of SpRDH</h2>" in body

    def test_reference_tail_not_stranded_outside_list(self, plos_html: str) -> None:
        # A page break dropped reference 33's head/number, stranding its tail
        # ("…dehydrogenase is independent of the oxidative fermentation. Biosci
        # Biotechnol Biochem…") as a loose <p>; it must be folded into the
        # bibliography <ol>, not left as a paragraph between numbered entries.
        body = _body(plos_html)
        assert "dehydrogenase is independent of the oxidative fermentation" in body
        # the stranded tail is inside the list, never a bare <p> right after the </ol>
        assert not re.search(r"</ol>\s*<p>dehydrogenase is independent", body)


@pytest.mark.integration
class TestPlosFigureCrops:
    """Figure crops on 32639976.pdf (PLOS ONE).  The model's bbox clips the right
    edge of the wide Fig 5 alignment, and for several figures sits low enough that
    the bottom-growth would bake the caption into the crop.  The right edge is
    recovered by ``_extend_edge``; ``_trim_swallowed_caption`` drops a
    recovered band that reads as the caption when the OCR also emitted that caption
    as its own text block."""

    def test_fig5_alignment_right_edge_recovered(self, plos_html: str) -> None:
        # The model clips Fig 5's right edge to ≈0.88·page-width; the alignment
        # actually spans the full text column, so the recovered crop must reach
        # well past that — independent of the model's clipped x1, which growth
        # ignores in favour of the figure's real right edge on the page.
        size = _figure_size_by_caption(
            plos_html, "Multiple sequence alignments", _OUTPUT_DIR
        )
        assert size is not None, "Fig 5 alignment figure not embedded"
        assert size[0] > 920, f"Fig 5 width {size[0]}px — right edge not recovered"

    def test_fig5_baked_caption_trimmed(self, plos_html: str) -> None:
        # The alignment is itself text, so the model boxes its caption + DOI inside
        # the figure; re-OCR confirms those trailing bands reproduce the caption and
        # trims them.  Left in, the crop runs ~130 px taller (caption + DOI baked in).
        size = _figure_size_by_caption(
            plos_html, "Multiple sequence alignments", _OUTPUT_DIR
        )
        assert size is not None, "Fig 5 alignment figure not embedded"
        assert size[1] < 830, f"Fig 5 height {size[1]}px — caption baked into crop"

    def test_carbon_source_plot_caption_not_baked_in(self, plos_html: str) -> None:
        # Fig 1 is a roughly square growth-curve plot; the model's box bottom abuts
        # its 3-line caption + DOI, which the bottom-growth pulls in.  With the
        # caption trimmed the crop stays near the plot's own height (~590 px); left
        # un-trimmed it grows past 700 px.
        size = _figure_size_by_caption(
            plos_html, "Effect of different carbon sources", _OUTPUT_DIR
        )
        assert size is not None, "Fig 1 plot figure not embedded"
        assert size[1] < 660, f"Fig 1 height {size[1]}px — caption baked into crop"


@pytest.mark.integration
class TestPlosTableReocr:
    """32639976.pdf Table 2: the full-page OCR pass drops the column-spanning
    subheader "Relative activity (%)" from part A.  Re-OCRing the table region as a
    tight crop (localized via the text layer) recovers it, with the spanning markup
    the full-page pass omitted."""

    def test_relative_activity_header_recovered(self, plos_html: str) -> None:
        body = _body(plos_html)
        assert "Relative activity (%)" in body

    def test_recovered_header_spans_its_columns(self, plos_html: str) -> None:
        # the crop re-OCR emits the subheader as a colspan cell over the two data
        # columns — structure the full-page pass never produced
        body = _body(plos_html)
        m = re.search(r'colspan="2"[^>]*>\s*Relative activity', body)
        assert m is not None, "spanning subheader not recovered with colspan"


@pytest.mark.integration
class TestTableReocrGate:
    """The coverage gate skips re-OCR for a table the page pass already captured in
    full, and the crops that remain are re-OCR'd in one batched (concurrent) call."""

    def test_complete_table_skipped_and_crops_batched(self, plos_run: object) -> None:
        # The coverage gate actually skipped at least one region (measured at the
        # gate itself, so a mere localization failure can't satisfy this): PLOS
        # Table 1 (Purification summary) is fully captured.  And the crops that
        # remain are re-OCR'd in a single batched call, not one request per region.
        # Spy data comes from the shared plos_run, so there is no second OCR.
        assert plos_run.gate_skips >= 1, "coverage gate skipped no complete table"
        assert len(plos_run.batches) == 1, "table crops were not OCR'd in one batch"
        assert plos_run.batches[0] >= 1, "no table region was re-OCR'd"


# The table-content needles below are not harvested from the pipeline's own OCR
# output (that would be circular — it would only pin whatever the OCR happens to
# emit, garbage included).  Each is verified to exist in the PDF's embedded text
# layer — the publisher's ground truth, independent of our OCR — so the test
# catches a re-OCR that drops or mislocates a table into wrong-region content.


@pytest.mark.integration
class TestHpcdhTableContent:
    """30592559.pdf carries four data tables (TABLE I–IV); each must keep its real
    cell content through the table re-OCR pass.  Needles confirmed in the PDF text
    layer (ground truth), not copied from OCR output."""

    def test_all_four_table_captions_present(self, article_html: str) -> None:
        body = _body(article_html)
        for caption in ("TABLE I", "TABLE II", "TABLE III", "TABLE IV"):
            assert caption in body, f"{caption!r} missing"

    def test_distinctive_cells_live_in_tables(self, article_html: str) -> None:
        # one distinctive ground-truth cell per table, asserted inside <table>
        # markup (not merely loose in the body prose)
        cells = _tables_text(article_html)
        for needle in (
            "Enantioselectivity",  # TABLE I header
            "2-Butanone",  # TABLE II column
            "2-Butanol production",  # TABLE III header
            "rR-HPCDH",  # TABLE IV row
        ):
            assert needle in cells, f"{needle!r} not found inside any table"


@pytest.mark.integration
class TestTropinoneTableContent:
    """31051047.pdf has two kinetics tables (TABLE 1 and a homolog comparison).
    Needles confirmed in the PDF text layer (ground truth)."""

    def test_kinetics_table_content(self, ad_prefix_html: str) -> None:
        cells = _tables_text(ad_prefix_html)
        # nKat is the kinetics table's activity unit; both isoforms are its rows
        assert "nKat" in cells
        assert "PtTRI" in cells and "PtTRII" in cells

    def test_homolog_table_species_and_reference(self, ad_prefix_html: str) -> None:
        # the second table lists homologs by species with a reference column; its
        # rows must not be lost
        cells = _tables_text(ad_prefix_html)
        assert "Przewalskia tangutica" in cells
        assert "In this study" in cells

    def test_homolog_table_caption_folded_not_heading(
        self, ad_prefix_html: str
    ) -> None:
        # The model promotes TABLE 2's caption to an <h2>; it must fold into the
        # table as a <caption>, not stay a stray section heading.
        html = ad_prefix_html
        assert "<caption>TABLE 2 Comparison between various tropinone" in html
        assert "<h2>TABLE 2" not in html


@pytest.mark.integration
class TestHpcdhSingleAuthorByline:
    """30592559.pdf has a lone author ("Daniel D. Clark") with no affiliation
    marker; the mid-name initial promotes it to the header byline rather than
    leaving it stranded at the top of the body."""

    def test_author_in_header_not_body(self, article_html: str) -> None:
        header = article_html[
            article_html.find("<header>") : article_html.find("</header>")
        ]
        assert "Daniel D. Clark" in header
        assert "Daniel D. Clark" not in _body(article_html)

    def test_keywords_in_metadata_panel(self, article_html: str) -> None:
        # With the byline removed from the body top, the keyword line is once
        # again the leading front-matter block and lands in the panel.
        assert "Biochemistry education" in _metadata(article_html)
        assert "Biochemistry education" not in _body(article_html)


@pytest.mark.integration
class TestPlosTableOverrun:
    """Table 3 of 32639976.pdf overruns its page; the OCR leaves it unclosed and the
    following page's prose ("SpRDH operon …", verified in the PDF text layer) would
    render inside the table.  It must stay body prose, and tables must be balanced."""

    def test_post_table_prose_not_in_table(self, plos_html: str) -> None:
        cells = _tables_text(plos_html)
        assert "SpRDH operon" in _body(plos_html)  # present as prose
        assert "SpRDH operon" not in cells
        assert "PAMC 26621 genome" not in cells

    def test_all_tables_balanced(self, plos_html: str) -> None:
        # the general invariant: no table is left open to absorb following content
        body = _body(plos_html)
        assert body.count("<table") == body.count("</table>")

    def test_truncated_last_row_recovered(self, plos_html: str) -> None:
        # the last row (Enterobacter aerogenes) was cut at the page bottom; once the
        # table is closed it becomes a region the crop re-OCR recovers in full,
        # restoring the row's tail — Km NAD⁺ 0.16, kcat 318, kcat/Km 30.9 (all
        # verified in the PDF text layer)
        cells = _tables_text(plos_html)
        for value in ("0.16", "318", "30.9"):
            assert value in cells, f"{value!r} not recovered into the table"

    def test_table_legend_recovered(self, plos_html: str) -> None:
        # the legend was truncated together with the overrun row; the crop re-OCR
        # (bbox extended one line to reach it) recovers it as a footnote under the
        # table — verified in the PDF text layer
        text = re.sub(r"<[^>]+>", "", _body(plos_html))
        assert "MW: molecular weight, NR: Not reported" in text

    def test_paragraph_split_across_tables_rejoined(self, plos_html: str) -> None:
        # the sentence is split at the page break ("…carbon metabolism. The" on one
        # page, "SpRDH operon …" on the next) with Tables 2 and 3 between the halves;
        # the cross-table merge must rejoin it into one paragraph
        text = re.sub(r"<[^>]+>", "", _body(plos_html))
        assert (
            "The SpRDH operon of Sphingomonas sp. PAMC 26621 genome "
            "contains a putative ABC transporter" in text
        )


@pytest.mark.integration
class TestPlosTableFormatting:
    """Inline markup the OCR leaves inside table blocks must render, not surface
    as literal source: organism names in Table 3 cells are italicised, and Table
    2's footnote keeps its real <sup> marker rather than an escaped tag."""

    def test_table3_organism_names_italicised(self, plos_html: str) -> None:
        # Table 3's Organism column carries ``*Klebsiella aerogenes*`` etc. as raw
        # markdown inside the HTML cell; it must come out as <em>, not bare "*".
        body = _body(plos_html)
        assert "<em>Klebsiella aerogenes</em>" in body
        assert "<em>Zymomonas mobilis</em>" in body
        assert "*Klebsiella" not in body

    def test_table2_footnote_marker_not_escaped(self, plos_html: str) -> None:
        # Table 2's footnote marker is recovered as ``<sup>a</sup>…``; it must render
        # as a superscript, never as an HTML-escaped "&lt;sup&gt;" literal anywhere.
        text = re.sub(r"<[^>]+>", "", _body(plos_html))
        assert (
            "Each value represents the mean ± SD of three independent experiments"
            in text
        )
        assert "&lt;sup&gt;" not in plos_html

    def test_table2_panels_merged_under_one_caption(self, plos_html: str) -> None:
        # Table 2 is a composite table with two panels ("A. Effect of EDTA…",
        # "B. ICP-MS analysis"); the model splits them into two adjacent <table>s on
        # the blank line between panels.  They must re-fuse into one table carrying the
        # single "Table 2. Metal ion analysis." caption — not a captionless orphan B.
        i = plos_html.find("Table 2. Metal ion analysis")
        assert i != -1
        # locate the <table> that owns the caption and its matching close
        tbl_open = plos_html.rfind("<table", 0, i)
        tbl_close = plos_html.find("</table>", i)
        table = plos_html[tbl_open : tbl_close + len("</table>")]
        # both panels live inside that single table block
        assert "A. Effect of EDTA" in table
        assert "B. ICP-MS analysis" in table
        # the model renders Zn²⁺ as either Unicode superscripts or <sup>2+</sup>
        # depending on batch composition — both are panel B's data under one caption
        assert "Zn²⁺" in table or "Zn<sup>2+</sup>" in table
        # exactly one caption for the composite table
        assert table.count("<caption>") == 1


@pytest.fixture(scope="session")
def frontiers_html(ocr_model: object) -> str:
    """Full pipeline HTML for the Frontiers 32117944.pdf fixture; skip if the model
    is absent.  Writes tests/fixtures/32117944.html for visual inspection."""
    if not _FRONTIERS_PDF.exists():
        pytest.skip(f"Fixture PDF not found: {_FRONTIERS_PDF}")
    return _run_pipeline_to_file(_FRONTIERS_PDF, ocr_model)


@pytest.mark.integration
class TestFrontiersSidebarMetadata:
    """32117944.pdf (Frontiers in Bioengineering and Biotechnology) prints a
    first-page sidebar headed by an "OPEN ACCESS" banner, carrying Edited by /
    Reviewed by / Correspondence / Citation entries plus a "Specialty section:"
    routing line.  The banner heading and specialty line must land in the Metadata
    panel — and the abstract (which has no "Abstract" heading and directly follows
    the multi-superscript affiliation run) must stay in the body, not be glued onto
    the affiliation and hidden in the panel with it."""

    def test_open_access_sidebar_in_panel_not_body(self, frontiers_html: str) -> None:
        panel, body = _metadata(frontiers_html), _body(frontiers_html)
        for fragment in (
            "OPEN ACCESS",
            "Specialty section",
            "submitted to Synthetic Biology",
            "Synthetic Biology and Bioengineering Research Center",
        ):
            assert fragment in panel, f"{fragment!r} missing from metadata panel"
            assert fragment not in body, f"{fragment!r} leaked into body"

    def test_byline_affiliation_markers_rendered_as_superscripts(
        self, frontiers_html: str
    ) -> None:
        # The numeric author markers ("1,2", "1,3") render as superscripts in the
        # byline, not flattened to inline text, with no markdown-emphasis corruption.
        header = frontiers_html[
            frontiers_html.find("<header>") : frontiers_html.find("</header>")
        ]
        byline = header[header.find("<p>") :]
        assert "<sup>" in byline
        assert "<em>" not in byline
        assert ",," not in byline

    def test_headingless_abstract_recovered_to_section(
        self, frontiers_html: str
    ) -> None:
        # The abstract has no "Abstract" heading and no inline label, and the
        # affiliation run before it ends "…South Korea" with no terminal punctuation;
        # it must be recovered into the abstract section, not left in the body nor
        # (opening with the affiliation's "¹") hidden in the collapsed panel.
        abstract_text = (
            "Bioconversion of C1 chemicals such as methane and methanol into higher"
        )
        abstract, panel, body = (
            _abstract(frontiers_html),
            _metadata(frontiers_html),
            _body(frontiers_html),
        )
        assert abstract_text in abstract
        assert abstract_text not in body
        assert abstract_text not in panel

    def test_keywords_relocated_to_panel_after_headingless_abstract(
        self, frontiers_html: str
    ) -> None:
        # The headingless abstract is recovered to its own section; the keyword line
        # that follows it is relocated to the panel (post-classify), not stranded in
        # the body.
        panel, body = _metadata(frontiers_html), _body(frontiers_html)
        assert "<strong>Keywords:</strong>" in panel
        assert "<strong>Keywords:</strong>" not in body

    def test_abbreviations_footnote_does_not_split_introduction(
        self, frontiers_html: str
    ) -> None:
        # The Introduction's first paragraph is split across a column by the
        # "Abbreviations:" glossary footnote; relocating the glossary to the panel
        # lets the two halves rejoin as one body paragraph.
        panel, body = _metadata(frontiers_html), _body(frontiers_html)
        assert (
            "crucial enzyme for bioconversion of valuable multi-carbon chemicals"
            in body
        )
        # The glossary footnote itself is in the panel; the bare phrase "endogenous
        # activator protein" also appears in body prose ("…enhanced by an endogenous
        # activator protein (ACT)"), so key on the glossary's bold label, not the
        # entry text.
        assert "<strong>Abbreviations:</strong>" in panel
        assert "<strong>Abbreviations:</strong>" not in body


@pytest.mark.integration
class TestFrontiersSidewaysTable:
    """32117944.pdf Table 2 is printed sideways (rotated 270°) on the page.  The
    model boxes it as a figure *and* mis-OCRs its column structure (4-wide alcohol
    groups collapse to colspan=2), and the localizer's upright line model swept the
    neighbouring "CONCLUSION" heading into the crop.  Rotation-aware re-OCR turns the
    crop upright (correct colspans, no stray heading) and the figure/table dedup
    drops the duplicate image, folding its caption into the real table."""

    def test_table2_columns_span_four_each(self, frontiers_html: str) -> None:
        # Each alcohol group (Methanol/Ethanol/n-Propanol) heads four measurement
        # columns; the sideways mis-OCR collapsed them to colspan=2.
        html = frontiers_html
        assert html.count('colspan="4"') >= 3
        for alcohol in ("Methanol", "Ethanol", "n-Propanol"):
            assert f'colspan="4">{alcohol}' in html, f"{alcohol} not a 4-wide group"
        # The mutant rows survive intact (the variant column is recovered, not lost).
        for variant in ("WT", "S101V", "T141S", "A164F"):
            assert f"<td>{variant}</td>" in html

    def test_conclusion_not_folded_into_table(self, frontiers_html: str) -> None:
        # CONCLUSION is the section heading after Table 2, not a table header row.
        assert "CONCLUSION</th>" not in frontiers_html
        assert "<h2>CONCLUSION</h2>" in frontiers_html

    def test_table2_not_duplicated_as_figure(self, frontiers_html: str) -> None:
        # The boxed-table image is dropped; the "TABLE 2" caption rides the real
        # table as its <caption>, never a <figcaption>.
        html = frontiers_html
        assert "<caption>TABLE 2 | Kinetic parameters" in html
        assert "<figcaption>TABLE 2" not in html


@pytest.mark.integration
class TestFrontiersReferenceList:
    """32117944.pdf's references trail off in DOIs (no terminal punctuation), so
    the cross-column paragraph merge used to chain the whole list into one <p>.
    Each entry must render as its own block, while an entry wrapped across the
    column break still rejoins."""

    def test_each_reference_is_its_own_paragraph(self, frontiers_html: str) -> None:
        html = frontiers_html
        for entry in ("Arfman, N.,", "Bradford, M. M.", "Cahn, J. K.,"):
            assert f"<p>{entry}" in html, f"{entry} not at the head of its own <p>"
        # The DOI of one entry must not be glued to the surname of the next.
        assert "00426.x Bradford" not in html

    def test_wrapped_reference_entry_rejoined(self, frontiers_html: str) -> None:
        # Marcal's entry breaks across the column; the lowercase continuation
        # rejoins instead of standing as a stray fragment.
        assert (
            "quaternary structure and possible subunit cooperativity" in frontiers_html
        )


@pytest.mark.integration
class TestFrontiersHeadingAndFootnote:
    """32117944.pdf mis-levels a Methods subsection as <h1> ("Molecular Mass
    Determination of Lxmdh") and prints a column-bottom URL footnote as a raw
    unicode superscript ("¹http://…/home.htm") that interrupts a paragraph split
    across the column break."""

    def test_body_section_h1_demoted(self, frontiers_html: str) -> None:
        # The article title is the only <h1>; the mis-levelled body heading is h2.
        assert frontiers_html.count("<h1>") == 1
        assert "<h2>Molecular Mass Determination of Lxmdh</h2>" in frontiers_html

    def test_unicode_footnote_routed_not_glued(self, frontiers_html: str) -> None:
        html = frontiers_html
        # The footnote is pulled out of the prose; the split sentence rejoins.
        assert "gel filtration chromatography" in html
        assert "gel filtration ¹http" not in html
        # ...and lands in the footnote run before the references.
        fn = html.find('<p class="footnote">¹http://schwarz')
        assert 0 < fn < html.find("<h2>REFERENCES</h2>")


@pytest.fixture(scope="session")
def bsr_html(ocr_model: object) -> str:
    """Full pipeline HTML for the Bioscience Reports 31123167.pdf fixture; skip if
    the model is absent.  Writes tests/fixtures/31123167.html for visual inspection."""
    if not _BSR_PDF.exists():
        pytest.skip(f"Fixture PDF not found: {_BSR_PDF}")
    return _run_pipeline_to_file(_BSR_PDF, ocr_model)


@pytest.mark.integration
class TestBioscienceReportsRunningHeader:
    """31123167.pdf (Bioscience Reports) repeats a per-page running-header journal
    citation ("Bioscience Reports (2019) 39 BSR20190715"), which LightOnOCR emits as
    a markdown heading; it must be stripped as running furniture, not promoted to a
    body <h2> on every page.  The first-page "OPEN ACCESS" banner must likewise not
    glue onto the front of the abstract."""

    _CITATION = "Bioscience Reports (2019) 39 BSR20190715"

    def test_running_header_citation_not_a_body_heading(self, bsr_html: str) -> None:
        assert f"<h2>{self._CITATION}</h2>" not in bsr_html
        # the repeated header is stripped, not scattered through the body
        assert _body(bsr_html).count(self._CITATION) == 0

    def test_open_access_banner_not_glued_to_abstract(self, bsr_html: str) -> None:
        # the headingless abstract is recovered into the abstract section, without the
        # OPEN ACCESS banner prefix the model bolds onto its front
        # (<strong>OPEN ACCESS</strong> Hydroxy…)
        assert "Hydroxyethylsulfonate" in _abstract(bsr_html)
        assert "Hydroxyethylsulfonate" not in _body(bsr_html)
        assert "<strong>OPEN ACCESS</strong>" not in bsr_html

    def test_license_footer_relocated_to_panel(self, bsr_html: str) -> None:
        # "© 2019 The Author(s). … (CC BY)." recurs on every page; the furniture strip
        # drops the per-page repeats, but the article's license is front matter — one
        # copy must land in the Metadata panel (not be discarded entirely), and never
        # in the body.
        assert "Creative Commons Attribution License" in _metadata(bsr_html)
        assert "Creative Commons Attribution License" not in _body(bsr_html)

    def test_title_and_references_intact(self, bsr_html: str) -> None:
        header = bsr_html[bsr_html.find("<header>") : bsr_html.find("</header>")]
        assert "sulfoacetaldehyde reductase" in header
        # author–year references, each its own <p>, not glued into one block
        refs = bsr_html[bsr_html.find("<h2>References</h2>") :]
        assert refs.count("<p>") >= 8

    def test_continuation_page_references_not_one_block(self, bsr_html: str) -> None:
        # The references continue onto a second page where the OCR drops the
        # markdown-list period from each marker ("9 Peck, …" not "9. Peck"), so the
        # entries arrive as plain <p>s that — each trailing off in a DOI rather than
        # terminal punctuation — were chained into one ~5 kB paragraph.  They must be
        # folded into the bibliography list, each its own item.
        refs = bsr_html[bsr_html.find("<h2>References</h2>") :]
        # no entry is left as a loose, period-less numbered <p> (the bug symptom)
        assert not re.search(r"<p>\s*\d+\s+[A-Z][a-z]+,", refs)
        # the entries render as list items, each its own <li>, not one giant blob
        items = re.findall(r"<li>.*?</li>", refs, re.DOTALL)
        assert len(items) >= 20
        assert max(len(it) for it in items) < 1500
        # a late continuation-page entry is folded into the list with its redundant
        # leading number dropped (the <ol> renders the number itself)
        assert re.search(r"<li>\s*<p>Suzek, B\.E\.", refs)

    def test_author_markers_rendered_as_superscripts_in_byline(
        self, bsr_html: str
    ) -> None:
        # The author markers ("1,*", "2,3") render as superscripts in the byline,
        # not flattened to inline text; the markdown emphasis parser must not eat
        # the "*" markers (which corrupted "1,*" into "1,," — the bug symptom).
        header = bsr_html[bsr_html.find("<header>") : bsr_html.find("</header>")]
        byline = header[header.find("<p>") :]
        assert "<sup>" in byline
        assert "<em>" not in byline
        assert ",," not in byline

    def test_author_contribution_footnote_in_metadata_panel(
        self, bsr_html: str
    ) -> None:
        # "These authors contributed equally to this work." is an author footnote
        # the OCR stranded among the Introduction paragraphs; it belongs in the
        # Metadata panel, not the body.
        assert "contributed equally" not in _body(bsr_html)
        panel = _metadata(bsr_html)
        assert "contributed equally" in panel
        # its "*" marker (swallowed into emphasis by the OCR) is restored so the
        # note still references the "*"-tagged authors in the byline
        assert re.search(r"\*\s*(?:<em>)?These authors contributed equally", panel)

    def test_chemical_super_and_subscripts_rendered(self, bsr_html: str) -> None:
        body = _body(bsr_html)
        # superscript: NAD$^{+}$ -> NAD⁺ (end-to-end, not just the unit test)
        assert "NAD⁺" in body
        # subscript: "toxic H$_{2}$S" -> H<sub>2</sub>S, locking the <sub> path
        # through a real fixture (otherwise only unit-tested)
        assert "H<sub>2</sub>S" in body

    def test_truncated_assay_clause_recovered_from_text_layer(
        self, bsr_html: str
    ) -> None:
        # On page 5 the OCR drops the tail of the last line above the footer,
        # transcribing "…with 15 s intervals" and losing "for 2–3 min at RT.";
        # _reconcile_text_layer restores it from the PDF text layer (en-dash kept).
        body = _body(bsr_html)
        assert "15 s intervals for 2–3 min at RT." in body

    def test_copyright_footer_stripped_from_body(self, bsr_html: str) -> None:
        # The full-sentence open-access footer ("© 2019 The Author(s). … Portland
        # Press Limited … (CC BY).") repeats on every page; the OCR emits it inline,
        # where — lacking it as furniture — the cross-page merge glued it onto the
        # prose it interrupts.  It must be stripped as running furniture, not appear
        # in the body.
        assert "Portland Press Limited" not in _body(bsr_html)

    def test_download_gutter_stamp_stripped(self, bsr_html: str) -> None:
        # The rotated "Downloaded from http://portlandpress.com/…​.pdf by guest on …"
        # stamp running up the gutter likewise recurs (differing only in digits) and
        # must not survive in the body breaking a paragraph.
        assert "Downloaded from" not in _body(bsr_html)
        assert "portlandpress.com/bioscience" not in _body(bsr_html)

    def test_cross_page_paragraph_merges_past_figure_legend(
        self, bsr_html: str
    ) -> None:
        # With the interrupting footer gone, the Discussion paragraph split across
        # the page break ("… (only 28% identity)," / "and putative substrate-binding
        # …") must rejoin — stepping over the two figures between its halves — rather
        # than glue onto Figure 7's legend, which the OCR stranded as a loose <p>
        # after the title-only "Figure 7. …" caption.
        body = _body(bsr_html)
        assert "(only 28% identity), and putative substrate-binding" in body
        # the stranded legend belongs in the figcaption, not the body
        legend = "Many of the close homologs of"
        in_figcaption = any(
            legend in c
            for c in re.findall(r"<figcaption>(.*?)</figcaption>", bsr_html, re.DOTALL)
        )
        assert in_figcaption
        assert f"<p>{legend}" not in body

    def test_split_panel_caption_folded_into_figure(self, bsr_html: str) -> None:
        # The model splits Figure 1's "(A) … (B) … (C) …" panel descriptions into a
        # paragraph separate from the caption header; they belong to the figcaption,
        # not the body.
        panel = "(A) Gene clusters containing the sulfoacetaldehyde reductases IsfD"
        in_figcaption = any(
            panel in c
            for c in re.findall(r"<figcaption>(.*?)</figcaption>", bsr_html, re.DOTALL)
        )
        assert in_figcaption
        assert f"<p>{panel}" not in _body(bsr_html)

    def test_figure_captions_have_balanced_emphasis(self, bsr_html: str) -> None:
        # The model bolds a caption and italicises the species at its end
        # ("**Figure 2. … of *BkTauF***"); the inline renderer must close the tags in
        # order, never the mis-ordered "<em>…</strong></em>" the old regex produced,
        # and never leak a stray "BkTauF***".
        captions = re.findall(r"<figcaption>(.*?)</figcaption>", bsr_html, re.DOTALL)
        assert any("BkTauF" in c for c in captions)
        for c in captions:
            assert "</strong></em>" not in c
            assert "***" not in c

    def test_dropped_figure_4_recovered(self, bsr_html: str) -> None:
        # The model drops Figure 4 entirely from page 7 — no ![image] placeholder
        # and no "Figure 4" caption — and reproduces the omission on a whole-page
        # re-OCR.  _recover_dropped_figures spots the gap in the caption numbering
        # (1,2,3,_,5,6,7) against the text layer, re-OCRs a tight crop of the figure
        # band, and splices the recovered figure back in.
        size = _figure_size_by_caption(bsr_html, "Crystal structures of", _OUTPUT_DIR)
        assert size is not None, "Figure 4 not recovered into the document"
        # a real two-panel structural figure, not a sliver
        assert size[0] > 400 and size[1] > 100, f"Figure 4 crop too small: {size}"
        # the recovered caption lives only in its figcaption, not duplicated as body
        # prose (the spliced figure block carries the caption with it)
        body_no_figures = re.sub(
            r"<figure>.*?</figure>", "", _body(bsr_html), flags=re.DOTALL
        )
        assert "Figure 4. Crystal structures" not in body_no_figures


@pytest.mark.integration
class TestBioscienceReportsTableBold:
    """Table 1's boldfaced section rows (Data collection, Refinement, …) are
    transcribed as plain cells by the OCR; the table-cell bold recovery re-applies
    <strong> from the PDF text-layer font weight."""

    _BOLD_ROWS = (
        "Data collection",
        "Cell dimension",
        "Refinement",
        "Number of atoms",
        "Ramachandran plot",
    )

    def test_bold_section_rows_recovered(self, bsr_html: str) -> None:
        bolded = {
            row
            for row in self._BOLD_ROWS
            if re.search(rf"<td[^>]*><strong>{re.escape(row)}", bsr_html)
        }
        # most section rows come back bold (tolerate an OCR cell-text garble that would
        # make one row miss the text-layer match)
        assert len(bolded) >= 3, f"only {bolded} bolded"

    def test_plain_data_cell_not_bolded(self, bsr_html: str) -> None:
        # a numeric data value (the X-ray wavelength) is not spuriously bolded
        assert "<strong>0.9795" not in bsr_html


@pytest.fixture(scope="session")
def jafc_html(ocr_model: object) -> str:
    """Full pipeline HTML for the J. Agric. Food Chem. 31298526.pdf fixture; skip if
    the model is absent.  Writes tests/fixtures/31298526.html for visual inspection."""
    if not _JAFC_PDF.exists():
        pytest.skip(f"Fixture PDF not found: {_JAFC_PDF}")
    return _run_pipeline_to_file(_JAFC_PDF, ocr_model)


@pytest.mark.integration
class TestJafcAbstractAndByline:
    """31298526.pdf (J. Agric. Food Chem., ACS) prints the abstract as an inline
    "ABSTRACT: …" bold label (no heading), and tags authors with LaTeX footnote
    symbols (\\ddagger, \\S).  The abstract must stay visible in the abstract section
    (not swept into the Metadata panel as front matter), and the LaTeX must render as
    the ‡/§ glyphs, not leak raw into the byline/affiliations."""

    _ABSTRACT = "L-Valine belongs to the branched-chain amino acids"

    def test_abstract_in_section_not_panel(self, jafc_html: str) -> None:
        assert self._ABSTRACT not in _metadata(jafc_html)
        sec = jafc_html.find("<section class='abstract'>")
        assert sec >= 0
        assert self._ABSTRACT in jafc_html[sec : jafc_html.find("</section>", sec)]

    def test_no_raw_latex_in_byline_or_affiliations(self, jafc_html: str) -> None:
        assert "\\ddagger" not in jafc_html
        assert "\\S" not in jafc_html
        # and positively: the \ddagger/\S fallback rendered the glyphs (a regression
        # that dropped the fallback to "" would pass the negative checks alone).
        assert "‡" in jafc_html
        assert "§" in jafc_html

    def test_byline_markers_rendered_as_superscripts(self, jafc_html: str) -> None:
        # The multi-symbol author markers (e.g. "*,†,‡,§", the ‡/§ from the LaTeX
        # fallback) render as superscripts, not flattened inline text, and the
        # markdown parser does not eat the "*" (the "1,," corruption symptom).
        header = jafc_html[jafc_html.find("<header>") : jafc_html.find("</header>")]
        byline = header[header.find("<p>") :]
        assert "<sup>" in byline
        assert "<em>" not in byline
        assert ",," not in byline

    def test_title_in_header(self, jafc_html: str) -> None:
        header = jafc_html[jafc_html.find("<header>") : jafc_html.find("</header>")]
        assert "Ketol-Acid Reductoisomerase" in header

    def test_submission_dates_in_panel_not_body(self, jafc_html: str) -> None:
        # The page-bottom submission-history footer ("Received: May 25, 2019" …,
        # US "Month DD, YYYY" order, unbolded) is relocated to the Metadata panel;
        # left in the body it would interrupt the cross-page paragraph it sits in.
        panel = _metadata(jafc_html)
        body = _body(jafc_html)
        for line in ("Received: May 25, 2019", "Published: July 12, 2019"):
            assert line in panel
            assert line not in body

    def test_keywords_relocated_to_panel_not_body(self, jafc_html: str) -> None:
        # The "KEYWORDS: …" label (colon outside the bold) terminates the inline
        # abstract, then is relocated from the body head to the Metadata panel rather
        # than stranded as the first body paragraph before INTRODUCTION.
        kw = "ketol-acid reductoisomerase"
        panel = _metadata(jafc_html)
        body = _body(jafc_html)
        assert "KEYWORDS" in panel
        assert kw in panel
        assert "<strong>KEYWORDS" not in body

    @pytest.mark.parametrize(
        "title",
        [
            "Table 1. Kinetic Analysis of CgKARI",
            "Table 2. Data Collection and Structural Refinement Statistics",
        ],
    )
    def test_table_title_rendered_as_caption(self, jafc_html: str, title: str) -> None:
        # The model bakes each table's title into the first row as a lone spanning
        # "<th colspan=N>Table N. …</th>" cell; it is hoisted into a <caption> so it
        # renders semantically like every other fixture's table title, not as a header.
        assert f"<caption>{title}</caption>" in jafc_html
        assert f'colspan="5">{title}' not in jafc_html
        assert f'colspan="2">{title}' not in jafc_html

    def test_paragraphs_after_citation_superscript_not_merged(
        self, jafc_html: str
    ) -> None:
        # Each of these prose paragraphs follows one that ends with a citation
        # superscript ("…humans.<sup>15–18</sup>", "…sequence.<sup>21,23</sup>",
        # "…software.³²"); the citation must not hide the terminal period and glue
        # the new paragraph onto the previous one, so each opens its own <p>.
        for opener in (
            "In <em>C. glutamicum</em>, L-valine is synthesized by the BCAA",
            "In the present study, we determined the crystal structure of KARI",
            "<strong>Metal Activity Assay.</strong> The metal activity was measured",
        ):
            assert f"<p>{opener}" in jafc_html

    def test_inline_equations_rendered_not_leaked(self, jafc_html: str) -> None:
        # The grid-size/cell-parameter equations the model wrapped in math mode
        # ("$x = 22$", "$a = 84.9$") must render with their '$' delimiters dropped,
        # not leak the raw "$… = …$" into the prose.  Their single-letter variables
        # render italic (<em>x</em>/<em>a</em>) per the math-variable italicization.
        assert "<em>x</em> = 22" in jafc_html
        assert "$x = " not in jafc_html
        assert "$a = " not in jafc_html
        assert "<em>a</em> = 84.9" in jafc_html


@pytest.mark.integration
class TestJafcTruncatedPageRecovery:
    """31298526.pdf page 2 carries the large Table 2; its OCR overruns the default
    token budget and truncates mid-table, dropping the rest of the table and all the
    prose after it (the pH-activity discussion).  ``_ocr_page`` detects the
    ``finish_reason == "length"`` cut and re-OCRs the page with the full remaining
    context window, so the table closes and the dropped prose returns."""

    def test_table_2_complete(self, jafc_html: str) -> None:
        # Table 2 must fully render — its last rows (and a closing </table>) are what
        # the truncation dropped.  "redundancy" and the cell after it sit near the
        # table's end, well past the 2048-token cut point.
        assert "Data Collection and Structural Refinement" in jafc_html
        assert "redundancy" in jafc_html
        # the truncated dump left "</tr" unclosed; a recovered page closes the table
        assert jafc_html.count("<table") == jafc_html.count("</table>")

    def test_post_table_prose_recovered(self, jafc_html: str) -> None:
        # The pH-activity prose that followed Table 2 on the page was dropped with the
        # truncation; the re-OCR brings it back.
        assert "highest enzyme activity at pH 8" in jafc_html

    def test_table_2_rebuilt_from_text_layer(self, jafc_html: str) -> None:
        # The OCR mangles Table 2 the same way every run (drops the empty header cell,
        # shifting the first rows off by one and losing the wavelength value); the
        # text-layer repair rebuilds it deterministically, so these are hard assertions.
        table = next(
            t
            for t in re.findall(r"<table\b.*?</table>", jafc_html, re.DOTALL)
            if "Data Collection and Structural" in t
        )
        # the empty header cell is restored: CgKARI_NADP⁺ alone in column 2
        assert "<tr><td></td><td>CgKARI_NADP⁺</td></tr>" in table
        # the off-by-one is fixed: 6JX2 is the PDB code's value, not wavelength's
        assert "<td>PDB code</td><td>6JX2</td>" in table
        assert re.search(r"wavelength \(Å\)</td><td>0\.97934", table)
        # the dropped tail (Ramachandran-plot stats) is recovered
        assert all(w in table for w in ("favored", "allowed", "outliers"))

    def test_table_2_has_no_decode_loop_explosion(self, jafc_html: str) -> None:
        # The page-level re-OCR (full-window retry on the truncated Table 2 page) can
        # decode-loop and trail the real table with dozens of byte-identical empty
        # rows ("RAMS Deviations" explosion).  ``test_table_2_complete`` stays green
        # through it — its anchor strings precede the loop and the table is still
        # balanced — so assert the explosion is gone directly: no row repeats more
        # than a sane handful of times in a row.
        rows = re.findall(r"<tr\b.*?</tr>", jafc_html, re.DOTALL | re.IGNORECASE)
        longest_run = max_run = 1
        for prev, cur in zip(rows, rows[1:], strict=False):
            max_run = max_run + 1 if cur == prev else 1
            longest_run = max(longest_run, max_run)
        assert longest_run <= 3, f"decode-loop explosion: {longest_run} identical rows"


@pytest.mark.integration
class TestStructuredDocument:
    """``lightonocr_pdf_to_document`` wires the full pipeline to the structured
    return — HTML plus the title/byline/DOI the annotation-hub Reference row needs
    without re-parsing the HTML (Tasks B).  The DOI is deterministic (text-layer), the
    title/byline track the live OCR's header."""

    def test_doi_from_text_layer(self, parsed_document: object) -> None:
        assert parsed_document.doi == "10.1002/bmb.21202"

    def test_title_and_byline_populated(self, parsed_document: object) -> None:
        assert "Hydroxypropyl-Coenzyme M Dehydrogenases" in parsed_document.title
        assert "Daniel D. Clark" in parsed_document.byline

    def test_html_is_a_full_document(self, parsed_document: object) -> None:
        html = parsed_document.html
        assert html.startswith("<!DOCTYPE html>")
        assert "<h1>" in html and html.rstrip().endswith("</html>")
