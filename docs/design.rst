Design notes
============

This page is the orientation a contributor needs before changing the pipeline:
how the stages fit together, and — more importantly — the recurring judgement
calls the code makes and *why* it makes them the way it does.  The per-function
docstrings in the :doc:`api/modules` carry the detail; this page is the shape
they hang on.

The pipeline at a glance
------------------------

``manuscribe`` converts a PDF to HTML — self-contained by default — by handing each
rendered page to **LightOnOCR-2-1B-bbox**, an end-to-end vision-language model that
returns the whole page as markdown — reading order, emphasis, ``<table>`` HTML,
LaTeX math, and figure crop boxes appended to ``![image]`` placeholders.  The model
runs out of process in a vLLM server; everything on this side of that call is
deterministic clean-up, and the entry point returns a
:class:`~manuscribe.pipeline.assemble.ParsedDocument` — the HTML plus the plain-text
title, byline and best-effort DOI a consumer would otherwise re-parse out of it.

The flow (design *B-prime*)::

    PDF
     │  render.py           rasterize each page to a PIL image (at the OCR budget)
     ▼
    page images
     │  model.py            LightOnOCR via the vLLM server → per-page markdown
     ▼                        ← the OCR seam (the GPU runs out of process)
    per-page markdown
     │                      post-OCR passes over the PDF text layer, held open once
     │                      by layers._DocumentLayers — recover what the page dropped:
     │  tables/             re-OCR a dropped table crop; rebuild a mangled 2-column
     │                        table; re-bold cells the text layer marks bold
     │  recover_figures.py  recover a whole figure the page pass dropped
     │  reconcile.py        splice a short OCR-truncated tail from the text layer
     │  doi.py              scan the first page for the article DOI
     ▼
    per-page markdown (recovered)
     │  markdown.py         markdown → list of block-HTML strings
     │  latex.py            inline $…$ → <sup>/<sub>/Unicode
     │  figures.py          ![image] box → cropped <figure> (via the ImageSink seam)
     ▼
    flat block stream
     │  merge.py            re-stitch column-split paragraphs; fold table captions
     │  classify.py         split title / byline / abstract / front matter / body
     ▼
    assemble.py             document shell → ParsedDocument(html, title, byline, doi)

The single most important structural decision is the **purity seam**.  The GPU work
— the LightOnOCR model itself — runs *out of process* in a vLLM server (see
``deploy/vllm/``), so :mod:`manuscribe.pipeline.model` is a thin HTTP client rather
than an in-process model load, and the package depends on neither torch nor
transformers.  What the seam buys is unchanged: everything downstream of the OCR call
— the entire clean-up, merge and classification surface, which is where the bugs and
the design live — is a pure function of ``(list[page_markdown], list[page_image])``.
That core is :func:`~manuscribe.pipeline.assemble._assemble_html`, which
:func:`~manuscribe.pipeline.assemble.lightonocr_pdf_to_document` wires rendering, OCR
and the recovery passes around.  The core is what makes the hard parts unit-testable
without a GPU *or* a server: feed synthetic markdown plus blank images straight in and
assert on the HTML — and the seam itself is now mockable with an
``httpx.MockTransport``, so the network contract is testable too.  When you add a
stage, keep it on the pure side of the seam unless it genuinely needs pixels or the
model.

Three kinds of work sit *outside* the pure core, upstream of it.
:mod:`~manuscribe.pipeline.render` turns the PDF into page images; the OCR seam turns
images into markdown; and a family of **recovery passes** repairs that markdown before
it reaches the core.  Each re-reads the page — some re-OCR a tight crop through the
seam, all of them consult the **PDF text layer** — to recover content the full-page
pass dropped or mangled: a clipped table (:mod:`manuscribe.pipeline.tables`), a whole
missing figure (:mod:`~manuscribe.pipeline.recover_figures`), a truncated sentence tail
(:mod:`~manuscribe.pipeline.reconcile`).  They are the IO-bound part by design, kept
upstream so the core still sees only ``(markdown, images)`` and stays model-free; their
own pure helpers — cell extraction, bbox-from-the-text-layer geometry, caption folding
— are unit-tested directly, and only the render-and-re-OCR wrappers touch the server.
All of them read one shared, already-open document
(:class:`manuscribe.pipeline.layers._DocumentLayers`), so a page is text-extracted once
rather than once per pass.

A note on testing: because real OCR output is the thing the heuristics actually
face, the highest-value regression tests replay *recorded* model markdown through
the pure core (split a dumped transcript on its page markers, pair each page with
one blank image) rather than hand-writing idealized markdown.

The trade-offs, and which way each one leans
--------------------------------------------

OCR clean-up is a long series of "is this X or Y?" decisions made on noisy
evidence.  There is no choice that is right every time, so each decision is
deliberately *biased* toward the failure that is cheaper to live with.  The three
biases below are not local quirks — they are applied consistently, and knowing
them tells you which way to push when you hit a new edge case.

Figures: rather lose a strip of margin than a strip of figure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A clipped figure is a serious, irrecoverable error — the reader simply never sees
that part of the image.  A crop that runs a few pixels into surrounding
whitespace is cosmetic.  So when the model's crop box and the actual figure
disagree, the code leans toward **including too much rather than too little**, and
that bias is intentional.  It shows up in three places:

* **Four-edge recovery**
  (:func:`~manuscribe.pipeline.figures._extend_edge`).
  The model's box routinely clips a figure's edges — most often the bottom, but
  any side too (a wide figure's right edge, a plot's left axis line, a top panel
  label or frame line the box cuts straight through).  Rather than trust the box,
  the crop grows over figure content and stops at the whitespace gap beyond it (the
  space before the caption below, or the page margin / inter-column gutter / prose
  above).  All four edges share one body, parameterized by ``_EDGE_PARAMS``: the
  axis to reduce when building the ink profile, and whether the edge grows toward
  zero (top/left, which also reverse the profile so the scan runs outward from the
  edge) or away from it (bottom/right).  The top edge is grown like the others but
  leans on the **leading-gap guard** to stay honest: it grows only when the box
  cuts straight through figure ink at the top and declines when a whitespace gap
  already separates the box from what is above (a caption or the preceding
  paragraph), so it recovers a clipped panel label without reaching up into the
  prose.  The "is this line blank?" threshold (``_FIGURE_BLANK_LINE_FRAC``) is kept
  near-empty on purpose: a sparse figure line — a thin axis, content narrower than
  the box — must count as *content* so growth doesn't stop early and behead the
  figure.  When a caption follows the figure, bottom growth tightens its stop gap
  to the smaller ``_FIGURE_BLOCK_GAP_FRAC`` so it halts at the figure↔caption gap
  rather than stepping over it into the caption.

* **Over-segmentation merge**
  (:func:`~manuscribe.pipeline.figures._cluster_figure_boxes`,
  :func:`~manuscribe.pipeline.figures._union_box`).  The model sometimes splits one
  tall figure into stacked boxes.  Boxes that share a column and sit close
  vertically are unioned back into one crop, so a panel never gets dropped on the
  floor between two boxes.  The same over-segmentation also emits the bare panel
  labels (``A``, ``B``) as their own text blocks;
  :func:`~manuscribe.pipeline.assemble._parse_page_blocks` drops a lone
  single-letter label that abuts a figure placeholder, since it belongs to the
  figure (baked into the crop), not the prose.

* **Render resolution** (:mod:`manuscribe.pipeline.render`).  Each page is rasterized
  straight to the model's long-side budget (capped at 200 DPI), so the crop has the
  model's full working resolution to recover from — without supersampling every page
  to 200 DPI only to shrink it back.

The bias is held in check in three ways, each keyed on a signal that growth would
swallow prose rather than recover figure.  First, *ambiguity*: if the ink past the
box runs all the way to the search cap with no clear whitespace gap, that is
usually a correct box sitting above caption or body text, and extending would
swallow prose — so the box is left untouched.  Second, a *leading gap*
(:func:`~manuscribe.pipeline.figures._ink_run_end`): growth recovers only content
*contiguous* with the box; a clean whitespace gap before the first ink means the
box already ends at the figure boundary and what follows is separated from it (a
caption below, a neighbouring column to the right), so growth is declined.  A real
clip leaves no such gap — it cuts straight through figure ink.

Third — and this is where page geometry alone is not enough — a caption sits close
*below* the figure.  When a clear gap separates the figure tail from the caption,
the tightened bottom-growth gap (above) already halts growth at it, so the caption
is never recovered in the first place.  But the caption can abut the recovered tail
with no gap, or — when the figure is *itself* text (a sequence alignment, a data
table) whose caption is pixel-identical to it — the model can draw the box low
enough to bake the caption *inside* it.  In both, the geometry waves it through, and
a *document-level* signal breaks the tie: the OCR already emitted the caption as its
own text block, so its words are known.  That signal is only as good as the caption
text actually attached to the figure: when the model splits a caption into a bare
``FIG. N`` label and a following descriptive block, :func:`~manuscribe.pipeline.assemble._parse_page_blocks`
rejoins them so the figure owns the *whole* caption — otherwise the trim is handed
only "FIG. N", finds none of those words in the baked band, and leaves the caption
baked in (and the descriptive sentence stranded in the body).

So when a caption is present and an OCR seam is available,
:func:`~manuscribe.pipeline.figures._trim_baked_caption` is **authoritative**.  It
re-OCRs the trailing text bands (cheap and rare — gated on the band being text-
rather than dense-figure-textured, and confined to the bottom of the crop) and cuts
at the caption's true top, dropping a band that *reproduces the caption's words*.
Two guards keep that honest: a high word-overlap bar, so a figure row the caption
merely *names* (an alignment labels its own rows — ``BsSDH``, ``HmSDH`` …) isn't
mistaken for the caption; and a distinct-word floor, so a row the model OCRs into a
repeated-token wall (``BMSDH BMSDH BMSDH …``) — which would otherwise score ~1.0
against a caption that names it — is rejected as degenerate.  Crucially, finding
*no* caption in the (gap-bounded) recovered region, it keeps the crop as grown — so
a clipped *sparse* tail (thin axes, tick labels), whose short ink runs a pixel-only
test would mistake for prose, survives, because its words (axis numbers) are not the
caption's.

The pixel-only run-length test
(:func:`~manuscribe.pipeline.figures._trim_swallowed_caption`,
``_FIGURE_PROSE_RUN_FRAC`` — caption text is short letter runs, recoverable figure
content long continuous runs) remains only as the backstop when no OCR seam is
available, or a band re-OCR fails on the shared GPU.  Without the OCR signal it
cannot tell a sparse tail from a caption and may trim such a tail — an accepted loss
in that degraded mode, since the caption is always re-emitted as ``<figcaption>``
and baking a whole caption into the image is worse.

The rule, then, is "expand toward the figure whenever the page shows you where it
ends; don't expand blindly into text — and when the OCR tells you a caption is
down there, trust prose-shaped pixels, or the caption's own words, to find it."

All of this assumes the page pass at least *boxed* the figure.  When a dense page
drops one whole — no ``![image]`` at all — the clip logic has nothing to grow, so a
separate recovery pass (:func:`~manuscribe.pipeline.recover_figures._recover_dropped_figures`)
takes over: it diffs the figure numbers the OCR emitted against the figure numbers in
the PDF text layer, localizes each missing one's caption from the layer, re-OCRs a
tight figure+caption crop, and splices the recovered ``![image]`` back through the
unchanged figure path.

Text merging: when the join is uncertain, don't join
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Re-stitching paragraphs that a two-column layout split across a column or page
break (:func:`~manuscribe.pipeline.merge._merge_split_paragraphs`) leans the
*opposite* way from figures: a wrong merge fuses two unrelated sentences into
grammatical nonsense, which is worse than leaving a paragraph visibly broken in
two.  So the merge fires only when the evidence is positive — the first fragment
lacks terminal punctuation *and* the continuation is not a new sentence, caption,
enumeration item, or heading (those act as hard barriers).  Two guards are worth
internalizing because they encode this caution:

* The **function-word guard** (``_FUNCTION_WORD_END_RE``):
  a fragment ending in "the", "of", "and"… is grammatically incomplete, so its
  continuation must be lowercase.  If the next block starts with a capital, the
  real continuation was probably dropped by OCR, and the merge is *refused*
  rather than gluing unrelated text together.

* The **identifier exception** (``_MIDSENTENCE_HEAD_RE``):
  a continuation opening with a scientific identifier — an all-caps acronym
  ("TRII", "DNA") or a mixed-case gene/protein name ("SpRDH", "PtTRI") — is
  mid-sentence, not a new-sentence capital, so it is allowed through.  The tell is
  an uppercase letter *inside* the first token (a sentence-opening word is
  capitalized only on its first letter).  Without this, a clause split as
  "…TRI and" / "TRII compete…", or "…carbon metabolism. The" / "SpRDH operon…",
  is wrongly left in two.

A subtlety that bites if you touch these predicates: terminal punctuation often
hides inside a closing inline tag (``…carboxylase.</em>``) or behind a trailing
citation superscript (``…humans.<sup>15–18</sup>``, ``…software.³²``).  Run the
sentence-end check through :func:`~manuscribe.pipeline.text._ends_sentence`, which
strips a trailing citation before testing the **visible** text
(:func:`~manuscribe.pipeline.text._visible_text`) — never a raw match on the block
HTML.  The same predicate is shared by the classification side
(``_looks_like_body_prose``, ``_is_frontmatter_text``, ``_ends_like_sentence``) so
a body sentence ending in a citation isn't mistaken for a label and hidden.

Classification: never hide the body
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Front matter (affiliations, keywords, correspondence, submission dates, the
PLOS-style citation sidebar) is relocated into a *collapsed* "Metadata" panel so
the article opens with prose.  That makes one error mode far worse than the
other: metadata left visible in the body is a blemish, but body prose
misclassified as metadata is **hidden from the reader entirely**.  So every
front-matter heuristic demands strong, positive evidence and is happy to
under-collect:

* An affiliation OCR'd without its author superscript is accepted only on a
  structural tell — a comma-separated address ending in a postal code or a
  recognised country (:func:`~manuscribe.pipeline.affiliations._is_affiliation_line`).
  An address with neither is left visible rather than risk hiding a sentence that
  merely names a university.

* Stray footer metadata is relocated position-independently only on its own
  strong evidence — a known publishing label, two or more metadata tokens (DOI,
  e-mail, date), or a fixed boilerplate phrase
  (:func:`~manuscribe.pipeline.classify._is_stray_metadata`) — never because of
  where it sits.

* As a final backstop, :func:`~manuscribe.pipeline.classify._extract_front_matter`
  refuses to hide the body when the run of "front matter" would be the *entire*
  document: an all-front-matter result signals a detection failure, not a
  metadata-only paper.

Other deliberate choices
------------------------

Math is reduced, not rendered.  :mod:`manuscribe.pipeline.latex` converts inline
``$…$`` to Unicode super/subscripts or HTML ``<sup>``/``<sub>``; full equation
rendering is out of scope (a later MathJax option).  The guiding rule when a
construct can't be reduced is *lossless degradation*: keep the TeX literal so a
later pass can render it, rather than crash the page or leak ``%s`` substitution
templates.  Symbol-command translation is delegated to ``pylatexenc``'s
maintained macro table instead of a hand-curated map, with one documented
override (``^\circ`` → "°", the degree idiom).  A ``$…$`` span with *no* TeX
markup is reduced too when it is plain inline math the model wrapped in math mode
— an identifier-led equation or variable list (``$x = 22$``, ``$a, b, c$``) sheds
its delimiters (:func:`~manuscribe.pipeline.latex._is_inline_math_span`).  Currency
is protected by the same test: it is digit-led (``$5``), so a stray ``$…$``
pairing over "$5 … $10" is left verbatim, and a span carrying a complete HTML tag
is rejected outright — the pre-markdown stream still holds raw ``<sup>``/``<td>``
whose ``<``/``>`` would otherwise read as a math relation.  A reduced span also
italicises a standalone single letter as a variable (``$x = 22$`` →
``<em>x</em> = 22``; :func:`~manuscribe.pipeline.latex._italicize_math_variables`),
with one carve-out that is itself an "is this X or Y?" call: a letter in *unit
position* — trailing a numeric magnitude, ``5 V``, ``9.8 m/s²`` — is a unit symbol,
not a variable, and stays upright.

OCR decoding is greedy (:func:`~manuscribe.pipeline.model._ocr_page`).  OCR wants
the single most-likely transcription, and a deterministic decode avoids
run-to-run drift — notably a figure box that occasionally over-segments into two
stacked crops on a sampled decode.  A page dense enough to overrun the token
budget truncates mid-output — dropping the rest of a large table and every block
after it — so ``_ocr_page`` detects the ``finish_reason == "length"`` cut and
re-OCRs once with the full remaining context window, keeping the longer result.

The pages of one document are OCR'd *concurrently* against the shared server
(:func:`~manuscribe.pipeline.model._ocr_pages`), which lets vLLM's continuous batching
engage and is most of the throughput.  The cost is that greedy decode no longer buys
byte-exact reproducibility — batch composition shifts with request timing, so two runs
can differ in the small — which is why the tests assert on structure and substrings,
never a byte-exact OCR snapshot.  A transient blip on one page (a connection reset, a
503, an out-of-memory) would otherwise abort the whole document through the propagated
exception, so a page POST carries a bounded backoff retry; a failure that outlives the
retries surfaces as a typed
:class:`~manuscribe.pipeline.errors.OcrUnavailableError` /
:class:`~manuscribe.pipeline.errors.OcrResponseError` a batch worker can classify for
re-queue, not a raw ``httpx`` traceback.

Leading image-only pages are skipped *before* OCR.  An ad-prefixed article opens with a
full-page-image cover the model would transcribe to garbage the assembly discards
anyway — one or two wasted ~10–24 s round trips, more wall-clock than any other saving
on this side of the seam.  :func:`~manuscribe.pipeline.layers._leading_image_only_pages`
reads the PDF text layer first and, for the *leading run* of pages whose layer is
essentially empty (below a small alphanumeric floor,
:func:`~manuscribe.pipeline.layers._is_trivial_text_layer`), sends nothing to the model;
each skipped page is padded with empty markdown so the page↔image alignment the recovery
passes and the assembly depend on survives, and the assembly's own article-start scan
then drops the pads exactly as it dropped an OCR'd-but-heading-less cover before.  This
is the one place the pipeline accepts a *possible* content loss for a performance gain,
the opposite lean from the content-preserving biases above, and that is worth being
explicit about: the empty-text-layer signal cannot tell an ad cover from a *scanned*
first page carrying a real title and abstract, so a mixed-source PDF — an image page 0
ahead of a born-digital body — would lose that page un-OCR'd.  It is accepted because it
is vanishingly rare (a modern article's first page has a text layer) and the only way to
be sure is to OCR the page, the exact cost the skip exists to avoid.  Two guards keep the
blast radius small: the skip is the leading run only, never an interior page, and a
document whose text layer is empty *everywhere* (a wholly-scanned paper) is left entirely
to the model.  Deciding ad-versus-article for a page that *does* carry text stays the
model's job (:func:`~manuscribe.pipeline.classify._is_article_page_md`); should a real
fixture ever lose content here, the fix is a cap on how many leading pages may be
skipped, not removing the skip.

Tables are re-OCR'd from a tight crop (:mod:`manuscribe.pipeline.tables`).  The
full-page pass silently drops small in-table content — a column-spanning
subheader, the ``colspan``/``rowspan`` structure of a header — and does so at
*every* render resolution; the loss is to the dense full page, not to pixels.
Re-OCRing the table on its own recovers it.  The hard part is locating the table
to crop: the model boxes figures but not *upright* tables (it does box a sideways
one — see below), and ignores any prompt asking it to, so the geometry has to come
from elsewhere.  The choice (Option 3) is to
use the **PDF text layer for geometry only** — a deliberately narrow carve-out of
B-prime's "OCR reads, nothing else does" rule.  The cells the full-page pass
*did* capture are matched against the text layer (after an NFKD-plus-alphanumeric
fold that closes the encoding gap — superscripts, the micro sign, the assorted
dashes — between OCR'd cell and text-layer glyph) to seed a bounding box, which
is then grown line-by-line through the table's rows and **halted at the wider gap
to body prose** (the threshold is the table's own median line spacing, robust to a
single wide internal row-group break).  For this localization pass the text
layer's *content* is never read into the document; it only supplies coordinates, so
the model stays the sole reader — with one deliberate exception, the deterministic
table rebuild below.  Two biases mirror the figure logic: localization may **over-shoot into
surrounding prose** — mostly harmless, because only ``<table>`` blocks are lifted
from the crop's re-OCR and the prose discarded — and a region is replaced only when
the re-OCR returns *at least as many non-empty cells* as the original, so a crop
that came back worse leaves the full-page transcription untouched.

Two limitations ride on this, both currently acceptable because the corpus is
single-column in the table band.  Horizontal extent is the union of the table's
lines, which on a **two-column page** would pull in the neighbouring column; if
that column's prose then re-OCRs back *as* table markup it can inflate the
cell-count guard and slip through, so the guard is a backstop, not a proof of
correctness.  And the guard counts cells, not structure — it cannot by itself tell
a table that traded a data row for header cells from one that genuinely improved;
the per-fixture table-content integration tests are what hold that line.  Pages
with no text layer (scanned PDFs) simply fail to localize and keep the full-page
table — a geometric/CV detector would be the fallback for either case if it ever
matters.

Sideways tables are the exception to "boxes figures, never tables."  A table printed
rotated 270° *is* boxed by the model — as a figure — while it also (badly) transcribes
it, mis-grouping the columns.  So here the geometry comes from a box after all, but the
crop is sideways: :func:`~manuscribe.pipeline.layers._page_text_and_boxes` records each
glyph's rotation, the localizer reads the region along its true reading axis, and the
re-OCR rotates the crop upright first (the model mis-groups a rotated table's columns
otherwise).  Because the model emitted both a ``<table>`` and an ``![image]`` for the
one table, :func:`~manuscribe.pipeline.assemble._dedup_table_figures` drops the duplicate
image.

One table defeats even a perfect crop, and it forces the single real exception to
the reader rule.  A dense two-column statistics table (label | value) that the OCR
mis-reads *the same way on every run* — dropping the empty top-left header cell so
every row shifts a column and a value falls off the end, then truncating the tail —
cannot be repaired by re-OCR, because a tighter crop just re-rolls the identical
error (measured 5/5 runs on one fixture).  The only fix is deterministic, so
:func:`~manuscribe.pipeline.tables.rebuild._repair_tables_from_text_layer` rebuilds
that table's *content* straight from the PDF text layer — reading the layer's words,
not just its coordinates, one of the two places B-prime's "OCR is the sole reader"
rule is broken on purpose (the other is the truncated-tail reconciliation below), and
a wider carve-out than the geometry-only localization above.
It is fenced in tightly: only a two-column table qualifies; the rebuild replaces the
OCR's only when it yields *more* rows (compared against the OCR table after
collapsing any decode-loop explosion — see below — so an inflated table can't mask a
truncated one); and the OCR's per-cell formatting is carried back by content-match,
so ``R<sub>sym</sub>`` survives and only the cells the OCR dropped outright fall back
to plain text.  The rule, then, holds everywhere except the narrow case where the
model is provably, repeatably wrong and the text layer is provably right.

Separately, a table that *overruns the bottom of a page* is transcribed without a
closing ``</table>`` — the model stops mid-table at the page edge — so the next
page's opening prose would render inside it.
:func:`~manuscribe.pipeline.tables.markup._close_unclosed_tables` balances the tags per
page (the unit the model transcribes) before block-splitting, scoped to the pure
core so it runs whether or not the re-OCR pass did.

One more shape pathology is deterministic too: a *decode loop*.  On a dense table
either OCR path can repeat a single row dozens of times (one fixture's table
ballooned to ~110 rows).  :func:`~manuscribe.pipeline.tables.markup._collapse_repeated_rows_md`
collapses any run of more than ``_MAX_IDENTICAL_ROW_RUN`` byte-identical adjacent
rows to one; like the tag-balancing it lives in the pure core and runs over every
page table, so it covers both the page-level length-retry and the crop re-OCR paths.
The lesson worth keeping: a table test has to bound *over*-generation, not just
assert the table is present — that 110-row explosion still had balanced tags and the
right caption, and slipped a presence-only assertion.

Reconciliation recovers what the OCR *truncated*, and is the second place the text
layer's words — not just its coordinates — are read.  When the model cuts a prose
block a few words short, :func:`~manuscribe.pipeline.reconcile._reconcile_text_layer`
anchors the block's tail uniquely in the PDF text layer and splices the faithful raw
layer slice up to the next block.  It leans the way classification does — a wrong
splice fabricates text, worse than a missing tail — so it fires only behind a stack of
guards (a length ceiling, a *unique* anchor, a prose-ratio test, a furniture-recurrence
reject, a replacement-glyph reject) tuned to zero wrong splices across the fixtures.
It runs *after* table and figure recovery, so an appended tail can neither feed the
table coverage gate's adjacent-token check nor make a figure number look already
emitted.  Like the table rebuild it is deterministic, so replaying recorded transcripts
through it is full verification — no server needed.

Stage order is load-bearing.  In
:func:`~manuscribe.pipeline.assemble._assemble_html` the clean-up runs in a fixed
sequence: caption-label rejoin → table-caption colocation → table-footnote
colocation → classify → paragraph merge.  Footnote colocation must precede
classify so a table's footnotes stay with the table instead of being swept into
the article's footnote section, and it must precede the merge so the table is a
single float the cross-table paragraph merge can step over.  Classify pulls
``<sup>``-marked footnotes out of the body *before* the merge, so the merge only
ever sees what classify left behind.  Reorder these at your peril.

After that sequence settles the body, one last pass re-levels heading *depth*.  The
OCR places every heading in the right reading order but jitters its ``<hN>`` level —
a subsection emitted as ``<h2>``, a top-level section as ``<h1>``.
:func:`~manuscribe.pipeline.classify._normalize_heading_levels` corrects depth from
two high-confidence signals only: a dotted section number sets depth by its dotted
depth (``2.`` → ``<h2>``, ``2.1.`` → ``<h3>``), and a canonical top-level section
name (Introduction, Results, References …) anchors to ``<h2>``.  Every other heading
keeps the OCR's level — the classifier's bias once more: an unrecognised real section
(a journal-specific name, an unnumbered subsection) is left at its OCR depth rather
than guessed at and possibly *demoted* out of the outline.  It runs last precisely
because it needs the body in its final order.  The aggressive half — a casing regime
and position-based inference — is deliberately deferred, so pure-title-case
subsection jitter still slips through: a known, bounded limitation.
