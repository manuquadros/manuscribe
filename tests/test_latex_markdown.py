"""Tests for LaTeX->HTML and markdown->HTML conversion."""


class TestLatexToHtml:
    """Inline `$…$` math is converted to deterministic sub/superscript HTML
    before markdown parsing."""

    def test_simple_subscript(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        assert _latex_to_html("$K_m$") == "<em>K</em><sub>m</sub>"

    def test_braced_subscript(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        assert _latex_to_html("$V_{max}$") == "<em>V</em><sub>max</sub>"

    def test_superscript_becomes_unicode(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # All-mappable superscript chars collapse to Unicode (matches "NAD⁺").
        assert _latex_to_html("NAD$^+$") == "NAD⁺"

    def test_superscript_letters_fall_back_to_tag(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        assert _latex_to_html("pH$^{S}$") == "pH<sup>S</sup>"

    def test_footnote_marker_asterisk_escaped_for_markdown(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # A '*' inside an author superscript ("1,*") is emitted as an HTML entity
        # so the downstream Markdown inline parser does not read it as emphasis and
        # pair it with the next author's marker (eating both).
        assert _latex_to_html("Zhou$^{1,*}$") == "Zhou<sup>1,&#42;</sup>"

    def test_literal_asterisk_in_math_span_escaped(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # A '*' *outside* the sub/superscript but inside a converted span (a
        # multiplication) is escaped too, so two of them are not paired into
        # emphasis by the markdown parser.  The single-letter variables a, b, c
        # render italic; the escaped '*' separates them.
        assert (
            _latex_to_html("$a^2*b*c^2$")
            == "<em>a</em>²&#42;<em>b</em>&#42;<em>c</em>²"
        )

    def test_ratio_of_kinetic_constants(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        assert (
            _latex_to_html("$k_{cat}/K_m$")
            == "<em>k</em><sub>cat</sub>/<em>K</em><sub>m</sub>"
        )

    def test_degree_command_superscript(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # ``$^\circ$`` is the LaTeX degree idiom; the single-char superscript
        # rule used to capture only the backslash, leaving a lone ``\`` inside
        # <sup> that markdown then mangled into "<sup></sup>circ".
        assert _latex_to_html(r"grown at 25 $\pm$ 1$^\circ$C") == "grown at 25 ± 1°C"

    def test_braced_degree_command(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        assert _latex_to_html(r"$^{\circ}$C") == "°C"

    def test_symbol_command_translated(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        assert _latex_to_html(r"$5 \times 10^{3}$ cells") == "5 × 10³ cells"

    def test_greek_command_as_subscript(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        assert _latex_to_html(r"$T_\alpha$") == "<em>T</em><sub>α</sub>"

    def test_command_matched_as_whole_token(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # Commands are matched as maximal "\name" tokens and looked up whole, so
        # a short command never eats the head of a longer one ("\to" vs "\top",
        # "\sim" vs "\simeq") — each resolves to its own glyph.
        assert _latex_to_html(r"$\to$") == "→"
        assert _latex_to_html(r"$\top$") == "⊤"
        assert _latex_to_html(r"$A \simeq B$") == "<em>A</em> ≃ <em>B</em>"

    def test_command_still_terminated_by_digit(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        assert _latex_to_html(r"$\alpha2$") == "α2"

    def test_unknown_command_left_literal(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # pylatexenc returns "" for an unknown macro; we keep the literal rather
        # than silently dropping it.
        assert (
            _latex_to_html(r"$x\notacommand y$") == r"<em>x</em>\notacommand <em>y</em>"
        )

    def test_extended_symbol_coverage(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # Coverage we get for free from pylatexenc that the old hand map lacked.
        assert _latex_to_html(r"$T_\beta + \nabla$") == "<em>T</em><sub>β</sub> + ∇"

    def test_supplementary_label_s_not_section_sign(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # The model misreads a supplementary label's leading "S" as the section
        # command (\S); "\S<digit>" means the letter S ("S4 Fig."), not "§", and the
        # plain identifier must not be math-variable italicised.
        assert _latex_to_html(r"$\S4$ Fig.") == "S4 Fig."
        assert _latex_to_html(r"$\S 4$ Fig.") == "S4 Fig."
        assert _latex_to_html(r"$\S1$ Raw images.") == "S1 Raw images."

    def test_supplementary_label_escapes_underscore(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # The captured label can carry a literal "_" ("$\S4_2$") which lands in the
        # pre-markdown stream; it is escaped to the HTML entity so the downstream
        # inline parser does not re-read it as emphasis (the documented "*"/"_"
        # pre-markdown-escape invariant).
        assert _latex_to_html(r"$\S4_2$ Data.") == "S4&#95;2 Data."

    def test_standalone_section_command_still_section_sign(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # A standalone "\S" (no following digit) is a real footnote/section marker and
        # still converts to "§" — the supplementary-label rule must not swallow it.
        assert _latex_to_html(r"$\S$") == "§"
        assert _latex_to_html(r"$^{1,\S}$") == "<sup>1,§</sup>"

    def test_literal_section_sign_label_recovered_at_block_start(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # The model sometimes emits the *resolved* section sign ("§4 Fig.") rather than
        # the "\S" command; a "§<digit>" at a line/block start (the label position) is
        # the same misread S and is recovered, including a bold-wrapped label.
        assert _latex_to_html("§4 Fig. The structure.") == "S4 Fig. The structure."
        assert _latex_to_html("§1 Raw images.") == "S1 Raw images."
        assert _latex_to_html("**§1 Raw images.**") == "**S1 Raw images.**"

    def test_genuine_midsentence_section_sign_preserved(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # A genuine section reference mid-sentence is not at a block start, so it is
        # left intact — the label rule must not rewrite "§3" into "S3" here.
        assert _latex_to_html("as described in §3 above") == "as described in §3 above"

    def test_arg_macro_that_raises_does_not_crash(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # pylatexenc raises when fed a bare arg-taking macro like \sqrt; the
        # exception must be swallowed and the span left intact, not propagated
        # up to crash the whole page conversion.
        assert _latex_to_html(r"$\sqrt{x}$") == r"\sqrtx"

    def test_arg_macro_template_not_leaked(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # \frac's substitution template ("%s/%s") must not reach the output; the
        # command stays literal so real math survives for a later MathJax pass.
        assert "%s" not in _latex_to_html(r"$\frac{a}{b}$")
        assert _latex_to_html(r"$\frac{a}{b}$") == r"\fracab"

    def test_extended_symbol_coverage_via_command_helper(self) -> None:
        from manuscribe.pipeline.latex import _latex_command_to_unicode

        assert _latex_command_to_unicode(r"\sqrt") == r"\sqrt"
        assert _latex_command_to_unicode(r"\frac") == r"\frac"
        assert _latex_command_to_unicode(r"\alpha") == "α"

    def test_plain_text_untouched(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        assert _latex_to_html("no math here") == "no math here"

    def test_currency_dollars_left_alone(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # No TeX markup between the '$' → not math; must not be stripped/merged.
        assert _latex_to_html("costs $5 and $10 total") == "costs $5 and $10 total"

    def test_math_wrapped_bare_number_unwrapped(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # A lone number in a math span is just a value the model wrapped — drop
        # the '$' delimiters but keep the number (and the surrounding spaces).
        assert _latex_to_html("was $42.26$ Sec") == "was 42.26 Sec"

    def test_inline_equation_unwrapped(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # A markup-free "variable = value" span the model wrapped in math mode must
        # shed its '$' delimiters like a bare number — the relational operator marks
        # it as math, not a currency pairing.
        assert _latex_to_html("size was $x = 22$ and $y = 16$ here") == (
            "size was <em>x</em> = 22 and <em>y</em> = 16 here"
        )
        assert _latex_to_html("cell $a = 84.9$ Å") == "cell <em>a</em> = 84.9 Å"
        assert _latex_to_html("at $P < 0.05$ level") == "at <em>P</em> < 0.05 level"

    def test_inline_variable_list_unwrapped(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # A comma-separated list of single-letter variables ("a, b, c") the model
        # wrapped in math mode — no relational operator — still sheds its delimiters,
        # since it opens with an identifier (not a currency digit).
        assert (
            _latex_to_html("the $a, b, c$ (Å) axes")
            == "the <em>a</em>, <em>b</em>, <em>c</em> (Å) axes"
        )

    def test_currency_left_alone(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # A spurious '$' pairing over prose (English words) and a digit-led currency
        # range (no relation) must both stay verbatim — neither is inline math.
        assert _latex_to_html("we paid $5 and lost $10 total") == (
            "we paid $5 and lost $10 total"
        )
        assert _latex_to_html("priced $5 - $10 each") == "priced $5 - $10 each"

    def test_currency_comparison_with_relation_left_alone(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # A currency pairing that brackets a relational operator ("$10 > $5") must not
        # be mistaken for an equation: currency is digit-led, so the identifier-lead
        # requirement keeps it (and its '$') intact even though it holds a '>'/'<'.
        assert (
            _latex_to_html("stock fell $10 > $5 today") == "stock fell $10 > $5 today"
        )
        assert _latex_to_html("the $5 < $10 rule") == "the $5 < $10 rule"

    def test_dollar_span_straddling_html_tag_left_alone(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # _latex_to_html runs on the pre-markdown stream, which still carries raw HTML
        # (<sub>, <td>, …).  A stray '$' pairing that straddles a tag must not be read
        # as inline math just because the tag's '<'/'>' look like a relation — doing so
        # drops a currency '$' and mangles the tag.
        assert (
            _latex_to_html("the var $a<sub>x</sub> equals $5 today")
            == "the var $a<sub>x</sub> equals $5 today"
        )
        assert (
            _latex_to_html("<td>$a</td><td>cost $5</td>")
            == "<td>$a</td><td>cost $5</td>"
        )

    def test_single_letter_variables_italicised(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # Math-mode convention: a standalone single Latin letter is a variable and
        # renders italic; numbers, operators, and multi-letter identifiers stay
        # upright.  "pH" is a two-letter token, so neither letter italicises.
        assert _latex_to_html("$pH = 7.4$") == "pH = 7.4"
        assert _latex_to_html("$n = 12$") == "<em>n</em> = 12"

    def test_script_marker_letter_not_italicised(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # A single letter inside a generated <sub>/<sup> — e.g. an affiliation or
        # table footnote marker that falls back to a tag — must NOT be italicised;
        # only top-level variables are.
        assert _latex_to_html(r"Author$^{A}$") == "Author<sup>A</sup>"
        assert _latex_to_html("$E_a$") == "<em>E</em><sub>a</sub>"

    def test_single_letter_units_not_italicised(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # A single letter in *unit position* — trailing a numeric magnitude — is a
        # unit symbol, not a variable, and stays upright; the variable left of the
        # relation still italicises.  The unit run extends across '/' and a
        # superscript exponent ("m/s²"), so both m and s stay upright.
        assert _latex_to_html("$P = 5 V$") == "<em>P</em> = 5 V"
        assert _latex_to_html("$9.8 m/s^2$") == "9.8 m/s²"
        assert _latex_to_html(r"$c = 3 \times 10^8 m/s$") == "<em>c</em> = 3 × 10⁸ m/s"

    def test_temperature_unit_not_italicised(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # "°C"/"°F" is a unit: the degree sign sits between the magnitude and the
        # letter, so the unit run must bridge it and keep C/F upright, not read the
        # lone letter as a variable ("25 ± 1°<em>C</em>").
        assert _latex_to_html(r"$25 \pm 1^\circ \text{C}$") == "25 ± 1°C"
        assert _latex_to_html(r"$T = 37^\circ C$") == "<em>T</em> = 37°C"

    def test_paren_led_stereodescriptor_unwrapped(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # The model wraps a Cahn-Ingold-Prelog stereodescriptor in math mode
        # ("$(R)$-2-alkanols"); a paren lead is not currency, so the delimiters drop
        # and the single letter italicises (the chemical convention for (R)/(S)),
        # instead of leaking a literal '$'.
        assert (
            _latex_to_html("the $(R)$-2-alkanols and $(S)$-pentanol")
            == "the (<em>R</em>)-2-alkanols and (<em>S</em>)-pentanol"
        )
        assert "$" not in _latex_to_html("$(R)$ vs $(S)$")
        # comma/slash-joined descriptors too
        assert _latex_to_html("$(R,S)$") == "(<em>R</em>,<em>S</em>)"

    def test_paren_led_panel_label_not_unwrapped(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # A bare parenthesized label has the same shape as a stereodescriptor but is
        # NOT one: a figure panel "(A)"/"(B)", a roman list item "(i)", or a letter
        # pair "(C, D)" must stay verbatim, never unwrapped and italicised as a
        # variable.  A genuine equation in parens ("(n = 5)") still unwraps.
        for label in ("$(A)$", "$(B)$", "$(i)$", "$(C, D)$"):
            assert _latex_to_html(label) == label
        assert _latex_to_html("$(n = 5)$") == "(<em>n</em> = 5)"

    def test_script_span_reattaches_to_preceding_token(self) -> None:
        from manuscribe.pipeline.latex import _latex_to_html

        # The model writes a unit and its exponent with a gap ("Sec $^{-1}$"); a
        # span opening with a script attaches to the previous token, no space.
        assert _latex_to_html("Sec $^{-1}$ mM $^{-1}$") == "Sec⁻¹ mM⁻¹"


class TestRenderInlineHtml:
    """The inline renderer for already-LaTeX'd fragments (figure captions, table
    cells): balanced emphasis, but no full-CommonMark code/link/autolink parsing of
    the model's caption prose."""

    def test_balanced_emphasis_and_escapes(self) -> None:
        from manuscribe.pipeline.markdown import _render_inline_html

        # The reason for using the parser over a regex sub: nested bold+italic closes
        # in order, and a backslash-escaped marker stays literal.
        assert (
            _render_inline_html("**Figure 2. analyses of *BkTauF***")
            == "<strong>Figure 2. analyses of <em>BkTauF</em></strong>"
        )
        assert _render_inline_html(r"Jang\* and Lee\*") == "Jang* and Lee*"

    def test_code_link_autolink_left_literal(self) -> None:
        from manuscribe.pipeline.markdown import _render_inline_html

        # A caption's OCR prose can carry backticks (mis-read primes), bracket+paren
        # adjacencies, and bare <addr> tokens; none must become <code>/<a>/autolinks.
        assert _render_inline_html("the 5`-end and 3`-end") == "the 5`-end and 3`-end"
        assert _render_inline_html("ref [12](2019) here") == "ref [12](2019) here"
        assert "href" not in _render_inline_html("contact <a@b.edu>")


class TestBreakCaptionTitle:
    """A bolded figure-caption title runs straight into its legend; the title gets a
    <br> before the legend, but only when it is genuinely bold and legend follows."""

    def test_bold_title_broken_from_panel_legend(self) -> None:
        from manuscribe.pipeline.markdown import _break_caption_title

        assert (
            _break_caption_title(
                "<strong>Figure 1. Gene clusters</strong> (A) Gene clusters"
            )
            == "<strong>Figure 1. Gene clusters</strong><br>(A) Gene clusters"
        )

    def test_nested_italic_title_closes_at_first_strong(self) -> None:
        from manuscribe.pipeline.markdown import _break_caption_title

        # the title's inner </em> closes before its </strong>, so the break lands after
        # the whole bold title, not inside it
        assert _break_caption_title(
            "<strong>Figure 2. analyses of <em>BkTauF</em></strong> (A) SDS/PAGE"
        ) == ("<strong>Figure 2. analyses of <em>BkTauF</em></strong><br>(A) SDS/PAGE")

    def test_non_panel_legend_also_broken(self) -> None:
        from manuscribe.pipeline.markdown import _break_caption_title

        assert (
            _break_caption_title(
                "<strong>Figure 7. Gene clusters</strong> Many of the close homologs"
            )
            == "<strong>Figure 7. Gene clusters</strong><br>Many of the close homologs"
        )

    def test_title_only_caption_unchanged(self) -> None:
        from manuscribe.pipeline.markdown import _break_caption_title

        # a caption that is wholly the bold title (no legend) gets no break
        assert (
            _break_caption_title("<strong>Figure 3. Overview of the assay.</strong>")
            == "<strong>Figure 3. Overview of the assay.</strong>"
        )

    def test_unbolded_caption_unchanged(self) -> None:
        from manuscribe.pipeline.markdown import _break_caption_title

        # no leading bold title → nothing to break, even if a term inside is bold
        cap = "Figure 4. Activity of the <strong>mutant</strong> enzyme over time"
        assert _break_caption_title(cap) == cap

    def test_punctuation_glued_to_bold_not_split(self) -> None:
        from manuscribe.pipeline.markdown import _break_caption_title

        # the </strong> is immediately followed by a period (the model un-bolded the
        # trailing dot) — no whitespace, so it must NOT be mistaken for a legend start
        cap = "<strong>Figure 5</strong>. continued title prose here"
        assert _break_caption_title(cap) == cap

    def test_bold_panel_label_not_a_title_unchanged(self) -> None:
        from manuscribe.pipeline.markdown import _break_caption_title

        # the leading bold is a panel label, not a figure/table title — no break, or the
        # label is split from its own description (the regression this guard prevents)
        cap = "<strong>(A)</strong> left panel shows the result"
        assert _break_caption_title(cap) == cap

    def test_bold_emphasis_word_not_a_title_unchanged(self) -> None:
        from manuscribe.pipeline.markdown import _break_caption_title

        # an emphasised first word in a legend ("**Asterisks** denote …") is not a title
        for cap in (
            "<strong>Asterisks</strong> denote p&lt;0.05 across all groups",
            "<strong>Note</strong> values are means of three replicates",
        ):
            assert _break_caption_title(cap) == cap

    def test_two_bold_run_title_breaks_after_last_run(self) -> None:
        from manuscribe.pipeline.markdown import _break_caption_title

        # the model emits the number and the title as two adjacent bold runs; the break
        # must land after the *whole* title (the last </strong>), not split it mid-title
        assert _break_caption_title(
            "<strong>Figure 2.</strong> <strong>Comparison of A and B.</strong> "
            "Panels show data"
        ) == (
            "<strong>Figure 2.</strong> <strong>Comparison of A and B.</strong>"
            "<br>Panels show data"
        )

    def test_all_bold_two_run_title_no_legend_unchanged(self) -> None:
        from manuscribe.pipeline.markdown import _break_caption_title

        # two bold runs, no non-bold legend after — nothing to break off, so the title
        # must not be split between its runs
        cap = "<strong>Figure 1.</strong> <strong>The whole title here.</strong>"
        assert _break_caption_title(cap) == cap


class TestMdToHtmlBlocks:
    """A page's markdown becomes one HTML string per top-level block, with raw
    HTML (tables, <sup>) passed through and thematic breaks dropped."""

    def test_heading_and_paragraph_split(self) -> None:
        from manuscribe.pipeline.markdown import _md_to_html_blocks

        blocks = _md_to_html_blocks("## Introduction\n\nSome prose here.")
        assert blocks == ["<h2>Introduction</h2>", "<p>Some prose here.</p>"]

    def test_emphasis_rendered(self) -> None:
        from manuscribe.pipeline.markdown import _md_to_html_blocks

        (block,) = _md_to_html_blocks("*Przewalskia tangutica* is **rare**.")
        assert (
            block == "<p><em>Przewalskia tangutica</em> is <strong>rare</strong>.</p>"
        )

    def test_table_passthrough(self) -> None:
        from manuscribe.pipeline.markdown import _md_to_html_blocks

        table = "<table><tbody><tr><td>1</td></tr></tbody></table>"
        assert _md_to_html_blocks(table) == [table]

    def test_table_cell_emphasis_rendered(self) -> None:
        from manuscribe.pipeline.markdown import _md_to_html_blocks

        # Raw-HTML table cells carry the model's ``*emphasis*`` unparsed; organism
        # names in a cell must still italicise instead of showing bare asterisks.
        table = (
            "<table><tbody><tr><td>*Sphingomonas* sp. PAMC 26621</td></tr>"
            "<tr><th>*Klebsiella aerogenes*</th></tr></tbody></table>"
        )
        (block,) = _md_to_html_blocks(table)
        assert "<td><em>Sphingomonas</em> sp. PAMC 26621</td>" in block
        assert "<th><em>Klebsiella aerogenes</em></th>" in block
        assert "*" not in block

    def test_table_cell_lone_asterisk_kept_literal(self) -> None:
        from manuscribe.pipeline.markdown import _md_to_html_blocks

        # A single footnote-marker asterisk is not an emphasis span — it must stay.
        (block,) = _md_to_html_blocks(
            "<table><tbody><tr><td>100*</td></tr></tbody></table>"
        )
        assert "<td>100*</td>" in block

    def test_table_cell_spaced_asterisks_not_emphasis(self) -> None:
        from manuscribe.pipeline.markdown import _md_to_html_blocks

        # Asterisks flanked by spaces (multiplication, paired footnote daggers) are
        # not CommonMark emphasis — they must stay literal, not wrap an <em>.
        (block,) = _md_to_html_blocks(
            "<table><tbody><tr><td>5 * 10 * 3</td></tr></tbody></table>"
        )
        assert "<td>5 * 10 * 3</td>" in block
        assert "<em>" not in block

    def test_table_cell_stray_lt_and_amp_escaped(self) -> None:
        from manuscribe.pipeline.markdown import _md_to_html_blocks

        # A bare "<" / "&" / ">" inside a cell must be escaped, not left to start a
        # bogus tag or a broken entity.
        (block,) = _md_to_html_blocks(
            "<table><tbody><tr><td>n<5 & p>0.05</td></tr></tbody></table>"
        )
        assert "<td>n&lt;5 &amp; p&gt;0.05</td>" in block

    def test_sup_passthrough(self) -> None:
        from manuscribe.pipeline.markdown import _md_to_html_blocks

        (block,) = _md_to_html_blocks("NAD<sup>+</sup> dependent.")
        assert block == "<p>NAD<sup>+</sup> dependent.</p>"

    def test_degree_does_not_bleed_into_superscript(self) -> None:
        from manuscribe.pipeline.markdown import _md_to_html_blocks

        # Regression: "$^\circ$" once produced "<sup>\</sup>", whose lone "\<"
        # markdown escaped into "&lt;/sup&gt;", swallowing the rest of the
        # sentence into a superscript.  No <sup> must survive here.
        (block,) = _md_to_html_blocks(
            r"grown at 25 $\pm$ 1$^\circ$C under 16 H of light."
        )
        assert block == "<p>grown at 25 ± 1°C under 16 H of light.</p>"

    def test_thematic_break_dropped(self) -> None:
        from manuscribe.pipeline.markdown import _md_to_html_blocks

        assert _md_to_html_blocks("A.\n\n---\n\nB.") == ["<p>A.</p>", "<p>B.</p>"]

    def test_list_kept_as_one_block(self) -> None:
        from manuscribe.pipeline.markdown import _md_to_html_blocks

        (block,) = _md_to_html_blocks("- one\n- two")
        assert block.startswith("<ul>")
        assert "<li>one</li>" in block and "<li>two</li>" in block
