"""Tests for the final document-assembly HTML sanitization boundary.

LightOnOCR transcribes whatever the source PDF's visible text/markup shows, so a
corrupted or crafted PDF can make the model emit ``<script>``/``<iframe>``/an
``on*`` handler that markdown.py's ``html: True`` passthrough would otherwise
carry verbatim into the document HTML the D3 Annotation Hub stores and serves.
``_sanitize_content_html`` (assemble.py) is the final allow-list pass every piece
of OCR-sourced content HTML goes through before reaching the shell.
"""

from helpers import _body, _byline, _fake_image, _header_h1, _run_lighton

from manuscribe.pipeline.assemble import _sanitize_content_html


class TestSanitizeContentHtml:
    """Direct coverage of the allow-list boundary itself — the same function
    assemble.py runs title_html/byline_html/abstract/metadata/body through."""

    def test_script_tag_and_content_stripped(self) -> None:
        assert _sanitize_content_html("<script>alert(1)</script>safe") == "safe"

    def test_inline_event_handler_attribute_stripped(self) -> None:
        html = _sanitize_content_html("Text with <img src=x onerror=alert(1)> inline.")
        assert "onerror" not in html
        assert "alert(1)" not in html
        assert '<img src="x">' in html

    def test_iframe_tag_and_content_stripped(self) -> None:
        html = _sanitize_content_html(
            '<iframe src="https://evil.example"></iframe>after'
        )
        assert "<iframe" not in html
        assert html == "after"

    def test_javascript_scheme_src_stripped(self) -> None:
        html = _sanitize_content_html('<img src="javascript:alert(1)" alt="">')
        assert "javascript:" not in html
        assert "src=" not in html

    def test_title_context_event_handler_stripped(self) -> None:
        html = _sanitize_content_html("Real Title <img src=x onerror=alert(1)>")
        assert "onerror" not in html
        assert "Real Title" in html

    def test_byline_context_script_stripped(self) -> None:
        html = _sanitize_content_html("Jane Doe<sup>1</sup><script>alert(1)</script>")
        assert "<script" not in html
        assert "alert(1)" not in html
        assert "Jane Doe<sup>1</sup>" in html

    def test_table_cell_script_stripped_structure_kept(self) -> None:
        html = _sanitize_content_html(
            '<table><tbody><tr><td colspan="2">'
            "<script>alert(1)</script>Cell <sup>a</sup> text</td></tr></tbody></table>"
        )
        assert "<script" not in html
        assert '<td colspan="2">Cell <sup>a</sup> text</td>' in html

    def test_load_bearing_tags_pass_through(self) -> None:
        html = (
            "<p>Some <em>italic</em> and <strong>bold</strong> prose with "
            "H<sub>2</sub>O and NAD<sup>+</sup>.<br>"
            "A second line.</p>"
        )
        assert _sanitize_content_html(html) == html

    def test_data_uri_figure_image_passes_through(self) -> None:
        html = (
            '<figure><img src="data:image/png;base64,AAAA" alt="">'
            "<figcaption>A caption.</figcaption></figure>"
        )
        assert _sanitize_content_html(html) == html

    def test_anchor_tag_unwrapped(self) -> None:
        # Nothing in manuscribe emits <a>; the block-level markdown-it link rule
        # can still turn OCR prose shaped like a citation adjacency into one, so
        # confirm it is unwrapped (text kept, tag/href dropped) rather than kept.
        html = _sanitize_content_html('<a href="javascript:alert(1)">12</a>')
        assert html == "12"


class TestSanitizeEndToEnd:
    """The seam is actually wired into ``_assemble_document`` (not just present
    as a standalone function) — dangerous markup injected at each content site
    a real OCR page could produce is gone from the assembled document."""

    def test_script_and_event_handler_stripped_from_body(self) -> None:
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\n"
            "<script>alert(1)</script>\n\n"
            "Text with <img src=x onerror=alert(1)> inline and "
            '<iframe src="javascript:alert(1)"></iframe> more.\n\n'
            '<table><tr><td colspan="2"><script>alert(1)</script>'
            "Cell <sup>a</sup> text</td></tr></table>\n"
        )
        html = _run_lighton([md], image=img)
        assert "<script" not in html
        assert "onerror" not in html
        assert "<iframe" not in html
        assert "alert(1)" not in html
        body = _body(html)
        assert "Text with" in body
        assert '<td colspan="2">Cell <sup>a</sup> text</td>' in body

    def test_script_stripped_from_figure_caption(self) -> None:
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\n"
            "![image](i.png)100,100,900,600\n\n"
            "Figure 1. <script>alert(1)</script>A real caption with "
            "<sup>a</sup> marker.\n"
        )
        html = _run_lighton([md], image=img)
        assert "<script" not in html
        assert "alert(1)" not in html
        assert (
            "<figcaption>Figure 1. A real caption with <sup>a</sup> marker."
            "</figcaption>" in html
        )

    def test_title_and_byline_event_handler_stripped(self) -> None:
        img = _fake_image(1190, 1540)
        md = (
            "# Real Title <img src=x onerror=alert(1)>\n\n"
            "Jane Doe<sup>1</sup><script>alert(1)</script>\n\n"
            "## Abstract\n\nA.\n\n## Body\n\nSome body text.\n"
        )
        html = _run_lighton([md], image=img)
        assert "onerror" not in html
        assert "<script" not in html
        assert "alert(1)" not in html
        assert "Real Title" in _header_h1(html)
        assert "Jane Doe" in _byline(html)
