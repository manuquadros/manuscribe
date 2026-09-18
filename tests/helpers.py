"""Shared pure helpers for the pipeline test suite (no model, no rendering)."""

import base64
import io
import re
from pathlib import Path

from PIL import Image

from pdfparser.pipeline.block import Block


def _fake_image(width: int = 800, height: int = 1000) -> Image.Image:
    return Image.new("RGB", (width, height), color="white")


def _as_blocks(parts: list[str]) -> list[Block]:
    """Wrap raw HTML strings into ``Block``s for a function migrated to take
    ``list[Block]`` — the test's own inputs/assertions stay plain strings."""
    return [Block.of(p) for p in parts]


def _img_size(fragment: str, base_dir: Path | None) -> tuple[int, int] | None:
    """Size of the first figure image in ``fragment``: a base64 data URI when
    ``base_dir`` is ``None``, else a sidecar PNG resolved relative to ``base_dir``."""
    if base_dir is None:
        uri = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", fragment)
        if not uri:
            return None
        return Image.open(io.BytesIO(base64.b64decode(uri.group(1)))).size
    src = re.search(r'<img src="([^"]+\.png)"', fragment)
    return Image.open(base_dir / src.group(1)).size if src else None


def _figure_sizes(html: str, base_dir: Path | None = None) -> list[tuple[int, int]]:
    """Every figure image's (w, h) — inlined base64, or sidecar PNGs under
    ``base_dir`` when figures were written to files."""
    sizes = [_img_size(fig, base_dir) for fig in re.findall(r"<img [^>]*>", html)]
    return [s for s in sizes if s is not None]


def _figure_size_by_caption(
    html: str, needle: str, base_dir: Path | None = None
) -> tuple[int, int] | None:
    """The (w, h) of the figure whose ``<figcaption>`` contains ``needle``;
    ``None`` if no such figure is present."""
    for fig in re.findall(r"<figure>.*?</figure>", html, re.DOTALL):
        cap = re.search(r"<figcaption>(.*?)</figcaption>", fig, re.DOTALL)
        if cap and needle in cap.group(1):
            return _img_size(fig, base_dir)
    return None


def _run_lighton(pages_md: list[str], image: Image.Image | None = None) -> str:
    """Assemble HTML from synthetic per-page markdown (no model, no rendering)."""
    from pdfparser.pipeline.assemble import _assemble_html

    img = image or _fake_image(1190, 1540)
    return _assemble_html(pages_md, [img for _ in pages_md])


def _body(html: str) -> str:
    """Extract the content of the <div class="body"> element.

    Depth-aware so nested </div> inside model-emitted HTML doesn't truncate.
    """
    start = html.find('<div class="body">')
    assert start >= 0, "body div not found"
    pos = start
    depth = 0
    while pos < len(html):
        if html.startswith("<div", pos):
            depth += 1
            pos += 4
        elif html.startswith("</div>", pos):
            depth -= 1
            if depth == 0:
                return html[start:pos]
            pos += 6
        else:
            pos += 1
    raise AssertionError("unclosed body div")


def _tables_text(html: str) -> str:
    """Visible text rendered *inside* any <table> in the body, tag-stripped.

    Depth-aware (not a `<table>...</table>` regex) so an unclosed table — one the
    OCR left open at a page bottom — is seen to extend to the document end and thus
    captures any following prose it swallows.  Used both to assert real cell content
    is present and to assert post-table prose is *not* absorbed."""
    body = _body(html)
    out: list[str] = []
    depth = 0
    i = 0
    while i < len(body):
        if body[i : i + 6].lower() == "<table":
            depth += 1
            i += 6
        elif body[i : i + 8].lower() == "</table>":
            depth = max(0, depth - 1)
            i += 8
        else:
            if depth > 0:
                out.append(body[i])
            i += 1
    return re.sub(r"<[^>]+>", "", "".join(out))


def _metadata(html: str) -> str:
    """Content of the collapsible <details class='metadata'> panel."""
    start = html.find("<details class='metadata'>")
    assert start >= 0, "metadata panel not found"
    return html[start : html.find("</details>", start)]


def _abstract(html: str) -> str:
    """Content of the <section class='abstract'> element; '' when absent."""
    start = html.find("<section class='abstract'>")
    if start < 0:
        return ""
    return html[start : html.find("</section>", start)]


def _header_h1(html: str) -> str:
    """Return the text of the document's <header><h1> title element."""
    m = re.search(r"<header>.*?<h1>(.*?)</h1>", html, re.DOTALL)
    assert m, "header <h1> not found"
    return m.group(1)


def _header(html: str) -> str:
    """The raw <header>…</header> slice (title + byline), markup intact."""
    start = html.find("<header>")
    assert start >= 0, "header not found"
    return html[start : html.find("</header>", start)]


def _byline(html: str) -> str:
    """The byline <p> (the paragraph after the <h1> title) inside the header.

    Scoped past the title because the title legitimately italicises a species name,
    so a header-wide `<em>` check would false-positive on it."""
    after_title = _header(html)[_header(html).find("</h1>") :]
    return after_title[after_title.find("<p>") : after_title.find("</p>") + 4]
