"""PDF → HTML pipeline using LightOnOCR-2-1B-bbox.

LightOnOCR (lightonai/LightOnOCR-2-1B-bbox, Apache-2.0) is an end-to-end VLM that
reconstructs each page as markdown — reading order, emphasis, ``<table>`` HTML,
LaTeX math, and figure crop boxes appended to ``![image]`` placeholders.  The
pipeline OCRs every page, converts the markdown to HTML, crops figures from the
rendered page, and assembles a document shell.

The package is split by concern:

* ``model`` — HTTP client seam: OCR one page via the vLLM server.
* ``render`` — PDF page → PIL image.
* ``latex`` / ``markdown`` — text → HTML conversion.
* ``figures`` — bbox geometry, crop recovery, ``<figure>`` emission.
* ``text`` / ``merge`` / ``classify`` — block-stream helpers and document
  structure classification.
* ``assemble`` — the render-free, model-free core plus the public entry point.

Typical use::

    ocr = load_ocr_model()
    html = lightonocr_pdf_to_html("paper.pdf", ocr=ocr)
    Path("out.html").write_text(html)
"""

from __future__ import annotations

from manuscribe.pipeline.assemble import (
    ParsedDocument,
    lightonocr_pdf_to_document,
    lightonocr_pdf_to_html,
)
from manuscribe.pipeline.errors import (
    ManuscribeError,
    OcrResponseError,
    OcrUnavailableError,
    PdfInputError,
)
from manuscribe.pipeline.figures import ImageSink
from manuscribe.pipeline.model import OcrModel, load_ocr_model

__all__ = [
    "ImageSink",
    "OcrModel",
    "OcrResponseError",
    "OcrUnavailableError",
    "ParsedDocument",
    "PdfInputError",
    "ManuscribeError",
    "lightonocr_pdf_to_document",
    "lightonocr_pdf_to_html",
    "load_ocr_model",
]
