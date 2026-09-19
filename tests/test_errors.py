"""The public exception contract.

Typed boundary errors a caller (e.g. a batch worker) classifies for retry vs.
permanent failure, with the original library exception preserved as ``__cause__``.
See plans/annotation-hub-integration.md (Tasks A).
"""

from pathlib import Path

import pypdfium2 as pdfium
import pytest

import manuscribe
import manuscribe.pipeline as pipeline
from manuscribe import (
    ManuscribeError,
    OcrResponseError,
    OcrUnavailableError,
    PdfInputError,
)
from manuscribe.pipeline.render import _render_page_images


class TestErrorContract:
    def test_public_names_exported_from_both_namespaces(self) -> None:
        for name in (
            "ManuscribeError",
            "OcrUnavailableError",
            "OcrResponseError",
            "PdfInputError",
        ):
            assert name in manuscribe.__all__
            assert name in pipeline.__all__
            # the top-level and package namespaces re-export the same class object
            assert getattr(manuscribe, name) is getattr(pipeline, name)

    def test_every_error_subclasses_the_base(self) -> None:
        for err in (OcrUnavailableError, OcrResponseError, PdfInputError):
            assert issubclass(err, ManuscribeError)

    def test_corrupt_pdf_raises_pdf_input_error_chaining_pdfium(
        self, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"%PDF-1.4 not a real pdf")
        with pytest.raises(PdfInputError) as excinfo:
            _render_page_images(bad)
        assert isinstance(excinfo.value.__cause__, pdfium.PdfiumError)

    def test_missing_pdf_raises_pdf_input_error_chaining_filenotfound(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(PdfInputError) as excinfo:
            _render_page_images(tmp_path / "nope.pdf")
        assert isinstance(excinfo.value.__cause__, FileNotFoundError)
