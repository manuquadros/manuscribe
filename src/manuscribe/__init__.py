"""PDF parser used to convert PDFs for the D3 Annotation Hub"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("manuscribe")
except PackageNotFoundError:
    __version__ = "unknown"

try:
    from beartype.claw import beartype_this_package

    beartype_this_package()
except ImportError:
    pass

from manuscribe.pipeline import (
    ImageSink,
    ManuscribeError,
    OcrResponseError,
    OcrUnavailableError,
    ParsedDocument,
    PdfInputError,
    lightonocr_pdf_to_document,
    lightonocr_pdf_to_html,
    load_ocr_model,
)

__all__ = [
    "ImageSink",
    "OcrResponseError",
    "OcrUnavailableError",
    "ParsedDocument",
    "PdfInputError",
    "ManuscribeError",
    "lightonocr_pdf_to_document",
    "lightonocr_pdf_to_html",
    "load_ocr_model",
]
