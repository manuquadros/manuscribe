"""Tests for the pre-OCR text-layer skip (layers.py) and its OCR wiring.

The decision — how many leading image-only pages to skip before OCR — is a pure,
GPU-free text-layer read, so it is fully unit-testable against the real fixtures and a
handful of synthetic flag lists.  The wiring (``_ocr_document_pages``) is exercised
offline with a mocked OCR seam, proving the ad page is never sent to the model.
"""

from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from manuscribe.pipeline.layers import (
    _MIN_TEXT_LAYER_ALNUM,
    _DocumentLayers,
    _is_trivial_text_layer,
    _leading_image_only_count,
    _leading_image_only_pages,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_AD_PREFIX_PDF = _FIXTURES / "31051047.pdf"  # full-page-image ad as page 0
_NO_AD_PDF = _FIXTURES / "30592559.pdf"  # article text on page 0


class TestIsTrivialTextLayer:
    def test_empty_string_is_trivial(self) -> None:
        assert _is_trivial_text_layer("")

    def test_whitespace_and_punctuation_only_is_trivial(self) -> None:
        # Only alphanumerics count, so a page of form feeds / stray punctuation reads
        # as empty rather than as content.
        assert _is_trivial_text_layer("\f\n  .,;—•  \t\n")

    def test_below_threshold_is_trivial(self) -> None:
        assert _is_trivial_text_layer("a" * (_MIN_TEXT_LAYER_ALNUM - 1))

    def test_at_threshold_is_not_trivial(self) -> None:
        assert not _is_trivial_text_layer("a" * _MIN_TEXT_LAYER_ALNUM)

    def test_paragraph_of_text_is_not_trivial(self) -> None:
        assert not _is_trivial_text_layer("Abstract " * 20)


class TestLeadingImageOnlyCount:
    def test_no_pages(self) -> None:
        assert _leading_image_only_count([]) == 0

    def test_first_page_text_bearing_skips_nothing(self) -> None:
        assert _leading_image_only_count([False, True, True]) == 0

    def test_single_leading_empty_page(self) -> None:
        assert _leading_image_only_count([True, False]) == 1

    def test_two_leading_empty_pages(self) -> None:
        assert _leading_image_only_count([True, True, False]) == 2

    def test_all_pages_empty_skips_nothing(self) -> None:
        # A fully-scanned paper has no usable text layer anywhere, so there is no
        # cover to distinguish from the article — the model must OCR every page.
        assert _leading_image_only_count([True, True, True]) == 0

    def test_only_the_leading_run_counts(self) -> None:
        # An interior empty page (a blank divider mid-document) is never skipped.
        assert _leading_image_only_count([True, False, True, False]) == 1


class TestLeadingImageOnlyPages:
    def test_ad_prefixed_pdf_skips_its_cover(self) -> None:
        with _DocumentLayers.open(_AD_PREFIX_PDF) as layers:
            assert _leading_image_only_pages(layers) == 1

    def test_normal_pdf_skips_nothing(self) -> None:
        with _DocumentLayers.open(_NO_AD_PDF) as layers:
            assert _leading_image_only_pages(layers) == 0


class TestOcrDocumentPagesSkip:
    """``_ocr_document_pages`` sends only the article pages to the OCR seam and pads the
    skipped leading pages with empty markdown to keep positional alignment."""

    def test_ad_page_is_never_ocred(self, monkeypatch) -> None:
        from manuscribe.pipeline import assemble
        from manuscribe.pipeline.model import OcrModel

        seen: list[list[Image.Image]] = []

        def _spy(images: list[Image.Image], ocr: object) -> list[str]:
            seen.append(images)
            return [f"page-{i}" for i in range(len(images))]

        monkeypatch.setattr(assemble, "_ocr_pages", _spy)

        # 31051047.pdf has 11 pages; page 0 is the image-only ad.  Distinct sizes so
        # PIL's content-equality (used by ``==``/``in``) tracks page identity.
        images = [Image.new("RGB", (i + 1, 4)) for i in range(11)]
        ocr = MagicMock(spec=OcrModel)

        with _DocumentLayers.open(_AD_PREFIX_PDF) as layers:
            result = assemble._ocr_document_pages(images, layers, ocr)

        # The seam saw exactly the 10 article pages — the ad image was excluded.
        assert len(seen) == 1
        assert len(seen[0]) == 10
        assert images[0] not in seen[0]
        assert seen[0] == images[1:]
        # Alignment preserved: the skipped page is empty markdown, the rest follow.
        assert len(result) == len(images)
        assert result[0] == ""
        assert result[1:] == [f"page-{i}" for i in range(10)]

    def test_no_ad_pdf_ocrs_every_page(self, monkeypatch) -> None:
        from manuscribe.pipeline import assemble
        from manuscribe.pipeline.model import OcrModel

        seen: list[list[Image.Image]] = []

        def _spy(images: list[Image.Image], ocr: object) -> list[str]:
            seen.append(images)
            return ["md"] * len(images)

        monkeypatch.setattr(assemble, "_ocr_pages", _spy)

        images = [Image.new("RGB", (i + 1, 4)) for i in range(3)]
        ocr = MagicMock(spec=OcrModel)

        with _DocumentLayers.open(_NO_AD_PDF) as layers:
            result = assemble._ocr_document_pages(images, layers, ocr)

        assert seen[0] == images  # nothing skipped
        assert result == ["md"] * 3
