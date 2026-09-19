"""PDF page rendering: a PDF path → one RGB PIL image per page."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — beartype reads annotations at runtime

import pypdfium2 as pdfium
from PIL import Image

from manuscribe.pipeline.errors import PdfInputError

_RENDER_SCALE = 200 / 72  # 200 DPI per the model card
_OCR_MAX_LONG_SIDE = 1540  # model-card target; VRAM ≈ 2.7/6.1 GiB at this size


def _downscale_to_long_side(img: Image.Image) -> Image.Image:
    """Shrink ``img`` so its long side is ≤ ``_OCR_MAX_LONG_SIDE`` (no-op if it
    already fits).  Shared by full-page render and the table-crop render so both
    honour the model's long-side budget from one place."""
    long_side = max(img.size)
    if long_side <= _OCR_MAX_LONG_SIDE:
        return img
    ratio = _OCR_MAX_LONG_SIDE / long_side
    return img.resize(
        (int(img.size[0] * ratio), int(img.size[1] * ratio)),
        Image.Resampling.LANCZOS,
    )


def _page_render_scale(width_pt: float, height_pt: float) -> float:
    """Scale that rasterizes a page directly at the OCR long-side budget.

    Capped at ``_RENDER_SCALE`` so a page already within budget at 200 DPI is not
    upscaled — that preserves the prior behaviour for such pages while avoiding the
    ~2× pixel-area waste of supersampling every standard page to 200 DPI only to
    LANCZOS-shrink it back to the budget."""
    return min(_RENDER_SCALE, _OCR_MAX_LONG_SIDE / max(width_pt, height_pt))


def _render_page_images(pdf_path: Path) -> list[Image.Image]:
    """Render every page to an RGB image, long side ≤ ``_OCR_MAX_LONG_SIDE``.

    Each page is rasterized once directly at the budget (``_page_render_scale``)
    rather than at a fixed 200 DPI then LANCZOS-downscaled; ``_downscale_to_long_side``
    then only clamps a sub-pixel rounding overshoot.  ``convert("RGB")`` detaches each
    image from the pdfium bitmap, so the document can be closed before the images are
    consumed.
    """
    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
    except (pdfium.PdfiumError, FileNotFoundError) as exc:
        raise PdfInputError(f"could not open PDF: {pdf_path}") from exc
    try:
        return [
            _downscale_to_long_side(
                page.render(scale=_page_render_scale(*page.get_size()))
                .to_pil()
                .convert("RGB")
            )
            for page in pdf
        ]
    except pdfium.PdfiumError as exc:
        raise PdfInputError(f"could not render PDF: {pdf_path}") from exc
    finally:
        pdf.close()
