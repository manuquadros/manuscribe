"""Tests for the pure (CPU-only, GPU-free) page-render helpers."""

from pathlib import Path

from PIL import Image

from manuscribe.pipeline.render import (
    _OCR_MAX_LONG_SIDE,
    _RENDER_SCALE,
    _downscale_to_long_side,
    _page_render_scale,
    _render_page_images,
)

_FIXTURES = Path(__file__).parent / "fixtures"


class TestPageRenderScale:
    def test_standard_page_renders_directly_to_the_budget(self) -> None:
        # US Letter (612x792 pt) at 200 DPI would overshoot the budget, so the
        # scale is chosen to land the long side exactly on it (no supersampling).
        scale = _page_render_scale(612.0, 792.0)
        assert scale == _OCR_MAX_LONG_SIDE / 792.0
        assert round(792.0 * scale) == _OCR_MAX_LONG_SIDE

    def test_sub_budget_page_is_not_upscaled(self) -> None:
        # A small page already within budget at 200 DPI keeps the 200 DPI scale
        # rather than being blown up to fill the budget.
        scale = _page_render_scale(200.0, 250.0)
        assert scale == _RENDER_SCALE

    def test_uses_the_long_side(self) -> None:
        assert _page_render_scale(792.0, 612.0) == _page_render_scale(612.0, 792.0)


class TestDownscaleToLongSide:
    def test_noop_when_within_budget(self) -> None:
        img = Image.new("RGB", (_OCR_MAX_LONG_SIDE, 800))
        assert _downscale_to_long_side(img) is img

    def test_shrinks_and_preserves_aspect(self) -> None:
        img = Image.new("RGB", (4000, 2000))
        out = _downscale_to_long_side(img)
        assert max(out.size) == _OCR_MAX_LONG_SIDE
        assert out.size == (_OCR_MAX_LONG_SIDE, _OCR_MAX_LONG_SIDE // 2)


class TestRenderPageImages:
    def test_every_page_within_long_side_budget(self) -> None:
        images = _render_page_images(_FIXTURES / "31298526.pdf")
        assert images
        for img in images:
            assert img.mode == "RGB"
            assert max(img.size) <= _OCR_MAX_LONG_SIDE
