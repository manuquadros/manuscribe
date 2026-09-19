"""Time the OCR phase against the running vLLM server and record one labelled row.

The attention backend is a *launch* flag, so comparing backends means one server
run each.  The label is passed in because ``/v1/models`` does not report the
backend -- only the startup log does.

    ATTENTION_BACKEND=FLASHINFER ./run-server.sh          # terminal 1
    pdm run python deploy/vllm/bench_attention.py flashinfer

Times the pipeline's own seam rather than a synthetic loop.  Speed alone would
mislead: a backend whose kernels do not suit the card returns HTTP 200 and a page
of ``!`` (see gpu-defaults.sh on the T4), so responses are checked before any
timing is reported, and a digest records whether two backends actually agree.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from manuscribe.pipeline.model import (
    OcrModel,
    _first_pass_new_tokens,
    _ocr_pages,
    load_ocr_model,
)
from manuscribe.pipeline.render import _render_page_images

# Annotation-only: beartype's import hook does not cover this script.
if TYPE_CHECKING:
    from PIL.Image import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"
_RESULTS = _REPO_ROOT / "spike_results" / "attention-backend-bench.tsv"

# Dense fixtures (tables, multi-column) keep the decode loop busy, and two of
# them keep a sweep to minutes rather than the integration suite's half hour.
_DEFAULT_PDFS = ("31051047.pdf", "31298526.pdf")

_MIN_CHARS_PER_PAGE = 200
# The T4 failure shape: overwhelmingly one repeated character.
_MAX_SINGLE_CHAR_SHARE = 0.9


@dataclass(frozen=True)
class Timing:
    """Wall-clock seconds for each timed repeat of the OCR phase."""

    seconds: list[float]
    pages: int

    @property
    def best(self) -> float:
        return min(self.seconds)

    @property
    def median(self) -> float:
        return statistics.median(self.seconds)

    @property
    def pages_per_second(self) -> float:
        return self.pages / self.best


def _degenerate_reason(text: str) -> str | None:
    """Why ``text`` cannot be a page transcription, or ``None`` if it looks real."""
    stripped = text.strip()
    if len(stripped) < _MIN_CHARS_PER_PAGE:
        return f"only {len(stripped)} chars"
    share = max(Counter(stripped).values()) / len(stripped)
    if share > _MAX_SINGLE_CHAR_SHARE:
        commonest = Counter(stripped).most_common(1)[0][0]
        return f"{share:.0%} of the response is {commonest!r}"
    return None


def _check_transcriptions(pages: list[str]) -> None:
    """Abort before reporting a time for output that is not a transcription."""
    for index, text in enumerate(pages):
        reason = _degenerate_reason(text)
        if reason is not None:
            sys.exit(
                f"page {index} is not a transcription ({reason}) -- this backend "
                f"is producing garbage, so its timing is meaningless. First 200 "
                f"chars:\n{text[:200]!r}"
            )


def _digest(pages: list[str]) -> str:
    joined = "\n".join(pages).encode()
    return hashlib.sha256(joined).hexdigest()[:12]


def _render(pdfs: list[Path], page_limit: int | None) -> list[Image]:
    images: list[Image] = []
    for pdf in pdfs:
        images.extend(_render_page_images(pdf))
    return images[:page_limit] if page_limit else images


def _time_ocr(
    images: list[Image], ocr: OcrModel, concurrency: int, repeats: int
) -> tuple[Timing, list[str]]:
    """Time ``repeats`` OCR passes after one untimed warm-up.

    The warm-up is not optional: FlashInfer JIT-compiles on first use, so a cold
    pass times the compiler and reports the backend as slow.
    """
    print(f"warm-up pass over {len(images)} pages...", file=sys.stderr)
    pages = _ocr_pages(images, ocr, concurrency=concurrency)
    _check_transcriptions(pages)

    seconds: list[float] = []
    for run in range(1, repeats + 1):
        started = time.perf_counter()
        pages = _ocr_pages(images, ocr, concurrency=concurrency)
        elapsed = time.perf_counter() - started
        seconds.append(elapsed)
        print(
            f"  run {run}/{repeats}: {elapsed:.1f}s "
            f"({len(images) / elapsed:.2f} pages/s)",
            file=sys.stderr,
        )
        _check_transcriptions(pages)
    return Timing(seconds, len(images)), pages


def _append_row(fields: dict[str, str]) -> None:
    _RESULTS.parent.mkdir(parents=True, exist_ok=True)
    is_new = not _RESULTS.exists()
    with _RESULTS.open("a", encoding="utf-8") as handle:
        if is_new:
            handle.write("\t".join(fields) + "\n")
        handle.write("\t".join(fields.values()) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "label",
        help="the backend the server was started with, e.g. FLASH_ATTN or "
        "FLASHINFER -- read it from the server's own startup log, which prints "
        "'Using <NAME> attention backend'",
    )
    parser.add_argument(
        "--pdfs",
        nargs="+",
        default=[str(_FIXTURES / name) for name in _DEFAULT_PDFS],
        help="fixture PDFs to OCR (default: two dense ones)",
    )
    parser.add_argument(
        "--pages", type=int, default=None, help="cap the page count (default: all)"
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="timed passes after the warm-up"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="in-flight requests (default: the server-resolved "
        "MANUSCRIBE_OCR_CONCURRENCY). Digests only compare at 1: above it vLLM's "
        "batch composition varies with request timing",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    images = _render([Path(p) for p in args.pdfs], args.pages)
    if not images:
        sys.exit("no pages rendered -- check the --pdfs paths")

    with load_ocr_model() as ocr:
        concurrency = args.concurrency or ocr.concurrency
        print(
            f"server: {ocr.base_url}  model: {ocr.model}  engine: {ocr.engine.value}\n"
            f"window: {ocr.context_len}  concurrency: {concurrency}  "
            f"pages: {len(images)}",
            file=sys.stderr,
        )
        timing, pages = _time_ocr(images, ocr, concurrency, args.repeats)
        row = {
            "backend": args.label,
            "model": ocr.model,
            "engine": ocr.engine.value,
            "window": str(ocr.context_len),
            # Derived from the window today, but recorded rather than left to be
            # recomputed: it sets how long a dense page decodes in one request,
            # so a row whose budget differs is not comparable on time.
            "first_pass_tokens": str(_first_pass_new_tokens(ocr.context_len)),
            "concurrency": str(concurrency),
            "pages": str(timing.pages),
            "best_s": f"{timing.best:.2f}",
            "median_s": f"{timing.median:.2f}",
            "pages_per_s": f"{timing.pages_per_second:.3f}",
            "chars": str(sum(len(p) for p in pages)),
            "digest": _digest(pages),
        }

    _append_row(row)
    print(
        f"\n{args.label}: {timing.pages_per_second:.3f} pages/s "
        f"(best {timing.best:.1f}s of {args.repeats})",
        file=sys.stderr,
    )
    print(f"appended to {_RESULTS.relative_to(_REPO_ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
