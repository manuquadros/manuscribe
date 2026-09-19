"""Command-line entry point: convert a PDF to a self-contained HTML file.

pdm run python -m manuscribe paper.pdf            # writes paper.html
pdm run python -m manuscribe paper.pdf out.html   # writes out.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from manuscribe.pipeline import (
    ManuscribeError,
    OcrUnavailableError,
    lightonocr_pdf_to_html,
)
from manuscribe.pipeline.model import _resolve_base_url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="manuscribe",
        description="Convert a PDF to a self-contained HTML document.",
    )
    parser.add_argument("pdf", type=Path, help="Path to the input PDF file.")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output HTML path (default: input path with a .html suffix).",
    )
    parser.add_argument(
        "--vllm-url",
        default=None,
        help="vLLM endpoint root (default: $MANUSCRIBE_VLLM_URL or "
        "http://127.0.0.1:8000/v1).",
    )
    parser.add_argument(
        "--vllm-model",
        default=None,
        help="Served model name (default: $MANUSCRIBE_VLLM_MODEL or lightonocr).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Write figure crops as sidecar PNGs into this directory (referenced "
        "relative to its parent) instead of inlining them as base64.  Default: "
        "inline, keeping the HTML self-contained.",
    )
    args = parser.parse_args(argv)

    if not args.pdf.is_file():
        parser.error(f"input PDF not found: {args.pdf}")

    output = args.output or args.pdf.with_suffix(".html")

    try:
        html = lightonocr_pdf_to_html(
            args.pdf,
            base_url=args.vllm_url,
            model=args.vllm_model,
            image_dir=args.image_dir,
        )
    except OcrUnavailableError as exc:
        # Raised when the server is down/unhealthy — the likeliest failure a human
        # hits.  Surface the underlying network error as a one-line message, not a raw
        # traceback (the typed wrapper's __cause__ carries the concrete detail).
        url = _resolve_base_url(args.vllm_url)
        detail = exc.__cause__ if exc.__cause__ is not None else exc
        print(
            f"error: could not reach the vLLM server at {url} ({detail}). "
            "Is it running? See deploy/vllm/run-server.sh.",
            file=sys.stderr,
        )
        return 1
    except ManuscribeError as exc:
        # A bad input PDF or a malformed OCR response — concise message, no traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
