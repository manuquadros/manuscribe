"""Public exception hierarchy raised at the pipeline's boundary.

A leaf module (it imports nothing else from the package) so the seam modules
(:mod:`~pdfparser.pipeline.model`, :mod:`~pdfparser.pipeline.render`) and the
entry point can raise typed errors that a caller — e.g. a batch-ingest worker —
classifies for *retry* vs. *permanent failure*
without reaching into ``httpx``/``pypdfium2`` internals.  The original library
exception is always preserved as ``__cause__`` (``raise … from``).
"""

from __future__ import annotations


class PdfParserError(Exception):
    """Base class for every error pdfparser raises at its public boundary."""


class OcrUnavailableError(PdfParserError):
    """The OCR server was unreachable, unhealthy, timed out, or out of memory.

    Retryable without bound: the condition is external and transient, so a worker
    should re-queue the document and retry once the server recovers.
    """


class OcrResponseError(PdfParserError):
    """The OCR server replied, but the payload was malformed or unconsumable.

    Retryable, but a worker should *cap* the attempts: it is usually a transient
    garble, yet may be a genuine defect that an uncapped retry would loop on forever.
    """


class PdfInputError(PdfParserError):
    """The input PDF could not be opened or rendered (corrupt/encrypted/missing).

    Permanent: re-running hits the same bad bytes, so a worker should fail the job
    rather than retry.
    """
