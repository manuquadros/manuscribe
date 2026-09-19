Embedding manuscribe as a library
=================================

``manuscribe`` is built to be imported by a long-lived process — the **D3 Annotation
Hub** drains a durable queue of PDFs with a single background worker that parses one
document at a time and drops each render into the annotation queue as it completes.
This page collects the integration contract that matters when you embed it that way.
(For the architecture and the OCR seam itself, start with :doc:`design`.)

Reuse one ``OcrModel`` for the batch
------------------------------------

:func:`~manuscribe.pipeline.model.load_ocr_model` opens an ``httpx`` connection pool
*and* probes the vLLM server for its context window, so it is the expensive part to
repeat.  Load it **once** at worker startup and pass it to every parse via ``ocr=``;
the bundle is closed by the caller that created it, not by the parse:

.. code-block:: python

    from manuscribe import (
        OcrUnavailableError,
        OcrResponseError,
        PdfInputError,
        load_ocr_model,
        lightonocr_pdf_to_document,
    )

    ocr = load_ocr_model()          # one probe + one pool for the worker's lifetime
    try:
        for job in queue:           # one document at a time — the GPU is the limiter
            doc = lightonocr_pdf_to_document(job.path, ocr=ocr)
            store(doc.html, title=doc.title, byline=doc.byline, doi=doc.doi)
    finally:
        ocr.close()

The GPU is the throughput ceiling: a single document already fans its pages out
``MANUSCRIBE_OCR_CONCURRENCY``-wide against the one shared card, so running documents
"in parallel" only queues more requests at the same GPU (and risks an out-of-memory).
One worker draining serially is the design — stream for latency, not for speedup.

The exception contract drives the retry policy
----------------------------------------------

Every failure the entry point can raise is one of the typed errors in
:mod:`manuscribe.pipeline.errors` (all subclasses of
:class:`~manuscribe.pipeline.errors.ManuscribeError`), so a worker classifies an outcome
without catching a grab-bag of ``httpx`` / ``pypdfium2`` / ``RuntimeError``.  The
original low-level error is always chained as ``__cause__``.

.. list-table::
   :header-rows: 1
   :widths: 26 18 56

   * - Raised
     - Disposition
     - When
   * - :class:`~manuscribe.pipeline.errors.OcrUnavailableError`
     - **Retry, uncapped**
     - Server unreachable / unhealthy, a timeout, a 5xx, or a page-level GPU
       out-of-memory.  The work is fine; wait for the server to come back and re-lease.
   * - :class:`~manuscribe.pipeline.errors.OcrResponseError`
     - **Retry, low cap (~2)**
     - A malformed or unexpected OCR payload.  Usually transient, but a persistent one
       is a real bug — cap the retries so it surfaces as a failed job, not a GPU-burning
       loop.
   * - :class:`~manuscribe.pipeline.errors.PdfInputError`
     - **Fail permanently**
     - Corrupt / encrypted / non-PDF / missing file.  Re-running re-fails identically;
       mark the job failed and move on — never retry, or it re-burns the GPU forever.

.. code-block:: python

    try:
        doc = lightonocr_pdf_to_document(job.path, ocr=ocr)
    except PdfInputError as exc:
        job.fail(reason=str(exc.__cause__ or exc))     # permanent
    except OcrResponseError:
        job.retry(cap=2)                               # transient, capped
    except OcrUnavailableError:
        job.requeue()                                    # transient, uncapped

Surviving a vLLM restart
------------------------

``OcrModel`` caches the context window from its one-time ``/models`` probe and holds a
live connection pool, so a server **restart** leaves both stale — the pooled sockets
are dead and the new server may even report a different window.  On a *persistent*
:class:`~manuscribe.pipeline.errors.OcrUnavailableError` (the server came back), call
:meth:`~manuscribe.pipeline.model.OcrModel.reconnect` to rebuild the pool and re-read
the window **in place** — every reference the worker threads around stays valid:

.. code-block:: python

    except OcrUnavailableError:
        if server_is_back_up():
            ocr.reconnect()        # rebuild pool + re-probe; keeps the same bundle
        job.requeue()

A failed ``reconnect`` raises ``OcrUnavailableError`` and leaves the existing bundle
untouched, so it is safe to keep retrying.  Dropping the bundle and calling
``load_ocr_model()`` again is an equivalent alternative; ``reconnect`` is the
convenience that avoids swapping the shared reference.

Set the lease timeout above the worst-case parse
-------------------------------------------------

``MANUSCRIBE_OCR_TIMEOUT`` defaults to **600 s per page**, so a wedged server can hold a
single parse for roughly ten minutes per page *legitimately* (manuscribe does not retry
a timeout — it lets the page fail so the pool tears down promptly).  Set the queue's
lease / visibility timeout **above** that worst case, or lower ``MANUSCRIBE_OCR_TIMEOUT``
for the worker — otherwise a second worker can lease and double-process a job whose
lease expired while manuscribe was still validly waiting on the server.

Image delivery
--------------

Figure crops inline as base64 ``data:`` URIs by default — self-contained, but the
bytes ride in every copy of the HTML.  When the HTML is stored in a database and
shipped on every fetch, pass an ``encode_image`` sink
(:class:`~manuscribe.pipeline.figures.ImageSink`, a ``(png_bytes, mime) -> src``
callback) that writes each crop to an asset store and returns its served URL; manuscribe
rewrites the ``<img src>`` and no image bytes land in the body.  ``image_dir=`` is the
middle ground (sidecar PNG files next to the HTML).  The sink receives PNG bytes, so it
never imports Pillow.

Production environment
----------------------

* **Python 3.13.**  manuscribe pins ``<3.14,>=3.13``; run the worker on 3.13.
* **Leave beartype out of the production environment.**  ``manuscribe/__init__.py``
  activates the beartype import hook *iff* ``beartype`` is importable (a guarded
  ``try/except ImportError``), which adds a runtime type-check on every call.  It is a
  dev/test aid — don't install ``beartype`` in the worker env, so the checks stay off.
* **No torch.**  The GPU work is entirely in the vLLM server; manuscribe's own
  dependencies are light (``httpx`` / ``pypdfium2`` / ``pillow`` / ``markdown-it`` /
  ``pylatexenc``), so embedding it pulls in no CUDA stack.
