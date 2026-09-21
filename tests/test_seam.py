"""Tests for the OCR seam, concurrency resolution, and CLI."""

import base64
import io
from pathlib import Path

import pytest
from helpers import (
    _fake_image,
)
from PIL import Image

from manuscribe.pipeline.errors import (
    ManuscribeError,
    OcrResponseError,
    OcrUnavailableError,
)

# Verbatim ``/v1/models`` entry from a live LightOnOCR vLLM server: ``id`` is the name
# run-server.sh pins whatever loads, ``root`` the weights actually served.
_LIGHTONOCR_MODEL_ENTRY: dict[str, object] = {
    "id": "lightonocr",
    "object": "model",
    "created": 1789821814,
    "owned_by": "vllm",
    "root": "lightonai/LightOnOCR-2-1B-bbox",
    "parent": None,
    "max_model_len": 8192,
    "permission": [
        {
            "id": "modelperm-a512f8e6e6918075",
            "object": "model_permission",
            "created": 1789821814,
            "allow_create_engine": False,
            "allow_sampling": True,
            "allow_logprobs": True,
            "allow_search_indices": False,
            "allow_view": True,
            "allow_fine_tuning": False,
            "organization": "*",
            "group": None,
            "is_blocking": False,
        }
    ],
}
_LIGHTONOCR_MODELS_BODY: dict[str, object] = {
    "object": "list",
    "data": [_LIGHTONOCR_MODEL_ENTRY],
}
# The same live server restarted onto chandra weights: ``id`` is unchanged, so the
# served name really does carry no signal; only ``root`` moved.
_CHANDRA_MODEL_ENTRY: dict[str, object] = {
    "id": "lightonocr",
    "object": "model",
    "created": 1789822306,
    "owned_by": "vllm",
    "root": "datalab-to/chandra-ocr-2",
    "parent": None,
    "max_model_len": 8192,
    "permission": [
        {
            "id": "modelperm-b0eed2e89d15a46a",
            "object": "model_permission",
            "created": 1789822306,
            "allow_create_engine": False,
            "allow_sampling": True,
            "allow_logprobs": True,
            "allow_search_indices": False,
            "allow_view": True,
            "allow_fine_tuning": False,
            "organization": "*",
            "group": None,
            "is_blocking": False,
        }
    ],
}
_CHANDRA_MODELS_BODY: dict[str, object] = {
    "object": "list",
    "data": [_CHANDRA_MODEL_ENTRY],
}


def _models_body_with_root(root: str | None) -> dict[str, object]:
    """The live capture with its ``root`` swapped for other weights — or dropped, as
    an older or non-vLLM OpenAI-compatible endpoint reports it."""
    entry = {k: v for k, v in _LIGHTONOCR_MODEL_ENTRY.items() if k != "root"}
    if root is not None:
        entry["root"] = root
    return {**_LIGHTONOCR_MODELS_BODY, "data": [entry]}


class TestOcrSeam:
    """The model seam is now an HTTP client to the vLLM server, so it is
    unit-testable with a mock transport — no GPU, no model load."""

    def test_ocr_page_request_shape_and_parsing(self) -> None:
        import json

        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "# OCR markdown"}}]}
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        result = _ocr_page(_fake_image(8, 8), ocr)

        assert result == "# OCR markdown"
        assert captured["url"] == "http://srv/v1/chat/completions"
        body = captured["body"]
        assert isinstance(body, dict)
        # Greedy decode (matches the former in-process do_sample=False) against
        # the served model name.
        assert body["temperature"] == 0.0
        assert body["model"] == "lightonocr"
        part = body["messages"][0]["content"][0]
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_ocr_upload_is_lossless(self) -> None:
        """The PNG upload must decode pixel-identical to the rendered page, so the
        encode's compression level is a pure speed knob with no effect on the OCR."""
        import json

        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        page = Image.frombytes(
            "RGB",
            (12, 9),
            bytes((i * 37 + c * 53) % 256 for i in range(12 * 9) for c in range(3)),
        )

        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]}
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        _ocr_page(page, ocr)

        body = captured["body"]
        assert isinstance(body, dict)
        uri = body["messages"][0]["content"][0]["image_url"]["url"]
        png = base64.b64decode(uri.split(",", 1)[1])
        decoded = Image.open(io.BytesIO(png)).convert("RGB")
        assert decoded.tobytes() == page.tobytes()

    def test_ocr_page_raises_on_server_error(self) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        with pytest.raises(OcrUnavailableError):
            _ocr_page(_fake_image(8, 8), ocr)

    def test_ocr_page_null_content_returns_empty_string(self) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        def handler(request: httpx.Request) -> httpx.Response:
            body = {"choices": [{"message": {"content": None}}]}
            return httpx.Response(200, json=body)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        # A degenerate page must yield "" (a str), not trip the return contract.
        assert _ocr_page(_fake_image(8, 8), ocr) == ""

    def test_ocr_page_raises_on_malformed_response(self) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": []})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        with pytest.raises(OcrResponseError, match="unexpected OCR response"):
            _ocr_page(_fake_image(8, 8), ocr)

    def test_ocr_page_null_choices_raises_ocr_response_error(self) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        # "choices": null is a null-valued key (None[0] -> TypeError), not a missing
        # one; it must still surface as the clear "unexpected OCR response" error.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": None})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        with pytest.raises(OcrResponseError, match="unexpected OCR response"):
            _ocr_page(_fake_image(8, 8), ocr)

    def test_ocr_page_non_string_content_raises(self) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        # Structured (non-null, non-string) content — e.g. OpenAI content parts —
        # is a response the pipeline can't consume; fail loudly rather than silently
        # transcribe the page as empty.
        def handler(request: httpx.Request) -> httpx.Response:
            body = {"choices": [{"message": {"content": [{"text": "x"}]}}]}
            return httpx.Response(200, json=body)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        with pytest.raises(OcrResponseError, match="unexpected OCR content type"):
            _ocr_page(_fake_image(8, 8), ocr)

    def test_truncated_page_is_reocrd_with_full_context(self) -> None:
        import json

        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        # First response truncates (finish_reason "length"); the seam must retry once
        # with the whole remaining context window (max-model-len 8192 − prompt − a
        # small margin) so the page's dropped tail is recovered.
        calls: list[int] = []
        truncated = "<table>the page up to the"
        # The full re-OCR reproduces the prefix and continues, so it is longer than
        # the truncated first response.
        full = truncated + " cut and the rest</table>"

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append(body["max_tokens"])
            content, reason = (
                (truncated, "length") if len(calls) == 1 else (full, "stop")
            )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": content}, "finish_reason": reason}
                    ],
                    "usage": {"prompt_tokens": 2500},
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        assert _ocr_page(_fake_image(8, 8), ocr) == full
        assert len(calls) == 2
        assert calls[0] == 2048
        # The retry claims the full remaining window, not just another fixed block.
        assert calls[1] == 8192 - 2500 - 64

    def test_first_pass_budget_follows_the_served_window(self) -> None:
        """The first pass is a share of the window, so the per-model max-model-len
        (deploy/vllm/model-defaults.sh) is the only place the figure is set: a
        wider window asks for more before it needs a retry at all."""
        from manuscribe.pipeline.model import _first_pass_new_tokens

        assert _first_pass_new_tokens(8192) == 2048
        assert _first_pass_new_tokens(32768) == 8192

    def test_first_pass_budget_leaves_room_for_the_page_image(self) -> None:
        """vLLM rejects a request whose prompt+max_tokens exceeds the window, so
        the budget has to leave the image most of it."""
        from manuscribe.pipeline.model import _first_pass_new_tokens

        for window in (4096, 8192, 32768, 262144):
            budget = _first_pass_new_tokens(window)
            assert 0 < budget <= window * 3 // 4

    def test_a_wide_window_asks_for_more_on_the_first_request(self) -> None:
        import json

        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(json.loads(request.content)["max_tokens"])
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "a page"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 2500},
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(
            client=client,
            base_url="http://srv/v1",
            model="lightonocr",
            context_len=32768,
        )
        _ocr_page(_fake_image(8, 8), ocr)
        assert calls == [8192]

    def test_natural_finish_does_not_retry(self) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "done"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 2500},
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        assert _ocr_page(_fake_image(8, 8), ocr) == "done"
        assert len(calls) == 1

    def test_truncation_without_headroom_keeps_best_effort(self) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        # The prompt already fills the window, so a retry has no more room than the
        # first call — keep the truncated text rather than re-OCRing for nothing.
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "partial"}, "finish_reason": "length"}
                    ],
                    "usage": {"prompt_tokens": 8000},
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        assert _ocr_page(_fake_image(8, 8), ocr) == "partial"
        assert len(calls) == 1

    def test_degenerate_retry_keeps_first_truncated_content(self) -> None:
        import json

        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        # First call truncates with usable text; the retry comes back degenerate
        # (null content -> ""). The good-but-truncated first response must be kept,
        # not replaced by the empty retry — otherwise the whole page is dropped.
        contents = ["a real but truncated page of markdown", None]

        def handler(request: httpx.Request) -> httpx.Response:
            json.loads(request.content)
            content = contents.pop(0)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": content}, "finish_reason": "length"}
                    ],
                    "usage": {"prompt_tokens": 2500},
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        assert (
            _ocr_page(_fake_image(8, 8), ocr) == "a real but truncated page of markdown"
        )

    def test_decode_loop_declines_retry(self) -> None:
        import json

        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        # The first response truncates on a fragment repeated well past the
        # collapse bar — a decode loop, not a dense page. A re-roll at the same
        # greedy temperature reproduces the loop, so the retry must never be
        # sent, and the raw (truncated) loop content is kept rather than a
        # cleaner retry response the model never got the chance to send.
        looped = "<p>Loop</p>" * 50
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(json.loads(request.content)["max_tokens"])
            content, reason = (
                (looped, "length") if len(calls) == 1 else ("clean retry", "stop")
            )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": content}, "finish_reason": reason}
                    ],
                    "usage": {"prompt_tokens": 2500},
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        assert _ocr_page(_fake_image(8, 8), ocr) == looped
        assert len(calls) == 1

    def test_retry_timeout_falls_back_to_truncated_first_pass(self) -> None:
        import json

        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        # The retry's budget can be several times the first pass's on the same
        # fixed request_timeout, so a page whose first pass was merely slow (not
        # looping) can time out on the retry. That must degrade the page to its
        # truncated-but-real first response, not abort the whole document.
        truncated = "a real but truncated page of markdown"
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(json.loads(request.content)["max_tokens"])
            if len(calls) == 1:
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {"content": truncated},
                                "finish_reason": "length",
                            }
                        ],
                        "usage": {"prompt_tokens": 2500},
                    },
                )
            raise httpx.ReadTimeout("simulated retry timeout", request=request)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        assert _ocr_page(_fake_image(8, 8), ocr) == truncated
        assert len(calls) == 2

    def test_decode_loop_partway_through_retries_at_bounded_budget(self) -> None:
        import json

        import httpx

        from manuscribe.pipeline.model import (
            _CONTEXT_SAFETY_MARGIN,
            OcrModel,
            _first_pass_new_tokens,
            _ocr_page,
        )

        # Real prose first, then a fragment repeated well past the collapse bar —
        # the loop clearly starts partway through, not at the response's start,
        # so there is real content worth spending one bounded retry to recover.
        prefix = (
            "Real prose describing the finding on this page in careful, "
            "ordinary detail. " * 5
        )[:400]
        content = prefix + "<p>Loop</p>" * 39
        # Must be at least as long as content's own collapsed length (prefix plus
        # one kept loop instance), or the final "not degenerate" comparison would
        # reject a perfectly good clean retry for merely being the shorter text.
        clean_retry = (
            prefix + "and the recovered tail finishes the caption in full this time, "
            "describing the specimen completely without interruption."
        )
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(json.loads(request.content)["max_tokens"])
            body, reason = (
                (content, "length") if len(calls) == 1 else (clean_retry, "stop")
            )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": body}, "finish_reason": reason}
                    ],
                    "usage": {"prompt_tokens": 100},
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(
            client=client,
            base_url="http://srv/v1",
            model="lightonocr",
            context_len=2048,
        )
        first_pass_budget = _first_pass_new_tokens(ocr.context_len)
        full_window_budget = ocr.context_len - 100 - _CONTEXT_SAFETY_MARGIN

        assert _ocr_page(_fake_image(8, 8), ocr) == clean_retry
        assert len(calls) == 2
        assert first_pass_budget < calls[1] < full_window_budget

    def test_ocr_pages_preserves_order_under_concurrency(self) -> None:
        import json

        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_pages

        # Each page carries a distinct width so the handler can echo its identity;
        # concurrent completion must still gather back in input (page) order.
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            url = body["messages"][0]["content"][0]["image_url"]["url"]
            png = base64.b64decode(url.split(",", 1)[1])
            width = Image.open(io.BytesIO(png)).size[0]
            return httpx.Response(
                200, json={"choices": [{"message": {"content": f"page-{width}"}}]}
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        images = [_fake_image(w, 8) for w in range(10, 18)]
        result = _ocr_pages(images, ocr, concurrency=4)

        assert result == [f"page-{w}" for w in range(10, 18)]

    def test_ocr_pages_empty_input_returns_empty(self) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_pages

        # No images → no requests, no thread pool; returns [] without touching the
        # client (a handler that would fail the test if called).
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be issued for empty input")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        assert _ocr_pages([], ocr) == []

    def test_ocr_pages_propagates_page_error(self) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_pages

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        with pytest.raises(OcrUnavailableError):
            _ocr_pages([_fake_image(8, 8), _fake_image(9, 8)], ocr, concurrency=2)

    def test_ocr_model_context_manager_closes_client(self) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrModel

        transport = httpx.MockTransport(lambda r: httpx.Response(200))
        client = httpx.Client(transport=transport)
        with OcrModel(client=client, base_url="http://srv/v1", model="m") as ocr:
            assert not ocr.client.is_closed
        assert client.is_closed

    @staticmethod
    def _patch_client(monkeypatch: pytest.MonkeyPatch, handler: object) -> list[object]:
        """Make ``model.load_ocr_model``'s internal ``httpx.Client(...)`` use a mock
        transport; return the list of clients it builds so a test can assert on the
        pool's lifecycle."""
        import httpx

        from manuscribe.pipeline import model

        built: list[object] = []
        real_client = httpx.Client

        def fake_client(**kwargs: object) -> httpx.Client:
            client = real_client(transport=httpx.MockTransport(handler), **kwargs)
            built.append(client)
            return client

        monkeypatch.setattr(model.httpx, "Client", fake_client)
        return built

    def test_load_ocr_model_probes_models_and_strips_trailing_slash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import load_ocr_model

        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(200, json={"data": []})

        self._patch_client(monkeypatch, handler)
        ocr = load_ocr_model(base_url="http://srv/v1/", model="m")
        # the trailing slash is stripped and the reachability probe hits /models
        assert seen["url"] == "http://srv/v1/models"
        assert ocr.base_url == "http://srv/v1"
        assert ocr.model == "m"
        ocr.close()

    def test_load_ocr_model_reads_env_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import load_ocr_model

        monkeypatch.setenv("MANUSCRIBE_VLLM_URL", "http://envhost:9/v1")
        monkeypatch.setenv("MANUSCRIBE_VLLM_MODEL", "envmodel")
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(200)

        self._patch_client(monkeypatch, handler)
        ocr = load_ocr_model()
        assert seen["url"] == "http://envhost:9/v1/models"
        assert ocr.model == "envmodel"
        ocr.close()

    def test_load_ocr_model_stores_resolved_concurrency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline import model
        from manuscribe.pipeline.model import load_ocr_model

        # Concurrency is resolved once at load and stored on the model *and* used to
        # size the httpx pool, so the pool cap and the _ocr_pages worker count can't
        # desync (a late env change is then ignored by both).
        monkeypatch.setenv("MANUSCRIBE_OCR_CONCURRENCY", "6")
        captured: dict[str, httpx.Limits] = {}
        real_client = httpx.Client

        def fake_client(**kwargs: object) -> httpx.Client:
            captured["limits"] = kwargs["limits"]  # type: ignore[assignment]
            return real_client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json={"data": []})
                ),
                **kwargs,
            )

        monkeypatch.setattr(model.httpx, "Client", fake_client)
        ocr = load_ocr_model(base_url="http://srv/v1", model="m")
        assert ocr.concurrency == 6
        assert captured["limits"].max_connections == 6
        ocr.close()

    def test_load_ocr_model_closes_client_when_probe_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import load_ocr_model

        built = self._patch_client(
            monkeypatch, lambda request: httpx.Response(503, json={"error": "down"})
        )
        with pytest.raises(OcrUnavailableError) as excinfo:
            load_ocr_model(base_url="http://srv/v1")
        assert isinstance(excinfo.value.__cause__, httpx.HTTPError)
        # the probe failed, so the pool must be torn down rather than leaked
        assert built and built[0].is_closed  # type: ignore[attr-defined]

    def test_reconnect_rebuilds_pool_and_reprobes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import load_ocr_model

        # The restarted server reports a different context window on the second probe.
        probes = {"n": 0}
        reported = [4096, 16384]

        def handler(request: httpx.Request) -> httpx.Response:
            i = probes["n"]
            probes["n"] += 1
            return httpx.Response(200, json={"data": [{"max_model_len": reported[i]}]})

        self._patch_client(monkeypatch, handler)
        ocr = load_ocr_model(base_url="http://srv/v1", model="m", timeout=123.0)
        assert ocr.context_len == 4096
        first_client = ocr.client

        returned = ocr.reconnect()
        assert returned is ocr  # in-place, chainable
        assert probes["n"] == 2  # re-probed the restarted server
        assert ocr.context_len == 16384  # context window re-read
        assert ocr.base_url == "http://srv/v1" and ocr.model == "m"
        assert ocr.request_timeout == 123.0  # explicit timeout preserved
        assert first_client.is_closed  # stale pool torn down
        assert not ocr.client.is_closed and ocr.client is not first_client
        ocr.close()

    def test_engine_defaults_to_lightonocr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrEngine, load_ocr_model

        monkeypatch.delenv("MANUSCRIBE_OCR_ENGINE", raising=False)
        self._patch_client(monkeypatch, lambda r: httpx.Response(200, json={}))
        with load_ocr_model(base_url="http://srv/v1") as ocr:
            assert ocr.engine is OcrEngine.LIGHTONOCR

    def test_engine_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrEngine, load_ocr_model

        monkeypatch.setenv("MANUSCRIBE_OCR_ENGINE", "chandra")
        self._patch_client(monkeypatch, lambda r: httpx.Response(200, json={}))
        with load_ocr_model(base_url="http://srv/v1") as ocr:
            assert ocr.engine is OcrEngine.CHANDRA

    def test_explicit_engine_wins_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrEngine, load_ocr_model

        monkeypatch.setenv("MANUSCRIBE_OCR_ENGINE", "chandra")
        self._patch_client(monkeypatch, lambda r: httpx.Response(200, json={}))
        with load_ocr_model(
            base_url="http://srv/v1", engine=OcrEngine.LIGHTONOCR
        ) as ocr:
            assert ocr.engine is OcrEngine.LIGHTONOCR

    def test_unknown_engine_env_raises_naming_accepted_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import load_ocr_model

        # A silent fallback would run a chandra server's div-tree through the
        # markdown parser and produce a wrong document, not a crash.
        monkeypatch.setenv("MANUSCRIBE_OCR_ENGINE", "tesseract")
        built = self._patch_client(monkeypatch, lambda r: httpx.Response(200))
        with pytest.raises(ValueError, match="lightonocr.*chandra"):
            load_ocr_model(base_url="http://srv/v1")
        assert built == []  # rejected before any pool is built

    def test_engine_is_not_inferred_from_served_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrEngine, load_ocr_model

        # run-server.sh serves every model under the fixed name "lightonocr", so
        # the served name carries no engine signal; only the knob does.
        monkeypatch.delenv("MANUSCRIBE_OCR_ENGINE", raising=False)
        self._patch_client(
            monkeypatch,
            lambda r: httpx.Response(200, json={"data": [{"id": "chandra"}]}),
        )
        with load_ocr_model(base_url="http://srv/v1", model="chandra") as ocr:
            assert ocr.engine is OcrEngine.LIGHTONOCR

    def test_engine_probed_from_lightonocr_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrEngine, load_ocr_model

        monkeypatch.delenv("MANUSCRIBE_OCR_ENGINE", raising=False)
        self._patch_client(
            monkeypatch, lambda r: httpx.Response(200, json=_LIGHTONOCR_MODELS_BODY)
        )
        with load_ocr_model(base_url="http://srv/v1") as ocr:
            assert ocr.engine is OcrEngine.LIGHTONOCR

    def test_engine_probed_from_chandra_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrEngine, load_ocr_model

        # Observed on the live server restarted onto chandra weights: id is still
        # "lightonocr", so root is the only signal — and it rides the probe the
        # context window already needs.
        monkeypatch.delenv("MANUSCRIBE_OCR_ENGINE", raising=False)
        probes = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            probes["n"] += 1
            return httpx.Response(200, json=_CHANDRA_MODELS_BODY)

        self._patch_client(monkeypatch, handler)
        with load_ocr_model(base_url="http://srv/v1") as ocr:
            assert ocr.engine is OcrEngine.CHANDRA
            assert ocr.context_len == 8192  # same body, one round trip
        assert probes["n"] == 1

    def test_engine_probed_from_local_weights_directory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrEngine, load_ocr_model

        # vLLM reports whatever path it was launched with, so a locally staged
        # snapshot of the same weights must be recognised by its directory name.
        monkeypatch.delenv("MANUSCRIBE_OCR_ENGINE", raising=False)
        self._patch_client(
            monkeypatch,
            lambda r: httpx.Response(
                200, json=_models_body_with_root("/srv/models/chandra-ocr-2/")
            ),
        )
        with load_ocr_model(base_url="http://srv/v1") as ocr:
            assert ocr.engine is OcrEngine.CHANDRA

    def test_explicit_engine_wins_over_probed_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrEngine, load_ocr_model

        monkeypatch.delenv("MANUSCRIBE_OCR_ENGINE", raising=False)
        self._patch_client(
            monkeypatch, lambda r: httpx.Response(200, json=_CHANDRA_MODELS_BODY)
        )
        with load_ocr_model(
            base_url="http://srv/v1", engine=OcrEngine.LIGHTONOCR
        ) as ocr:
            assert ocr.engine is OcrEngine.LIGHTONOCR

    def test_env_engine_wins_over_probed_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrEngine, load_ocr_model

        monkeypatch.setenv("MANUSCRIBE_OCR_ENGINE", "lightonocr")
        self._patch_client(
            monkeypatch, lambda r: httpx.Response(200, json=_CHANDRA_MODELS_BODY)
        )
        with load_ocr_model(base_url="http://srv/v1") as ocr:
            assert ocr.engine is OcrEngine.LIGHTONOCR

    def test_unknown_root_falls_back_without_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrEngine, load_ocr_model

        # A fine-tune or a mirror names weights we don't know; unlike a bad env var
        # that is not a wrong instruction, so it defaults rather than raising.
        monkeypatch.delenv("MANUSCRIBE_OCR_ENGINE", raising=False)
        self._patch_client(
            monkeypatch,
            lambda r: httpx.Response(
                200, json=_models_body_with_root("/local/weights/some-finetune")
            ),
        )
        with load_ocr_model(base_url="http://srv/v1") as ocr:
            assert ocr.engine is OcrEngine.LIGHTONOCR

    def test_missing_root_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrEngine, load_ocr_model

        monkeypatch.delenv("MANUSCRIBE_OCR_ENGINE", raising=False)
        self._patch_client(
            monkeypatch,
            lambda r: httpx.Response(200, json=_models_body_with_root(None)),
        )
        with load_ocr_model(base_url="http://srv/v1") as ocr:
            assert ocr.engine is OcrEngine.LIGHTONOCR

    def test_non_json_probe_body_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrEngine, load_ocr_model

        monkeypatch.delenv("MANUSCRIBE_OCR_ENGINE", raising=False)
        self._patch_client(monkeypatch, lambda r: httpx.Response(200, text="OK"))
        with load_ocr_model(base_url="http://srv/v1") as ocr:
            assert ocr.engine is OcrEngine.LIGHTONOCR

    def test_reconnect_preserves_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrEngine, load_ocr_model

        monkeypatch.delenv("MANUSCRIBE_OCR_ENGINE", raising=False)
        self._patch_client(monkeypatch, lambda r: httpx.Response(200, json={}))
        ocr = load_ocr_model(base_url="http://srv/v1", engine=OcrEngine.CHANDRA)
        # The env is unset at reconnect time, so only an explicit pass-through
        # can keep the bundle on chandra.
        assert ocr.reconnect().engine is OcrEngine.CHANDRA
        ocr.close()

    def test_reconnect_keeps_old_bundle_when_reprobe_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import load_ocr_model

        probes = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            probes["n"] += 1
            if probes["n"] == 1:
                return httpx.Response(200, json={"data": [{"max_model_len": 8192}]})
            return httpx.Response(503, json={"error": "restarting"})

        built = self._patch_client(monkeypatch, handler)
        ocr = load_ocr_model(base_url="http://srv/v1", model="m")
        first_client = ocr.client

        with pytest.raises(OcrUnavailableError):
            ocr.reconnect()
        # the freshly built (failed) pool is closed; the original bundle is untouched
        assert built[1].is_closed  # type: ignore[attr-defined]
        assert ocr.client is first_client and not ocr.client.is_closed
        assert ocr.context_len == 8192
        ocr.close()


class TestDecodeLoopOnset:
    """Direct tests of ``_decode_loop_onset``, so the onset-finding logic behind
    ``_ocr_page``'s bounded retry is proven general-purpose rather than tuned to
    one repeated string."""

    def test_partial_loop_onset_is_where_the_repeats_start(self) -> None:
        from manuscribe.pipeline.model import _decode_loop_onset

        prefix = (
            "Table 3 summarizes the measured concentrations across all sampled sites. "
        )
        content = prefix + "<li>N/A</li>" * 10
        assert _decode_loop_onset(content) == len(prefix)

    def test_partial_loop_onset_generalizes_to_a_different_unit(self) -> None:
        from manuscribe.pipeline.model import _decode_loop_onset

        prefix = "Figure 4 shows the cumulative distribution for each cohort. "
        content = prefix + "<span>err</span>" * 12
        assert _decode_loop_onset(content) == len(prefix)

    def test_whole_response_loop_has_no_onset(self) -> None:
        from manuscribe.pipeline.model import _decode_loop_onset

        # None signals nothing precedes the loop worth retrying for — the
        # existing test_decode_loop_declines_retry keeps this content declined.
        assert _decode_loop_onset("<p>Loop</p>" * 50) is None

    def test_no_repeat_run_has_no_onset(self) -> None:
        from manuscribe.pipeline.model import _decode_loop_onset

        assert _decode_loop_onset("<p>one</p><p>two</p><p>three</p>") is None


class TestOcrTransientRetry:
    """A transient connection blip or vLLM overload status on one page must not
    abort the whole document — it is retried with backoff; a non-retryable status
    and an exhausted retry still propagate."""

    @staticmethod
    def _record_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
        """Stub out the backoff sleep and return the list it records each delay into."""
        from manuscribe.pipeline import model

        delays: list[float] = []
        monkeypatch.setattr(model.time, "sleep", lambda s: delays.append(s))
        return delays

    @staticmethod
    def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
        from manuscribe.pipeline import model

        monkeypatch.setattr(model.time, "sleep", lambda _s: None)

    def test_retryable_status_then_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        self._no_sleep(monkeypatch)
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) < 3:  # two transient 503s, then a good response
                return httpx.Response(503, json={"error": "overloaded"})
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "recovered"}}]}
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="m")
        assert _ocr_page(_fake_image(8, 8), ocr) == "recovered"
        assert len(calls) == 3

    def test_transport_error_then_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        self._no_sleep(monkeypatch)
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                raise httpx.ConnectError("connection reset")
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]}
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="m")
        assert _ocr_page(_fake_image(8, 8), ocr) == "ok"
        assert len(calls) == 2

    def test_remote_protocol_error_then_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        self._no_sleep(monkeypatch)
        calls: list[int] = []

        # A peer disconnect mid-response (RemoteProtocolError) is the canonical
        # transient on a busy shared vLLM; it is a *sibling* of NetworkError under
        # TransportError, not a subclass, so it must be retried explicitly rather than
        # fall through to the re-raise.
        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                raise httpx.RemoteProtocolError(
                    "server disconnected without sending a complete response"
                )
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]}
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="m")
        assert _ocr_page(_fake_image(8, 8), ocr) == "ok"
        assert len(calls) == 2

    def test_backoff_has_jitter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        from manuscribe.pipeline import model
        from manuscribe.pipeline.model import _RETRY_BACKOFF_BASE_S, OcrModel, _ocr_page

        delays = self._record_sleeps(monkeypatch)
        # Pin the jitter term so the exact sleep is assertable; the point under test is
        # that it is *added* to the deterministic exponential backoff, so several pages
        # retrying in lockstep don't stampede the just-overloaded server in sync.
        monkeypatch.setattr(model.random, "uniform", lambda _a, _b: 0.123)
        calls: list[int] = []

        # A 503 with no Retry-After header takes the backoff branch (not the honored
        # Retry-After path, which stays jitter-free).
        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(503, json={"error": "overloaded"})
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]}
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="m")
        assert _ocr_page(_fake_image(8, 8), ocr) == "ok"
        # attempt-0 backoff (base * 2**0) plus the pinned jitter term
        assert delays == [_RETRY_BACKOFF_BASE_S + 0.123]

    def test_retries_exhausted_reraises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        from manuscribe.pipeline.model import _MAX_OCR_RETRIES, OcrModel, _ocr_page

        self._no_sleep(monkeypatch)
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(503, json={"error": "down"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="m")
        with pytest.raises(OcrUnavailableError):
            _ocr_page(_fake_image(8, 8), ocr)
        # the initial attempt plus exactly _MAX_OCR_RETRIES retries, then it gives up
        assert len(calls) == _MAX_OCR_RETRIES + 1

    @pytest.mark.parametrize(
        ("status", "expected", "retried"),
        [
            # A rejected request re-fails identically on every lease, so it must carry
            # the *capped*-retry error — the uncapped one has the worker re-render and
            # re-send the whole document forever.
            (400, OcrResponseError, False),
            (404, OcrResponseError, False),
            (413, OcrResponseError, False),
            # 4xx by number only: a timeout and a back-pressure hint are transient, so
            # they keep the uncapped disposition (429 is retried in-request too).
            (408, OcrUnavailableError, False),
            (429, OcrUnavailableError, True),
            (500, OcrUnavailableError, False),
        ],
    )
    def test_status_classification_and_attempt_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
        status: int,
        expected: type[ManuscribeError],
        retried: bool,
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import _MAX_OCR_RETRIES, OcrModel, _ocr_page

        self._no_sleep(monkeypatch)
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(status, json={"error": "nope"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="m")
        with pytest.raises(expected) as excinfo:
            _ocr_page(_fake_image(8, 8), ocr)
        # exact type, not a subclass: the two dispositions are siblings, so an
        # isinstance-style check would pass on the wrong one if they ever merged.
        assert type(excinfo.value) is expected
        assert isinstance(excinfo.value.__cause__, httpx.HTTPStatusError)
        assert len(calls) == (_MAX_OCR_RETRIES + 1 if retried else 1)

    def test_read_timeout_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_page

        self._no_sleep(monkeypatch)
        calls: list[int] = []

        # A read timeout already spent the full per-request budget; retrying would
        # re-spend it (~3× the 600s timeout) and defeats _ocr_pages' fast-fail of a
        # wedged page — so it must propagate on the first attempt, unlike a connection
        # blip (ConnectError, a NetworkError) which IS retried.
        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            raise httpx.ReadTimeout("read timed out")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="m")
        with pytest.raises(OcrUnavailableError) as excinfo:
            _ocr_page(_fake_image(8, 8), ocr)
        assert isinstance(excinfo.value.__cause__, httpx.ReadTimeout)
        assert len(calls) == 1

    def test_retry_after_header_is_honored_and_capped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import _MAX_RETRY_AFTER_S, OcrModel, _ocr_page

        delays = self._record_sleeps(monkeypatch)
        calls: list[int] = []
        # First a 503 with a sane Retry-After (honored verbatim), then a 503 with an
        # absurd one (capped), then success — the client waits the server's hint rather
        # than piling load back onto a busy server.
        retry_afters = ["2", "9999"]

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if retry_afters:
                return httpx.Response(503, headers={"Retry-After": retry_afters.pop(0)})
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]}
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="m")
        assert _ocr_page(_fake_image(8, 8), ocr) == "ok"
        assert delays == [2.0, _MAX_RETRY_AFTER_S]


class TestOcrConcurrencyResolution:
    """``MANUSCRIBE_OCR_CONCURRENCY`` parsing — defaulting, flooring, and the
    misconfiguration warning — without spinning up the thread pool."""

    def test_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from manuscribe.pipeline.model import (
            _DEFAULT_OCR_CONCURRENCY,
            _resolve_ocr_concurrency,
        )

        monkeypatch.delenv("MANUSCRIBE_OCR_CONCURRENCY", raising=False)
        assert _resolve_ocr_concurrency() == _DEFAULT_OCR_CONCURRENCY

    def test_valid_integer_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from manuscribe.pipeline.model import _resolve_ocr_concurrency

        monkeypatch.setenv("MANUSCRIBE_OCR_CONCURRENCY", "7")
        assert _resolve_ocr_concurrency() == 7

    def test_floored_at_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from manuscribe.pipeline.model import _resolve_ocr_concurrency

        monkeypatch.setenv("MANUSCRIBE_OCR_CONCURRENCY", "0")
        assert _resolve_ocr_concurrency() == 1

    def test_non_integer_warns_and_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from manuscribe.pipeline.model import (
            _DEFAULT_OCR_CONCURRENCY,
            _resolve_ocr_concurrency,
        )

        monkeypatch.setenv("MANUSCRIBE_OCR_CONCURRENCY", "lots")
        # a misconfigured deployment is surfaced, not silently defaulted
        with pytest.warns(UserWarning, match="not an integer"):
            assert _resolve_ocr_concurrency() == _DEFAULT_OCR_CONCURRENCY

    def test_ocr_pages_uses_model_concurrency_not_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        import httpx

        from manuscribe.pipeline import model
        from manuscribe.pipeline.model import OcrModel, _ocr_pages

        # Env raised *after* load must not change the worker count: it is taken from the
        # value resolved at load time (OcrModel.concurrency), which also sized the httpx
        # pool — so a late env change can't desync the workers from the pool cap.
        monkeypatch.setenv("MANUSCRIBE_OCR_CONCURRENCY", "9")
        captured: list[int] = []
        real_pool = model.ThreadPoolExecutor

        def spy_pool(max_workers: int) -> ThreadPoolExecutor:
            captured.append(max_workers)
            return real_pool(max_workers=max_workers)

        monkeypatch.setattr(model, "ThreadPoolExecutor", spy_pool)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]}
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(
            client=client, base_url="http://srv/v1", model="m", concurrency=3
        )
        images = [_fake_image(8, 8) for _ in range(5)]
        assert _ocr_pages(images, ocr) == ["ok"] * 5
        # worker count is ocr.concurrency (3), floored by the image count — not the
        # post-load env value (9)
        assert captured == [3]

    def test_ocr_pages_fast_fail_does_not_wait_on_slow_sibling(self) -> None:
        import json
        import threading

        import httpx

        from manuscribe.pipeline.model import OcrModel, _ocr_pages

        # Page 0 blocks until released; page 1 fails fast.  The failure must surface
        # without first waiting out the slow earlier-submitted page — the property a
        # submission-order ``[f.result()]`` gather would defeat.
        release = threading.Event()

        def handler(request: httpx.Request) -> httpx.Response:
            width = Image.open(
                io.BytesIO(
                    base64.b64decode(
                        json.loads(request.content)["messages"][0]["content"][0][
                            "image_url"
                        ]["url"].split(",", 1)[1]
                    )
                )
            ).size[0]
            if width == 10:
                release.wait(timeout=5)
                return httpx.Response(
                    200, json={"choices": [{"message": {"content": "slow"}}]}
                )
            return httpx.Response(500, json={"error": "boom"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="m")
        try:
            with pytest.raises(OcrUnavailableError):
                _ocr_pages([_fake_image(10, 8), _fake_image(11, 8)], ocr, concurrency=2)
        finally:
            release.set()  # let the still-running slow page unwind


class TestOcrTimeoutResolution:
    """``MANUSCRIBE_OCR_TIMEOUT`` / ``MANUSCRIBE_OCR_HEALTH_TIMEOUT`` parsing —
    defaulting, honoring a value, and the misconfiguration warnings — plus the
    load-time wiring onto the client and the reachability probe."""

    def test_request_timeout_default_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from manuscribe.pipeline.model import (
            _DEFAULT_REQUEST_TIMEOUT_S,
            _resolve_request_timeout,
        )

        monkeypatch.delenv("MANUSCRIBE_OCR_TIMEOUT", raising=False)
        assert _resolve_request_timeout() == _DEFAULT_REQUEST_TIMEOUT_S

    def test_health_timeout_default_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from manuscribe.pipeline.model import (
            _DEFAULT_HEALTH_TIMEOUT_S,
            _resolve_health_timeout,
        )

        monkeypatch.delenv("MANUSCRIBE_OCR_HEALTH_TIMEOUT", raising=False)
        assert _resolve_health_timeout() == _DEFAULT_HEALTH_TIMEOUT_S

    def test_valid_float_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from manuscribe.pipeline.model import _resolve_request_timeout

        monkeypatch.setenv("MANUSCRIBE_OCR_TIMEOUT", "42.5")
        assert _resolve_request_timeout() == 42.5

    def test_non_number_warns_and_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from manuscribe.pipeline.model import (
            _DEFAULT_HEALTH_TIMEOUT_S,
            _resolve_health_timeout,
        )

        monkeypatch.setenv("MANUSCRIBE_OCR_HEALTH_TIMEOUT", "soon")
        with pytest.warns(UserWarning, match="not a number"):
            assert _resolve_health_timeout() == _DEFAULT_HEALTH_TIMEOUT_S

    def test_non_positive_warns_and_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from manuscribe.pipeline.model import (
            _DEFAULT_REQUEST_TIMEOUT_S,
            _resolve_request_timeout,
        )

        # A zero/negative timeout is a misconfiguration with no sane clamp (unlike the
        # integer concurrency floor) — it warns and falls back rather than coercing.
        monkeypatch.setenv("MANUSCRIBE_OCR_TIMEOUT", "0")
        with pytest.warns(UserWarning, match="must be positive"):
            assert _resolve_request_timeout() == _DEFAULT_REQUEST_TIMEOUT_S

    def test_load_ocr_model_wires_env_timeouts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline import model
        from manuscribe.pipeline.model import load_ocr_model

        # The request timeout reaches the client default; the shorter health timeout is
        # applied as a per-request override on the /models probe (httpx surfaces the
        # resolved per-request value in request.extensions["timeout"]).
        monkeypatch.setenv("MANUSCRIBE_OCR_TIMEOUT", "123")
        monkeypatch.setenv("MANUSCRIBE_OCR_HEALTH_TIMEOUT", "7")
        captured: dict[str, object] = {}
        real_client = httpx.Client

        def handler(request: httpx.Request) -> httpx.Response:
            captured["probe_timeout"] = request.extensions["timeout"]["read"]
            return httpx.Response(200, json={"data": []})

        def fake_client(**kwargs: object) -> httpx.Client:
            captured["client_timeout"] = kwargs["timeout"]
            return real_client(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(model.httpx, "Client", fake_client)
        ocr = load_ocr_model(base_url="http://srv/v1", model="m")
        assert captured["client_timeout"] == 123.0
        assert captured["probe_timeout"] == 7.0
        ocr.close()

    def test_explicit_timeout_arg_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline import model
        from manuscribe.pipeline.model import load_ocr_model

        monkeypatch.setenv("MANUSCRIBE_OCR_TIMEOUT", "123")
        captured: dict[str, object] = {}
        real_client = httpx.Client

        def fake_client(**kwargs: object) -> httpx.Client:
            captured["client_timeout"] = kwargs["timeout"]
            return real_client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json={"data": []})
                ),
                **kwargs,
            )

        monkeypatch.setattr(model.httpx, "Client", fake_client)
        ocr = load_ocr_model(base_url="http://srv/v1", model="m", timeout=5.0)
        # the explicit arg wins over the env value, mirroring base_url/model
        assert captured["client_timeout"] == 5.0
        ocr.close()


class TestCli:
    """The ``python -m manuscribe`` entry point: output-path defaulting and option
    forwarding, with the conversion itself stubbed (no model, no rendering)."""

    @staticmethod
    def _make_pdf(tmp_path: Path) -> Path:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        return pdf

    def test_default_output_path_derived_from_pdf_suffix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import manuscribe.__main__ as cli

        pdf = self._make_pdf(tmp_path)
        captured: dict[str, object] = {}

        def fake_convert(p: Path, **kwargs: object) -> str:
            captured["pdf"] = p
            captured["kwargs"] = kwargs
            return "<html>ok</html>"

        monkeypatch.setattr(cli, "lightonocr_pdf_to_html", fake_convert)
        assert cli.main([str(pdf)]) == 0
        # default output is the input path with a .html suffix
        out = (tmp_path / "paper.html").read_text(encoding="utf-8")
        assert out == "<html>ok</html>"
        assert captured["pdf"] == pdf
        assert captured["kwargs"] == {
            "base_url": None,
            "model": None,
            "image_dir": None,
        }

    def test_explicit_output_and_options_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import manuscribe.__main__ as cli

        pdf = self._make_pdf(tmp_path)
        out = tmp_path / "custom.html"
        image_dir = tmp_path / "imgs"
        captured: dict[str, object] = {}

        def fake_convert(p: Path, **kwargs: object) -> str:
            captured.update(kwargs)
            return "<html></html>"

        monkeypatch.setattr(cli, "lightonocr_pdf_to_html", fake_convert)
        rc = cli.main(
            [
                str(pdf),
                str(out),
                "--vllm-url",
                "http://h/v1",
                "--vllm-model",
                "m",
                "--image-dir",
                str(image_dir),
            ]
        )
        assert rc == 0
        assert out.read_text(encoding="utf-8") == "<html></html>"
        assert captured["base_url"] == "http://h/v1"
        assert captured["model"] == "m"
        assert captured["image_dir"] == image_dir

    def test_missing_pdf_errors_before_conversion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import manuscribe.__main__ as cli

        called: list[int] = []
        monkeypatch.setattr(
            cli,
            "lightonocr_pdf_to_html",
            lambda *a, **k: called.append(1) or "",
        )
        # argparse's parser.error exits with code 2 and the convert is never reached
        with pytest.raises(SystemExit) as exc:
            cli.main([str(tmp_path / "absent.pdf")])
        assert exc.value.code == 2
        assert not called

    def test_unreachable_server_prints_message_returns_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import httpx

        import manuscribe.__main__ as cli

        pdf = self._make_pdf(tmp_path)

        def boom(*a: object, **k: object) -> str:
            raise OcrUnavailableError("server unreachable") from httpx.ConnectError(
                "connection refused"
            )

        monkeypatch.setattr(cli, "lightonocr_pdf_to_html", boom)
        # An unreachable server must yield a concise stderr message naming the resolved
        # base URL and a non-zero exit, not a raw traceback or a written output file.
        rc = cli.main([str(pdf), "--vllm-url", "http://down:9/v1"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "http://down:9/v1" in err
        assert "connection refused" in err
        assert not (tmp_path / "paper.html").exists()


class TestServerContextWindow:
    """The server's ``max_model_len`` (the truncation-retry budget) is resolved once
    at load time from ``/models``, with the env var as override — not re-read from
    ``os.environ`` per page."""

    def test_parse_server_context_len_extracts_max_model_len(self) -> None:
        from manuscribe.pipeline.model import _parse_server_context_len

        payload = {"object": "list", "data": [{"id": "m", "max_model_len": 16384}]}
        assert _parse_server_context_len(payload) == 16384

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            {},
            {"data": []},
            {"data": "nope"},
            {"data": [{"id": "m"}]},  # no max_model_len (older server)
            {"data": [{"id": "m", "max_model_len": 0}]},  # non-positive
            {"data": [{"id": "m", "max_model_len": "8192"}]},  # wrong type
        ],
    )
    def test_parse_server_context_len_none_on_missing_or_bad_shape(
        self, payload: object
    ) -> None:
        from manuscribe.pipeline.model import _parse_server_context_len

        assert _parse_server_context_len(payload) is None

    def test_resolve_prefers_reported_over_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from manuscribe.pipeline.model import _resolve_context_len

        monkeypatch.delenv("MANUSCRIBE_VLLM_MAX_MODEL_LEN", raising=False)
        assert _resolve_context_len(16384) == 16384

    def test_resolve_falls_back_to_default_when_unreported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from manuscribe.pipeline.model import (
            _DEFAULT_MODEL_CONTEXT_LEN,
            _resolve_context_len,
        )

        monkeypatch.delenv("MANUSCRIBE_VLLM_MAX_MODEL_LEN", raising=False)
        assert _resolve_context_len(None) == _DEFAULT_MODEL_CONTEXT_LEN

    def test_env_override_beats_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from manuscribe.pipeline.model import _resolve_context_len

        monkeypatch.setenv("MANUSCRIBE_VLLM_MAX_MODEL_LEN", "5000")
        assert _resolve_context_len(16384) == 5000

    def test_bad_env_override_warns_and_falls_back_to_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from manuscribe.pipeline.model import _resolve_context_len

        monkeypatch.setenv("MANUSCRIBE_VLLM_MAX_MODEL_LEN", "lots")
        with pytest.warns(UserWarning):
            assert _resolve_context_len(16384) == 16384

    def test_load_ocr_model_stores_reported_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import load_ocr_model

        monkeypatch.delenv("MANUSCRIBE_VLLM_MAX_MODEL_LEN", raising=False)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"max_model_len": 16384}]})

        TestOcrSeam._patch_client(monkeypatch, handler)
        ocr = load_ocr_model(base_url="http://srv/v1")
        assert ocr.context_len == 16384
        ocr.close()

    def test_load_ocr_model_env_override_beats_reported_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import load_ocr_model

        monkeypatch.setenv("MANUSCRIBE_VLLM_MAX_MODEL_LEN", "4096")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"max_model_len": 16384}]})

        TestOcrSeam._patch_client(monkeypatch, handler)
        ocr = load_ocr_model(base_url="http://srv/v1")
        assert ocr.context_len == 4096
        ocr.close()

    def test_load_ocr_model_defaults_window_when_unreported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from manuscribe.pipeline.model import (
            _DEFAULT_MODEL_CONTEXT_LEN,
            load_ocr_model,
        )

        monkeypatch.delenv("MANUSCRIBE_VLLM_MAX_MODEL_LEN", raising=False)

        # A bare-200 probe (no JSON body) must not crash resolution.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        TestOcrSeam._patch_client(monkeypatch, handler)
        ocr = load_ocr_model(base_url="http://srv/v1")
        assert ocr.context_len == _DEFAULT_MODEL_CONTEXT_LEN
        ocr.close()

    def test_ocr_page_retry_budget_uses_stored_window(self) -> None:
        import json

        import httpx

        from manuscribe.pipeline.model import (
            _CONTEXT_SAFETY_MARGIN,
            OcrModel,
            _ocr_page,
        )

        # The truncation retry must size its max_tokens off the bundle's context_len,
        # not a hardcoded/env-read default — proving the per-page os.environ re-read is
        # gone and a non-default server window is honored.
        calls: list[int] = []
        truncated = "<table>cut"
        full = truncated + " and the rest</table>"

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append(body["max_tokens"])
            content, reason = (
                (truncated, "length") if len(calls) == 1 else (full, "stop")
            )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": content}, "finish_reason": reason}
                    ],
                    "usage": {"prompt_tokens": 1000},
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(
            client=client, base_url="http://srv/v1", model="m", context_len=4096
        )
        assert _ocr_page(_fake_image(8, 8), ocr) == full
        assert calls[1] == 4096 - 1000 - _CONTEXT_SAFETY_MARGIN

    def test_client_limits_track_concurrency(self) -> None:
        from manuscribe.pipeline.model import _client_limits

        # the pool is sized to exactly the resolved worker count, both caps (the
        # env→worker-count resolution is exercised end-to-end by the load test below)
        limits = _client_limits(9)
        assert limits.max_connections == 9
        assert limits.max_keepalive_connections == 9
