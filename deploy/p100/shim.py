"""P100 fallback OCR server: LightOnOCR-2-1B via HF transformers behind the two
OpenAI-compatible routes manuscribe's model seam (``pipeline/model.py``) uses.

Why this exists: the pinned vLLM image has no sm_60 (Pascal) kernels, so vLLM
crashes on a P100 with ``cudaErrorNoKernelImageForDevice``.  ``transformers``
runs the model with standard ops, so a Pascal-capable torch build is enough.
This is a keep-the-P100 compromise — serialized, eager, no continuous batching,
and outside the pinned-vLLM determinism/fidelity gate.

Not a general OpenAI server: only ``GET /v1/models`` (reachability + context
probe) and ``POST /v1/chat/completions`` (single image, greedy) are implemented,
matching exactly what the client parses.  No streaming, no batching, one GPU.

Run (in a venv carrying torch+transformers, NOT the manuscribe venv)::

    python deploy/p100/shim.py

Env knobs: ``SHIM_MODEL_ID`` / ``SHIM_SERVED_NAME`` / ``SHIM_MAX_MODEL_LEN`` /
``SHIM_PORT``.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import re
import threading
from contextlib import asynccontextmanager
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

MODEL_ID = os.environ.get("SHIM_MODEL_ID", "lightonai/LightOnOCR-2-1B-bbox")
SERVED_NAME = os.environ.get("SHIM_SERVED_NAME", "lightonocr")
# The client (pipeline/model.py) reads this off /v1/models to size its
# truncation-retry token budget; it is not enforced here.
MAX_MODEL_LEN = int(os.environ.get("SHIM_MAX_MODEL_LEN", "8192"))
# Bind address.  Default 127.0.0.1 keeps the unauthenticated server local — never
# expose it directly on a public interface.  For tailnet access, either leave it on
# 127.0.0.1 and front it with `tailscale serve`, or set SHIM_HOST to the Tailscale
# IP (100.x.y.z) so only tailnet peers can reach it (WireGuard encrypts the hop).
HOST = os.environ.get("SHIM_HOST", "127.0.0.1")
PORT = int(os.environ.get("SHIM_PORT", "8000"))
# Pascal has no bfloat16 (the model card's recommended dtype); fp16 is the P100 path.
DTYPE = torch.float16

_DATA_URI_RE = re.compile(r"^data:[\w/+.\-]*;base64,(?P<b64>.+)$", re.DOTALL)

_state: dict[str, Any] = {}
# generate() is not concurrency-safe and there is one GPU; the client issues
# pages in parallel (default 16), so serialize decoding here.  Also set
# MANUSCRIBE_OCR_CONCURRENCY=1 client-side to avoid pointless queueing.
_gpu_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # attn_implementation="eager": no FlashAttention on Pascal.  If a transformers
    # build rejects the kwarg for this model, drop it (torch-SDPA's math backend
    # also runs on sm_60).
    _state["model"] = (
        LightOnOcrForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE, attn_implementation="eager"
        )
        .to("cuda")
        .eval()
    )
    _state["processor"] = LightOnOcrProcessor.from_pretrained(MODEL_ID)
    yield
    _state.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/v1/models")
def models() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": SERVED_NAME, "object": "model", "max_model_len": MAX_MODEL_LEN}
        ],
    }


def _decode_image(url: str) -> Image.Image:
    match = _DATA_URI_RE.match(url)
    if match is None:
        raise HTTPException(400, "image_url must be a base64 data URI")
    try:
        raw = base64.b64decode(match.group("b64"), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(400, f"bad base64 image: {exc}") from exc
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _extract_image(body: dict) -> Image.Image:
    for message in body.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                return _decode_image(part["image_url"]["url"])
    raise HTTPException(400, "no image_url part in request")


@app.post("/v1/chat/completions")
def chat(body: dict) -> dict:
    image = _extract_image(body)
    max_new = int(body.get("max_tokens", 2048))
    processor = _state["processor"]
    model = _state["model"]

    # Pass the decoded PIL directly.  If a transformers version rejects a PIL under
    # the "image" key, fall back to the original data URI as "url" (recent
    # transformers.image_utils.load_image decodes base64 data URIs).
    conversation = [{"role": "user", "content": [{"type": "image", "image": image}]}]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    prompt_tokens = int(inputs["input_ids"].shape[1])

    with _gpu_lock, torch.inference_mode():
        try:
            # do_sample=False (greedy) matches the client's temperature=0.0 contract
            # and keeps the transcription reproducible run-to-run.
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            new_ids = out[0, prompt_tokens:]
            text = processor.decode(new_ids, skip_special_tokens=True)
            completion_tokens = int(new_ids.shape[0])
        finally:
            # Page images and context lengths vary request-to-request, so the CUDA
            # caching allocator fragments; on a 16 GiB card a later *normal* page
            # then OOMs in the eager vision-encoder attention (which materializes a
            # full fp32 seq×seq matrix) even though no single request is too big.
            # Return cached blocks between requests — safe because the GPU lock
            # serializes generation.  Pair with, on the server env,
            # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
            del inputs
            torch.cuda.empty_cache()
    # finish_reason="length" is load-bearing: it triggers the client's dense-page
    # re-OCR with the full context window (pipeline/model.py _ocr_page).
    finish_reason = "length" if completion_tokens >= max_new else "stop"

    return {
        "id": "chatcmpl-shim",
        "object": "chat.completion",
        "model": SERVED_NAME,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
