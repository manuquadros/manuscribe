#!/usr/bin/env bash
# GPU-capability-derived vLLM flags, shared by the two deployment shapes
# (run-server.sh on the host, entrypoint.sh inside the unified image) so the same
# invocation serves an Ada card and a T4.
#
# Two of the server's settings depend on the card, and they fail very differently:
#
#   --dtype bfloat16        vLLM refuses it below sm_80 and says so at startup
#                           ("Bfloat16 is only supported on GPUs with compute
#                           capability of at least 8.0"), so that half is a
#                           legible stop.
#   --attention-backend     vLLM's auto-selection picks FlashInfer on a T4 --
#                           FlashInferBackend.supports_compute_capability(7.5)
#                           answers True -- and the image's prebuilt kernels then
#                           kill the engine on the first request with
#                           "BatchPrefillWithPagedKVCache failed with error
#                           invalid argument".  Because that gate is optimistic
#                           rather than wrong-by-omission, auto-selection never
#                           corrects itself: the backend has to be named.
#
# The backend is a *command-line flag*.  VLLM_ATTENTION_BACKEND, which most
# advice online still names, does not exist in vLLM 0.22.1 -- the string appears
# nowhere in the package -- so setting it is silently inert.
#
# Whatever the caller already exported wins; this only fills blanks.

# Lowest compute capability among the visible GPUs -- the server has to run on
# every one of them -- as "<major>.<minor>", or empty when nvidia-smi is absent
# or predates the compute_cap query.
_gpu_compute_cap() {
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
    tr -d '[:blank:]' |
    grep -E '^[0-9]+\.[0-9]+$' |
    sort -t. -k1,1n -k2,2n |
    head -1
}

# Report the flags that will actually be used, which is not always the branch's
# own preference: an explicit override reaches here already set.
_report_gpu_defaults() {
  if [ -n "${ATTENTION_BACKEND:-}" ]; then
    echo "gpu-defaults: $1 -> --dtype $DTYPE," \
         "--attention-backend $ATTENTION_BACKEND" >&2
  else
    echo "gpu-defaults: $1 -> --dtype $DTYPE, vLLM's own backend choice" >&2
  fi
}

# bf16 is right on Blackwell (GB10 is sm_121), but vLLM's backend choice is
# only as good as the image's kernels, and older tags have none for sm_121 --
# which, as on the T4, kills the engine on the first request, not at startup.
_report_blackwell_caveat() {
  echo "gpu-defaults: Blackwell ($1) is unverified -- run smoke-test.sh" \
       "first; see deploy/vllm/README.md's GB10 section." >&2
}

# Fill in DTYPE and ATTENTION_BACKEND to match the detected card.  An empty
# ATTENTION_BACKEND means "let vLLM choose", which is right on Ampere and newer.
apply_gpu_defaults() {
  local cap major
  cap="$(_gpu_compute_cap)"
  major="${cap%%.*}"

  if [ -n "$cap" ] && [ "$major" -ge 8 ]; then
    # Ampere or newer: the configuration the determinism spike validated, with
    # the backend left to vLLM (FlashInfer, which does run here).
    : "${DTYPE:=bfloat16}"
    : "${ATTENTION_BACKEND:=}"
    _report_gpu_defaults "compute capability $cap"
    if [ "$major" -ge 10 ]; then
      _report_blackwell_caveat "$cap"
    fi
    return 0
  fi

  # Pre-Ampere, or a card we could not identify.  TRITON_ATTN is the backend that
  # reports support down to sm_60; FlashInfer would be auto-selected instead and
  # its prebuilt kernels abort the engine on the first request.
  #
  # float32, not float16, and that is the expensive part.  This model's weights
  # are bfloat16, and casting them to fp16 does not merely lose precision -- the
  # server comes up healthy and then emits nothing but "!" (token 0, the
  # signature of non-finite logits), measured here on both FlashInfer and
  # TRITON_ATTN with bfloat16 correct on the same image.  Below sm_80 there is no
  # bfloat16 to fall back on, so fp32 is the only width left that is known not to
  # overflow.  It doubles the weight footprint to ~4.8 GiB and is slower, which
  # is a poor trade -- but silent garbage is a worse one, and a pipeline that
  # OCRs every page to "!" fails much later and much less legibly than one that
  # runs out of memory at startup.
  : "${DTYPE:=float32}"
  : "${ATTENTION_BACKEND:=TRITON_ATTN}"
  if [ -n "$cap" ]; then
    _report_gpu_defaults "compute capability $cap is pre-Ampere"
  else
    _report_gpu_defaults "no compute capability readable (no nvidia-smi?), assuming pre-Ampere"
    echo "gpu-defaults: set DTYPE / ATTENTION_BACKEND to override that guess." >&2
  fi
  echo "gpu-defaults: this path is NOT covered by the determinism spike -- run" \
       "deploy/vllm/smoke-test.sh and check the output is markdown, not \"!!!\"," \
       "before trusting a document. DTYPE=float16 is known to produce exactly" \
       "that." >&2
}
