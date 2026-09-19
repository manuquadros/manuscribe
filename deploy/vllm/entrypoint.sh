#!/usr/bin/env bash
# Unified-image entrypoint: resolve the card-dependent flags at container *start*
# (the GPU is only visible then, never at build time), then hand off to the base
# image's `vllm serve`.
set -euo pipefail

# shellcheck source=deploy/vllm/gpu-defaults.sh
. /usr/local/bin/gpu-defaults.sh
apply_gpu_defaults

# vLLM takes the model as the first positional. No model leaves this empty and
# gets the fallback window, as an unrecognised one does.
# shellcheck source=deploy/vllm/model-defaults.sh
. /usr/local/bin/model-defaults.sh
apply_model_defaults "${1:-}"

# vLLM rejects a command whose model is not the *first* positional ("`model`
# should be provided as the first positional argument"), so derived flags can
# only be appended, never prepended.  Appended, they would also win over an
# explicit setting in the command — vLLM takes the last --dtype, and rejects
# --attention-backend outright when --attention-config already carries one — so
# add each only when the caller supplied nothing equivalent.
derived=()
caller_dtype=""
caller_backend=""
caller_max_len=""
caller_mem_util=""
caller_eager=""
for arg in "$@"; do
  case "$arg" in
    --dtype | --dtype=*) caller_dtype=1 ;;
    --attention-backend | --attention-backend=* | \
      --attention-config | --attention-config=* | -ac) caller_backend=1 ;;
    --max-model-len | --max-model-len=*) caller_max_len=1 ;;
    --gpu-memory-utilization | --gpu-memory-utilization=*) caller_mem_util=1 ;;
    --enforce-eager | --no-enforce-eager) caller_eager=1 ;;
  esac
done

[ -n "$caller_dtype" ] || derived+=(--dtype "$DTYPE")
if [ -z "$caller_backend" ] && [ -n "${ATTENTION_BACKEND:-}" ]; then
  derived+=(--attention-backend "$ATTENTION_BACKEND")
fi
[ -n "$caller_max_len" ] || derived+=(--max-model-len "$MAX_MODEL_LEN")
[ -n "$caller_mem_util" ] ||
  derived+=(--gpu-memory-utilization "$GPU_MEM_UTIL")
# ENFORCE_EAGER is the whole flag or empty (CUDA graphs on), not a value.
if [ -z "$caller_eager" ] && [ -n "${ENFORCE_EAGER:-}" ]; then
  derived+=("$ENFORCE_EAGER")
fi

exec vllm serve "$@" "${derived[@]}"
