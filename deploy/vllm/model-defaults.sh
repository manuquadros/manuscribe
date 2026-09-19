#!/usr/bin/env bash
# Weights-dependent vLLM flags, sourced by run-server.sh and entrypoint.sh like
# gpu-defaults.sh next door.  Whatever the caller exported wins.
#
# Too small a window loses data rather than speed: the client derives its
# truncation-retry budget from the server's max_model_len and gives up on the
# retry once the remainder no longer covers a generation, so a truncated page's
# tail is simply dropped.

# Trailing path component, lowercased -- the key model.py's _ENGINE_BY_WEIGHTS
# uses, so a mirror or local snapshot resolves the same on both sides.  Keep the
# two tables in step.
_model_weights_name() {
  printf '%s\n' "$1" | tr '[:upper:]' '[:lower:]' | sed 's#/*$##; s#.*/##'
}

apply_model_defaults() {
  case "$(_model_weights_name "${1:-}")" in
    # Native window is 262144; 32 k clears a dense page without sizing every
    # request's KV reservation for one no page approaches.
    chandra-ocr-2) : "${MAX_MODEL_LEN:=32768}" ;;
  esac

  # LightOnOCR's window, and the floor for anything unrecognised.
  : "${MAX_MODEL_LEN:=8192}"

  echo "model-defaults: ${1:-<no model>} -> --max-model-len $MAX_MODEL_LEN" >&2
}
