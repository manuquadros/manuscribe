#!/usr/bin/env bash
# Start the LightOnOCR vLLM OpenAI-compatible server in a rootless podman
# container. Keeps the model warm across documents so the pipeline pays the
# ~35 s load once, not per run.
#
# Card-dependent settings (dtype, attention backend, CUDA graphs, memory
# fraction) come from gpu-defaults.sh, the weights-dependent one (context
# window) from model-defaults.sh, so this runs unedited on a 6 GiB card, a T4
# and a GB10, under either engine.
#
# The flashinfer JIT sampler stays off: the 6 GiB box has no nvcc, and decoding
# is greedy, so the Torch-native fallback costs nothing measurable.
# VLLM_USE_FLASHINFER_SAMPLER=1 re-enables it.
#
# Override any value via env:
#   PORT=8001 ./run-server.sh
#   MODEL=datalab-to/chandra-ocr-2 ./run-server.sh
#   GPU_MEM_UTIL=0.6 ENFORCE_EAGER= ./run-server.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-docker.io/vllm/vllm-openai:v0.29.0}"
NAME="${NAME:-lighton-vllm}"
MODEL="${MODEL:-lightonai/LightOnOCR-2-1B-bbox}"
PORT="${PORT:-8000}"
# Loopback by default: vLLM has no auth of its own, so publishing on every
# interface — podman's behaviour when -p names no address — would serve the
# model to anyone who reaches the host.  Reaching it from another machine is
# therefore an explicit act:  BIND_ADDR="$(tailscale ip -4)" ./run-server.sh
# Matches deploy/p100/shim.py's SHIM_HOST default; see deploy/SECURITY.md.
BIND_ADDR="${BIND_ADDR:-127.0.0.1}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# DTYPE and ATTENTION_BACKEND default to whatever the card supports; an explicit
# value here or in the environment wins.  See gpu-defaults.sh for why the backend
# cannot be left to vLLM's own selection on a pre-Ampere card — and why it is a
# flag rather than the VLLM_ATTENTION_BACKEND variable most advice still names.
# shellcheck source=deploy/vllm/gpu-defaults.sh
. "$SCRIPT_DIR/gpu-defaults.sh"
apply_gpu_defaults

# MAX_MODEL_LEN follows the weights; see model-defaults.sh on why too small a
# window drops a truncated page's tail outright.
# shellcheck source=deploy/vllm/model-defaults.sh
. "$SCRIPT_DIR/model-defaults.sh"
apply_model_defaults "$MODEL"

# Mount the host HF cache at a fixed path and point HF_HOME at it explicitly,
# rather than guessing the image's home dir — the cache is reused (no re-pull of
# the 2.7 GiB model) whether the container runs as root or a non-root user.
# Rootless podman maps the host user onto the container user, so the existing
# weights remain readable through the mount.
# Accept a bracketed IPv6 literal as well, so the value can be pasted straight
# from a -p flag; the bare form is what the interface check below needs.
bind_bare="${BIND_ADDR#"["}"
bind_bare="${bind_bare%"]"}"
case "$bind_bare" in
  # An IPv6 literal must be bracketed in -p or its colons read as the field
  # separators between address, host port and container port.
  *:*) publish="[${bind_bare}]:${PORT}:8000" ;;
  *) publish="${bind_bare}:${PORT}:8000" ;;
esac

# A boot-time unit can start this before tailscaled (or any other late
# interface) has brought the address up, and podman's own bind failure names
# neither the address nor the reason.  The wildcards are exempt: they are valid
# binds that belong to no interface, so the check would otherwise reject the one
# deliberate way of asking for every interface.
case "$bind_bare" in
  0.0.0.0 | ::) wildcard=yes ;;
  *) wildcard=no ;;
esac
if [ "$wildcard" = no ] && command -v ip >/dev/null 2>&1 &&
  ! ip -o addr show | grep -qF " ${bind_bare}/"; then
  echo "BIND_ADDR ${bind_bare} is not assigned to any local interface." >&2
  echo "If it is a tailnet address, tailscaled may not be up yet." >&2
  exit 1
fi

podman_args=(
  --rm
  --name "$NAME"
  --device nvidia.com/gpu=all
  --ipc=host
  -p "$publish"
  -v "${HF_CACHE}:/hf-cache:rw"
  -e HF_HOME=/hf-cache
  -e VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
)

# Name the backend only when one was chosen; "let vLLM decide" is the Ampere+
# path, and vLLM rejects --attention-backend alongside --attention-config.
backend_args=()
if [ -n "${ATTENTION_BACKEND:-}" ]; then
  backend_args=(--attention-backend "$ATTENTION_BACKEND")
fi

# The official vLLM image's entrypoint already runs `vllm serve`, while NVIDIA's
# NGC entrypoint executes the supplied arguments as a complete command.
server_command=()
case "$IMAGE" in
  nvcr.io/nvidia/vllm:* | nvcr.io/nvidia/vllm@*) server_command=(vllm serve) ;;
esac

# ENFORCE_EAGER is deliberately unquoted below: empty must expand to *no*
# argument, which a quoted "" would not do. "$@" passes through any of this
# script's own positional args as extra vLLM flags (e.g. --trust-remote-code
# for a model that ships custom modeling code, like dots.ocr) — appended last
# so they can override anything above if vLLM's own last-flag-wins applies.
# shellcheck disable=SC2086
exec podman run "${podman_args[@]}" \
  "$IMAGE" \
    "${server_command[@]}" \
    "$MODEL" \
    --served-model-name lightonocr \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --limit-mm-per-prompt '{"image": 1}' \
    "${backend_args[@]}" \
    $ENFORCE_EAGER \
    "$@"
