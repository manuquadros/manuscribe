#!/usr/bin/env bash
# Render page 1 of a fixture PDF to a base64 PNG and OCR it through the running
# vLLM server's OpenAI-compatible chat endpoint. Confirms GPU passthrough, model
# load, and the multimodal path end-to-end. Prints the first lines of markdown.
#
#   ./smoke-test.sh                       # uses tests/fixtures/31051047.pdf p1
#   PDF=tests/fixtures/30592559.pdf ./smoke-test.sh
#   BASE_URL=http://10.0.0.5:8000/v1 ./smoke-test.sh   # server on another host
set -euo pipefail

PORT="${PORT:-8000}"
PDF="${PDF:-tests/fixtures/31051047.pdf}"
# Defaults to the same env var the pipeline's model seam reads, so one export
# points both the smoke test and a conversion at the same server. 127.0.0.1
# rather than localhost in the fallback: rootless podman's port-forwarder
# answers on IPv4 only, so a localhost resolving to ::1 gets "connection reset".
BASE_URL="${BASE_URL:-${MANUSCRIBE_VLLM_URL:-http://127.0.0.1:${PORT}/v1}}"
BASE_URL="${BASE_URL%/}"
MODEL="${MODEL:-${MANUSCRIBE_VLLM_MODEL:-lightonocr}}"
ENDPOINT="${BASE_URL}/chat/completions"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "server: ${BASE_URL}   model: ${MODEL}" >&2
# Probe /models before rendering: against a remote server the likeliest failure
# is an unreachable host, a closed port or a dead tunnel, and it reads far more
# clearly here than as a curl error after the multi-second render below. The
# body is also read for data[0].root: run-server.sh pins the served id to
# "lightonocr" whatever weights actually load (see CLAUDE.md's model.py notes
# on _parse_server_engine), so the "model:" line above can name the wrong
# engine on a server restarted onto different weights. root is the field
# model.py itself trusts, so echoing it here is what keeps this script's own
# diagnostic honest instead of repeating the same trap.
MODELS_BODY="$(curl -sS -m 20 "${BASE_URL}/models")" || {
  echo "cannot reach ${BASE_URL}/models - is the server up and the port reachable?" >&2
  exit 1
}
echo "  served id: ${MODEL}   weights root (from /models): $(
  printf '%s' "$MODELS_BODY" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)["data"]
    print(data[0].get("root", "<not reported>"))
except (json.JSONDecodeError, KeyError, IndexError, TypeError):
    print("<unparseable /models response>")
'
)" >&2

# Render with the project's own renderer so the image matches what the pipeline
# would send (200 DPI, long side <= 1540).
IMG_B64="$(pdm run python - "$PDF" <<'PY'
import base64, io, sys
from pathlib import Path
from manuscribe.pipeline.render import _render_page_images
img = _render_page_images(Path(sys.argv[1]))[0]
buf = io.BytesIO()
img.save(buf, format="PNG")
print(base64.b64encode(buf.getvalue()).decode())
PY
)"

# -sS keeps curl quiet but still prints transport errors; HTTP error *bodies*
# (e.g. a 503 while the model loads, or an {"error": ...} JSON) come back with
# exit 0, so the parser below must surface them rather than KeyError on them.
curl -sS "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d @- <<JSON | python3 -c '
import sys, json
raw = sys.stdin.read()
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    sys.exit(f"server returned non-JSON (still loading?):\n{raw[:500]}")
if "choices" not in data:
    sys.exit(f"server error:\n{json.dumps(data, indent=2)[:500]}")
print(data["choices"][0]["message"]["content"][:1200])
'
{
  "model": "${MODEL}",
  "temperature": 0.0,
  "max_tokens": 2048,
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,${IMG_B64}"}}
    ]}
  ]
}
JSON
