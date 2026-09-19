# vLLM server for LightOnOCR (podman)

Runs `lightonai/LightOnOCR-2-1B-bbox` under vLLM as an OpenAI-compatible
server, in a rootless podman container. All GPU/torch work lives in vLLM;
`manuscribe` itself carries no torch or transformers — its model seam
(`pipeline/model.py`) is a thin HTTP client that talks to this server.

Unless noted otherwise, shell commands assume the repository root as the
working directory.

The server stays up and keeps the model resident, so each document pays the
HTTP round-trip, not a ~35 s model reload.

Two deployment shapes, both covered below:

- **Single container** (`Containerfile`) — vLLM **and** manuscribe in one image;
  convert PDFs with `podman exec`. Simplest when both run on the same box.
- **Server only** (`run-server.sh`) — just the vLLM server; run manuscribe from
  any environment (it only needs `pip install manuscribe`, no GPU) pointed at the
  server via `MANUSCRIBE_VLLM_URL`.

## One-time host setup

1. **Install podman** (needs sudo; the only step that does):

   ```
   sudo apt-get install -y podman
   ```

2. **GPU passthrough is already configured.** NVIDIA CDI is generated at
   `/var/run/cdi/nvidia.yaml` (`kind: nvidia.com/gpu`). Verify podman sees it:

   ```
   podman run --rm --device nvidia.com/gpu=all \
     docker.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

   If the device is ever missing (the `/var/run/cdi` copy is on tmpfs and is
   lost on reboot), regenerate it to the **same** path:
   `sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml`. For a
   reboot-persistent spec, write to `/etc/cdi/nvidia.yaml` instead **and delete
   the `/var/run/cdi` copy** — the same qualified device name in two spec dirs
   makes podman reject the CDI registry as a duplicate.

## GPU compute capability (dtype and attention backend)

Four of the server's settings depend on the card, and `gpu-defaults.sh` derives
all of them at start time for both `run-server.sh` and the unified image, so
neither has to be edited per machine. Two follow the GPU's compute capability:

| Card | Example | `--dtype` | `--attention-backend` |
|---|---|---|---|
| Ampere or newer (sm_80+) | RTX Ada, A10, A100 | `bfloat16` | omitted — vLLM chooses (FlashInfer) |
| Blackwell (sm_100+) | GB10 (Dell Pro Max with GB10, DGX Spark) | `bfloat16` | omitted — vLLM chooses, **unverified** — see [GB10 section](#gb10--dgx-spark-class-unified-memory-arm64-workstations) below |
| Turing (sm_75) | Tesla T4 | `float32` | `TRITON_ATTN` |
| Undetectable (no `nvidia-smi`) | — | `float32` | `TRITON_ATTN`, with a warning |

The other two follow how much memory the card has — `--enforce-eager` and
`--gpu-memory-utilization`, covered under
[Notes / caveats](#notes--caveats).

An explicit value always wins, so any of them can be forced:

```
DTYPE=float16 ATTENTION_BACKEND=FLEX_ATTENTION ./deploy/vllm/run-server.sh
```

**The backend is a command-line flag, not an environment variable.**
`VLLM_ATTENTION_BACKEND` — which most advice online, and this repo's own earlier
comments, still name — **did not exist as of vLLM 0.22.1**: the string appears
nowhere in the package, so exporting it is silently inert and the server goes on
selecting FlashInfer. The live knob is `--attention-backend <NAME>` (or
`--attention-config '{"backend": "<NAME>"}'`; the two are mutually exclusive and
vLLM raises if both are given). `ATTENTION_BACKEND` above is *this script's* env
knob, which becomes that flag.

**Why both are needed on a T4, and why only one of them is obvious.** The dtype
half fails loudly at startup — vLLM refuses `bfloat16` below sm_80 and says so
(*"Bfloat16 is only supported on GPUs with compute capability of at least
8.0"*). The backend half fails on the **first request**, killing the engine and
taking the whole server down with it:

```
RuntimeError: BatchPrefillWithPagedKVCache failed with error invalid argument
EngineCore encountered a fatal error … EngineDeadError
```

That one will not resolve itself by leaving the choice to vLLM.
`FlashInferBackend`'s own gate answers `supports_compute_capability(7.5) → True`,
so auto-selection picks FlashInfer on a T4 and the image's prebuilt kernels then
fail at launch — an optimistic gate, not a missing one. The backend has to be
named. `TRITON_ATTN` is the one that reports support down to sm_60 and takes
fp16; **`FLASH_ATTN` does not support sm_75** and is not an alternative there.

### float16 is not an option for this model

The obvious dtype below sm_80 is fp16, and it is **silently wrong here**. The
weights are bfloat16; cast to float16 the server starts, reports healthy, serves
200s at full throughput — and every page comes back as `!!!!!!!!` (token 0, the
signature of non-finite logits). Measured on one page of
`tests/fixtures/31051047.pdf`, same image and request for all three:

| `--dtype` | backend | output |
|---|---|---|
| `bfloat16` | FlashInfer | correct markdown |
| `float16` | FlashInfer | `!!!!` |
| `float16` | `TRITON_ATTN` | `!!!!` |

So it is the **dtype**, not the backend — and a pre-Ampere card has no bfloat16
to fall back on. That leaves `float32`, the only remaining width that cannot
overflow. It costs ~4.8 GiB of weights (fine on a T4's 16 GiB, an OOM on a 6 GiB
card) and it is slow. That is a bad trade taken deliberately: a server that OOMs
at startup fails immediately and legibly, whereas one that OCRs every page to
`!` fails at the far end of the pipeline, as empty documents.

**The pre-Ampere path is not covered by the determinism spike, and fp32 output
quality has not been verified on a T4** — it could not be measured on the 6 GiB
development card, which OOMs before serving. Run
`./deploy/vllm/smoke-test.sh` and confirm the output is markdown before trusting
a document to it. If fp32 proves too slow,
`deploy/p100/`'s transformers shim is the fallback — but note it runs fp16 too,
so smoke-test its output as well rather than assuming it escapes this.

Expect Turing to be materially slower than an Ampere+ card: Triton attention
replaces FlashInfer, fp32 replaces bf16, and the run sits outside the pinned-vLLM
determinism gate the spike validated. A **P100 (sm_60) is out of scope for this
image entirely** — it has no Pascal kernels at all; see `deploy/p100/`.

If a vLLM upgrade ever renames a backend, the valid names are its own enum:

```
podman run --rm --device nvidia.com/gpu=all --entrypoint python3 \
  docker.io/vllm/vllm-openai:v0.29.0 -c \
  'from vllm.v1.attention.backends.registry import AttentionBackendEnum as E; print(sorted(m.name for m in E))'
```

Check the flag itself the same way — `vllm serve --help | grep attention` — since
it is the half of this that a version bump is most likely to move.

## GB10 / DGX Spark-class unified-memory ARM64 workstations

A Grace-Blackwell unified-memory workstation — the Dell Pro Max with GB10,
NVIDIA DGX Spark, or anything else built on the GB10 Superchip — differs on
every axis this deployment cares about from the x86_64 dedicated-VRAM hosts
(a 6 GiB dev card, a T4) the scripts were first built against. The defaults now
cover it, each for a reason worth knowing:

- **ARM64 host, not x86_64.** The Grace CPU is `aarch64`, so an image has to
  ship an arm64 build, not merely a matching version number. The default tag
  does, and is what this box runs; a different one may not, and podman then
  fails the pull with no matching manifest.
- **Blackwell GPU, compute capability sm_121** (reported by `nvidia-smi` as
  `12.1`). `gpu-defaults.sh` already routes this into the Ampere-or-newer
  branch (`bfloat16`, backend left to vLLM) — bf16 is genuinely correct for
  Blackwell — but it also prints a Blackwell-specific caveat, because "vLLM's
  own backend choice" only means whatever attention kernels the *image*
  actually built for this compute capability, and image support for sm_121
  arrived well after the x86_64/Ampere images this repo was validated against.
- **128 GB of memory coherently shared between the CPU and the GPU** (not a
  dedicated VRAM pool). `--gpu-memory-utilization` here reserves a fraction of
  memory the host OS also needs to run in, unlike on a discrete card, so the
  naive "bigger card → raise it" reflex is backwards: a card this size is
  exactly where the fraction over-reserves. `gpu-defaults.sh` already drops to
  `0.35` (and turns CUDA graphs back on) at 64 GiB and above, so the defaults
  need no adjustment here — but watch `free -h` / `nvidia-smi` before raising
  `GPU_MEM_UTIL` past it. It reaches that figure via `/proc/meminfo`, because
  `nvidia-smi --query-gpu=memory.total` answers `[N/A]` on this part — there is
  no dedicated pool to report. The startup line names the source it used
  (`… MiB (host)` here, `(device)` on a discrete card, `(unknown)` when neither
  answered and the conservative pair is in force). The startup log states what it reserved and what
  reached the KV cache: `GPU KV cache size: N tokens` divided by
  `MANUSCRIBE_OCR_CONCURRENCY` × `--max-model-len` is how many times more
  cache was reserved than this pipeline can put to work.

### Picking an image

The `IMAGE` env var (`run-server.sh`) and Podman's `--from` option override the
base image for the server-only and unified-image paths, respectively. Override
the repository, not just the tag, since a sm_121-capable build may not live under
`docker.io/vllm/vllm-openai` at all:

```
GPU_MEM_UTIL=0.4 IMAGE=nvcr.io/nvidia/vllm:26.06-py3 ./deploy/vllm/run-server.sh
# or, building the unified image:
podman build --from nvcr.io/nvidia/vllm:26.06-py3 \
  -f deploy/vllm/Containerfile -t lighton-manuscribe .
```

Use NGC 26.06 or newer. NGC 25.09 contains vLLM 0.10.1, which does not support
`LightOnOCRForConditionalGeneration`; NGC 26.06 contains vLLM 0.22.1, the
version this deployment used before v0.29.0 became the default.

`docker.io/vllm/vllm-openai:v0.29.0` is the default and is preferred over the
NGC route above: measured on a DGX Spark (GB10, `GPU_MEM_UTIL=0.4`), the same
document went from 6:38 wall-clock on v0.22.1 to 3:03 on v0.29.0 — roughly 2x,
consistent across a full document not a single page. Output was byte-identical
apart from one figure crop's PNG encode. Likely a Blackwell attention-kernel
improvement between the two releases, not confirmed further than that. It
publishes an arm64 manifest and is what the GB10 box runs. None of this
displaces the "smoke-test before trusting a document" rule below.

`run-server.sh` accounts for the image entrypoint difference: the official
vLLM image already runs `vllm serve`, while the NGC image requires that full
command after its generic NVIDIA entrypoint.

Before trusting whatever tag you land on:

1. Confirm the manifest actually carries arm64: `podman manifest inspect
   <image>` or `podman inspect --format '{{.Os}}/{{.Architecture}}' <image>`.
2. Start the server and watch for a death on the **first request** rather than
   at startup — that shape means the image's attention kernels don't cover
   sm_121, the same failure mode `gpu-defaults.sh` works around for the T4 by
   naming `TRITON_ATTN` explicitly. If it happens here, try
   `ATTENTION_BACKEND=TRITON_ATTN` (or whatever the image's own
   `AttentionBackendEnum` reports as supporting this capability — see the
   enum-dump command above) before assuming the hardware itself doesn't work.
3. Run `./deploy/vllm/smoke-test.sh` and confirm markdown, not `!!!!` or
   garbage — the fp16-looks-healthy-but-emits-token-0 failure documented above
   for pre-Ampere cards is a dtype bug, not an architecture-specific one, and
   nothing rules it out on a new image/kernel combination you haven't run
   before.

**This path has not been through the determinism spike** (`spike_results/
vllm_determinism.md`, Ampere-only) or the integration test suite. Treat its
output as unverified — smoke-test before trusting a document, the same rule
this doc already applies to the untested pre-Ampere/fp32 path — until someone
runs that verification on GB10 specifically and this section is updated to
say so.

### GPU passthrough

The CDI setup in "One-time host setup" above is architecture-agnostic (`nvidia-ctk`
generates the same CDI spec shape on ARM64), but the verification image named
there (`docker.io/nvidia/cuda:12.4.1-base-ubuntu22.04`) needs its own arm64
manifest check before you rely on it as a smoke test — don't assume a tag that
works on x86_64 publishes arm64 too.

## Single container (manuscribe + vLLM)

Build the unified image from the repo root (the build context needs `src/` and
`pyproject.toml`):

```
podman build -f deploy/vllm/Containerfile -t lighton-manuscribe .
```

Start it as a warm server (mount the HF cache to skip the model download; the
`HF_HOME` env points vLLM at the mount regardless of the image's home dir):

```
podman run -d --name lighton \
  --device nvidia.com/gpu=all --ipc=host \
  -v "$HOME/.cache/huggingface:/hf-cache:rw" -e HF_HOME=/hf-cache \
  lighton-manuscribe
```

Then convert PDFs against the in-container server with `podman exec` (the image
sets `MANUSCRIBE_VLLM_URL=http://127.0.0.1:8000/v1`, so no flags needed):

```
podman exec lighton python3 -m manuscribe /data/in.pdf /data/out.html
```

Mount your input/output dir with `-v` on `podman run` to make `/data` visible.
The server stays resident between `exec`s — that's the whole point of one
long-lived container over a one-shot-per-PDF run.

## Server only

```
./deploy/vllm/run-server.sh
```

First run pulls the ~16 G `vllm/vllm-openai:v0.29.0` image (disk is tight — see
below). Confirm that exact tag is published first — the image tag set can lag
the pip release the spike used; check with
`podman search --list-tags docker.io/vllm/vllm-openai` (or the Docker Hub tags
page) and override via
`IMAGE=…/vllm-openai:<tag> ./deploy/vllm/run-server.sh` if needed.
The model weights are **not** re-downloaded: `run-server.sh` mounts your
existing `~/.cache/huggingface` (already holds the 2.7 G bbox model). The server
is ready when the log prints `Application startup complete` on port 8000.

Tunables are env overrides, e.g.:

```
PORT=8001 GPU_MEM_UTIL=0.80 ./deploy/vllm/run-server.sh
```

### Serving a different OCR model (chandra-ocr-2)

`MODEL=` swaps the weights, and the client picks the matching response parser on
its own: the `/models` probe reports the model path the server was launched with
as `root`, which `load_ocr_model()` matches against the weights it knows. The
served *name* is no signal — it stays `lightonocr` whatever is loaded — so it is
never consulted.

```
MODEL=datalab-to/chandra-ocr-2 ./deploy/vllm/run-server.sh
pdm run python -m manuscribe in.pdf out.html
```

`--max-model-len` follows the weights too, from `model-defaults.sh`: chandra gets
32 k (its native window is 262 144), LightOnOCR keeps the 8 k the spike
validated. That is the one server flag worth getting right per model — the
client derives its truncation-retry budget from whatever the server reports, and
gives up on the retry entirely once the remaining budget no longer covers a
generation, so a window too small for the model does not slow a dense page down,
it silently drops its tail. `MAX_MODEL_LEN=` overrides. The table is keyed on the
model basename, the same key `model.py` uses to pick the parser, and
`tests/test_deploy_defaults.py` fails if the two drift apart.

`MANUSCRIBE_OCR_ENGINE` (`lightonocr` or `chandra`) overrides that detection, for
a deployment whose `root` names no known repository — a fine-tune, a mirror, or a
locally staged checkpoint under an unrelated name. An unrecognised `root` falls
back to `lightonocr`; an unrecognised env var is rejected at `load_ocr_model()`
rather than silently defaulted.

The request is identical for both models; only the ingestion differs.

### Which interface the port lands on

**The default is `127.0.0.1`** — the server is unreachable from other machines
until you say otherwise. vLLM has no authentication of its own, so the port is
the whole access control; publishing on every interface (podman's behaviour when
`-p` names no address) would serve the model to anyone who reaches the host.
`BIND_ADDR` names the one address to publish on:

```
BIND_ADDR="$(tailscale ip -4)" ./deploy/vllm/run-server.sh   # tailnet only
./deploy/vllm/run-server.sh                                  # loopback: podman exec, SSH tunnel
BIND_ADDR=0.0.0.0 ./deploy/vllm/run-server.sh                # every interface, deliberately
```

An IPv6 literal is bracketed automatically, and a value that is already
bracketed is taken as-is. `0.0.0.0` and `::` are accepted as the explicit way to
ask for every interface — that is a decision you now have to make out loud
rather than get by omission.

The address must be up before the server starts. A boot-time unit can easily
beat `tailscaled` to it, and podman's own bind failure names neither the address
nor the cause, so the script checks first and says which address is missing (the
two wildcards are exempt, belonging to no interface).

Note this is the *only* control over the public interface. A Tailscale ACL
governs what may cross the tailnet; it has nothing to say about a port that is
also listening on a public IP. The two are not substitutes — you want both, and
`BIND_ADDR` is the half that lives here.

The other half, and the network design around it — tailnet ACL and tags, host
firewall, optional app-layer token, and how to verify the result actually
holds — is in [`../SECURITY.md`](../SECURITY.md). It covers this server and the
P100 shim alike; read it before exposing either on a VM with a public IP.

## Smoke test

With the server up, in another shell:

```
./deploy/vllm/smoke-test.sh
```

Renders fixture page 1 with the project's own renderer and OCRs it through the
chat endpoint; prints the first ~1200 chars of markdown.
`PDF=… ./deploy/vllm/smoke-test.sh` selects another file.

### Comparing attention backends

On a card where more than one backend is viable, which one vLLM auto-selects is
not necessarily the fastest — on Blackwell it picks `FLASH_ATTN` and reports
`Using FlashAttention version 2`, out of a candidate list that also holds
`FLASHINFER`. The backend is a launch flag, so comparing them means restarting
the server, not switching at runtime:

```
# terminal 1, once per backend
ATTENTION_BACKEND=FLASHINFER ./deploy/vllm/run-server.sh

# terminal 2, after each restart — the label is yours to supply, since
# /v1/models does not report the backend; read it off the server's startup log
pdm run python deploy/vllm/bench_attention.py FLASHINFER
```

Each run appends a row to `spike_results/attention-backend-bench.tsv`: pages/s
plus the model, window, concurrency and a digest of the transcription. It
renders before starting the clock, and discards one warm-up pass — FlashInfer
JIT-compiles its kernels on first use, so a cold pass times the compiler.

**Read the digest column, not just the times.** A backend whose kernels do not
suit the card can return HTTP 200 and a page of `!` (see
[float16](#float16-is-not-an-option-for-this-model) for the same shape from a
different cause); the script refuses to report a timing for output like that,
but a subtler disagreement between two backends shows up only as differing
digests. Those are comparable at `--concurrency 1` — above it vLLM's batch
composition varies with request arrival and the digest changes run to run on one
backend.

### Against a server on another host

The GPU box and the machine holding the PDFs need not be the same: the script
renders locally and sends only the page image, so point it at the remote server.
`BASE_URL` (or `MANUSCRIBE_VLLM_URL`, the same variable the pipeline reads)
overrides the endpoint, and `MODEL`/`MANUSCRIBE_VLLM_MODEL` the served name:

```
BASE_URL=http://<vm-host>:8000/v1 ./deploy/vllm/smoke-test.sh
```

For this to reach anything, the server must have been started with a
`BIND_ADDR` that is reachable from here — it binds loopback otherwise, so a
remote client gets a refused connection no matter what the firewall allows. On
a tailnet that is `BIND_ADDR="$(tailscale ip -4)" ./deploy/vllm/run-server.sh`.

Alternatively leave the server on loopback and forward the port over SSH, which
keeps the default endpoint working unchanged:

```
ssh -N -L 8000:127.0.0.1:8000 <vm-host> &     # in one shell
./deploy/vllm/smoke-test.sh                   # in another
```

An unreachable host, a closed port or a dead tunnel is reported by the `/models`
probe before the render, so it fails in a second rather than after rendering.

Once the smoke test passes, the same variable drives a full conversion:

```
MANUSCRIBE_VLLM_URL=http://<vm-host>:8000/v1 pdm run python -m manuscribe in.pdf out.html
```

A remote server makes the round-trip latency per page visible; raise
`MANUSCRIBE_OCR_CONCURRENCY` (default 4) if the link is slow but the GPU is idle.

## Calling it from the pipeline

OpenAI-compatible chat completions, one image per request, greedy:

```python
import base64, io
from openai import OpenAI

# 127.0.0.1, not localhost — rootless podman forwards the port on IPv4 only.
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")


def ocr_image(img) -> str:  # img: PIL.Image
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    resp = client.chat.completions.create(
        model="lightonocr",
        temperature=0.0,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    )
    return resp.choices[0].message.content
```

This is the drop-in for `_ocr_page`'s model call. Per the determinism spike,
keep **substring/structural** acceptance — greedy vLLM is byte-stable run-to-run
but jitters bbox low-order digits across batch composition.

## Stop

`run-server.sh` uses `--rm`, so:

```
podman stop lighton-vllm
```

removes the container. To keep it running across reboots, install it as a
podman **quadlet** (systemd user unit) — ask and I'll generate the
`~/.config/containers/systemd/lighton-vllm.container` file.

## Notes / caveats

- **Disk:** the host is at ~95 % (53 G free). The image is ~16 G; it fits, but
  prune old images (`podman image prune`) if a pull fails for space.
- **Card memory:** `--enforce-eager` (no CUDA-graph capture) and
  `--gpu-memory-utilization 0.85` are what fit the 6 GiB GPU in the spike, and
  `gpu-defaults.sh` still picks them below 64 GiB. At or above that it enables
  CUDA graphs and drops to `0.35` — CUDA graphs cost a couple of GiB and buy
  decode throughput, which is most of an OCR run, and the high fraction only
  ever existed to fit weights on a small card. `ENFORCE_EAGER` / `GPU_MEM_UTIL`
  override either. On a unified-memory box the lower fraction matters for a
  second reason: see the
  [GB10 section](#gb10--dgx-spark-class-unified-memory-arm64-workstations).
- **Context window** comes from `model-defaults.sh`, keyed on the model, so
  `MODEL=datalab-to/chandra-ocr-2 ./run-server.sh` serves a 32 k window while
  LightOnOCR keeps 8 k. Too small a window is not a slowdown but silent data
  loss — the client gives up on its truncation retry when the remaining budget
  no longer covers a generation, so a dense page's tail is simply gone.
  `MAX_MODEL_LEN` overrides.
- **Image tag** defaults to `v0.29.0`, which is what recent engine work — the
  chandra-ocr-2 / LightOnOCR comparison included — actually ran on. The §4
  determinism spike was run against `v0.22.1`, so the determinism claims carry
  that tag's evidence, not this one's; `IMAGE=…:v0.22.1` pins it back. Moving
  the tag again re-opens the determinism/fidelity question — re-run the spike
  against the new one before trusting it.
- **flashinfer, twice over:** two unrelated settings carry the name.
  `VLLM_USE_FLASHINFER_SAMPLER=0` (always set) avoids the startup nvcc JIT
  failure on this runtime-only host — the container has no CUDA toolkit either,
  so leave it set unless you switch to a `flashinfer-jit-cache` image. The
  flashinfer *attention backend* is a separate matter, handled by capability
  above; disabling the sampler does not disable it.
