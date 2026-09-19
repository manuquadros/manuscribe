"""The parts of ``deploy/vllm/`` that decide something on their own.

All of them fail silently rather than loudly, which is why they are pinned here:
too small a window drops a truncated page's tail, a 6 GiB card's memory fraction
reserves tens of GiB of unusable KV cache on a large one, and a garbage
transcription still arrives as HTTP 200.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from manuscribe.pipeline.model import _ENGINE_BY_WEIGHTS

_DEPLOY = Path(__file__).resolve().parents[1] / "deploy" / "vllm"
_SCRIPT = _DEPLOY / "model-defaults.sh"
_GPU_SCRIPT = _DEPLOY / "gpu-defaults.sh"


def _load_bench() -> object:
    """``bench_attention.py`` -- a script, not a module of the package."""
    spec = importlib.util.spec_from_file_location(
        "bench_attention", _DEPLOY / "bench_attention.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: dataclasses resolves the module's string annotations
    # through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BENCH = _load_bench()


def _run(model: str, **env: str) -> str:
    """``MAX_MODEL_LEN`` after sourcing the script and resolving ``model``."""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; apply_model_defaults "$2"; printf %s "$MAX_MODEL_LEN"',
            "_",
            str(_SCRIPT),
            model,
        ],
        capture_output=True,
        check=True,
        text=True,
        env={**os.environ, **env},
    )
    return completed.stdout


def test_chandra_gets_a_window_past_the_lighton_default() -> None:
    assert int(_run("datalab-to/chandra-ocr-2")) > 8192


def test_lightonocr_keeps_the_spike_validated_window() -> None:
    assert _run("lightonai/LightOnOCR-2-1B-bbox") == "8192"


def test_a_mirror_of_the_same_weights_resolves_like_the_canonical_id() -> None:
    """Keyed on the basename, case-folded, so a local snapshot is not a new model."""
    canonical = _run("datalab-to/chandra-ocr-2")
    assert _run("/srv/models/Chandra-OCR-2/") == canonical


def test_an_unknown_model_falls_back_rather_than_inheriting_another_window() -> None:
    assert _run("some-org/an-unreleased-finetune") == "8192"
    assert _run("") == "8192"


def test_an_exported_window_wins_over_the_models_own() -> None:
    assert _run("datalab-to/chandra-ocr-2", MAX_MODEL_LEN="4096") == "4096"


def test_every_engine_the_client_knows_resolves_to_a_usable_window() -> None:
    """The script's table and ``_ENGINE_BY_WEIGHTS`` are keyed the same way and are
    meant to move together; this fails when one grows a model the other lacks."""
    for weights in _ENGINE_BY_WEIGHTS:
        assert int(_run(f"an-org/{weights}")) >= 8192


def _memory_defaults(total_mib: str, **env: str) -> tuple[str, str]:
    """``(GPU_MEM_UTIL, ENFORCE_EAGER)`` for a card with ``total_mib`` of memory."""
    completed = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; _apply_memory_defaults "$2"; '
            'printf "%s\\n%s" "$GPU_MEM_UTIL" "$ENFORCE_EAGER"',
            "_",
            str(_GPU_SCRIPT),
            total_mib,
        ],
        capture_output=True,
        check=True,
        text=True,
        env={**os.environ, **env},
    )
    util, _, eager = completed.stdout.partition("\n")
    return util, eager


def test_a_small_card_keeps_the_spike_validated_pair() -> None:
    util, eager = _memory_defaults("6144")
    assert (util, eager) == ("0.85", "--enforce-eager")


def test_a_large_card_turns_cuda_graphs_on_and_stops_hoarding_kv_cache() -> None:
    util, eager = _memory_defaults("124416")
    assert eager == ""
    assert float(util) < 0.85


def test_an_unreadable_card_size_keeps_the_settings_that_fit_anywhere() -> None:
    """No nvidia-smi is not a licence to assume headroom: the conservative pair
    runs on a small card, and the permissive one does not."""
    util, eager = _memory_defaults("")
    assert (util, eager) == ("0.85", "--enforce-eager")


def test_an_explicitly_empty_enforce_eager_survives_on_a_small_card() -> None:
    """``ENFORCE_EAGER=`` means "CUDA graphs on" and must not be refilled — the
    reason the script assigns it with ``${VAR=}`` rather than ``${VAR:=}``."""
    _, eager = _memory_defaults("6144", ENFORCE_EAGER="")
    assert eager == ""


def test_exported_memory_settings_win_on_a_large_card() -> None:
    util, eager = _memory_defaults(
        "124416", GPU_MEM_UTIL="0.9", ENFORCE_EAGER="--enforce-eager"
    )
    assert (util, eager) == ("0.9", "--enforce-eager")


def test_a_real_transcription_passes_the_bench_gate() -> None:
    page = (
        "## Results\n\nThe reductase activity of PtTRI was assayed against "
        "NAD⁺ at 25 °C in 50 mM phosphate buffer, and the specific "
        "activity of the purified enzyme reached 4.2 U/mg after the final "
        "chromatography step. Table 2 lists the kinetic constants for each "
        "substrate tested in this work.\n"
    )
    assert _BENCH._degenerate_reason(page) is None


def test_the_bench_gate_catches_the_repeated_glyph_failure() -> None:
    """The fp16/wrong-kernel shape: HTTP 200, finish_reason stop, and a page of
    one character.  It has to fail here or a broken backend gets a timing."""
    reason = _BENCH._degenerate_reason("!" * 5000)
    assert reason is not None
    assert "!" in reason


def test_the_bench_gate_catches_a_response_too_short_to_be_a_page() -> None:
    assert _BENCH._degenerate_reason("## Results\n") is not None
    assert _BENCH._degenerate_reason("") is not None
