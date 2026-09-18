import importlib.util
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "check_lock_python_target.py"
)
_spec = importlib.util.spec_from_file_location("check_lock_python_target", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
check_lock_python_target = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_lock_python_target)

find_requires_python_mismatches = (
    check_lock_python_target.find_requires_python_mismatches
)

_PYPROJECT = """
[project]
requires-python = "<3.14,>=3.12"
"""

_LOCK_CONSISTENT = """
[metadata]
content_hash = "sha256:whatever"

[[metadata.targets]]
requires_python = ">=3.12,<3.14"
"""

_LOCK_STALE = """
[metadata]
content_hash = "sha256:whatever"

[[metadata.targets]]
requires_python = ">=3.13,<3.14"
"""


def test_consistent_targets_report_no_mismatch() -> None:
    assert find_requires_python_mismatches(_PYPROJECT, _LOCK_CONSISTENT) == []


def test_stale_target_is_reported() -> None:
    mismatches = find_requires_python_mismatches(_PYPROJECT, _LOCK_STALE)
    assert len(mismatches) == 1
    assert ">=3.13" in mismatches[0]
    assert ">=3.12" in mismatches[0]
