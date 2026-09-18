"""Verify pdm.lock's requires_python target matches pyproject.toml.

``pdm lock --check`` only compares pyproject.toml's content hash against the
one recorded in pdm.lock; it does not re-derive the lock's
``[[metadata.targets]] requires_python`` from the current
``project.requires-python``. An incremental ``pdm lock`` run from an
interpreter inside both the old and new range can leave that target stale,
so the hash check passes while ``pdm install``/``pdm sync`` later fails on
an interpreter outside the stale range.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
LOCKFILE_PATH = REPO_ROOT / "pdm.lock"


def find_requires_python_mismatches(pyproject_text: str, lock_text: str) -> list[str]:
    """Return one message per pdm.lock target whose requires_python disagrees
    with pyproject.toml's requires-python; empty if all targets agree."""
    pyproject = tomllib.loads(pyproject_text)
    lock = tomllib.loads(lock_text)

    expected = SpecifierSet(pyproject["project"]["requires-python"])
    targets = lock.get("metadata", {}).get("targets", [])
    if not targets:
        return ["pdm.lock has no [[metadata.targets]] entries"]

    mismatches = []
    for target in targets:
        actual = SpecifierSet(target["requires_python"])
        if actual != expected:
            mismatches.append(
                f"pdm.lock target requires_python {actual!s} does not match "
                f"pyproject.toml requires-python {expected!s}"
            )
    return mismatches


def main() -> int:
    mismatches = find_requires_python_mismatches(
        PYPROJECT_PATH.read_text(), LOCKFILE_PATH.read_text()
    )
    if mismatches:
        print(
            "pdm.lock's requires_python target is out of sync with "
            "pyproject.toml (run `pdm lock` from an interpreter matching the "
            "new requires-python floor and re-check):",
            file=sys.stderr,
        )
        for mismatch in mismatches:
            print(f"  - {mismatch}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
