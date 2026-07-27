"""Git provenance for the run manifest.

A dirty tree on a headline run is a reproducibility hole. It is recorded rather
than blocked -- pilot and smoke runs legitimately happen on dirty trees, and a
hard block would just get bypassed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return out.stdout.strip()


def git_sha(cwd: Path | None = None) -> str:
    return _git("rev-parse", "HEAD", cwd=cwd) or "unknown"


def git_dirty(cwd: Path | None = None) -> bool:
    status = _git("status", "--porcelain", cwd=cwd)
    if status is None:
        return True  # unknown state is not a clean state
    return bool(status)
