"""Loading configs from YAML, and the calibration-lock guard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from loopguard.hashing import hash_json_file_content
from loopguard.schemas.config import LoopGuardConfig


class CalibrationLockError(RuntimeError):
    """The frozen calibration does not match what the config expects."""


def load_config(path: str | Path) -> LoopGuardConfig:
    """Load, validate, and resolve a config.

    Resolution steps that happen here (and nowhere else, so the hash is stable):

    * the system prompt file is read and inlined into ``semantic.prompt.text``
    * relative paths resolve against the config file's directory
    * ``loop.max_steps`` defaults to ``2*d_max+4`` (in the model validator)
    """
    config_path = Path(path).resolve()
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} did not parse as a mapping")

    base_dir = config_path.parent
    prompt = raw.get("semantic", {}).get("prompt", {})
    if "text" in prompt:
        raise ValueError(
            "semantic.prompt.text is populated from semantic.prompt.path at load time; "
            "setting it in YAML would let the hashed prompt and the file diverge."
        )
    prompt_path = Path(prompt.get("path", ""))
    if not prompt_path.is_absolute():
        prompt_path = (base_dir / prompt_path).resolve()
    prompt["text"] = prompt_path.read_text(encoding="utf-8")
    prompt["path"] = str(prompt_path)

    config = LoopGuardConfig.model_validate(raw)

    out_dir = config.runtime.out_dir
    if not out_dir.is_absolute():
        config.runtime.out_dir = (base_dir / out_dir).resolve()
    return config


def verify_calibration_lock(config: LoopGuardConfig, lock_path: str | Path) -> str | None:
    """Refuse to run when the frozen calibration has drifted (TRD 5.2).

    Re-tuning difficulty after seeing headline results is PRD 3.4's explicit
    hazard. Making it require a deliberate edit to *both* a lock file and a
    config is the guard: it cannot happen by accident.

    Returns the lock's content hash, or ``None`` when the config declares no lock
    (only legitimate before Phase 1 exit).
    """
    expected = config.semantic.task.calibration_lock_hash
    lock = Path(lock_path)

    if expected is None:
        if lock.exists():
            raise CalibrationLockError(
                f"{lock} exists but task.calibration_lock_hash is null. "
                "Pin the hash in the config, or delete the lock."
            )
        return None

    if not lock.exists():
        raise CalibrationLockError(
            f"config pins calibration_lock_hash={expected} but {lock} does not exist"
        )
    actual = hash_json_file_content(lock.read_text(encoding="utf-8"))
    if actual != expected:
        raise CalibrationLockError(
            f"calibration lock mismatch: {lock} hashes to {actual}, "
            f"config expects {expected}. Refusing to start."
        )
    return actual


def write_calibration_lock(path: str | Path, payload: dict[str, Any]) -> str:
    """Freeze the distractor knobs at Phase 1 exit. Returns the content hash."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    return hash_json_file_content(text)
