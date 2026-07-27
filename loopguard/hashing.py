"""Config hashing and seed derivation (TRD 5.1).

All seeds derive from a single root, so a hash fully determines the task set::

    task_hash     = blake2b(canonical_json(semantic.task), 16)
    config_hash   = blake2b(canonical_json(semantic),      16)
    task_seed     = blake2b(f"{task_hash}|{depth}|{task_index}")[:8]
    decoding_seed = blake2b(f"{task_seed}|{repeat_index}")[:8]

Note the root of ``task_seed``: ``task_hash``, not ``config_hash``. See the
module docstring of ``loopguard.schemas.config`` for why -- deriving task seeds
from the full semantic hash would move them when the mitigation arm changes the
prompt, breaking the paired comparison the PRD requires.
"""

from __future__ import annotations

import json
from hashlib import blake2b
from typing import Any

from loopguard.schemas.config import SemanticConfig, TaskConfig

#: Providers reject seeds outside signed-32-bit range; the internal seed keeps
#: full width so it stays a stable identifier even if this clamp changes.
_PROVIDER_SEED_MODULUS = 2**31 - 1


def canonical_json(obj: Any) -> str:
    """Stable serialization for hashing. Key order and separators are fixed."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hex16(payload: str) -> str:
    return blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def _seed64(payload: str) -> int:
    return int.from_bytes(blake2b(payload.encode("utf-8")).digest()[:8], "big")


def task_hash(task: TaskConfig) -> str:
    """Hash of the task-determining config only. Equal across arms by design."""
    return _hex16(canonical_json(task.model_dump(mode="json")))


def config_hash(semantic: SemanticConfig) -> str:
    """Hash of every field that can move a result. Runtime config is excluded."""
    return _hex16(canonical_json(semantic.model_dump(mode="json")))


def derive_task_seed(task_hash_hex: str, depth: int, task_index: int) -> int:
    return _seed64(f"{task_hash_hex}|{depth}|{task_index}")


def derive_decoding_seed(task_seed: int, repeat_index: int) -> int:
    """A retry reuses this value. A reseeded retry is a different sample, not a retry."""
    return _seed64(f"{task_seed}|{repeat_index}")


def provider_seed(decoding_seed: int) -> int:
    """Narrow a 64-bit derived seed to what OpenAI-compatible endpoints accept."""
    return decoding_seed % _PROVIDER_SEED_MODULUS


def hash_text(text: str) -> str:
    """Used for prompt hashes in :class:`~loopguard.schemas.manifest.JudgePin`."""
    return _hex16(text)


def hash_json_file_content(raw: str) -> str:
    """Hash of a lock file's *parsed* content, so formatting churn is not a change."""
    return _hex16(canonical_json(json.loads(raw)))
