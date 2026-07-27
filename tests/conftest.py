from __future__ import annotations

from pathlib import Path

import pytest

from loopguard.config_io import load_config
from loopguard.schemas.config import LoopGuardConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


@pytest.fixture
def smoke_config() -> LoopGuardConfig:
    return load_config(CONFIGS / "smoke.yaml")


@pytest.fixture
def baseline_config() -> LoopGuardConfig:
    return load_config(CONFIGS / "baseline.yaml")
