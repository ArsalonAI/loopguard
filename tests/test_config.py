from __future__ import annotations

import json

import pytest
import yaml

from loopguard.config_io import (
    CalibrationLockError,
    load_config,
    verify_calibration_lock,
    write_calibration_lock,
)
from loopguard.tools.smoke import build_registry

from .conftest import CONFIGS


def test_baseline_config_loads_and_resolves(baseline_config):
    semantic = baseline_config.semantic
    assert semantic.task.depths == [1, 2, 3, 4, 5]
    # TRD 6.2: max_steps defaults to 2*d_max+4 and the resolved value is hashed.
    assert semantic.loop.max_steps == 14
    assert semantic.prompt.text.strip(), "prompt file was not inlined"


def test_prompt_text_cannot_be_set_in_yaml(tmp_path):
    raw = yaml.safe_load((CONFIGS / "smoke.yaml").read_text())
    raw["semantic"]["prompt"]["path"] = str(CONFIGS / "prompts" / "baseline_v1.txt")
    raw["semantic"]["prompt"]["text"] = "an inline prompt that would diverge from the file"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="populated from"):
        load_config(path)


def test_smoke_tool_tags_match_the_registry(smoke_config):
    """An untagged tool must not silently default -- it is a frozen degree of freedom."""
    build_registry().validate_against_policy(smoke_config.semantic.tools)


def test_exposed_tool_count_excludes_the_terminal_tool():
    registry = build_registry()
    assert "submit_answer" not in registry.exposed_names()
    assert len(registry.exposed_names()) == 6


def test_depths_must_be_sorted_and_unique(tmp_path):
    raw = yaml.safe_load((CONFIGS / "smoke.yaml").read_text())
    raw["semantic"]["task"]["depths"] = [3, 1, 2]
    raw["semantic"]["prompt"]["path"] = str(CONFIGS / "prompts" / "baseline_v1.txt")
    path = tmp_path / "unsorted.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(Exception, match="sorted and unique"):
        load_config(path)


def test_calibration_lock_guard(tmp_path, baseline_config):
    lock = tmp_path / "calibration.lock.json"

    # Pre-Phase-1: no lock declared, no lock file. Legitimate.
    assert verify_calibration_lock(baseline_config, lock) is None

    digest = write_calibration_lock(lock, {"distractors_per_hop": 2, "id_edit_distance": 1})

    # A lock exists but the config does not pin it -> refuse.
    with pytest.raises(CalibrationLockError, match="null"):
        verify_calibration_lock(baseline_config, lock)

    baseline_config.semantic.task.calibration_lock_hash = digest
    assert verify_calibration_lock(baseline_config, lock) == digest

    # Re-tuning difficulty after seeing results must not be possible by accident.
    lock.write_text(json.dumps({"distractors_per_hop": 4, "id_edit_distance": 2}))
    with pytest.raises(CalibrationLockError, match="mismatch"):
        verify_calibration_lock(baseline_config, lock)


def test_lock_hash_ignores_formatting_churn(tmp_path, baseline_config):
    lock = tmp_path / "calibration.lock.json"
    digest = write_calibration_lock(lock, {"b": 2, "a": 1})
    lock.write_text('{"a":1,"b":2}')
    baseline_config.semantic.task.calibration_lock_hash = digest
    assert verify_calibration_lock(baseline_config, lock) == digest
