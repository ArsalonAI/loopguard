"""Determinism and the semantic/runtime/task hash split (TRD 5.1).

These are validity tests, not unit tests. If the runtime tier leaks into a hash,
changing concurrency reseeds the task set and the baseline<->mitigation pairing
silently breaks -- producing numbers that look fine and mean nothing.
"""

from __future__ import annotations

import copy

from loopguard.hashing import (
    canonical_json,
    config_hash,
    derive_decoding_seed,
    derive_task_seed,
    provider_seed,
    task_hash,
)


def test_hashes_are_deterministic(baseline_config):
    other = copy.deepcopy(baseline_config)
    assert task_hash(baseline_config.semantic.task) == task_hash(other.semantic.task)
    assert config_hash(baseline_config.semantic) == config_hash(other.semantic)


def test_runtime_changes_move_neither_hash(baseline_config):
    """Concurrency, rate limits, retry budget, and out_dir are excluded by construction."""
    before = (task_hash(baseline_config.semantic.task), config_hash(baseline_config.semantic))

    baseline_config.runtime.rate_limits.max_concurrency = 32
    baseline_config.runtime.retry.max_attempts = 9
    baseline_config.runtime.out_dir = baseline_config.runtime.out_dir / "elsewhere"
    baseline_config.runtime.log_level = "DEBUG"
    baseline_config.runtime.max_spend_usd = 999.0

    assert (
        task_hash(baseline_config.semantic.task),
        config_hash(baseline_config.semantic),
    ) == before


def test_mitigation_arm_keeps_task_seeds_but_changes_config_hash(baseline_config):
    """The paired-comparison invariant, enforced.

    A mitigation changes the arm and the prompt. Task seeds must not move -- the
    comparison is paired, and unpaired sampling is materially weaker at n=40/cell.
    The config hash *must* move, so the change is visible in a manifest diff.
    """
    baseline_task_hash = task_hash(baseline_config.semantic.task)
    baseline_config_hash = config_hash(baseline_config.semantic)

    baseline_config.semantic.arm = "mitigation"
    baseline_config.semantic.prompt.version = "mitigation_v1"
    baseline_config.semantic.prompt.text += "\nAfter each lookup, restate the id you received.\n"

    assert task_hash(baseline_config.semantic.task) == baseline_task_hash
    assert config_hash(baseline_config.semantic) != baseline_config_hash


def test_task_config_changes_move_task_seeds(baseline_config):
    before = task_hash(baseline_config.semantic.task)
    baseline_config.semantic.task.tasks_per_depth = 41
    assert task_hash(baseline_config.semantic.task) != before


def test_seed_chain_is_stable_and_distinct(baseline_config):
    t_hash = task_hash(baseline_config.semantic.task)

    seeds = {
        (d, i): derive_task_seed(t_hash, d, i)
        for d in baseline_config.semantic.task.depths
        for i in range(20)
    }
    assert len(set(seeds.values())) == len(seeds), "task seeds collided"
    assert derive_task_seed(t_hash, 3, 7) == derive_task_seed(t_hash, 3, 7)

    task_seed = seeds[(3, 7)]
    repeats = [derive_decoding_seed(task_seed, r) for r in range(3)]
    assert len(set(repeats)) == 3
    assert derive_decoding_seed(task_seed, 1) == repeats[1]


def test_provider_seed_fits_signed_32_bit(baseline_config):
    t_hash = task_hash(baseline_config.semantic.task)
    for i in range(200):
        seed = derive_decoding_seed(derive_task_seed(t_hash, 5, i), 2)
        assert 0 <= provider_seed(seed) < 2**31


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": [2, {"d": 4, "c": 3}]}) == canonical_json(
        {"a": [2, {"c": 3, "d": 4}], "b": 1}
    )
