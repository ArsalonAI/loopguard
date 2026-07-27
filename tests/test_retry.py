"""Retry policy (TRD 6.3).

The measurement requirements matter more than the mechanics here: latency must
exclude backoff, and a retry must not become a differently-seeded sample.
"""

from __future__ import annotations

import pytest

from loopguard.agent.retry import RetryBudgetExhausted, backoff_delays, with_retries
from loopguard.schemas.config import RetryConfig

CONFIG = RetryConfig(max_attempts=4, base_delay_s=1.0, max_delay_s=30.0)


def test_backoff_is_bounded_and_seeded():
    first = list(backoff_delays(CONFIG, seed=99))
    assert first == list(backoff_delays(CONFIG, seed=99)), "jitter must be reproducible"
    assert first != list(backoff_delays(CONFIG, seed=100))
    assert len(first) == CONFIG.max_attempts - 1
    for i, delay in enumerate(first):
        assert 0.0 <= delay <= min(CONFIG.max_delay_s, CONFIG.base_delay_s * 2**i)


def test_succeeds_after_transient_failures_and_reports_attempt_count():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TimeoutError("upstream capacity")
        return "ok"

    slept: list[float] = []
    result, count = with_retries(
        flaky, config=CONFIG, seed=7, is_retryable=lambda e: True, sleep=slept.append
    )
    assert (result, count) == ("ok", 3)
    assert len(slept) == 2, "backoff happened between attempts, and is not part of latency"


def test_non_retryable_errors_propagate_immediately():
    def bad_request():
        raise ValueError("400 invalid tool schema")

    with pytest.raises(ValueError):
        with_retries(
            bad_request, config=CONFIG, seed=1, is_retryable=lambda e: False, sleep=lambda _: None
        )


def test_exhaustion_raises_and_carries_the_last_error():
    def always_429():
        raise TimeoutError("429")

    with pytest.raises(RetryBudgetExhausted, match="exhausted 4 attempts"):
        with_retries(
            always_429, config=CONFIG, seed=1, is_retryable=lambda e: True, sleep=lambda _: None
        )
