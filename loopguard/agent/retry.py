"""Retry policy (TRD 6.3).

Exponential backoff with full jitter, bounded per episode. The jitter draws from
a generator seeded by the episode's decoding seed rather than global ``random``,
so a replayed run sleeps the same way -- retry behavior stays reproducible even
though it does not enter any reported number.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator

from loopguard.schemas.config import RetryConfig


class RetryBudgetExhausted(RuntimeError):
    """All attempts failed. The caller marks the episode ``infrastructure_failure``."""


def backoff_delays(config: RetryConfig, seed: int) -> Iterator[float]:
    """Yield ``max_attempts - 1`` full-jitter delays in seconds."""
    rng = random.Random(seed)
    for attempt in range(config.max_attempts - 1):
        ceiling = min(config.max_delay_s, config.base_delay_s * (2**attempt))
        yield rng.uniform(0.0, ceiling)


def with_retries(
    call: Callable[[], object],
    *,
    config: RetryConfig,
    seed: int,
    is_retryable: Callable[[BaseException], bool],
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[object, int]:
    """Run ``call`` with backoff. Returns ``(result, attempt_count)``.

    ``attempt_count`` is recorded on the step so retries stay visible without
    contaminating ``latency_ms``.
    """
    delays = list(backoff_delays(config, seed))
    last: BaseException | None = None
    for attempt in range(config.max_attempts):
        try:
            return call(), attempt + 1
        except BaseException as exc:
            if not is_retryable(exc):
                raise
            last = exc
            if attempt < len(delays):
                sleep(delays[attempt])
    raise RetryBudgetExhausted(
        f"exhausted {config.max_attempts} attempts; last error: {last!r}"
    ) from last
