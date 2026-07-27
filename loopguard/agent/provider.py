"""The provider interface (TRD 6.1).

One implementation covers every candidate provider, because they all speak the
same OpenAI-compatible wire format; swapping providers is a ``base_url`` and a
model string. A second implementation replays recorded traces with no network,
so grading logic is testable without spending tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from loopguard.schemas.config import DecodingParams
from loopguard.schemas.trace import ToolCall


class ProviderError(RuntimeError):
    """A call failed after the retry budget was exhausted.

    The episode is marked ``infrastructure_failure`` and excluded from
    failure-rate denominators (TRD 6.3). Grading a provider timeout as an agent
    failure is the single easiest way to fabricate a depth effect.
    """


@dataclass
class CompletionResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: Successful attempt only -- retry backoff must never enter it.
    latency_ms: int | None = None
    attempt_count: int = 1
    finish_reason: str | None = None
    #: Provider-reported served revision, when exposed. Recorded in the manifest.
    model_revision: str | None = None
    raw: dict[str, Any] | None = None


@runtime_checkable
class ProviderClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        decoding: DecodingParams,
        seed: int,
    ) -> CompletionResult: ...
