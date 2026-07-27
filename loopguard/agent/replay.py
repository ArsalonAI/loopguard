"""Network-free provider implementations (TRD 6.1, 12).

``ReplayClient`` replays a recorded trace so resolver and loop logic are testable
without spending tokens; CI runs the agent loop entirely through these.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loopguard.agent.provider import CompletionResult
from loopguard.schemas.config import DecodingParams
from loopguard.schemas.trace import EpisodeTrace
from loopguard.trace_io import read_trace


class ScriptedClient:
    """Returns pre-built completions in order. The unit-test workhorse."""

    def __init__(self, results: list[CompletionResult]) -> None:
        self._results = list(results)
        self._index = 0
        #: Every (messages, tools, seed) the loop asked for, for assertions.
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        decoding: DecodingParams,
        seed: int,
    ) -> CompletionResult:
        self.calls.append({"messages": list(messages), "tools": tools, "seed": seed})
        if self._index >= len(self._results):
            raise AssertionError(
                f"ScriptedClient exhausted after {len(self._results)} completions; "
                "the loop asked for another"
            )
        result = self._results[self._index]
        self._index += 1
        return result


class ReplayClient:
    """Replays the assistant turns of a recorded episode.

    Tool responses are *not* replayed -- they are recomputed by the real tools
    over the real graph. That is deliberate: if a tool change would alter what
    the agent saw, replay must diverge visibly rather than paper over it with the
    old recorded response.
    """

    def __init__(self, trace: EpisodeTrace) -> None:
        self._assistant_steps = [s for s in trace.steps if s.role == "assistant"]
        self._index = 0

    @classmethod
    def from_file(cls, path: str | Path) -> ReplayClient:
        return cls(read_trace(path))

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        decoding: DecodingParams,
        seed: int,
    ) -> CompletionResult:
        if self._index >= len(self._assistant_steps):
            raise AssertionError("ReplayClient exhausted: recorded trace has no further turns")
        step = self._assistant_steps[self._index]
        self._index += 1
        return CompletionResult(
            content=step.content,
            tool_calls=[step.tool_call] if step.tool_call else [],
            prompt_tokens=step.prompt_tokens,
            completion_tokens=step.completion_tokens,
            latency_ms=step.latency_ms,
            attempt_count=step.attempt_count,
        )
