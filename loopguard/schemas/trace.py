"""Episode trace records (TRD 3.2).

One JSONL file per episode, append-only:

* line 1      -- :class:`EpisodeHeader`
* lines 2..n  -- :class:`StepRecord`, one per agent step
* final line  -- :class:`EpisodeFooter`

An episode is complete **iff its footer line parses**. That is what makes a
1,200-episode run resumable after an interruption without re-spending tokens on
work already done.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from loopguard.schemas.base import VersionedModel

TerminalReason = Literal[
    "final_answer",
    "max_steps",
    "max_completion_tokens",
    "infrastructure_failure",
]

RecordType = Literal["header", "step", "footer"]


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    name: str
    raw_arguments: str = Field(description="Verbatim provider string, before parsing")
    parsed_arguments: dict[str, Any] | None = None
    parse_error: str | None = Field(
        default=None,
        description="Set when raw_arguments is not valid JSON -> resolver rule 3, malformed_args",
    )


class EpisodeHeader(VersionedModel):
    record_type: Literal["header"] = "header"

    task_id: str
    task_seed: int
    decoding_seed: int
    repeat_index: int
    depth: int
    arm: str
    model_id: str
    provider_model: str
    provider: str
    #: Provenance of everything that can move a result.
    config_hash: str
    #: Provenance of the task set only; equal across baseline and mitigation arms,
    #: which is what makes the comparison paired.
    task_hash: str
    started_at: datetime


class StepRecord(VersionedModel):
    record_type: Literal["step"] = "step"

    step_index: int
    role: Literal["assistant", "tool"]
    content: str | None = None
    tool_call: ToolCall | None = None
    tool_response: dict[str, Any] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: Successful attempt only. Including backoff sleep would make the cost
    #: comparison a measure of provider load rather than of the mitigation.
    latency_ms: int | None = None
    #: >1 means retries occurred (TRD 6.3).
    attempt_count: int = 1


class EpisodeFooter(VersionedModel):
    record_type: Literal["footer"] = "footer"

    terminal_reason: TerminalReason
    final_answer: str | None = None
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    step_count: int = 0
    #: Sum of successful-attempt latencies; excludes retry backoff by construction.
    wall_clock_ms: int = 0
    completed_at: datetime
    error: str | None = Field(
        default=None, description="Set when terminal_reason == infrastructure_failure"
    )


class EpisodeTrace(BaseModel):
    """A fully-read trace file. Produced by ``loopguard.trace_io.read_trace``."""

    model_config = ConfigDict(extra="forbid")

    header: EpisodeHeader
    steps: list[StepRecord]
    footer: EpisodeFooter
