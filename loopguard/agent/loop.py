"""The agent loop.

Runs one episode against a :class:`~loopguard.agent.provider.ProviderClient` and
writes an append-only JSONL trace. It makes no judgements -- every classification
belongs to the resolver (Phase 2). The loop's only job is to produce a complete,
faithful record of what happened, including the parts that look like failures.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loopguard.agent.provider import CompletionResult, ProviderError
from loopguard.schemas.config import LoopGuardConfig, ModelSpec
from loopguard.schemas.task import TaskInstance
from loopguard.schemas.trace import (
    EpisodeFooter,
    EpisodeHeader,
    StepRecord,
    TerminalReason,
    ToolCall,
)
from loopguard.tools import ToolRegistry
from loopguard.trace_io import TraceWriter

#: Appended verbatim when the model replies with prose instead of calling a tool.
#:
#: Deliberately lives in code, not in the prompt config: it must be byte-identical
#: in every condition including the mitigation arm, and a mitigation is allowed to
#: rewrite the system prompt. Keeping it here means a prompt edit cannot silently
#: change the harness's control flow and confound the comparison.
NUDGE = (
    "Continue by calling a tool. When you have the final value, call submit_answer "
    "with the exact value and nothing else."
)


def _now() -> datetime:
    return datetime.now(UTC)


def _assistant_message(result: CompletionResult) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.content or ""}
    if result.tool_calls:
        message["tool_calls"] = [
            {
                "id": tc.call_id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.raw_arguments},
            }
            for tc in result.tool_calls
        ]
    return message


def build_messages(system_prompt: str, question: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def run_episode(
    *,
    client: Any,
    task: TaskInstance,
    registry: ToolRegistry,
    config: LoopGuardConfig,
    model: ModelSpec,
    repeat_index: int,
    decoding_seed: int,
    config_hash: str,
    task_hash: str,
    trace_path: str | Path,
) -> EpisodeFooter:
    """Run one episode to termination and return its footer.

    The trace file is written incrementally, so an interrupted episode leaves a
    footerless file that the runner will re-run rather than a silently truncated
    one it would treat as done.
    """
    semantic = config.semantic
    max_steps = semantic.loop.max_steps
    assert max_steps is not None  # resolved in SemanticConfig validator

    messages = build_messages(semantic.prompt.text, task.question)
    wire_tools = registry.to_wire()

    started = _now()
    step_index = 0
    prompt_tokens_total = 0
    completion_tokens_total = 0
    wall_clock_ms = 0
    final_answer: str | None = None
    terminal_reason: TerminalReason = "max_steps"
    error: str | None = None

    writer = TraceWriter(trace_path)
    writer.write_header(
        EpisodeHeader(
            task_id=task.task_id,
            task_seed=task.task_seed,
            decoding_seed=decoding_seed,
            repeat_index=repeat_index,
            depth=task.depth,
            arm=semantic.arm,
            model_id=model.id,
            provider_model=model.provider_model,
            provider=semantic.provider.name,
            config_hash=config_hash,
            task_hash=task_hash,
            started_at=started,
        )
    )

    try:
        for _turn in range(max_steps):
            try:
                result = client.complete(
                    messages=messages,
                    tools=wire_tools,
                    decoding=semantic.decoding,
                    seed=decoding_seed,
                )
            except ProviderError as exc:
                # Retry budget exhausted. Not an agent failure: excluded from
                # failure-rate denominators and counted separately (TRD 6.3).
                terminal_reason = "infrastructure_failure"
                error = str(exc)
                break

            prompt_tokens_total += result.prompt_tokens or 0
            completion_tokens_total += result.completion_tokens or 0
            wall_clock_ms += result.latency_ms or 0
            messages.append(_assistant_message(result))

            if not result.tool_calls:
                writer.write_step(
                    StepRecord(
                        step_index=step_index,
                        role="assistant",
                        content=result.content,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        latency_ms=result.latency_ms,
                        attempt_count=result.attempt_count,
                    )
                )
                step_index += 1
                # Prose instead of a tool call: nudge and continue. The nudge is a
                # user message, not a step, and is reconstructable from its position.
                messages.append({"role": "user", "content": NUDGE})
                if completion_tokens_total >= semantic.loop.max_completion_tokens:
                    terminal_reason = "max_completion_tokens"
                    break
                continue

            terminated = False
            for call_position, tool_call in enumerate(result.tool_calls):
                writer.write_step(
                    StepRecord(
                        step_index=step_index,
                        role="assistant",
                        content=result.content if call_position == 0 else None,
                        tool_call=tool_call,
                        prompt_tokens=result.prompt_tokens if call_position == 0 else None,
                        completion_tokens=result.completion_tokens if call_position == 0 else None,
                        latency_ms=result.latency_ms if call_position == 0 else None,
                        attempt_count=result.attempt_count,
                    )
                )
                step_index += 1

                response = _dispatch(registry, tool_call, task)
                writer.write_step(
                    StepRecord(
                        step_index=step_index,
                        role="tool",
                        tool_call=tool_call,
                        tool_response=response,
                    )
                )
                step_index += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.call_id,
                        "content": json.dumps(response, ensure_ascii=False),
                    }
                )

                if tool_call.name == registry.terminal_tool and tool_call.parse_error is None:
                    args = tool_call.parsed_arguments or {}
                    final_answer = str(args.get("answer", ""))
                    terminal_reason = "final_answer"
                    terminated = True
                    break

            if terminated:
                break
            if completion_tokens_total >= semantic.loop.max_completion_tokens:
                terminal_reason = "max_completion_tokens"
                break

        footer = EpisodeFooter(
            terminal_reason=terminal_reason,
            final_answer=final_answer,
            prompt_tokens_total=prompt_tokens_total,
            completion_tokens_total=completion_tokens_total,
            step_count=step_index,
            wall_clock_ms=wall_clock_ms,
            completed_at=_now(),
            error=error,
        )
        writer.write_footer(footer)
        return footer
    finally:
        writer.close()


def _dispatch(registry: ToolRegistry, tool_call: ToolCall, task: TaskInstance) -> dict[str, Any]:
    """Execute a tool call, turning every failure into a normal tool response.

    Malformed arguments and unknown tools are findings (TOOL_MISUSE), not runner
    errors -- the episode continues and the resolver classifies it.
    """
    if tool_call.parse_error is not None:
        return {"error": "malformed_arguments", "message": tool_call.parse_error}
    if tool_call.name not in registry:
        return {"error": "unknown_tool", "message": f"no tool named {tool_call.name!r}"}
    try:
        return registry.call(tool_call.name, tool_call.parsed_arguments or {}, task.graph)
    except KeyError as exc:
        return {"error": "missing_argument", "message": f"required argument {exc} not provided"}
