"""Agent loop behaviour, entirely through network-free clients (TRD 12)."""

from __future__ import annotations

import json

import pytest

from loopguard.agent.loop import NUDGE, run_episode
from loopguard.agent.provider import CompletionResult, ProviderError
from loopguard.agent.replay import ReplayClient, ScriptedClient
from loopguard.schemas.trace import ToolCall
from loopguard.tools.smoke import build_registry, build_task
from loopguard.trace_io import is_complete, read_trace


def _call(name: str, args: dict, call_id: str = "c") -> ToolCall:
    return ToolCall(
        call_id=call_id,
        name=name,
        raw_arguments=json.dumps(args),
        parsed_arguments=args,
    )


def _turn(*tool_calls: ToolCall, content: str | None = None) -> CompletionResult:
    return CompletionResult(
        content=content,
        tool_calls=list(tool_calls),
        prompt_tokens=100,
        completion_tokens=10,
        latency_ms=7,
    )


def _gold_turns(task):
    turns = [
        _turn(_call(h.tool_name, dict(h.arguments), f"c{h.hop_index}")) for h in task.gold_trace
    ]
    turns.append(_turn(_call("submit_answer", {"answer": task.gold_answer}, "cf")))
    return turns


def _run(client, config, tmp_path, name="ep.jsonl"):
    task = build_task(task_seed=42)
    path = tmp_path / name
    footer = run_episode(
        client=client,
        task=task,
        registry=build_registry(),
        config=config,
        model=config.semantic.models[0],
        repeat_index=0,
        decoding_seed=987654321,
        config_hash="cfg",
        task_hash="tsk",
        trace_path=path,
    )
    return task, footer, path


def test_happy_path_writes_a_valid_trace(smoke_config, tmp_path):
    task = build_task(task_seed=42)
    client = ScriptedClient(_gold_turns(task))
    task, footer, path = _run(client, smoke_config, tmp_path)

    assert footer.terminal_reason == "final_answer"
    assert footer.final_answer == task.gold_answer
    assert is_complete(path)

    trace = read_trace(path)
    assert trace.header.depth == 3
    assert trace.header.task_hash == "tsk"
    names = [s.tool_call.name for s in trace.steps if s.role == "assistant" and s.tool_call]
    assert names == ["lookup_service", "lookup_datacenter", "lookup_region", "submit_answer"]

    # Tool responses were produced by the real tools over the real graph.
    responses = [s.tool_response for s in trace.steps if s.role == "tool"]
    assert responses[0]["record"]["datacenter"] == "DC-NORTH-12"


def test_every_call_reuses_the_same_decoding_seed(smoke_config, tmp_path):
    """A retry -- or any subsequent turn -- must not be a differently-seeded sample."""
    task = build_task(task_seed=42)
    client = ScriptedClient(_gold_turns(task))
    _run(client, smoke_config, tmp_path)
    assert {c["seed"] for c in client.calls} == {987654321}


def test_prose_reply_is_nudged_rather_than_terminating(smoke_config, tmp_path):
    task = build_task(task_seed=42)
    client = ScriptedClient([_turn(content="Let me think about this."), *_gold_turns(task)])
    _, footer, _path = _run(client, smoke_config, tmp_path)

    assert footer.terminal_reason == "final_answer"
    second_call_messages = client.calls[1]["messages"]
    assert second_call_messages[-1] == {"role": "user", "content": NUDGE}


def test_max_steps_exhaustion(smoke_config, tmp_path):
    """No final answer and a budget cap hit -- the NON_TERMINATION precondition."""
    turns = [_turn(content="still thinking") for _ in range(smoke_config.semantic.loop.max_steps)]
    _, footer, path = _run(ScriptedClient(turns), smoke_config, tmp_path)

    assert footer.terminal_reason == "max_steps"
    assert footer.final_answer is None
    assert is_complete(path)


def test_completion_token_cap(smoke_config, tmp_path):
    smoke_config.semantic.loop.max_completion_tokens = 15
    turns = [_turn(content="a"), _turn(content="b"), _turn(content="c")]
    _, footer, _ = _run(ScriptedClient(turns), smoke_config, tmp_path)
    assert footer.terminal_reason == "max_completion_tokens"


def test_provider_failure_is_infrastructure_not_agent_failure(smoke_config, tmp_path):
    """Retry-budget exhaustion is excluded from failure-rate denominators (TRD 6.3)."""

    class FailingClient:
        def complete(self, **_kwargs):
            raise ProviderError("exhausted 4 attempts; last error: RateLimitError")

    _, footer, path = _run(FailingClient(), smoke_config, tmp_path)

    assert footer.terminal_reason == "infrastructure_failure"
    assert footer.final_answer is None
    assert "exhausted" in (footer.error or "")
    assert is_complete(path), "an infra failure must still write a parseable footer"


def test_malformed_arguments_continue_the_episode(smoke_config, tmp_path):
    """Malformed args are a finding (TOOL_MISUSE), not a runner crash."""
    task = build_task(task_seed=42)
    bad = ToolCall(
        call_id="c0",
        name="lookup_service",
        raw_arguments="{service_id: SVC-4471",
        parsed_arguments=None,
        parse_error="Expecting property name enclosed in double quotes",
    )
    client = ScriptedClient([_turn(bad), *_gold_turns(task)])
    _, footer, path = _run(client, smoke_config, tmp_path)

    assert footer.terminal_reason == "final_answer"
    responses = [s.tool_response for s in read_trace(path).steps if s.role == "tool"]
    assert responses[0]["error"] == "malformed_arguments"


def test_unknown_tool_continues_the_episode(smoke_config, tmp_path):
    task = build_task(task_seed=42)
    client = ScriptedClient([_turn(_call("lookup_planet", {"planet_id": "X"})), *_gold_turns(task)])
    _, footer, path = _run(client, smoke_config, tmp_path)

    assert footer.terminal_reason == "final_answer"
    responses = [s.tool_response for s in read_trace(path).steps if s.role == "tool"]
    assert responses[0]["error"] == "unknown_tool"


def test_nonexistent_entity_is_a_tool_error_not_a_crash(smoke_config, tmp_path):
    task = build_task(task_seed=42)
    client = ScriptedClient(
        [_turn(_call("lookup_service", {"service_id": "SVC-9999"})), *_gold_turns(task)]
    )
    _, footer, path = _run(client, smoke_config, tmp_path)

    responses = [s.tool_response for s in read_trace(path).steps if s.role == "tool"]
    assert responses[0]["error"] == "tool_error"
    assert footer.terminal_reason == "final_answer"


def test_replay_reproduces_a_recorded_episode(smoke_config, tmp_path):
    task = build_task(task_seed=42)
    _, original, path = _run(
        ScriptedClient(_gold_turns(task)), smoke_config, tmp_path, "first.jsonl"
    )

    _, replayed, replay_path = _run(
        ReplayClient.from_file(path), smoke_config, tmp_path, "replay.jsonl"
    )

    assert replayed.final_answer == original.final_answer
    assert replayed.terminal_reason == original.terminal_reason
    original_calls = [s.tool_call.name for s in read_trace(path).steps if s.tool_call]
    replay_calls = [s.tool_call.name for s in read_trace(replay_path).steps if s.tool_call]
    assert replay_calls == original_calls


def test_multiple_tool_calls_in_one_turn_are_recorded_linearly(smoke_config, tmp_path):
    task = build_task(task_seed=42)
    parallel = _turn(
        _call("lookup_service", {"service_id": "SVC-4471"}, "a"),
        _call("search_entities", {"query": "DC-"}, "b"),
    )
    client = ScriptedClient([parallel, *_gold_turns(task)])
    _, _, path = _run(client, smoke_config, tmp_path)

    steps = read_trace(path).steps
    assert [s.role for s in steps[:4]] == ["assistant", "tool", "assistant", "tool"]
    assert [s.step_index for s in steps] == list(range(len(steps)))


def test_scripted_client_exhaustion_is_loud(smoke_config, tmp_path):
    with pytest.raises(AssertionError, match="exhausted"):
        _run(ScriptedClient([]), smoke_config, tmp_path)
