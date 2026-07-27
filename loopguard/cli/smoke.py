"""``loopguard smoke`` -- the Phase 0 provider smoke test.

Phase 0's exit criterion (PRD 5): one end-to-end episode runs against both models
and writes a valid trace. This command is that criterion, and it is also how the
provider gets chosen (TRD 14) -- run it against each candidate and compare
tool-calling fidelity.

It reports *fidelity*, not correctness. Grading is the mechanical resolver's job
(Phase 2) and is deliberately not duplicated here; two implementations of
attribution would drift, and this one would be the one people quote.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loopguard.agent.loop import run_episode
from loopguard.agent.provider import CompletionResult
from loopguard.agent.replay import ScriptedClient
from loopguard.config_io import load_config
from loopguard.gitinfo import git_dirty, git_sha
from loopguard.hashing import config_hash, derive_decoding_seed, derive_task_seed, task_hash
from loopguard.providers import get as get_provider
from loopguard.schemas.config import LoopGuardConfig, ModelSpec, ProviderConfig
from loopguard.schemas.manifest import ModelPin, RunManifest
from loopguard.schemas.trace import EpisodeFooter, ToolCall
from loopguard.tools import smoke as smoke_fixture
from loopguard.trace_io import read_trace


@dataclass
class ModelOutcome:
    model_id: str
    provider_model: str
    trace_path: Path
    footer: EpisodeFooter
    tool_sequence: list[str]
    gold_sequence: list[str]
    unparsable_calls: int
    unknown_tools: int
    reached_terminal_tool: bool
    answer_matches_gold: bool
    error: str | None = None

    @property
    def tool_calling_ok(self) -> bool:
        """Fidelity, not correctness: did the provider produce usable tool calls at all?"""
        return (
            self.error is None
            and self.unparsable_calls == 0
            and self.unknown_tools == 0
            and self.reached_terminal_tool
        )


def _normalize(value: str | None) -> str:
    return (value or "").strip().strip(".,;:'\"").casefold()


def _override_provider(config: LoopGuardConfig, provider_name: str) -> LoopGuardConfig:
    candidate = get_provider(provider_name)
    config.semantic.provider = ProviderConfig(name=candidate.name, base_url=candidate.base_url)
    config.runtime.api_key_env = candidate.api_key_env
    config.semantic.models = [
        ModelSpec(id="llama-3.3-70b", provider_model=candidate.llama_model),
        ModelSpec(id="qwen3", provider_model=candidate.qwen_model),
    ]
    return config


def _scripted_client(task: Any) -> ScriptedClient:
    """A canned perfect trajectory, for exercising the plumbing with no network."""
    results = [
        CompletionResult(
            content=None,
            tool_calls=[
                ToolCall(
                    call_id=f"call_{hop.hop_index}",
                    name=hop.tool_name,
                    raw_arguments=_json(hop.arguments),
                    parsed_arguments=dict(hop.arguments),
                )
            ],
            prompt_tokens=100 + 50 * hop.hop_index,
            completion_tokens=20,
            latency_ms=5,
        )
        for hop in task.gold_trace
    ]
    results.append(
        CompletionResult(
            content=None,
            tool_calls=[
                ToolCall(
                    call_id="call_final",
                    name="submit_answer",
                    raw_arguments=_json({"answer": task.gold_answer}),
                    parsed_arguments={"answer": task.gold_answer},
                )
            ],
            prompt_tokens=400,
            completion_tokens=15,
            latency_ms=5,
        )
    )
    return ScriptedClient(results)


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def run_smoke(
    *,
    config_path: str | Path,
    provider: str | None,
    dry_run: bool,
    only_model: str | None,
    llama_model: str | None,
    qwen_model: str | None,
) -> int:
    config = load_config(config_path)
    if provider:
        config = _override_provider(config, provider)
    if llama_model:
        config.semantic.models[0].provider_model = llama_model
    if qwen_model:
        config.semantic.models[1].provider_model = qwen_model

    models = config.semantic.models
    if only_model:
        models = [config.model_by_id(only_model)]

    t_hash = task_hash(config.semantic.task)
    c_hash = config_hash(config.semantic)
    task_seed = derive_task_seed(t_hash, depth=3, task_index=0)
    decoding_seed = derive_decoding_seed(task_seed, repeat_index=0)

    task = smoke_fixture.build_task(task_seed)
    registry = smoke_fixture.build_registry()
    registry.validate_against_policy(config.semantic.tools)

    api_key = os.environ.get(config.runtime.api_key_env)
    if not dry_run and not api_key:
        print(
            f"error: ${config.runtime.api_key_env} is not set. "
            f"Set it in .env or the environment, or pass --dry-run to exercise the "
            f"harness with no network.",
            file=sys.stderr,
        )
        return 3

    started = datetime.now(UTC)
    run_id = f"smoke-{started:%Y%m%d-%H%M%S}-{config.semantic.provider.name}"
    run_dir = Path(config.runtime.out_dir) / run_id
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    (run_dir / "tasks").mkdir(parents=True, exist_ok=True)
    (run_dir / "tasks" / f"{task.task_id}.json").write_text(
        task.model_dump_json(indent=2), encoding="utf-8"
    )

    print(f"run_id      : {run_id}")
    print(f"provider    : {config.semantic.provider.name} ({config.semantic.provider.base_url})")
    print(f"models      : {', '.join(m.provider_model for m in models)}")
    print(f"task_hash   : {t_hash}")
    print(f"config_hash : {c_hash}")
    print(f"episodes    : {len(models)}  (fixture d=3, {len(registry.exposed_names())} tools)")
    print()

    outcomes: list[ModelOutcome] = []
    for model in models:
        trace_path = run_dir / "traces" / f"{task.task_id}-{model.id}-r0.jsonl"
        client = (
            _scripted_client(task)
            if dry_run
            else _live_client(config, model.provider_model, api_key or "")
        )
        error: str | None = None
        try:
            footer = run_episode(
                client=client,
                task=task,
                registry=registry,
                config=config,
                model=model,
                repeat_index=0,
                decoding_seed=decoding_seed,
                config_hash=c_hash,
                task_hash=t_hash,
                trace_path=trace_path,
            )
        except Exception as exc:
            print(f"  {model.id}: FAILED to complete an episode: {exc!r}", file=sys.stderr)
            error = repr(exc)
            outcomes.append(
                ModelOutcome(
                    model_id=model.id,
                    provider_model=model.provider_model,
                    trace_path=trace_path,
                    footer=EpisodeFooter(
                        terminal_reason="infrastructure_failure",
                        completed_at=datetime.now(UTC),
                        error=error,
                    ),
                    tool_sequence=[],
                    gold_sequence=[h.tool_name for h in task.gold_trace],
                    unparsable_calls=0,
                    unknown_tools=0,
                    reached_terminal_tool=False,
                    answer_matches_gold=False,
                    error=error,
                )
            )
            continue

        # Read the trace back rather than trusting in-memory state: this is also
        # the Phase 0 check that what we wrote is a valid, re-readable artifact.
        trace = read_trace(trace_path)
        calls: list[ToolCall] = [
            s.tool_call for s in trace.steps if s.role == "assistant" and s.tool_call is not None
        ]
        outcomes.append(
            ModelOutcome(
                model_id=model.id,
                provider_model=model.provider_model,
                trace_path=trace_path,
                footer=footer,
                tool_sequence=[c.name for c in calls],
                gold_sequence=[h.tool_name for h in task.gold_trace],
                unparsable_calls=len([c for c in calls if c.parse_error]),
                unknown_tools=len([c for c in calls if c.name not in registry]),
                reached_terminal_tool=any(c.name == registry.terminal_tool for c in calls),
                answer_matches_gold=_normalize(footer.final_answer) == _normalize(task.gold_answer),
            )
        )

    manifest = RunManifest(
        run_id=run_id,
        git_sha=git_sha(),
        git_dirty=git_dirty(),
        config_hash=c_hash,
        task_hash=t_hash,
        config_resolved=config.model_dump(mode="json"),
        calibration_lock_hash=None,
        provider=config.semantic.provider.name,
        models=[
            ModelPin(id=m.id, provider_model=m.provider_model, served_revision=None) for m in models
        ],
        decoding=config.semantic.decoding,
        arm=config.semantic.arm,
        task_seeds=[task_seed],
        started_at=started,
        completed_at=datetime.now(UTC),
        episode_count=len(outcomes),
        infrastructure_failure_count=sum(
            1 for o in outcomes if o.footer.terminal_reason == "infrastructure_failure"
        ),
    )
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    _print_report(outcomes, run_dir)
    return 0 if all(o.tool_calling_ok for o in outcomes) else 1


def _live_client(config: LoopGuardConfig, provider_model: str, api_key: str) -> Any:
    from loopguard.agent.openai_compat import OpenAICompatClient

    return OpenAICompatClient(
        base_url=config.semantic.provider.base_url,
        api_key=api_key,
        model=provider_model,
        retry=config.runtime.retry,
    )


def _print_report(outcomes: list[ModelOutcome], run_dir: Path) -> None:
    print("tool-calling fidelity (not a grade -- attribution is Phase 2)")
    print("-" * 72)
    for o in outcomes:
        status = "PASS" if o.tool_calling_ok else "FAIL"
        print(f"[{status}] {o.model_id}  ({o.provider_model})")
        if o.error:
            print(f"        error            : {o.error}")
            continue
        print(f"        terminal_reason  : {o.footer.terminal_reason}")
        print(f"        tool sequence    : {' -> '.join(o.tool_sequence) or '(none)'}")
        print(f"        gold sequence    : {' -> '.join(o.gold_sequence)} -> submit_answer")
        print(f"        unparsable args  : {o.unparsable_calls}")
        print(f"        unknown tools    : {o.unknown_tools}")
        print(f"        final answer     : {o.footer.final_answer!r}")
        print(f"        matches gold     : {o.answer_matches_gold}")
        print(
            f"        tokens           : {o.footer.prompt_tokens_total} prompt / "
            f"{o.footer.completion_tokens_total} completion"
        )
        print(f"        wall clock       : {o.footer.wall_clock_ms} ms (successful attempts only)")
        print(f"        trace            : {o.trace_path}")
    print("-" * 72)
    print(f"manifest: {run_dir / 'manifest.json'}")
    print(
        "\nnote: `matches gold` is a smoke signal only. One episode at temperature 0 "
        "is not evidence about either model."
    )
