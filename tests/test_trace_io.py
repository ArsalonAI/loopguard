"""Trace IO, and the completeness rule that makes a 1,200-episode run resumable.

An episode is complete iff its footer parses. Getting this wrong in either
direction is expensive: too strict re-runs finished episodes and re-spends
tokens; too loose treats a crashed episode as data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from loopguard.schemas.base import SchemaVersionError
from loopguard.schemas.trace import EpisodeFooter, EpisodeHeader, StepRecord, ToolCall
from loopguard.trace_io import TraceFormatError, TraceWriter, is_complete, read_trace


def _header():
    return EpisodeHeader(
        task_id="d3-0000",
        task_seed=11,
        decoding_seed=22,
        repeat_index=0,
        depth=3,
        arm="baseline",
        model_id="llama-3.3-70b",
        provider_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        provider="together",
        config_hash="cafe",
        task_hash="beef",
        started_at=datetime.now(UTC),
    )


def _write_complete(path):
    with TraceWriter(path) as w:
        w.write_header(_header())
        w.write_step(
            StepRecord(
                step_index=0,
                role="assistant",
                tool_call=ToolCall(
                    call_id="c0",
                    name="lookup_service",
                    raw_arguments='{"service_id": "SVC-4471"}',
                    parsed_arguments={"service_id": "SVC-4471"},
                ),
                prompt_tokens=100,
                completion_tokens=12,
                latency_ms=340,
            )
        )
        w.write_step(
            StepRecord(step_index=1, role="tool", tool_response={"record": {"id": "SVC-4471"}})
        )
        w.write_footer(
            EpisodeFooter(
                terminal_reason="final_answer",
                final_answer="j.okafor@corp.example",
                prompt_tokens_total=100,
                completion_tokens_total=12,
                step_count=2,
                wall_clock_ms=340,
                completed_at=datetime.now(UTC),
            )
        )


def test_roundtrip(tmp_path):
    path = tmp_path / "ep.jsonl"
    _write_complete(path)

    assert is_complete(path)
    trace = read_trace(path)
    assert trace.header.task_id == "d3-0000"
    assert len(trace.steps) == 2
    assert trace.steps[0].tool_call.parsed_arguments == {"service_id": "SVC-4471"}
    assert trace.footer.final_answer == "j.okafor@corp.example"


def test_footerless_trace_is_incomplete(tmp_path):
    """The crash case: header plus partial steps must be re-run, not counted."""
    path = tmp_path / "crashed.jsonl"
    writer = TraceWriter(path)
    writer.write_header(_header())
    writer.write_step(StepRecord(step_index=0, role="assistant", content="thinking"))
    writer.close()

    assert not is_complete(path)
    with pytest.raises(TraceFormatError, match="episode incomplete"):
        read_trace(path)


def test_missing_and_empty_files_are_incomplete(tmp_path):
    assert not is_complete(tmp_path / "nope.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert not is_complete(empty)


def test_truncated_json_line_is_incomplete(tmp_path):
    path = tmp_path / "torn.jsonl"
    _write_complete(path)
    text = path.read_text()
    path.write_text(text[: -len(text.splitlines()[-1]) // 2])
    assert not is_complete(path)


def test_foreign_schema_version_is_not_silently_rerun(tmp_path):
    """A trace this build cannot read must raise, not be overwritten as 'incomplete'."""
    path = tmp_path / "future.jsonl"
    _write_complete(path)
    lines = path.read_text().splitlines()
    footer = json.loads(lines[-1])
    footer["schema_version"] = "99.0"
    lines[-1] = json.dumps(footer)
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(SchemaVersionError):
        is_complete(path)


def test_writer_appends_rather_than_truncates(tmp_path):
    path = tmp_path / "ep.jsonl"
    _write_complete(path)
    before = path.read_text()
    TraceWriter(path).close()
    assert path.read_text() == before
