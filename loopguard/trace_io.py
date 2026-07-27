"""Append-only JSONL trace IO (TRD 3.2).

Kept out of ``schemas/`` because it does file IO, and out of ``agent/`` because
``grading/`` reads traces too -- one reader, so writer and resolver cannot drift
in how they interpret a file.

Completeness rule: an episode is complete **iff its footer line parses**. Every
line is flushed as it is written, so a crash mid-episode leaves a
header-plus-partial-steps file that ``is_complete`` correctly rejects and the
runner re-runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any

from loopguard.schemas.base import SchemaVersionError
from loopguard.schemas.trace import (
    EpisodeFooter,
    EpisodeHeader,
    EpisodeTrace,
    StepRecord,
)


class TraceFormatError(RuntimeError):
    """A trace file is structurally malformed (wrong record order, missing header)."""


class TraceWriter:
    """Append-only writer. One instance per episode.

    Opened in append mode so a resumed run never truncates a file it decided not
    to re-run; the runner deletes incomplete traces before re-running instead.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._closed = False

    def _write(self, record: Any) -> None:
        self._fh.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def write_header(self, header: EpisodeHeader) -> None:
        self._write(header)

    def write_step(self, step: StepRecord) -> None:
        self._write(step)

    def write_footer(self, footer: EpisodeFooter) -> None:
        self._write(footer)
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._fh.close()
            self._closed = True

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _iter_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise TraceFormatError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
    return records


def is_complete(path: str | Path) -> bool:
    """True iff the file's last record parses as a footer.

    A :class:`SchemaVersionError` from a foreign artifact propagates: an
    unreadable trace must not be silently treated as "incomplete, re-run it",
    because that would quietly overwrite data this build cannot interpret.
    """
    p = Path(path)
    if not p.exists():
        return False
    try:
        records = _iter_records(p)
    except TraceFormatError:
        return False
    if not records:
        return False
    last = records[-1]
    if last.get("record_type") != "footer":
        return False
    EpisodeFooter.model_validate(last)  # raises SchemaVersionError on a foreign version
    return True


def read_trace(path: str | Path) -> EpisodeTrace:
    """Read and validate a complete trace. Raises on anything unexpected."""
    p = Path(path)
    records = _iter_records(p)
    if not records:
        raise TraceFormatError(f"{p} is empty")
    if records[0].get("record_type") != "header":
        raise TraceFormatError(f"{p}: first record is not a header")
    if records[-1].get("record_type") != "footer":
        raise TraceFormatError(f"{p}: last record is not a footer (episode incomplete)")

    header = EpisodeHeader.model_validate(records[0])
    footer = EpisodeFooter.model_validate(records[-1])
    steps = []
    for i, rec in enumerate(records[1:-1], start=1):
        if rec.get("record_type") != "step":
            raise TraceFormatError(f"{p}: record {i} has record_type={rec.get('record_type')!r}")
        steps.append(StepRecord.model_validate(rec))
    return EpisodeTrace(header=header, steps=steps, footer=footer)


__all__ = [
    "SchemaVersionError",
    "TraceFormatError",
    "TraceWriter",
    "is_complete",
    "read_trace",
]
