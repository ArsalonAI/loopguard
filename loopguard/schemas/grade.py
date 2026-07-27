"""Grading output (TRD 3.4) -- ``runs/<run_id>/grades.jsonl``.

Phase 0 defines the contract; the resolver that produces these is Phase 2.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from loopguard.schemas.base import VersionedModel


class FailureCategory(StrEnum):
    """PRD 4.1, plus UNRESOLVED (TRD 3.4)."""

    CORRECT = "CORRECT"
    HALLUCINATION = "HALLUCINATION"
    TOOL_MISUSE = "TOOL_MISUSE"
    CONTEXT_POLLUTION = "CONTEXT_POLLUTION"
    NON_TERMINATION = "NON_TERMINATION"
    AMBIGUOUS = "AMBIGUOUS"
    #: What AMBIGUOUS becomes when judge labels fail the kappa >= 0.70 gate.
    #: Applied automatically in code, not by discipline (TRD 8).
    UNRESOLVED = "UNRESOLVED"


class EpisodeGrade(VersionedModel):
    task_id: str
    repeat_index: int
    model_id: str
    depth: int
    category: FailureCategory
    #: None iff CORRECT. A first-class reported quantity, not a debug field:
    #: it is what distinguishes "depth degrades reasoning" from "depth offers
    #: more chances to fail" (PRD 4.1).
    first_error_hop: int | None = None
    subcode: str | None = Field(
        default=None, description="e.g. malformed_args, near_id, premature_termination"
    )
    resolution_source: Literal["mechanical", "judge", "human"]
    #: Non-optional in spirit: every mechanical label records *why* -- the
    #: divergent step index, the rule number that fired, the compared values,
    #: and P_i membership results -- so a disputed label is auditable without
    #: re-running the resolver (TRD 7.4).
    evidence: dict[str, Any] = Field(default_factory=dict)
    judge_confidence: float | None = None
