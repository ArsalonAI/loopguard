"""Generator output.

Phase 0 defines the contract only; the generator that fills it is Phase 1.

The load-bearing property (PRD 3.3): the generator emits a **gold trace**, not a
gold answer. Ground truth for every intermediate hop is what makes mechanical
failure attribution possible. Any change here that drops per-hop ground truth
collapses the taxonomy onto an LLM judge -- the exact thing the design exists to
avoid.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from loopguard.schemas.base import VersionedModel

DistractorKind = Literal["near_id", "cross_branch", "stale"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Entity(StrictModel):
    entity_id: str
    entity_type: str
    attributes: dict[str, str]


class EntityGraph(StrictModel):
    """The full generated world, for tool binding.

    Tools are pure deterministic functions over this graph -- no network, no
    clock, no randomness. Replayability depends on it.
    """

    entities: dict[str, Entity] = Field(default_factory=dict)

    def by_type(self, entity_type: str) -> list[Entity]:
        return [e for e in self.entities.values() if e.entity_type == entity_type]


class GoldHop(StrictModel):
    hop_index: int = Field(description="0-based")
    tool_name: str
    arguments: dict[str, str]
    returned_entity_id: str
    resolved_value: str = Field(description="Feeds the next hop, or is the final answer")


class DistractorRecord(StrictModel):
    """Why a value is a distractor.

    The registry is the key to attribution: without it, CONTEXT_POLLUTION cannot
    be separated from AMBIGUOUS. The generator must register **every** distractor
    value it emits, including ones it emits incidentally (TRD 3.1).
    """

    value: str
    kind: DistractorKind
    introduced_at_hop: int = Field(description="Which tool response first surfaced it")
    correct_counterpart: str


class TokenAccounting(StrictModel):
    """Realized per-hop payload token counts under both tokenizers (TRD 5.3).

    Length-matching is done against a reference tokenizer and asserted within a
    looser tolerance under the second one. Both realized counts are recorded here
    and surfaced in the dashboard's method panel, so the compromise is visible
    rather than presented as exact matching.
    """

    reference_tokenizer: str
    secondary_tokenizer: str
    per_hop_reference: list[int] = Field(default_factory=list)
    per_hop_secondary: list[int] = Field(default_factory=list)


class TaskInstance(VersionedModel):
    task_id: str = Field(description='Deterministic: f"d{depth}-{task_index:04d}"')
    task_seed: int
    depth: int
    question: str
    gold_trace: list[GoldHop]
    gold_answer: str
    graph: EntityGraph
    distractor_registry: dict[str, DistractorRecord] = Field(
        default_factory=dict, description="value -> why it's a distractor"
    )
    exposed_tools: list[str] = Field(
        description="Length N, constant across depths. Depth changes how many are needed."
    )
    token_accounting: TokenAccounting | None = None
