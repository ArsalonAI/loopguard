"""Phase-0 smoke fixture: a hand-built d=3 world and tool set.

**This is not the generator.** Phase 1 replaces it with the seeded generator that
emits graph, chain, distractors, and gold trace together. It exists only so that
Phase 0's exit criterion -- one end-to-end episode against both models, writing a
valid trace -- can be met before the generator lands, and so the provider smoke
test exercises real multi-hop tool calling rather than a single trivial call.

It deliberately mirrors the real task's shape (PRD 3.3): a chained entity lookup
with near-ID and cross-branch distractors in every response, six lookup tools of
which three are needed, and a gold trace with per-hop ground truth.
"""

from __future__ import annotations

from typing import Any

from loopguard.schemas.base import SCHEMA_VERSION
from loopguard.schemas.task import DistractorRecord, Entity, EntityGraph, GoldHop, TaskInstance
from loopguard.tools import ToolError, ToolRegistry, ToolSpec

_ENTITIES = [
    Entity(
        entity_id="SVC-4471",
        entity_type="service",
        attributes={"name": "checkout-api", "datacenter": "DC-NORTH-12", "tier": "1"},
    ),
    Entity(
        entity_id="SVC-4472",
        entity_type="service",
        attributes={"name": "checkout-api-canary", "datacenter": "DC-SOUTH-04", "tier": "3"},
    ),
    Entity(
        entity_id="DC-NORTH-12",
        entity_type="datacenter",
        attributes={"name": "north-primary", "region": "REG-EMEA-3", "rack_count": "412"},
    ),
    Entity(
        entity_id="DC-NORTH-13",
        entity_type="datacenter",
        attributes={"name": "north-secondary", "region": "REG-EMEA-7", "rack_count": "118"},
    ),
    Entity(
        entity_id="DC-SOUTH-04",
        entity_type="datacenter",
        attributes={"name": "south-primary", "region": "REG-EMEA-7", "rack_count": "260"},
    ),
    Entity(
        entity_id="REG-EMEA-3",
        entity_type="region",
        attributes={
            "name": "emea-central",
            "escalation_contact": "j.okafor@corp.example",
            "timezone": "Europe/Berlin",
        },
    ),
    Entity(
        entity_id="REG-EMEA-7",
        entity_type="region",
        attributes={
            "name": "emea-south",
            "escalation_contact": "m.laurent@corp.example",
            "timezone": "Europe/Lisbon",
        },
    ),
]

GRAPH = EntityGraph(entities={e.entity_id: e for e in _ENTITIES})

QUESTION = (
    "What is the escalation contact for the region that owns the datacenter "
    "hosting service SVC-4471?"
)

GOLD_TRACE = [
    GoldHop(
        hop_index=0,
        tool_name="lookup_service",
        arguments={"service_id": "SVC-4471"},
        returned_entity_id="SVC-4471",
        resolved_value="DC-NORTH-12",
    ),
    GoldHop(
        hop_index=1,
        tool_name="lookup_datacenter",
        arguments={"datacenter_id": "DC-NORTH-12"},
        returned_entity_id="DC-NORTH-12",
        resolved_value="REG-EMEA-3",
    ),
    GoldHop(
        hop_index=2,
        tool_name="lookup_region",
        arguments={"region_id": "REG-EMEA-3"},
        returned_entity_id="REG-EMEA-3",
        resolved_value="j.okafor@corp.example",
    ),
]

GOLD_ANSWER = "j.okafor@corp.example"

#: Every distractor value the fixture's tool responses can surface, and why.
#: The Phase-1 generator must register these exhaustively -- an unregistered
#: distractor lands in AMBIGUOUS instead of CONTEXT_POLLUTION.
DISTRACTORS = {
    "SVC-4472": DistractorRecord(
        value="SVC-4472", kind="near_id", introduced_at_hop=0, correct_counterpart="SVC-4471"
    ),
    "DC-SOUTH-04": DistractorRecord(
        value="DC-SOUTH-04",
        kind="cross_branch",
        introduced_at_hop=0,
        correct_counterpart="DC-NORTH-12",
    ),
    "DC-NORTH-13": DistractorRecord(
        value="DC-NORTH-13", kind="near_id", introduced_at_hop=1, correct_counterpart="DC-NORTH-12"
    ),
    "REG-EMEA-7": DistractorRecord(
        value="REG-EMEA-7",
        kind="cross_branch",
        introduced_at_hop=1,
        correct_counterpart="REG-EMEA-3",
    ),
    "m.laurent@corp.example": DistractorRecord(
        value="m.laurent@corp.example",
        kind="cross_branch",
        introduced_at_hop=2,
        correct_counterpart="j.okafor@corp.example",
    ),
    # Stale distractor: the datacenter name is surfaced beside the region so an
    # agent can terminate early on an earlier hop's value (PRD 3.3).
    "north-primary": DistractorRecord(
        value="north-primary", kind="stale", introduced_at_hop=1, correct_counterpart="emea-central"
    ),
}


def _record(entity: Entity) -> dict[str, Any]:
    return {"id": entity.entity_id, "type": entity.entity_type, **entity.attributes}


def _siblings(graph: EntityGraph, entity: Entity, limit: int = 2) -> list[dict[str, Any]]:
    """Plausible-but-wrong records returned alongside the correct one."""
    return [
        _record(e) for e in graph.by_type(entity.entity_type) if e.entity_id != entity.entity_id
    ][:limit]


def _lookup(graph: EntityGraph, entity_id: str, expected_type: str) -> dict[str, Any]:
    entity = graph.entities.get(entity_id)
    if entity is None:
        raise ToolError(f"no entity with id {entity_id!r}")
    if entity.entity_type != expected_type:
        raise ToolError(f"{entity_id!r} is a {entity.entity_type}, not a {expected_type}")
    return {"record": _record(entity), "related": _siblings(graph, entity)}


def _id_param(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {name: {"type": "string", "description": description}},
        "required": [name],
        "additionalProperties": False,
    }


def _tool_lookup_service(graph: EntityGraph, args: dict[str, Any]) -> dict[str, Any]:
    return _lookup(graph, str(args["service_id"]), "service")


def _tool_lookup_datacenter(graph: EntityGraph, args: dict[str, Any]) -> dict[str, Any]:
    return _lookup(graph, str(args["datacenter_id"]), "datacenter")


def _tool_lookup_region(graph: EntityGraph, args: dict[str, Any]) -> dict[str, Any]:
    return _lookup(graph, str(args["region_id"]), "region")


def _tool_lookup_service_legacy(graph: EntityGraph, args: dict[str, Any]) -> dict[str, Any]:
    """Semantically adjacent and schema-plausible, but serves a stale field only.

    Present so tool selection is a real decision rather than a trivial one.
    """
    result = _lookup(graph, str(args["service_id"]), "service")
    record = dict(result["record"])
    record.pop("datacenter", None)
    record["deprecated"] = "true"
    return {"record": record, "related": result["related"]}


def _tool_search_entities(graph: EntityGraph, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("query", "")).upper()
    hits = [_record(e) for e in graph.entities.values() if query in e.entity_id.upper()]
    return {"matches": hits[:5], "match_count": len(hits)}


def _tool_list_regions(graph: EntityGraph, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "regions": [
            {"id": e.entity_id, "name": e.attributes["name"]} for e in graph.by_type("region")
        ]
    }


def _tool_submit_answer(graph: EntityGraph, args: dict[str, Any]) -> dict[str, Any]:
    return {"received": str(args.get("answer", ""))}


def build_registry() -> ToolRegistry:
    """Six lookup tools (N=6), three of which are needed at d=3, plus the terminal tool."""
    return ToolRegistry(
        [
            ToolSpec(
                "lookup_service",
                "Look up a service record by its service ID.",
                _id_param("service_id", "Service ID, e.g. SVC-1234"),
                "resolving",
                _tool_lookup_service,
            ),
            ToolSpec(
                "lookup_datacenter",
                "Look up a datacenter record by its datacenter ID.",
                _id_param("datacenter_id", "Datacenter ID, e.g. DC-NORTH-12"),
                "resolving",
                _tool_lookup_datacenter,
            ),
            ToolSpec(
                "lookup_region",
                "Look up a region record by its region ID.",
                _id_param("region_id", "Region ID, e.g. REG-EMEA-3"),
                "resolving",
                _tool_lookup_region,
            ),
            ToolSpec(
                "lookup_service_legacy",
                "Deprecated service lookup. Returns the legacy service record.",
                _id_param("service_id", "Service ID, e.g. SVC-1234"),
                "resolving",
                _tool_lookup_service_legacy,
            ),
            ToolSpec(
                "search_entities",
                "Free-text search across all entity IDs.",
                _id_param("query", "Substring to search for in entity IDs"),
                "exploratory",
                _tool_search_entities,
            ),
            ToolSpec(
                "list_regions",
                "List every region with its ID and name.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                "exploratory",
                _tool_list_regions,
            ),
            ToolSpec(
                "submit_answer",
                "Submit the final answer. Provide the exact value, with no surrounding prose.",
                _id_param("answer", "The exact terminal attribute value"),
                "terminal",
                _tool_submit_answer,
            ),
        ]
    )


def build_task(task_seed: int) -> TaskInstance:
    """The fixture task. ``task_seed`` is recorded but does not vary the world."""
    registry = build_registry()
    return TaskInstance(
        schema_version=SCHEMA_VERSION,
        task_id="smoke-d3-0000",
        task_seed=task_seed,
        depth=3,
        question=QUESTION,
        gold_trace=GOLD_TRACE,
        gold_answer=GOLD_ANSWER,
        graph=GRAPH,
        distractor_registry=DISTRACTORS,
        exposed_tools=registry.exposed_names(),
    )
