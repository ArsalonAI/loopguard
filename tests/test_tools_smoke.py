"""The Phase-0 fixture world.

Phase 1's generator inherits these invariants as property tests; asserting them
on the fixture now means the contract is exercised before the generator exists.
"""

from __future__ import annotations

import itertools

import pytest

from loopguard.tools import ToolError, ToolRegistry, ToolSpec
from loopguard.tools.smoke import DISTRACTORS, GOLD_ANSWER, GOLD_TRACE, GRAPH, build_registry


def test_gold_chain_resolves_to_the_gold_answer():
    registry = build_registry()
    for hop in GOLD_TRACE:
        response = registry.call(hop.tool_name, dict(hop.arguments), GRAPH)
        assert response["record"]["id"] == hop.returned_entity_id
        assert hop.resolved_value in response["record"].values()
    assert GOLD_TRACE[-1].resolved_value == GOLD_ANSWER


def test_gold_hops_chain_into_one_another():
    for previous, current in itertools.pairwise(GOLD_TRACE):
        assert previous.resolved_value in current.arguments.values()


def test_every_distractor_surfaced_by_a_tool_is_registered():
    """An unregistered distractor lands in AMBIGUOUS instead of CONTEXT_POLLUTION."""
    registry = build_registry()
    surfaced: set[str] = set()
    for hop in GOLD_TRACE:
        response = registry.call(hop.tool_name, dict(hop.arguments), GRAPH)
        for related in response["related"]:
            surfaced.update(str(v) for v in related.values())

    gold_values = {h.resolved_value for h in GOLD_TRACE} | {
        h.returned_entity_id for h in GOLD_TRACE
    }
    plausible = {v for v in surfaced if any(v.startswith(p) for p in ("SVC-", "DC-", "REG-"))}
    plausible |= {v for v in surfaced if "@" in v}
    unregistered = plausible - set(DISTRACTORS) - gold_values
    assert not unregistered, f"unregistered distractor values: {sorted(unregistered)}"


def test_tools_are_pure_functions_of_the_graph():
    registry = build_registry()
    first = registry.call("lookup_datacenter", {"datacenter_id": "DC-NORTH-12"}, GRAPH)
    second = registry.call("lookup_datacenter", {"datacenter_id": "DC-NORTH-12"}, GRAPH)
    assert first == second


def test_wrong_entity_type_is_rejected():
    with pytest.raises(ToolError, match="not a datacenter"):
        from loopguard.tools.smoke import _tool_lookup_datacenter

        _tool_lookup_datacenter(GRAPH, {"datacenter_id": "SVC-4471"})


def test_wire_format_is_openai_compatible():
    entry = build_registry()["lookup_service"].to_wire()
    assert entry["type"] == "function"
    assert entry["function"]["name"] == "lookup_service"
    assert entry["function"]["parameters"]["required"] == ["service_id"]


def test_registry_requires_exactly_one_terminal_tool():
    spec = ToolSpec("t", "d", {"type": "object", "properties": {}}, "resolving", lambda g, a: {})
    with pytest.raises(ValueError, match="terminal tool"):
        ToolRegistry([spec])


def test_registry_rejects_a_tag_mismatch_between_code_and_config(smoke_config):
    smoke_config.semantic.tools.tags["search_entities"] = "resolving"
    with pytest.raises(ValueError, match="tagged"):
        build_registry().validate_against_policy(smoke_config.semantic.tools)
