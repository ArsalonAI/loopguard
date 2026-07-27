"""Tool specification and dispatch.

Tools are **pure deterministic functions over a generated graph** -- no network,
no clock, no randomness. Replayability depends on it, and so does the resolver's
ability to attribute a failure to the agent rather than to the environment.

Tagging (TRD 7.2) is what keeps exploration from being scored as misuse:

* ``resolving``   -- advances the gold-hop pointer; a wrong one is divergence
* ``exploratory`` -- does not advance the pointer and does not count as
  divergence, up to ``exploration_budget_per_hop``
* ``terminal``    -- ``submit_answer``; excluded from the exposed tool count N
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loopguard.schemas.config import ToolPolicy, ToolTag
from loopguard.schemas.task import EntityGraph

#: A tool body: (graph, parsed arguments) -> JSON-serializable response.
ToolFn = Callable[[EntityGraph, dict[str, Any]], dict[str, Any]]


class ToolError(Exception):
    """Raised by a tool body for an argument the schema allows but the graph rejects.

    Surfaced to the agent as a normal tool response so the episode continues --
    the resolver, not the runner, decides whether it was TOOL_MISUSE.
    """


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    tag: ToolTag
    fn: ToolFn

    def to_wire(self) -> dict[str, Any]:
        """Render as an OpenAI-compatible ``tools`` entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """The tool set exposed for one episode.

    PRD 3.2: the same ``N`` tools are exposed at every depth. Depth changes how
    many are *needed*, never how many are *available*. :meth:`exposed_names`
    excludes the terminal tool, so ``N`` counts lookup surface only and a change
    to the answer mechanism cannot masquerade as a change to tool-set size.
    """

    def __init__(self, specs: list[ToolSpec]) -> None:
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate tool names: {names}")
        terminal = [s.name for s in specs if s.tag == "terminal"]
        if len(terminal) != 1:
            raise ValueError(f"expected exactly one terminal tool, got {terminal}")
        self._specs = {s.name: s for s in specs}
        self.terminal_tool = terminal[0]

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __getitem__(self, name: str) -> ToolSpec:
        return self._specs[name]

    @property
    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def exposed_names(self) -> list[str]:
        """Tool-set size N: everything the agent chooses between, minus the terminal tool."""
        return [s.name for s in self._specs.values() if s.tag != "terminal"]

    def to_wire(self) -> list[dict[str, Any]]:
        return [s.to_wire() for s in self._specs.values()]

    def validate_against_policy(self, policy: ToolPolicy) -> None:
        """Every exposed tool must be tagged in config, and vice versa.

        An untagged tool would silently default to some behavior in the resolver,
        and the exploration allowance is a frozen researcher degree of freedom --
        it must not be settable by omission.
        """
        registry_names = set(self._specs)
        policy_names = set(policy.tags)
        if missing := registry_names - policy_names:
            raise ValueError(f"tools not tagged in config: {sorted(missing)}")
        if extra := policy_names - registry_names:
            raise ValueError(f"config tags tools not in the registry: {sorted(extra)}")
        for name, tag in policy.tags.items():
            if self._specs[name].tag != tag:
                raise ValueError(
                    f"tool {name!r} is tagged {self._specs[name].tag!r} in code "
                    f"but {tag!r} in config"
                )

    def call(self, name: str, arguments: dict[str, Any], graph: EntityGraph) -> dict[str, Any]:
        if name not in self._specs:
            return {"error": "unknown_tool", "message": f"no tool named {name!r}"}
        try:
            return self._specs[name].fn(graph, arguments)
        except ToolError as exc:
            return {"error": "tool_error", "message": str(exc)}


__all__ = ["ToolError", "ToolFn", "ToolRegistry", "ToolSpec"]
