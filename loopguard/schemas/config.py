"""Configuration model.

The semantic/runtime split (TRD 5.1) is structural, not a filter applied at hash
time: ``config_hash`` is computed over :class:`SemanticConfig` and nothing else,
so a field's placement in the model *is* the decision about whether it affects
results.

Three tiers, not two
--------------------
TRD 5.1 writes ``task_seed = f(config_hash, depth, task_index)``. Taken
literally that conflicts with the PRD invariant that the mitigation arm reuses
the baseline's task seeds: a mitigation changes the prompt, the prompt is
semantic, so ``config_hash`` changes and the seeds move -- destroying pairing,
which is the thing the invariant exists to protect.

Resolved by splitting the semantic tier:

* :class:`TaskConfig` -- the only inputs that determine *which tasks exist*.
  Hashed into ``task_hash``; task seeds derive from that.
* the rest of :class:`SemanticConfig` -- prompt, arm, models, decoding, loop
  caps, grading policy. Hashed into ``config_hash`` for provenance.
* :class:`RuntimeConfig` -- hashed into neither.

Baseline and mitigation therefore share ``task_hash`` (identical task set, paired
comparison) while differing in ``config_hash`` (the change is visible in a
manifest diff, not silent). Both are recorded in the manifest and in every trace
header. See ``docs/deviations.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from loopguard.schemas.base import VersionedModel

ToolTag = Literal["exploratory", "resolving", "terminal"]
Arm = Literal["baseline", "mitigation"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Tier 1: task-determining (hashed into task_hash)
# ---------------------------------------------------------------------------


class TaskConfig(StrictModel):
    """Everything that determines the generated task set -- and nothing else.

    Adding a field here moves every task seed. Adding a field that does *not*
    change what the generator emits belongs in :class:`SemanticConfig` instead.
    """

    template: str = Field(description="Task family identifier, e.g. chained_entity_lookup_v1")
    depths: list[int] = Field(description="Loop depths under test; the sole independent variable")
    tasks_per_depth: int = Field(description="Distinct generator seeds per (depth, model) cell")
    repeats: int = Field(description="Decoding seeds per task")
    calibration_lock_hash: str | None = Field(
        default=None,
        description=(
            "Hash of configs/calibration.lock.json. None only before Phase 1 exit; "
            "`loopguard run` refuses to start on a mismatch (TRD 5.2)."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> TaskConfig:
        if not self.depths:
            raise ValueError("task.depths must not be empty")
        if sorted(set(self.depths)) != self.depths:
            raise ValueError("task.depths must be sorted and unique (it is hashed verbatim)")
        if self.tasks_per_depth < 1 or self.repeats < 1:
            raise ValueError("task.tasks_per_depth and task.repeats must be >= 1")
        return self

    @property
    def d_max(self) -> int:
        return max(self.depths)


# ---------------------------------------------------------------------------
# Tier 2: semantic but not task-determining (hashed into config_hash)
# ---------------------------------------------------------------------------


class ProviderConfig(StrictModel):
    """OpenAI-compatible endpoint for the models under test.

    ``base_url`` is semantic: a different serving stack is a different
    measurement, and that should be visible in the config hash.
    """

    name: str = Field(description="together | groq | fireworks | vllm | ollama")
    base_url: str


class ModelSpec(StrictModel):
    id: str = Field(description="Short stable label used in filenames and reports")
    provider_model: str = Field(description="Exact provider model string")


class DecodingParams(StrictModel):
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    seed_policy: Literal["derived"] = Field(
        default="derived",
        description="Seeds derive from task_seed|repeat_index (TRD 5.1); retries reuse them.",
    )


class LoopConfig(StrictModel):
    """Hard caps, both recorded as ``terminal_reason`` (TRD 6.2)."""

    max_steps: int | None = Field(
        default=None,
        description="None resolves to 2*d_max+4 at load time; the resolved value is hashed.",
    )
    max_completion_tokens: int = Field(description="Per-episode completion-token cap")


class ToolPolicy(StrictModel):
    """Tool tagging and the exploration allowance (TRD 7.2).

    ``exploration_budget_per_hop`` is a genuine researcher degree of freedom: it
    moves the TOOL_MISUSE rate. It is frozen with the calibration lock and
    sensitivity-checked at +/-1 in Phase 2.
    """

    exploration_budget_per_hop: int = 2
    tags: dict[str, ToolTag] = Field(
        default_factory=dict,
        description="tool name -> tag. Untagged tools are rejected at load time.",
    )


class PromptConfig(StrictModel):
    """System prompt, held byte-identical across conditions except the mitigation arm.

    ``text`` is populated from ``path`` at load time and is what gets hashed, so
    an edit to the prompt file changes ``config_hash`` even though the path did
    not move.
    """

    version: str
    path: Path
    text: str = Field(default="", description="Populated at load time from `path`; do not set.")


class SemanticConfig(StrictModel):
    """Everything that can change a result. Hashed in full into ``config_hash``."""

    task: TaskConfig
    arm: Arm
    provider: ProviderConfig
    models: list[ModelSpec]
    decoding: DecodingParams
    loop: LoopConfig
    tools: ToolPolicy
    prompt: PromptConfig

    @model_validator(mode="after")
    def _resolve(self) -> SemanticConfig:
        if self.loop.max_steps is None:
            # TRD 6.2: headroom for exploration without unbounded looping.
            self.loop.max_steps = 2 * self.task.d_max + 4
        if not self.models:
            raise ValueError("semantic.models must not be empty")
        ids = [m.id for m in self.models]
        if len(set(ids)) != len(ids):
            raise ValueError(f"semantic.models ids must be unique, got {ids}")
        return self


# ---------------------------------------------------------------------------
# Tier 3: runtime (hashed into nothing)
# ---------------------------------------------------------------------------


class RateLimits(StrictModel):
    max_concurrency: int = 4
    requests_per_minute: int = 60
    tokens_per_minute: int = 200_000


class RetryConfig(StrictModel):
    """Bounded retry budget per episode (TRD 6.3).

    Exhausting it marks the episode ``infrastructure_failure``, which is excluded
    from failure-rate denominators and counted separately.
    """

    max_attempts: int = 4
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0


class RuntimeConfig(StrictModel):
    """Deliberately excluded from every hash.

    Hashing any of this would change task seeds when parallelism changes, which
    silently breaks baseline<->mitigation pairing (TRD 5.1).
    """

    out_dir: Path = Path("runs")
    api_key_env: str = Field(description="Env var holding the provider key; never the key itself")
    rate_limits: RateLimits = RateLimits()
    retry: RetryConfig = RetryConfig()
    max_spend_usd: float = Field(default=25.0, description="Hard abort ceiling (TRD 6.4)")
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


class LoopGuardConfig(VersionedModel):
    semantic: SemanticConfig
    runtime: RuntimeConfig

    def model_by_id(self, model_id: str) -> ModelSpec:
        for spec in self.semantic.models:
            if spec.id == model_id:
                return spec
        known = [m.id for m in self.semantic.models]
        raise KeyError(f"unknown model id {model_id!r}; configured: {known}")
