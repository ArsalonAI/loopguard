"""Run manifest (TRD 3.3) -- ``runs/<run_id>/manifest.json``.

Written at run start, finalized at run end. This is the reproducibility record
(PRD 3.5) and the input `diff` uses to decide whether two runs can be compared
paired or only unpaired.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from loopguard.schemas.base import VersionedModel
from loopguard.schemas.config import Arm, DecodingParams


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelPin(StrictModel):
    id: str
    provider_model: str
    served_revision: str | None = Field(
        default=None,
        description="Provider-reported revision when exposed; None when the provider hides it",
    )


class JudgePin(StrictModel):
    """The judge is provider-independent from the harness (TRD 8).

    Recording ``provider`` alongside the model makes a judge swap visible in a
    manifest diff rather than silent.
    """

    provider: str
    base_url: str
    model: str
    prompt_version: str
    prompt_hash: str


class RunManifest(VersionedModel):
    run_id: str
    git_sha: str
    #: A dirty tree on a headline run is a reproducibility hole; recorded, not blocked.
    git_dirty: bool
    config_hash: str
    task_hash: str
    config_resolved: dict[str, Any] = Field(description="Fully-resolved config, inlined")
    calibration_lock_hash: str | None = None
    provider: str
    models: list[ModelPin]
    decoding: DecodingParams
    arm: Arm
    #: Enables paired diff detection (TRD 10).
    task_seeds: list[int] = Field(default_factory=list)
    judge: JudgePin | None = None
    started_at: datetime
    completed_at: datetime | None = None
    episode_count: int = 0
    #: Counted separately and excluded from failure-rate denominators (TRD 6.3).
    infrastructure_failure_count: int = 0
    cost_estimate_usd: float | None = None
