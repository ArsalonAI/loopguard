"""Schema versioning, with fail-loud semantics.

TRD 3: every artifact on disk carries ``schema_version``. A component reading an
artifact whose version it does not recognize must fail loudly, never best-effort
parse -- silently misreading a trace produces a plausible wrong number, which is
the worst failure this project can have.

The enforcement lives here rather than at each call site so that "read an
artifact" and "check its version" cannot come apart.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, field_validator

#: Bumped whenever an on-disk artifact shape changes incompatibly.
SCHEMA_VERSION = "1.0"


class SchemaVersionError(RuntimeError):
    """Raised when an artifact declares a schema version we cannot read.

    Deliberately *not* a ``ValueError``: pydantic v2 converts ``ValueError`` and
    ``AssertionError`` raised inside validators into a ``ValidationError``, which
    callers routinely catch and treat as "bad row, skip it". This must not be
    skippable, so it propagates out of validation untouched.
    """


class VersionedModel(BaseModel):
    """Base for every artifact that is written to disk.

    Subclasses may widen :attr:`supported_schema_versions` when they can read
    older payloads. Anything outside that set raises :class:`SchemaVersionError`.
    """

    model_config = ConfigDict(extra="forbid")

    supported_schema_versions: ClassVar[frozenset[str]] = frozenset({SCHEMA_VERSION})

    schema_version: str = SCHEMA_VERSION

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, value: str) -> str:
        if value not in cls.supported_schema_versions:
            supported = ", ".join(sorted(cls.supported_schema_versions))
            raise SchemaVersionError(
                f"{cls.__name__} cannot read schema_version={value!r}; "
                f"this build supports [{supported}]. Refusing to best-effort parse."
            )
        return value

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-safe dict (paths, enums, and datetimes rendered as strings)."""
        return self.model_dump(mode="json")
