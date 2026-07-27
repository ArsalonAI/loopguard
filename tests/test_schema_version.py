"""A component reading an unrecognized schema_version must fail loudly (TRD 3).

Best-effort parsing of a foreign trace produces a plausible wrong number, which
is the worst failure this project can have. These tests pin the failure mode --
including that it is not a ``ValidationError``, because callers catch those and
skip the row.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from loopguard.schemas.base import SchemaVersionError
from loopguard.schemas.trace import EpisodeFooter, EpisodeHeader


def _header_payload(**overrides):
    payload = {
        "record_type": "header",
        "task_id": "d3-0000",
        "task_seed": 1,
        "decoding_seed": 2,
        "repeat_index": 0,
        "depth": 3,
        "arm": "baseline",
        "model_id": "llama-3.3-70b",
        "provider_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "provider": "together",
        "config_hash": "abc",
        "task_hash": "def",
        "started_at": datetime.now(UTC).isoformat(),
    }
    payload.update(overrides)
    return payload


def test_known_version_parses():
    assert EpisodeHeader.model_validate(_header_payload(schema_version="1.0"))


def test_unknown_version_raises_loudly():
    with pytest.raises(SchemaVersionError, match="Refusing to best-effort parse"):
        EpisodeHeader.model_validate(_header_payload(schema_version="2.0"))


def test_schema_version_error_is_not_a_validation_error():
    """Callers routinely catch ValidationError and skip the row. This must not be skippable."""
    with pytest.raises(SchemaVersionError):
        try:
            EpisodeFooter.model_validate(
                {
                    "record_type": "footer",
                    "schema_version": "0.9",
                    "terminal_reason": "final_answer",
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            )
        except ValidationError:  # pragma: no cover -- would mean the guard is bypassable
            pytest.fail("SchemaVersionError was converted into a ValidationError")


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        EpisodeHeader.model_validate(_header_payload(unexpected_field=1))
