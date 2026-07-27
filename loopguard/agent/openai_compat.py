"""OpenAI-compatible provider client.

"OpenAI-compatible" names a *wire format* -- ``POST /v1/chat/completions`` with
a ``messages`` array, a ``tools`` array, and a ``model`` string. It is the de
facto standard for serving open-weight models, spoken by Together, Groq,
Fireworks, vLLM, and Ollama. The ``openai`` package is used here purely as a
well-maintained HTTP client for that format, pointed at an open-weight
provider's ``base_url``. No OpenAI model, key, endpoint, or service is involved.
"""

from __future__ import annotations

import json
import time
from typing import Any

from loopguard.agent.provider import CompletionResult, ProviderError
from loopguard.agent.retry import RetryBudgetExhausted, with_retries
from loopguard.hashing import provider_seed
from loopguard.schemas.config import DecodingParams, RetryConfig
from loopguard.schemas.trace import ToolCall

#: Transient by nature: rate limits, upstream capacity, and connection blips.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    if isinstance(exc, RateLimitError | APIConnectionError | APITimeoutError):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status in _RETRYABLE_STATUS


class OpenAICompatClient:
    """One client per (provider, model). Thread-safe enough for the run pool."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        retry: RetryConfig,
        timeout_s: float = 120.0,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.base_url = base_url
        self._retry = retry
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        decoding: DecodingParams,
        seed: int,
    ) -> CompletionResult:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": decoding.temperature,
            "top_p": decoding.top_p,
            "max_tokens": decoding.max_tokens,
            # A retry reuses this value. A reseeded retry is a different sample.
            "seed": provider_seed(seed),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        started = time.monotonic()

        def _call() -> Any:
            nonlocal started
            started = time.monotonic()  # reset so backoff sleep never enters latency
            return self._client.chat.completions.create(**kwargs)

        try:
            response, attempts = with_retries(
                _call, config=self._retry, seed=seed, is_retryable=_is_retryable
            )
        except RetryBudgetExhausted as exc:
            raise ProviderError(str(exc)) from exc

        latency_ms = int((time.monotonic() - started) * 1000)
        return _to_result(response, latency_ms=latency_ms, attempts=attempts)


def _to_result(response: Any, *, latency_ms: int, attempts: int) -> CompletionResult:
    choice = response.choices[0]
    message = choice.message
    usage = getattr(response, "usage", None)

    tool_calls: list[ToolCall] = []
    for index, raw in enumerate(getattr(message, "tool_calls", None) or []):
        raw_args = raw.function.arguments or ""
        parsed: dict[str, Any] | None = None
        parse_error: str | None = None
        try:
            decoded = json.loads(raw_args) if raw_args.strip() else {}
            if isinstance(decoded, dict):
                parsed = decoded
            else:
                parse_error = f"arguments decoded to {type(decoded).__name__}, expected object"
        except json.JSONDecodeError as exc:
            # Not an error here: malformed arguments are TOOL_MISUSE, a finding,
            # so the raw string is preserved for the resolver rather than raised.
            parse_error = str(exc)
        tool_calls.append(
            ToolCall(
                call_id=getattr(raw, "id", None) or f"call_{index}",
                name=raw.function.name,
                raw_arguments=raw_args,
                parsed_arguments=parsed,
                parse_error=parse_error,
            )
        )

    return CompletionResult(
        content=message.content,
        tool_calls=tool_calls,
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        latency_ms=latency_ms,
        attempt_count=attempts,
        finish_reason=choice.finish_reason,
        model_revision=getattr(response, "model", None),
    )
