"""Candidate provider registry for the Phase 0 smoke test.

TRD 14 leaves the provider open until Phase 0, to be "chosen on tool-calling
fidelity in the smoke test, then rate limits". This table exists so that choice
is made by running against each candidate rather than by editing YAML three
times. Once chosen, ``configs/baseline.yaml`` pins one provider and this module
stops being load-bearing.

**Model strings are candidates, not facts.** Providers rename and retire model
strings; ``loopguard smoke`` verifies a given string resolves before any headline
run depends on it. Override with ``--llama-model`` / ``--qwen-model``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCandidate:
    name: str
    base_url: str
    api_key_env: str
    llama_model: str
    qwen_model: str


CANDIDATES: dict[str, ProviderCandidate] = {
    "together": ProviderCandidate(
        name="together",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        llama_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        qwen_model="Qwen/Qwen3-235B-A22B-fp8-tput",
    ),
    "groq": ProviderCandidate(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        llama_model="llama-3.3-70b-versatile",
        qwen_model="qwen/qwen3-32b",
    ),
    "fireworks": ProviderCandidate(
        name="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        llama_model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        qwen_model="accounts/fireworks/models/qwen3-235b-a22b",
    ),
}


def get(name: str) -> ProviderCandidate:
    try:
        return CANDIDATES[name]
    except KeyError:
        raise KeyError(f"unknown provider {name!r}; known: {sorted(CANDIDATES)}") from None
