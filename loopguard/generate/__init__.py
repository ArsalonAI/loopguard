"""Entity graph, chain, distractors, gold trace.

**Phase 1.** The contract this module must satisfy is already fixed in
:mod:`loopguard.schemas.task`: it emits a gold *trace*, not a gold answer, and it
registers **every** distractor value it surfaces. Both are load-bearing --
without per-hop ground truth the taxonomy collapses onto an LLM judge, and
without a complete distractor registry CONTEXT_POLLUTION cannot be separated
from AMBIGUOUS.
"""
