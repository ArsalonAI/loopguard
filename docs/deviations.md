# Documented deviations from the PRD/TRD

The design's validity depends on a small set of invariants. Where implementation
forced a choice the specification did not make — or made differently — it is
recorded here rather than absorbed quietly. Each entry states what changed, why,
and what it would cost to reverse.

---

## D1 — Task seeds derive from a three-tier hash, not from `config_hash`

**Phase:** 0 · **Affects:** TRD §5.1

TRD §5.1 specifies `task_seed = f(config_hash, depth, task_index)`. Taken
literally this contradicts the PRD invariant that *the mitigation arm reuses the
baseline's task seeds*: a mitigation changes the system prompt, the prompt is
semantic, so `config_hash` moves and every task seed moves with it. The paired
comparison — materially tighter than independent samples at n=40/cell — would be
destroyed by the very mechanism meant to make the change visible.

Resolved by splitting the semantic tier in the config model itself
(`loopguard/schemas/config.py`):

| Tier | Contents | Hash |
|---|---|---|
| `semantic.task` | template, depths, tasks-per-depth, repeats, calibration-lock hash | `task_hash` → task seeds |
| `semantic.*` (rest) | arm, provider, models, decoding, loop caps, tool policy, prompt | `config_hash` → provenance |
| `runtime.*` | out dir, concurrency, rate limits, retry budget, spend cap, log level | neither |

Baseline and mitigation therefore share `task_hash` (identical task set, paired
comparison intact) and differ in `config_hash` (the change is visible in a
manifest diff, not silent). Both are recorded in the run manifest and in every
trace header, so `loopguard diff` can detect pairing from either.

`tests/test_hashing.py::test_mitigation_arm_keeps_task_seeds_but_changes_config_hash`
pins this behaviour.

**Reversal cost:** cheap now, expensive after Phase 2 — task seeds appear in
every trace header and manifest.

---

## D2 — The final answer is submitted through a tool, not free text

**Phase:** 0 · **Affects:** PRD §3.2 (tool-set size), TRD §7.3 (answer comparison)

TRD §7.3 requires answer comparison to be exact-match against gold, normalized
for case/whitespace/punctuation but **never fuzzy-matched** — fuzzy matching
hands the correctness decision to the resolver. A free-text final answer makes
that requirement unworkable in practice: `"The escalation contact is
j.okafor@corp.example."` is a correct answer that exact-match rejects, and every
workaround for it is a fuzzy match wearing a different name.

The harness therefore exposes a `submit_answer(answer: str)` tool, tagged
`terminal`. It is a third tool tag alongside `exploratory` and `resolving`, and:

- it is **excluded from the exposed tool count `N`**, so a change to the answer
  mechanism cannot masquerade as a change to tool-set size;
- it never advances the gold-hop pointer;
- it is exposed identically at every depth and in both arms.

**Consequence to watch:** an episode can now fail by never calling
`submit_answer` despite having the right value in context. That is
`NON_TERMINATION` by the taxonomy, and it is a genuine failure mode of tool-using
agents rather than an artifact — but if its rate turns out to differ sharply
between the two models, it is confounded with tool-calling fidelity and must be
reported separately.

---

## D3 — A fixed nudge is appended when the model replies with prose

**Phase:** 0 · **Affects:** PRD §3.2 (byte-identical prompt)

When a model returns prose with no tool call, the loop appends one constant user
message (`loopguard.agent.loop.NUDGE`) and continues. Without it, a single
conversational turn would end the episode and be graded `NON_TERMINATION`,
measuring chattiness rather than reasoning depth.

The nudge lives **in code, not in the prompt config**, deliberately: it must be
byte-identical in every condition including the mitigation arm, and a mitigation
is permitted to rewrite the system prompt. Keeping it out of the prompt file
means a prompt edit cannot silently change the harness's control flow and
confound the comparison.

It consumes step budget like any other turn, uniformly across depths and arms.

---

## D4 — `.python-version` is tracked

**Phase:** 0 · **Affects:** repo conventions

`.gitignore` excluded it. TRD §1 makes reproducibility of the analysis
environment part of the deliverable, and `uv.lock` pins packages but not the
interpreter; without the pin, a fresh clone resolves to whatever Python is
newest. Pinned to 3.11, the floor stated in TRD §1, so 3.12+ syntax cannot creep
in unnoticed.

---

## Open — to be recorded when they land

- **Provider choice** (Phase 0). `loopguard smoke --provider {together|groq|fireworks}`
  produces the tool-calling fidelity comparison; the winner and the rejected
  candidates' failure modes belong here.
- **Any per-model prompt adaptation** (PRD §10 risk row). None so far. If a model
  cannot be made to call tools reliably under the shared prompt, the adaptation
  is a deviation from "byte-identical prompt" and is recorded here with the
  evidence that forced it.
- **Tool-set size `N`** (Phase 1). Phase 0's fixture exposes 6; TRD §14 expects
  8–12 after calibration.
