# LoopGuard — Technical Requirements Document

**Status:** Draft v1
**Companion to:** `prd.md` (source of truth for scope, experimental design, and success criteria)
**Last updated:** 2026-07-27

This document decides what the PRD deliberately left open and specifies the mechanics that the PRD's validity claims depend on. Where the two conflict, `prd.md` wins on *what* and *why*; this document wins on *how*.

---

## 1. Stack decisions

| Layer | Choice | Rationale |
|---|---|---|
| Harness language | **Python 3.11+** | The statistics (Wilson, McNemar, trend tests), tokenizer access, and provider SDKs all live here. Nothing else is close for the analysis half. |
| Package manager | **uv** | Lockfile-based, fast, single tool for venv + deps. Reproducibility of the analysis environment is part of the deliverable. |
| Config | **YAML** on disk → **Pydantic v2** models in memory | Validation at load time, JSON-Schema export for free, and a canonical serialization for config hashing (§5.1). |
| Provider access | **OpenAI-compatible wire format**, via the `openai` package as HTTP client | The wire format is the de facto standard for open-weight serving; all three candidate providers speak it. One client, swap `base_url` + model string. See the note below. |
| Stats | **scipy** + **statsmodels** | Wilson intervals, McNemar exact, Cochran–Armitage trend. Do not hand-roll. |
| Test runner | **pytest** | Plus `pytest-xdist` for the generator property tests. |
| Lint / format | **ruff** (lint + format) | Single tool, no black/isort split. |
| Types | **mypy**, strict on `loopguard/schemas` and `loopguard/grading` | The schemas and the resolver are where a silent type error becomes a wrong number. Strictness elsewhere is optional. |
| Dashboard | **Vite + React + TypeScript**, static export | Per PRD §9. No backend. |
| Charts | **Recharts** | CI bands and stacked bars without a d3 build-out. |
| Dashboard package manager | **pnpm** | |

**On the `openai` package — no OpenAI model is involved.** "OpenAI-compatible" names a *wire format*: `POST /v1/chat/completions` carrying a `messages` array, a `tools` array, and a `model` string. It has become the de facto standard for serving open-weight models, spoken by Together, Groq, and Fireworks as well as local runtimes like vLLM and Ollama. The `openai` package is used here purely as a well-maintained HTTP client for that format — the way `requests` is used to call a non-Python server. No OpenAI model, key, endpoint, or service is part of this project, and no data reaches OpenAI.

```python
client = OpenAI(base_url="https://api.together.xyz/v1", api_key=TOGETHER_API_KEY)
client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", ...)
```

**Every model under test is open-weight** (PRD §2 non-goals: closed/frontier models are out of scope). The single exception is the LLM judge, which is not under test and may be a closed model — see §8 and §14.

**Reversibility note.** The Python/TypeScript split is cheap to change now and expensive after Phase 2, once traces exist in a format the analysis code assumes. Raise objections before Phase 0 exits.

---

## 2. Repository layout

```
loopguard/
├── prd.md, trd.md, CLAUDE.md
├── pyproject.toml, uv.lock
├── configs/
│   ├── baseline.yaml
│   ├── mitigation.yaml            # written at Phase 3
│   ├── policy.yaml                # CI gate thresholds
│   └── calibration.lock.json      # FROZEN at Phase 1 — see §5.2
├── loopguard/
│   ├── schemas/                   # Pydantic models; the contract between all stages
│   ├── generate/                  # entity graph, chain, distractors, gold trace
│   ├── tools/                     # pure functions over a generated graph
│   ├── agent/                     # loop, provider client, rate limiting, retries
│   ├── grading/                   # mechanical resolver, judge, calibration metrics
│   ├── stats/                     # intervals, paired tests, trend tests
│   ├── report/                    # tables + JSON artifacts for the dashboard
│   └── cli/                       # run · grade · report · diff · gate
├── fixtures/
│   └── judge_calibration.jsonl    # hand labels — TRACKED (see CLAUDE.md)
├── tests/
├── dashboard/                     # Vite app; reads run artifacts as JSON
└── runs/                          # gitignored; per-run output (§4)
```

---

## 3. Data model

All schemas are Pydantic models with a `schema_version` field. Every artifact on disk carries it. A resolver reading a trace whose `schema_version` it does not recognize must **fail loudly**, never best-effort parse — silently misreading a trace produces a plausible wrong number, which is the worst failure this project can have.

### 3.1 `TaskInstance` — generator output

```python
class TaskInstance:
    schema_version: str
    task_id: str                  # deterministic: f"d{depth}-{task_index:04d}"
    task_seed: int                # derived per §5.1
    depth: int
    question: str
    gold_trace: list[GoldHop]
    gold_answer: str
    graph: EntityGraph            # the full generated world, for tool binding
    distractor_registry: dict[str, DistractorRecord]   # value -> why it's a distractor
    exposed_tools: list[str]      # length N, constant across depths

class GoldHop:
    hop_index: int                # 0-based
    tool_name: str
    arguments: dict[str, str]
    returned_entity_id: str
    resolved_value: str           # feeds the next hop, or is the final answer

class DistractorRecord:
    value: str
    kind: Literal["near_id", "cross_branch", "stale"]
    introduced_at_hop: int        # which tool response first surfaced it
    correct_counterpart: str
```

`distractor_registry` is the key to attribution. Without it, `CONTEXT_POLLUTION` cannot be separated from `AMBIGUOUS`. The generator must register **every** distractor value it emits, including ones it emits incidentally.

### 3.2 `EpisodeTrace` — JSONL, append-only

One file per episode: `runs/<run_id>/traces/<task_id>-r<repeat>.jsonl`.

- **Line 1** — `EpisodeHeader`: `task_id`, `task_seed`, `decoding_seed`, `repeat_index`, `model`, `provider`, `config_hash`, `depth`, `started_at`.
- **Lines 2..n** — `StepRecord`, one per agent step.
- **Final line** — `EpisodeFooter`: `terminal_reason`, `final_answer`, token totals, `wall_clock_ms`, `completed_at`.

```python
class StepRecord:
    step_index: int
    role: Literal["assistant", "tool"]
    content: str | None
    tool_call: ToolCall | None         # name + raw arguments string + parsed
    tool_response: dict | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int | None             # successful attempt only
    attempt_count: int                 # >1 means retries occurred (§6.3)
```

Append-only with a terminal footer gives crash-resumability for free: an episode is complete iff its footer line parses. See §6.4.

### 3.3 `RunManifest` — `runs/<run_id>/manifest.json`

Written at run start, finalized at run end. Required for reproducibility (PRD §3.5) and for `diff` to detect pairing (§8).

```python
class RunManifest:
    schema_version: str
    run_id: str
    git_sha: str
    git_dirty: bool               # a dirty tree on a headline run is a reproducibility hole
    config_hash: str              # §5.1
    config_resolved: dict         # fully-resolved config, inlined
    calibration_lock_hash: str
    provider: str
    models: list[ModelPin]        # provider model string + served revision if exposed
    decoding: DecodingParams
    arm: Literal["baseline", "mitigation"]
    task_seeds: list[int]         # enables paired diff detection
    judge: JudgePin | None        # model, prompt version, prompt hash
    started_at, completed_at, episode_count, cost_estimate_usd
```

### 3.4 `EpisodeGrade` — `runs/<run_id>/grades.jsonl`

```python
class EpisodeGrade:
    task_id: str
    repeat_index: int
    category: FailureCategory     # per PRD §4.1, plus UNRESOLVED
    first_error_hop: int | None   # None iff CORRECT
    resolution_source: Literal["mechanical", "judge", "human"]
    evidence: dict                # what the resolver matched on — see §7.4
    judge_confidence: float | None
```

`evidence` is non-optional in spirit: every mechanical label must record *why*, so a disputed label can be audited without re-running the resolver.

---

## 4. Run directory layout

```
runs/<run_id>/
├── manifest.json
├── tasks/<task_id>.json        # TaskInstance, so grading never re-generates
├── traces/<task_id>-r<n>.jsonl
├── grades.jsonl
├── report.json                 # dashboard input (§9)
└── logs/
```

`tasks/` is materialized rather than regenerated on demand. Regenerating at grade time would make grading depend on generator determinism holding across code versions — a dependency that will eventually break quietly.

---

## 5. Determinism and configuration

### 5.1 Seed derivation and config hashing

All seeds derive from a single root, so a config hash fully determines the task set:

```
config_hash  = blake2b(canonical_json(config_semantic), digest_size=16).hex()
task_seed    = int.from_bytes(blake2b(f"{config_hash}|{depth}|{task_index}").digest()[:8])
decoding_seed= int.from_bytes(blake2b(f"{task_seed}|{repeat_index}").digest()[:8])
```

**`config_semantic` excludes runtime-only fields** — output directory, concurrency, rate limits, retry budget, log level. If these were hashed, running the same experiment with different parallelism would produce a different `config_hash` and different task seeds, silently breaking pairing between the baseline and mitigation arms. The split between semantic and runtime config must be explicit in the Pydantic model (two nested sections), not a filter applied at hash time.

### 5.2 `calibration.lock.json`

Written once at Phase 1 exit. Contains the frozen distractor knobs and its own hash. Every headline run records `calibration_lock_hash` in its manifest, and `loopguard run` **refuses to start** if the lock file's hash does not match the value recorded in the config. Re-tuning after seeing results (PRD §3.4's explicit hazard) then requires deliberately editing a lock file and a config together, which is hard to do by accident.

### 5.3 The payload length-matching problem

PRD §3.2 requires tool responses length-matched to ±10% tokens across hops and depths. **Llama 3.3 and Qwen3 use different tokenizers**, so "±10% tokens" is not a single well-defined constraint.

Resolution:
- The generator length-matches against a **reference tokenizer** (the Llama 3.3 tokenizer, via `tokenizers`), padding a neutral filler field to hit the target.
- At generation time it **asserts the constraint also holds within ±15% under the Qwen3 tokenizer**, and fails generation if not.
- The realized per-hop token counts under *both* tokenizers are recorded in `TaskInstance`, and reported in the method panel.

Two tolerances rather than one is a compromise, and it is stated as such in the writeup rather than presented as exact matching.

---

## 6. Agent loop and provider layer

### 6.1 Interface

```python
class ProviderClient(Protocol):
    def complete(self, messages, tools, decoding: DecodingParams,
                 seed: int) -> CompletionResult: ...
```

One implementation (OpenAI-compatible) covers all three candidate providers. A `ReplayClient` implementation replays a recorded trace without network access — used by resolver tests so grading logic is testable without spending tokens.

### 6.2 Loop termination

Hard caps, both recorded as `terminal_reason`: `max_steps = 2 * d_max + 4` (headroom for exploration without unbounded looping) and a per-episode completion-token cap. Both are semantic config and therefore hashed.

### 6.3 Rate limiting, retries, and their effect on measurement

Concurrency semaphore plus a token-bucket limiter sized to the chosen provider's published limits. Retries on 429 and 5xx with exponential backoff and full jitter, bounded retry budget per episode.

Three requirements that exist specifically to keep retries from corrupting results:

1. A retry **reuses the same `decoding_seed`**. Changing it turns a retry into a different sample.
2. `latency_ms` records the **successful attempt only**; `attempt_count` records the rest. Including backoff sleep in latency would make the cost comparison in PRD §7 a measure of provider load, not of the mitigation.
3. Exhausting the retry budget marks the episode `terminal_reason="infrastructure_failure"`. These episodes are **excluded from failure-rate denominators and counted separately.** Grading an infrastructure timeout as an agent failure is the single easiest way to fabricate a depth effect — deep episodes take longer and so fail infrastructurally more often. The report must surface the infrastructure-failure count per cell; if it is not roughly flat across depths, the depth curve is suspect.

### 6.4 Resumability and cost control

`loopguard run` skips any episode whose trace file has a parseable footer. A 1,200-episode matrix cannot be restarted from scratch on every interruption.

Before execution, `run` prints an episode count and a cost estimate and requires `--yes` for non-interactive execution. A hard `max_spend_usd` in runtime config aborts the run when the running token total crosses it.

---

## 7. Mechanical resolver

This is the most correctness-critical component in the repo. Its output *is* the result.

### 7.1 Provenance set

Maintained incrementally: `P_i` = the set of all string values appearing in tool responses returned before step `i`, plus values appearing in the question. Values are extracted by walking the response JSON and collecting all leaf strings, plus regex-extracted entity IDs from free-text fields.

`P` is what distinguishes `HALLUCINATION` (value ∉ P — invented) from `CONTEXT_POLLUTION` (value ∈ P but wrong). Getting extraction wrong in either direction moves episodes between the two headline categories, so extraction has its own unit tests against fixtures.

### 7.2 Exploration allowance

A strict "any non-gold tool call is `TOOL_MISUSE`" rule would be wrong and would inflate that category. Agents may legitimately call a broad tool (`search_entities`) before the correct narrow one; that is reasonable behavior, not misuse.

Policy:
- Tools are tagged in config as `exploratory` or `resolving`.
- Calls to `exploratory` tools do not advance the gold-hop pointer and do not count as divergence, up to `exploration_budget` calls per episode (semantic config, default 2 per hop).
- Exceeding the budget without progress is `TOOL_MISUSE`, subcode `unproductive_exploration`.
- A `resolving` call that is wrong for the current hop is divergence, evaluated per §7.3.

`exploration_budget` is a real degree of freedom that affects the `TOOL_MISUSE` rate. It is frozen with the calibration lock and reported alongside results. A sensitivity check — resolver re-run at budget ±1 to show category rates do not swing wildly — is a Phase 2 deliverable.

### 7.3 Classification order

For each episode, walk agent steps aligned against `gold_trace`, and classify at the **first divergence**. Order matters; the first matching rule wins:

1. No final answer and a budget cap was hit → `NON_TERMINATION`
2. Retry budget exhausted → `infrastructure_failure` (excluded, §6.3)
3. Unknown tool name, or arguments failing the tool's JSON schema → `TOOL_MISUSE` (`malformed_args` / `unknown_tool`)
4. Exploration budget exceeded without progress → `TOOL_MISUSE` (`unproductive_exploration`)
5. `resolving` tool that cannot serve the current hop's entity type → `TOOL_MISUSE` (`wrong_tool_for_hop`)
6. Correct tool, argument ∉ `P_i` → `HALLUCINATION` (`fabricated_argument`)
7. Correct tool, argument ∈ `P_i`, ≠ gold argument, ∈ `distractor_registry` → `CONTEXT_POLLUTION` (subcode = distractor kind)
8. Correct tool, argument ∈ `P_i`, ≠ gold argument, ∉ registry → `AMBIGUOUS`
9. No tool divergence, final answer == `gold_answer` → `CORRECT`
10. Final answer ∉ `P` → `HALLUCINATION` (`fabricated_answer`)
11. Final answer ∈ `distractor_registry` → `CONTEXT_POLLUTION` (`wrong_selection`)
12. Terminated early on a value that was an intermediate hop's `resolved_value` → `CONTEXT_POLLUTION` (`premature_termination`)
13. Otherwise → `AMBIGUOUS`

Answer comparison is normalized (case, whitespace, surrounding punctuation) but **not** fuzzy-matched. Fuzzy matching would let the resolver decide correctness, which belongs to exact-match against gold.

### 7.4 Evidence

Every label records: the divergent step index, the rule number that fired, the compared values, and `P_i` membership results. This makes the ≥80% mechanical-resolution requirement auditable and makes resolver regressions visible in diffs.

---

## 8. Judge

**Interface constraint:** the judge must be runnable on *any* episode, not only `AMBIGUOUS` ones — PRD §4.3 calibrates it against a stratified subset that includes mechanically-resolved episodes. In production grading it is invoked only on `AMBIGUOUS`; the two paths share one code path with different selection.

**Provider independence.** The judge is configured with its own `base_url`, key, and model string, separate from the models under test. It is the one component permitted to use a closed model (§14) — it is not under test, it labels only the ambiguous residual, and its output is κ-gated regardless of provenance. `JudgePin` therefore records provider alongside model, prompt version, and prompt hash, so a judge swap is visible in the manifest diff rather than silent.

- Judge sees the trace and the gold trace; it returns a category from the same enum plus a confidence. It never sees, and never emits, a correctness verdict.
- Prompt is versioned and hashed into `JudgePin`. Changing the prompt without bumping the version is a defect.
- `loopguard grade --calibrate` runs the judge blind against `fixtures/judge_calibration.jsonl` and reports Cohen's κ overall and per category.
- κ ≥ 0.70 → judge labels count. κ < 0.70 → the pipeline **automatically** relabels judge-assigned episodes as `UNRESOLVED` and the report renders them as their own bucket. This is enforced in code, not left to discipline, because the temptation to accept a 0.68 will be real.

---

## 9. Statistics

| Quantity | Method | Library |
|---|---|---|
| Failure-rate CI | Wilson score, 95% | `statsmodels.stats.proportion.proportion_confint(method="wilson")` |
| Depth trend | Cochran–Armitage | `statsmodels` |
| Mitigation effect (paired) | McNemar exact | `statsmodels.stats.contingency_tables.mcnemar(exact=True)` |
| Category-rate deltas | Paired bootstrap, 10k resamples, seeded | `numpy` |

**Unit of analysis.** Repeats are nested within tasks, so the 1,200 episodes are *not* 1,200 independent observations. Two reported quantities, labeled distinctly:
- **Task-level** (primary): a task counts as failed if it fails in ≥2 of 3 repeats. n=40 per cell. This is the unit for CIs and paired tests.
- **Episode-level** (secondary): raw 1,200, reported for failure-mode composition where per-episode variation is the point.

Conflating these inflates apparent significance roughly threefold. The report emits both with explicit `n` on every figure.

---

## 10. CLI

Implements PRD §8. Additional technical requirements:

- Every subcommand accepts `--json` and writes machine-readable output to stdout, human tables to stderr, so `gate` is scriptable in CI.
- `diff` detects pairing by comparing `task_seeds` and `config_hash` across the two manifests; it prints which mode it chose and refuses to silently fall back.
- `gate` exit codes per PRD §8. Exit `2` (insufficient data) triggers when `episode_count < min_episodes` **or** when required categories are absent from either run — a run that never produced a `CONTEXT_POLLUTION` episode cannot certify that category as non-regressed.
- `report` writes `report.json` (dashboard input) and never mutates `grades.jsonl`.

---

## 11. Dashboard

Static per PRD §9. Technical constraints:

- Input is `report.json` plus a `traces/` subset. The build copies a **capped** trace set (all failures up to a limit, plus a sample of successes) into `dashboard/public/` — shipping 1,200 full traces to a browser is not viable.
- Trace inspector renders agent-vs-gold side by side with the first divergence highlighted, using `EpisodeGrade.evidence` for the highlight rather than recomputing resolver logic in TypeScript. **The resolver must never be reimplemented client-side** — two implementations will disagree, and the browser's will be the one people look at.
- Run selection is static: the build embeds a manifest of available runs. No query backend (PRD §9, deferred).
- Must work from `vite preview` against a local run directory with no AWS involvement.

---

## 12. Testing

| Layer | Requirement |
|---|---|
| Generator | Property tests: chain resolves to `gold_answer`; every distractor registered; `exposed_tools` length constant across depths; payload token counts within tolerance under both tokenizers (§5.3). |
| Resolver | Fixture-driven, one hand-built trace per classification rule in §7.3, asserting rule number and `first_error_hop`. This is the highest-value test suite in the repo. |
| Provenance extraction | Unit tests on nested/edge-case tool responses (§7.1). |
| Agent loop | Runs against `ReplayClient` — no network in CI. |
| Stats | Golden-value tests against hand-computed Wilson/McNemar examples. |
| CLI | `gate` returns each of exit codes 0/1/2/3 on synthetic run pairs. |
| Determinism | Same config hash ⇒ identical `task_seeds` and identical generated `TaskInstance` bytes. |

Live-provider tests are marked and excluded from CI by default.

---

## 13. CI

GitHub Actions:
- **On PR:** ruff, mypy (scoped per §1), pytest excluding live-provider tests, dashboard typecheck + build.
- **On PR touching `loopguard/agent/**` or `loopguard/generate/**`:** the reduced-matrix `gate` job per PRD §8, against a committed baseline run summary.
- **Nightly:** full-matrix gate.
- A deliberately-regressed fixture run is committed so the gate's failure path is itself tested — PRD §11 criterion 6 requires the gate to actually fail something, not merely to run.

Secrets: provider keys via GitHub Actions secrets; `.env` locally, gitignored.

---

## 14. Open technical decisions

Resolved at the phase indicated; none block starting Phase 0.

| Decision | Phase | Notes |
|---|---|---|
| Provider (Together / Groq / Fireworks) | 0 | Chosen on tool-calling fidelity in the smoke test, then rate limits. |
| Tool-set size `N` | 1 | With `d_max = 5`, expect `N` in 8–12. Set during calibration. |
| ~~Judge model~~ **decided** | — | **May be a closed model.** It is not under test, sees only the ambiguous residual, and is κ-gated identically regardless of provenance — so judge strength is worth buying. Hard constraint: must not be either model under test. Cost: exact replication of judge-assigned labels requires that vendor's access (§8, PRD §5 Phase 7). Specific model chosen at Phase 2. |
| `exploration_budget` default | 1 | Frozen with the calibration lock; sensitivity-checked at Phase 2 (§7.2). |
| Trace cap for dashboard | 6 | Driven by payload size once real traces exist. |
