# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

Pre-implementation: `prd.md`, `trd.md`, `.gitignore` only. **No source, no build system, no tests exist yet — do not infer working commands from this file.** The stack is decided (below) but not yet scaffolded; that is Phase 0.

Two documents, different jobs. `prd.md` is the source of truth for scope, experimental design, and success criteria — *what* and *why*. `trd.md` decides stack, schemas, and algorithms — *how*. Where they conflict, PRD wins on intent, TRD wins on mechanism. Read both before implementing; the constraints below are summaries, not substitutes.

## Stack (decided in `trd.md` §1)

Python 3.11+ with **uv**, Pydantic v2 for config and schemas, scipy/statsmodels for statistics, pytest, ruff, mypy (strict on `loopguard/schemas` and `loopguard/grading`). Dashboard is Vite + React + TypeScript with Recharts, pnpm, static export.

**Models under test are open-weight, always.** Provider access uses the **OpenAI-compatible wire format** (`POST /v1/chat/completions`) — the de facto standard for open-weight serving, spoken by Together, Groq, Fireworks, vLLM, and Ollama. The `openai` package appears in the dependency list purely as an HTTP client for that format, pointed at a provider `base_url`; no OpenAI model, key, or service is involved. Do not read that dependency as a vendor choice, and do not propose swapping the models under test for closed ones (PRD §2 non-goals). The judge is the sole exception — see below.

Planned CLI surface — `loopguard run | grade | report | diff | gate` (PRD §8, TRD §10). **Not yet implemented.** Once Phase 0 lands, this file should gain a Commands section covering: one episode end-to-end, one matrix cell, grading a run, judge calibration, and running a single test.

## What LoopGuard is

A controlled experiment measuring how tool-using agent reliability degrades as loop depth (number of sequential tool calls) increases, across two hosted open-weight models (Llama 3.3, Qwen3). Its deliverable is a trustworthy measurement, not a performance win. A negative result is a success.

The pipeline: a **seeded generator** builds a synthetic entity graph plus a chained-lookup task, and emits the **gold trace** alongside it → an **agent loop** runs the task against hosted OpenAI-compatible endpoints, writing an append-only JSONL trace → a **mechanical resolver** diffs trace against gold trace to attribute failures → an **LLM judge** handles only the unresolvable residual → a **CLI** reports and diffs runs → a **static dashboard** renders the artifacts.

## Load-bearing invariants

These are the design's validity conditions. Violating one does not produce a bug — it produces plausible numbers that mean nothing. Treat changes to them as requiring explicit sign-off, not as ordinary refactors.

**Loop depth is the sole independent variable.** Tool-set size, distractor density, per-hop payload size (±10% tokens), task template, system prompt, and decoding params are constant across every depth. A per-model prompt tweak, an extra tool at high depth, or a longer payload for a later hop silently destroys the depth curve. If a model-specific adaptation is genuinely unavoidable, it must be recorded as a documented deviation, not absorbed quietly.

**The generator emits a gold trace, not a gold answer.** Ground truth for every intermediate hop is what makes mechanical failure attribution possible. Any generator change must preserve per-hop ground truth or the taxonomy collapses to an LLM judge — the exact thing this design exists to avoid.

**Tools are pure deterministic functions over the generated graph.** No network, no clock, no randomness. Replayability depends on it.

**`calibration.lock.json` is frozen after Phase 1.** Difficulty is tuned only via distractor knobs (count, ID edit distance, cross-branch overlap), only before headline runs, and never re-tuned after seeing results. Pilot traces are excluded from all reported numbers.

**Failures are attributed to the first divergence from the gold trace**, not the final answer. `first_error_hop` is a reported quantity, not a debug field — it is what distinguishes "depth degrades reasoning" from "depth offers more chances to fail."

**The judge never grades correctness.** Correctness is exact-match against gold, normalized for case/whitespace/punctuation but never fuzzy-matched — fuzzy matching hands the correctness decision to the resolver. In production the judge sees only `AMBIGUOUS` episodes, though it must remain runnable on any episode since calibration spans mechanically-resolved ones too. Two pre-committed gates: mechanical resolution covers ≥80% of failed episodes, and judge/human Cohen's κ ≥ 0.70 for judge labels to enter headline numbers. The κ gate is **enforced in code** — below threshold, judge labels are auto-relabeled `UNRESOLVED`. It is deliberately not left to discipline, because the temptation to accept a 0.68 will be real.

The judge is the one component permitted a **closed model** — it is not under test and labels only the residual. It must not be either model under test, it is configured with its own `base_url`/key independent of the harness provider, and the κ gate applies identically. The cost is that judge-assigned labels are not exactly replicable without that vendor's access; this is a stated limitation in the writeup (PRD §5 Phase 7), not a silent one.

**Exploration is not tool misuse.** Tools are tagged `exploratory` or `resolving`; calls to exploratory tools don't advance the gold-hop pointer or count as divergence, up to `exploration_budget`. Without this, every reasonable `search_entities` call inflates `TOOL_MISUSE`. The budget is a genuine researcher degree of freedom — it is frozen with the calibration lock and sensitivity-checked at ±1 in Phase 2.

**The mitigation arm reuses the baseline's task seeds.** Comparisons are paired; unpaired sampling is materially weaker at this n. The mitigation's primary outcome is the rate of its *targeted* failure category — aggregate accuracy is secondary, and all off-target category rates are reported so displacement between categories is visible rather than hidden by an aggregate win.

**Token count is the primary cost metric; latency is secondary and caveated.** Inference is hosted, so wall-clock is network-noisy — report median and p90 across the ≥3 repeats, never a single timing. Latency is measured on the *successful* attempt only; retry backoff must never enter it, or the cost comparison measures provider load instead of the mitigation.

**Infrastructure failures are excluded from failure-rate denominators and counted separately.** Retry-budget exhaustion is not an agent failure. Grading it as one is the easiest way to fabricate a depth effect, since deeper episodes run longer and time out more. If the per-cell infrastructure-failure count is not roughly flat across depths, treat the depth curve as suspect.

**Config hashing covers semantic fields only.** Output directory, concurrency, rate limits, and retry budget are excluded — hashing them would change task seeds when parallelism changes and silently break baseline↔mitigation pairing. The semantic/runtime split lives in the config model itself, not in a filter at hash time.

**Never reimplement the resolver client-side.** The dashboard highlights divergences from `EpisodeGrade.evidence`. Two implementations will drift, and the browser's is the one people will believe.

## Statistical conventions

Wilson score intervals on proportions; McNemar (exact) on paired per-task flips; Cochran–Armitage for the depth trend. Use statsmodels — do not hand-roll.

**Unit of analysis matters more than it looks.** Repeats nest within tasks, so 1,200 episodes are not 1,200 independent observations. Task-level (failed if ≥2 of 3 repeats fail, n=40/cell) is primary and is the unit for all CIs and paired tests; episode-level is secondary and used for failure-mode composition. Conflating them inflates apparent significance roughly threefold. Every reported figure carries an explicit `n`.

At n=40/cell small effects are not resolvable — report intervals and decline to narrate differences the data cannot support.

## Reproducibility requirements

Every run writes a manifest: provider model string and served revision, decoding params, config hash, calibration-lock hash, git SHA and dirty flag, task seeds, arm, and judge pin. Traces are append-only JSONL per episode with a terminal footer — an episode is complete iff its footer parses, which is what makes a 1,200-episode run resumable.

All artifacts carry `schema_version`. A component reading an unrecognized version must **fail loudly rather than best-effort parse** — a misread trace yields a plausible wrong number, the worst failure mode this project has.

Seeds derive deterministically from the config hash (`config_hash → task_seed → decoding_seed`), so a config hash fully determines the task set. Retries reuse the same decoding seed; a reseeded retry is a different sample, not a retry.

## Phase order and cut lines

Phases run 0→7 (`prd.md` §5) and are sequential; the mitigation cannot be chosen before the baseline diagnosis exists. If scope must be cut, the order is **Phase 6 (dashboard) first, then Phase 5 (CLI)** — the study and taxonomy are the deliverable, the rest is packaging.

Mitigation selection is deliberately not specified in advance. It is chosen from baseline results under the §6 criteria, with a pre-registration note (mechanism + predicted direction and magnitude) committed **before** the mitigation arm runs.

## Repo conventions

`.gitignore` excludes `runs/`, `traces/`, and `*.jsonl` — traces are large and regenerable from a committed config + manifest + git SHA. Two consequences to keep in mind:

- The hand-labeled judge-calibration set is ground truth that **cannot** be regenerated. It is tracked only via the `!fixtures/**/*.jsonl` exception, so it must live under `fixtures/`. Anywhere else and it is silently ignored.
- Dashboard artifacts under `runs/` are ignored. Committing a demo run for a deployed dashboard needs a new explicit exception.

Commits use Conventional Commits (`docs:`, `chore:`, …) with a body explaining rationale rather than restating the diff.
