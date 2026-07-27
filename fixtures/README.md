# fixtures/

Ground truth that **cannot be regenerated**. Everything here is tracked in git.

`.gitignore` excludes `*.jsonl` repo-wide because traces are large and
regenerable from a committed config + manifest + git SHA. The hand-labeled
judge-calibration set is neither: it is human labour, and losing it means
re-labelling 100+ episodes by hand. It survives only via the
`!fixtures/**/*.jsonl` exception, so it must live under this directory.
Anywhere else and it is silently ignored.

## Contents

| File | Phase | Purpose |
|---|---|---|
| `judge_calibration.jsonl` | 2 | ≥100 hand-labeled episodes, stratified across all depths, both models, and both mechanically-resolved and ambiguous cases. Input to `loopguard grade --calibrate`, which reports Cohen's κ overall and per category. Doubles as a regression test for judge-prompt changes. |

The κ ≥ 0.70 gate is enforced in code: below threshold, judge-assigned labels are
automatically relabeled `UNRESOLVED` rather than entering headline numbers.
