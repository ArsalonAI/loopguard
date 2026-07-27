"""``loopguard`` CLI.

Surface per PRD 8: ``run | grade | report | diff | gate``, plus ``smoke`` for the
Phase 0 provider check. Only ``smoke`` and ``config`` are implemented at Phase 0;
the rest are declared so the surface is fixed and argument names cannot drift as
each phase lands.
"""

from __future__ import annotations

import argparse
import sys

from loopguard import __version__
from loopguard.envfile import load_dotenv

_PHASE = {
    "run": "Phase 1 (needs the task generator)",
    "grade": "Phase 2 (needs the mechanical resolver)",
    "report": "Phase 2",
    "diff": "Phase 5",
    "gate": "Phase 5",
}


def _not_yet(command: str) -> int:
    print(
        f"`loopguard {command}` is not implemented yet: {_PHASE[command]}.\n"
        f"Phase 0 implements `loopguard smoke` and `loopguard config`.",
        file=sys.stderr,
    )
    return 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loopguard", description=__doc__)
    parser.add_argument("--version", action="version", version=f"loopguard {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Phase 0 provider smoke test on both models")
    smoke.add_argument("--config", default="configs/smoke.yaml")
    smoke.add_argument(
        "--provider",
        choices=["together", "groq", "fireworks"],
        help="Override provider, base_url, key env, and both model strings",
    )
    smoke.add_argument("--model", dest="only_model", help="Run one model only, by config id")
    smoke.add_argument("--llama-model", help="Override the Llama provider model string")
    smoke.add_argument("--qwen-model", help="Override the Qwen provider model string")
    smoke.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise the loop, trace writer, and manifest with no network",
    )

    cfg = sub.add_parser("config", help="Resolve a config and print its hashes")
    cfg.add_argument("--config", default="configs/baseline.yaml")
    cfg.add_argument("--json", action="store_true", help="Emit the resolved config as JSON")

    run = sub.add_parser("run", help="Execute a run matrix")
    run.add_argument("--config", required=True)
    run.add_argument("--out")
    run.add_argument("--yes", action="store_true", help="Skip the cost-estimate confirmation")

    grade = sub.add_parser("grade", help="Mechanical + judge attribution")
    grade.add_argument("run_dir")
    grade.add_argument(
        "--calibrate", action="store_true", help="Judge calibration against fixtures"
    )

    report = sub.add_parser("report", help="Tables, curves, and report.json")
    report.add_argument("run_dir")

    diff = sub.add_parser("diff", help="Per-category, per-depth deltas between two runs")
    diff.add_argument("baseline")
    diff.add_argument("candidate")

    gate = sub.add_parser("gate", help="CI regression gate")
    gate.add_argument("baseline")
    gate.add_argument("candidate")
    gate.add_argument("--policy", default="configs/policy.yaml")

    # TRD 10: every subcommand takes --json (machine-readable to stdout, human
    # tables to stderr) so `gate` is scriptable in CI.
    for p in (smoke, run, grade, report, diff, gate):
        p.add_argument("--json", action="store_true", help="Machine-readable output")

    return parser


def _cmd_config(args: argparse.Namespace) -> int:
    import json

    from loopguard.config_io import load_config
    from loopguard.hashing import config_hash, derive_task_seed, task_hash

    config = load_config(args.config)
    t_hash = task_hash(config.semantic.task)
    c_hash = config_hash(config.semantic)

    if args.json:
        print(
            json.dumps(
                {
                    "task_hash": t_hash,
                    "config_hash": c_hash,
                    "resolved": config.model_dump(mode="json"),
                },
                indent=2,
            )
        )
        return 0

    task = config.semantic.task
    n_models = len(config.semantic.models)
    episodes = len(task.depths) * task.tasks_per_depth * task.repeats * n_models
    print(f"task_hash   : {t_hash}   (task set; equal across arms -> paired comparison)")
    print(f"config_hash : {c_hash}   (everything that can move a result)")
    print(f"arm         : {config.semantic.arm}")
    print(f"depths      : {task.depths}   max_steps: {config.semantic.loop.max_steps}")
    print(
        f"episodes    : {episodes}"
        f"  ({task.tasks_per_depth} tasks x {task.repeats} repeats x "
        f"{len(task.depths)} depths x {n_models} models)"
    )
    print(f"calibration : {task.calibration_lock_hash or '(unfrozen -- pre-Phase-1)'}")
    print("first task seeds:")
    for depth in task.depths:
        seeds = [derive_task_seed(t_hash, depth, i) for i in range(3)]
        print(f"  d={depth}: {seeds} ...")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    if args.command == "smoke":
        from loopguard.cli.smoke import run_smoke

        return run_smoke(
            config_path=args.config,
            provider=args.provider,
            dry_run=args.dry_run,
            only_model=args.only_model,
            llama_model=args.llama_model,
            qwen_model=args.qwen_model,
        )
    if args.command == "config":
        return _cmd_config(args)
    return _not_yet(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
