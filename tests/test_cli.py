from __future__ import annotations

import json

import pytest
import yaml

from loopguard.cli.main import main

from .conftest import CONFIGS, REPO_ROOT


def test_config_command_prints_both_hashes(capsys):
    assert main(["config", "--config", str(CONFIGS / "baseline.yaml")]) == 0
    out = capsys.readouterr().out
    assert "task_hash" in out and "config_hash" in out
    assert "1200" in out, "baseline matrix should be 5 depths x 40 tasks x 3 repeats x 2 models"


def test_config_json_output_is_machine_readable(capsys):
    assert main(["config", "--config", str(CONFIGS / "baseline.yaml"), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"task_hash", "config_hash", "resolved"}
    assert payload["resolved"]["semantic"]["arm"] == "baseline"


def test_unimplemented_subcommands_exit_nonzero_and_say_which_phase(capsys):
    for command, argv in [
        ("run", ["run", "--config", "x"]),
        ("grade", ["grade", "runs/x"]),
        ("report", ["report", "runs/x"]),
        ("diff", ["diff", "a", "b"]),
        ("gate", ["gate", "a", "b"]),
    ]:
        assert main(argv) == 3
        assert "Phase" in capsys.readouterr().err, command


def test_smoke_dry_run_writes_a_complete_run_directory(tmp_path, capsys):
    """Phase 0's exit criterion, exercised with no network.

    The live version of this is `loopguard smoke --provider <candidate>`.
    """
    raw = yaml.safe_load((CONFIGS / "smoke.yaml").read_text())
    raw["runtime"]["out_dir"] = str(tmp_path / "runs")
    raw["semantic"]["prompt"]["path"] = str(CONFIGS / "prompts" / "baseline_v1.txt")
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(raw))

    assert main(["smoke", "--config", str(config_path), "--dry-run"]) == 0

    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["episode_count"] == 2
    assert manifest["infrastructure_failure_count"] == 0
    assert len(manifest["models"]) == 2
    assert manifest["task_seeds"]

    from loopguard.trace_io import read_trace

    traces = sorted((run_dir / "traces").glob("*.jsonl"))
    assert len(traces) == 2
    for path in traces:
        trace = read_trace(path)
        assert trace.footer.terminal_reason == "final_answer"

    assert (run_dir / "tasks" / "smoke-d3-0000.json").exists()
    assert "tool-calling fidelity" in capsys.readouterr().out


def test_smoke_without_a_key_fails_with_a_useful_message(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    raw = yaml.safe_load((CONFIGS / "smoke.yaml").read_text())
    raw["runtime"]["out_dir"] = str(tmp_path / "runs")
    raw["semantic"]["prompt"]["path"] = str(CONFIGS / "prompts" / "baseline_v1.txt")
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(raw))

    monkeypatch.chdir(tmp_path)  # so a repo-root .env is not picked up
    assert main(["smoke", "--config", str(config_path)]) == 3
    assert "TOGETHER_API_KEY" in capsys.readouterr().err


@pytest.mark.live_provider
def test_smoke_against_a_live_provider():
    """Excluded from CI by default. Run with `pytest -m live_provider`."""
    assert main(["smoke", "--config", str(REPO_ROOT / "configs" / "smoke.yaml")]) == 0
