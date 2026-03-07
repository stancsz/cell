"""
Realistic SOTA evaluation harness for Cell agent.

Each scenario is loaded from realistic_eval_scenarios.yaml and run as a
separate pytest test. Scenarios are parameterized so each appears with its
own name in the pytest report.

Usage:
    # Run all scenarios (live LLM — slow):
    python -m pytest tests/test_realistic_eval.py -v

    # Run a single named scenario:
    python -m pytest tests/test_realistic_eval.py -k "debug_broken_pytest_suite" -v

    # Run only easy scenarios:
    python -m pytest tests/test_realistic_eval.py -k "easy" -v
"""

import asyncio
import os
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from unittest.mock import patch, AsyncMock

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCENARIO_FILE = Path(__file__).parent / "realistic_eval_scenarios.yaml"
EVAL_LOG = Path(__file__).parent / "eval_results.jsonl"


def load_scenarios():
    """Parse realistic_eval_scenarios.yaml and return list of scenario dicts."""
    data = yaml.safe_load(SCENARIO_FILE.read_text(encoding="utf-8"))
    return data["scenarios"]


def _get_timeout():
    """Read the runner.timeout_seconds from the YAML (default 300)."""
    data = yaml.safe_load(SCENARIO_FILE.read_text(encoding="utf-8"))
    return data.get("runner", {}).get("timeout_seconds", 300)


def _scenario_id(scenario):
    return scenario["name"]


# ---------------------------------------------------------------------------
# Helpers for success-criteria checking
# ---------------------------------------------------------------------------

def _check_criteria(scenario: dict, base_dir: Path) -> list[str]:
    """
    Evaluate success_criteria against `base_dir`.
    Returns a list of failure messages (empty = all pass).
    """
    criteria = scenario.get("success_criteria", {})
    failures = []

    # --- files_must_exist ---
    for rel in criteria.get("files_must_exist", []):
        p = base_dir / rel
        if not p.exists():
            failures.append(f"MISSING FILE: {rel}")

    # --- files_must_not_exist ---
    for rel in criteria.get("files_must_not_exist", []):
        p = base_dir / rel
        if p.exists():
            failures.append(f"FILE SHOULD NOT EXIST: {rel}")

    # --- file_must_contain ---
    for rule in criteria.get("file_must_contain", []):
        p = base_dir / rule["path"]
        if not p.exists():
            failures.append(f"FILE MISSING (for content check): {rule['path']}")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "contains" in rule and rule["contains"] not in text:
            failures.append(
                f"CONTENT MISSING in {rule['path']}: expected to find {rule['contains']!r}"
            )
        if "not_contains" in rule and rule["not_contains"] in text:
            failures.append(
                f"FORBIDDEN CONTENT in {rule['path']}: must not contain {rule['not_contains']!r}"
            )

    # --- commands_must_pass ---
    for cmd in criteria.get("commands_must_pass", []):
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(base_dir),
            )
            if result.returncode != 0:
                failures.append(
                    f"COMMAND FAILED (exit {result.returncode}): {cmd}\n"
                    f"  stdout: {result.stdout[:500]}\n"
                    f"  stderr: {result.stderr[:500]}"
                )
        except subprocess.TimeoutExpired:
            failures.append(f"COMMAND TIMED OUT: {cmd}")
        except Exception as e:
            failures.append(f"COMMAND ERROR ({cmd}): {e}")

    return failures


def _log_result(scenario_name: str, passed: bool, failures: list[str]) -> None:
    """Append a JSON line to eval_results.jsonl for live progress tracking."""
    record = {
        "scenario": scenario_name,
        "passed": passed,
        "failures": failures,
    }
    with open(EVAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Core test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", load_scenarios(), ids=_scenario_id)
@pytest.mark.asyncio
@pytest.mark.timeout(380)
async def test_realistic_scenario(scenario, tmp_path):
    """
    Run a single realistic eval scenario end-to-end using the live Cell agent.

    The agent is started with the scenario's `instructions` as its initial
    prompt. We poll every 2 s for the sentinel file, then verify all criteria.
    Results are streamed to tests/eval_results.jsonl as each scenario finishes.
    """
    from cell.app import CellApp
    import cell.app as cell_app

    # Isolate memory so tests don't corrupt real ~/.cell/context.md
    original_memory = cell_app.MEMORY_FILE
    cell_app.MEMORY_FILE = tmp_path / "test_context.md"

    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))

    try:
        # Real MCP tools required — no mock.
        app = CellApp(initial_prompt=scenario["instructions"])

        # Sentinel: last file in files_must_exist signals task completion.
        expected_files = scenario.get("success_criteria", {}).get("files_must_exist", [])
        sentinel = tmp_path / expected_files[-1] if expected_files else None

        poll_timeout = _get_timeout()
        async with app.run_test() as pilot:
            for _ in range(int(poll_timeout / 2)):
                await asyncio.sleep(2)
                if sentinel and sentinel.exists():
                    break

        failures = _check_criteria(scenario, tmp_path)
        _log_result(scenario["name"], passed=not failures, failures=failures)
        assert not failures, (
            f"\nScenario '{scenario['name']}' failed {len(failures)} criteria:\n"
            + "\n".join(f"  ✗ {f}" for f in failures)
        )

    except Exception as exc:
        _log_result(scenario["name"], passed=False, failures=[str(exc)])
        raise

    finally:
        os.chdir(original_cwd)
        cell_app.MEMORY_FILE = original_memory


# ---------------------------------------------------------------------------
# Smoke tests — no LLM calls
# ---------------------------------------------------------------------------

def test_scenario_file_is_valid():
    """Verify the YAML is well-formed and all required keys are present."""
    data = yaml.safe_load(SCENARIO_FILE.read_text(encoding="utf-8"))
    assert "scenarios" in data, "Top-level 'scenarios' key missing"
    for s in data["scenarios"]:
        assert "name" in s, f"Scenario missing 'name': {s}"
        assert "instructions" in s, f"Scenario '{s.get('name')}' missing 'instructions'"
        assert "success_criteria" in s, f"Scenario '{s.get('name')}' missing 'success_criteria'"


def test_all_scenarios_have_unique_names():
    """Each scenario name must be unique (to avoid pytest collision)."""
    names = [s["name"] for s in load_scenarios()]
    assert len(names) == len(set(names)), f"Duplicate scenario names: {names}"
