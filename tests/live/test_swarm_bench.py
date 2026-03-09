"""
Swarm Node Benchmark — end-to-end agent evaluation harness.

Loads scenarios from scenarios.yaml and runs each as a live Cell agent task.
Results are streamed to tests/live/bench_results.jsonl.

Usage:
    # Run ALL benchmark scenarios (live LLM — very slow):
    RUN_SOTA=1 python -m pytest tests/live/test_swarm_bench.py -v

    # Run a single named scenario:
    RUN_SOTA=1 python -m pytest tests/live/test_swarm_bench.py -k "debug_broken_pytest_suite" -v

    # Run only easy scenarios:
    RUN_SOTA=1 python -m pytest tests/live/test_swarm_bench.py -k "easy" -v
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCENARIO_FILE = Path(__file__).parent.parent / "realistic_eval_scenarios.yaml"
BENCH_LOG = Path(__file__).parent / "bench_results.jsonl"


def load_scenarios():
    """Parse realistic_eval_scenarios.yaml and return list of scenario dicts."""
    data = yaml.safe_load(SCENARIO_FILE.read_text(encoding="utf-8"))
    return data["scenarios"]


def _get_timeout():
    data = yaml.safe_load(SCENARIO_FILE.read_text(encoding="utf-8"))
    return data.get("runner", {}).get("timeout_seconds", 300)


def _scenario_id(scenario):
    return scenario["name"]


# ---------------------------------------------------------------------------
# Success criteria checker
# ---------------------------------------------------------------------------

def _check_criteria(scenario: dict, base_dir: Path) -> list[str]:
    criteria = scenario.get("success_criteria", {})
    failures = []

    for rel in criteria.get("files_must_exist", []):
        if not (base_dir / rel).exists():
            failures.append(f"MISSING FILE: {rel}")

    for rel in criteria.get("files_must_not_exist", []):
        if (base_dir / rel).exists():
            failures.append(f"FILE SHOULD NOT EXIST: {rel}")

    for rule in criteria.get("file_must_contain", []):
        p = base_dir / rule["path"]
        if not p.exists():
            failures.append(f"FILE MISSING (for content check): {rule['path']}")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "contains" in rule and rule["contains"] not in text:
            failures.append(f"CONTENT MISSING in {rule['path']}: expected {rule['contains']!r}")
        if "not_contains" in rule and rule["not_contains"] in text:
            failures.append(f"FORBIDDEN CONTENT in {rule['path']}: must not contain {rule['not_contains']!r}")

    for cmd in criteria.get("commands_must_pass", []):
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60, cwd=str(base_dir))
            if result.returncode != 0:
                failures.append(f"COMMAND FAILED (exit {result.returncode}): {cmd}\n  stdout: {result.stdout[:500]}\n  stderr: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            failures.append(f"COMMAND TIMED OUT: {cmd}")
        except Exception as e:
            failures.append(f"COMMAND ERROR ({cmd}): {e}")

    return failures


def _log_result(scenario_name: str, passed: bool, failures: list[str]) -> None:
    record = {"scenario": scenario_name, "passed": passed, "failures": failures}
    with open(BENCH_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", load_scenarios(), ids=_scenario_id)
@pytest.mark.asyncio
@pytest.mark.timeout(380)
@pytest.mark.skipif(os.getenv("RUN_SOTA") != "1", reason="Benchmark tests require live LLM; set RUN_SOTA=1 to enable")
async def test_swarm_benchmark(scenario, tmp_path):
    """
    Run a single benchmark scenario end-to-end using the live Cell node agent.
    The agent is started with the scenario instructions as its initial prompt.
    We poll for the .eval_done sentinel, then verify all success criteria.
    """
    from cell.app import CellApp
    import cell.app as cell_app

    original_memory = cell_app.MEMORY_FILE
    cell_app.MEMORY_FILE = tmp_path / "bench_context.md"

    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    os.environ["CELL_EVAL_MODE"] = "1"

    try:
        app = CellApp(initial_prompt=scenario["instructions"])
        sentinel = tmp_path / ".eval_done"
        poll_timeout = _get_timeout()

        async with app.run_test():
            for _ in range(int(poll_timeout / 2)):
                await asyncio.sleep(2)
                if sentinel.exists():
                    break

        # Dump messages for post-mortem analysis
        with open(tmp_path / "agent_messages.json", "w", encoding="utf-8") as fm:
            dumpable = [m.model_dump() if hasattr(m, "model_dump") else (m if isinstance(m, dict) else str(m)) for m in app.messages]
            fm.write(json.dumps(dumpable, indent=2))

        failures = _check_criteria(scenario, tmp_path)
        _log_result(scenario["name"], passed=not failures, failures=failures)
        assert not failures, (
            f"\nBenchmark '{scenario['name']}' failed {len(failures)} criteria:\n"
            + "\n".join(f"  ✗ {f}" for f in failures)
        )

    except Exception as exc:
        _log_result(scenario["name"], passed=False, failures=[str(exc)])
        raise

    finally:
        os.chdir(original_cwd)
        os.environ.pop("CELL_EVAL_MODE", None)
        cell_app.MEMORY_FILE = original_memory
