"""
Live integration tests — require a real LLM API key and network access.

Usage:
    RUN_LIVE=1 python -m pytest tests/live/test_live_integration.py -v
"""

import asyncio
import os
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, AsyncMock

import cell.app
from cell.app import CellApp, load_memory, save_memory, MEMORY_FILE


@pytest.fixture(autouse=True)
def isolate_memory(tmp_path):
    original = cell.app.MEMORY_FILE
    cell.app.MEMORY_FILE = tmp_path / "test_context.md"
    yield
    cell.app.MEMORY_FILE = original


@pytest.fixture
def mock_mcp():
    with patch("cell.app.load_mcp_servers", new_callable=AsyncMock) as m:
        m.return_value = ({}, [])
        yield m


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("RUN_LIVE") != "1", reason="Set RUN_LIVE=1 to run live LLM integration tests")
async def test_remember_tool_called(mock_mcp):
    """Live: agent calls the remember tool when asked to remember a fact."""
    app = CellApp()
    async with app.run_test():
        app.submit_chat_input("Remember that the secret password is 'SQUIRREL_123'.")
        for _ in range(60):
            await asyncio.sleep(0.5)
            tool_msgs = [m for m in app.messages if m.get("role") == "tool" and m.get("name") == "remember"]
            if tool_msgs:
                break
    assert tool_msgs, f"remember tool never called. messages={app.messages}"
    assert "Fact remembered" in tool_msgs[-1]["content"]


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.skipif(os.getenv("RUN_SOTA") != "1", reason="Set RUN_SOTA=1 to run SOTA live completion test")
async def test_sota_completion(tmp_path, mock_mcp):
    """Live: agent autonomously completes a full SOTA eval scenario."""
    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    os.environ["CELL_EVAL_MODE"] = "1"
    try:
        scenario_path = Path(__file__).parent.parent / "sota_eval_scenario.yaml"
        data = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
        prompt = data["instructions"]
        app = CellApp(initial_prompt=prompt)
        async with app.run_test() as pilot:
            success_file = tmp_path / "sota_eval_test" / "SUCCESS.txt"
            for _ in range(120):
                await pilot.pause(2.0)
                if success_file.exists():
                    break
            assert success_file.exists(), "Agent failed to create SUCCESS.txt within 240s."
    finally:
        os.chdir(original_cwd)
        os.environ.pop("CELL_EVAL_MODE", None)
