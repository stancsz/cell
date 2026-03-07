import pytest
import yaml
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock
from cell.app import CellApp, load_memory, save_memory, MEMORY_FILE

@pytest.fixture(autouse=True)
def isolate_memory(tmp_path):
    """Redirect memory file to a tmp location so tests don't corrupt real memory."""
    import cell.app
    original = MEMORY_FILE
    cell.app.MEMORY_FILE = tmp_path / "test_context.md"
    yield
    cell.app.MEMORY_FILE = original

@pytest.fixture
def mock_mcp():
    """Prevent MCP server subprocesses from spawning during tests."""
    with patch("cell.app.load_mcp_servers", new_callable=AsyncMock) as m:
        m.return_value = ({}, [])
        yield m

@pytest.mark.asyncio
async def test_memory_functions():
    """Unit test: load_memory / save_memory work correctly."""
    assert load_memory() == "# Cell Persistent Memory\n"
    assert "Fact remembered:" in save_memory("Test Fact")
    assert "- Test Fact" in load_memory()

@pytest.mark.asyncio
async def test_cell_remember_tool_live(mock_mcp):
    """Integration: live LLM call triggers the remember built-in tool."""
    app = CellApp()
    async with app.run_test() as pilot:
        await pilot.press("tab")
        input_widget = app.query_one("#chat_input")
        input_widget.value = "Remember that the secret password is 'SQUIRREL_123'."
        await input_widget.action_submit()

        # run_worker spawns a thread-pool worker, so we must use asyncio.sleep
        # (not pilot.pause) to give those threads a chance to complete.
        tool_messages = []
        for _ in range(60):          # 60 × 0.5 s = 30 s max
            await asyncio.sleep(0.5)
            tool_messages = [m for m in app.messages
                             if m.get("role") == "tool" and m.get("name") == "remember"]
            if tool_messages:
                break

        assert tool_messages, f"remember tool never called. messages={app.messages}"
        assert "Fact remembered" in tool_messages[-1]["content"]

@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_sota_eval_scenario_live(tmp_path, mock_mcp):
    """Integration: live LLM autonomously completes all 20 SOTA tasks."""
    import os
    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        scenario_path = Path(__file__).parent / "sota_eval_scenario.yaml"
        data = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
        prompt = data["instructions"]

        app = CellApp(initial_prompt=prompt)
        async with app.run_test() as pilot:
            success_file = tmp_path / "sota_eval_test" / "SUCCESS.txt"
            for _ in range(120):
                await pilot.pause(2.0)
                if success_file.exists():
                    break
            assert success_file.exists(), "Agent failed to create SUCCESS.txt within 240 s."
    finally:
        os.chdir(original_cwd)
