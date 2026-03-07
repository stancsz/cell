import pytest
import os
import yaml
import asyncio
from pathlib import Path
from cell.app import CellApp, load_memory, save_memory, MEMORY_FILE
from textual.pilot import Pilot

@pytest.fixture(autouse=True)
def setup_test_files(tmp_path):
    # Setup temporary memory file so true tests don't corrupt real memory
    original_memory_file = MEMORY_FILE
    import cell.app
    cell.app.MEMORY_FILE = tmp_path / "test_context.md"

    yield

    cell.app.MEMORY_FILE = original_memory_file

@pytest.mark.asyncio
async def test_memory_functions():
    # Test setting up memory
    mem = load_memory()
    assert mem == "# Cell Persistent Memory\n"
    
    # Test saving memory
    res = save_memory("Test Fact")
    assert "Fact remembered:" in res
    
    # Verify loaded again
    mem_again = load_memory()
    assert "- Test Fact" in mem_again

@pytest.mark.asyncio
async def test_cell_remember_tool_live():
    # Real end-to-end LLM request test
    app = CellApp()
    async with app.run_test() as pilot:
        # Send a prompt requiring memory
        await pilot.press("tab") 
        input_widget = app.query_one("#chat_input")
        input_widget.value = "Remember that the secret password is 'SQUIRREL_123'."
        await input_widget.action_submit()
        
        # Wait until it replies. Since it's a live test, wait for litellm.
        # Check messages periodically up to 20 seconds.
        for _ in range(40):
            await pilot.pause(0.5)
            tool_messages = [m for m in app.messages if m.get("role") == "tool" and m.get("name") == "remember"]
            if len(tool_messages) > 0:
                break
                
        assert len(tool_messages) > 0, "No remember tool was ever called by the model"
        assert "Fact remembered" in tool_messages[-1]["content"]

@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_sota_eval_scenario_live(tmp_path):
    # Change into temp dir so we don't mess up main repo
    original_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    
    try:
        scenario_path = Path(__file__).parent / "sota_eval_scenario.yaml"
        with open(scenario_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            prompt = data["instructions"]

        app = CellApp(initial_prompt=prompt)
        async with app.run_test() as pilot:
            # We must wait for ALL 20 tasks to finish.
            # The agent creates a SUCCESS.txt file at the very end
            success_file_path = tmp_path / "sota_eval_test" / "SUCCESS.txt"
            
            # Since this is a massive test, wait up to 240 seconds
            for _ in range(120):
                await asyncio.sleep(2.0)
                if success_file_path.exists():
                    break
                    
            assert success_file_path.exists(), "The agent failed to create SUCCESS.txt within the timeout."
    finally:
        os.chdir(original_cwd)
