"""Fast unit tests — no LLM calls, no subprocesses. Runs in < 5 seconds."""
import pytest
import cell.app
from cell.app import load_memory, save_memory, MEMORY_FILE


@pytest.fixture(autouse=True)
def isolate_memory(tmp_path):
    """Redirect memory file to a tmp location so tests don't corrupt real memory."""
    original = cell.app.MEMORY_FILE
    cell.app.MEMORY_FILE = tmp_path / "test_context.md"
    yield
    cell.app.MEMORY_FILE = original


@pytest.mark.asyncio
async def test_memory_load_default():
    """load_memory() creates and returns an empty header on first call."""
    content = load_memory()
    assert content == "# Cell Persistent Memory\n"


@pytest.mark.asyncio
async def test_memory_save_and_load():
    """save_memory() writes a fact; load_memory() reads it back."""
    result = save_memory("Test Fact")
    assert "Fact remembered:" in result
    assert "- Test Fact" in load_memory()


def test_build_system_prompt_contains_directives():
    """System prompt contains the core identity and swarm directives."""
    from cell.app import build_system_prompt
    prompt = build_system_prompt()
    assert "CELL" in prompt
    assert "swarm" in prompt.lower() or "node" in prompt.lower()
    assert "tool" in prompt.lower()


def test_env_context():
    """get_env_context() returns a non-empty string with CWD."""
    from cell.app import get_env_context
    ctx = get_env_context()
    assert "CWD" in ctx
    assert len(ctx) > 0
