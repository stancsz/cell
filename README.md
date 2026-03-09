# Cell

> A small footprint, cost-effective pseudo-node agent for massive agentic swarms.

Cell is a tiny, highly token-efficient agent framework. It operates on a single idea: an LLM connected to tools via [MCP](https://modelcontextprotocol.io/). There are smarter and more capable agents out there. Cell is explicitly designed to be a cheaper, "worker bee" unit that excels on lower-tier, low-cognitive models (like `gpt-5-mini`), allowing it to perform impressive routine ops reliably while more capable models handle higher-level reasoning. It acts as a perfect node worker in a massive swarm.

```
┌──────────────────────────────────────────┐
│  User → LLM → Tool Calls → LLM → ...    │
│         (the vascular loop)              │
└──────────────────────────────────────────┘
```

---

## The Swarm Philosophy

Cell isn't trying to be the most advanced cognitive orchestrator in the world. Instead:
- **Cost-effective** — Designed specifically for minimal token footprints. It gets right to the point without verbose logic.
- **Lower-tier Model Champion** — Extensively tuned to extract surprising competence from smaller models like `gpt-5-mini`, freeing up expensive API limits for master-planner models.
- **Small Footprint Worker** — Has just enough capabilities (file I/O, bash, persistent memory) to autonomously execute assigned tasks in a wider swarm. 

---

## Features

- **Streaming TUI** — Textual-based terminal UI with a hacker aesthetic
- **Built-in tools** — `remember` for persistent long-term memory
- **MCP tool loading** — dynamically connects to any MCP server defined in `~/.cell/mcp.json`
- **Core skills** — `read_file`, `write_file`, `replace_in_file`, `run_command`, `schedule_task` via a local MCP server
- **Async subprocess** — `run_command` times out after 20 s, preventing hanging
- **Persistent memory** — facts saved to `~/.cell/context.md`, injected into every system prompt
- **LiteLLM backend** — works with OpenAI, Anthropic, Gemini, Ollama — any provider LiteLLM supports
- **pip installable** — single `pip install` gets you the `cell` CLI entry point

---

## Install

```bash
git clone https://github.com/stancsz/cell
cd cell
pip install -e .
```

For development (includes pytest + pyyaml):

```bash
pip install -e ".[dev]"
```

---

## Usage

Launch the interactive TUI:

```bash
cell
```

Pass a one-shot prompt directly (non-interactive):

```bash
cell "Summarize the file README.md"
```

On Windows, use the included wrapper:

```bat
.\cell.bat "Your prompt here"
```

---

## Configuration

### API Key

Create a `.env` file in the project root (or export to your shell):

```env
OPENAI_API_KEY=sk-...
```

Cell uses [LiteLLM](https://github.com/BerriAI/litellm), so you can swap to any provider:

```env
ANTHROPIC_API_KEY=...      # for claude-3-5-sonnet
GEMINI_API_KEY=...         # for gemini/gemini-2.0-flash
OLLAMA_BASE_URL=http://localhost:11434  # for local ollama
```

Change the model by editing `src/cell/app.py`:

```python
self.model = os.environ.get("CELL_MODEL", "openai/gpt-5.2-codex")
```

Or set it via env var:

```env
CELL_MODEL=anthropic/claude-3-5-sonnet-20241022
```

### MCP Servers

Cell auto-creates `~/.cell/mcp.json` on first run, pointing to the built-in core skills server:

```json
{
  "mcpServers": {
    "core_skills": {
      "command": "python",
      "args": ["-m", "cell.skills.core"]
    }
  }
}
```

Add any MCP-compatible server here. Cell will load all tools from all servers and make them available to the LLM automatically.

---

## Built-in Tools

| Tool | Description |
|------|-------------|
| `remember(fact)` | Saves a fact to `~/.cell/context.md` for persistent memory |
| `read_file(path)` | Reads a file |
| `write_file(path, content)` | Writes/overwrites a file |
| `replace_in_file(path, target, replacement)` | Targeted in-place edits |
| `run_command(command)` | Runs a shell command (20 s timeout) |
| `schedule_task(delay_seconds, command)` | Schedules a command to run later |

---

## Project Structure

```
cell/
├── src/cell/
│   ├── app.py          # Core TUI + vascular loop (LLM ↔ tools)
│   ├── mcp_loader.py   # Connects to MCP servers from ~/.cell/mcp.json
│   ├── cli.py          # CLI entry point + argparse
│   └── skills/
│       └── core.py     # Built-in MCP tool server
├── tests/
│   ├── test_app.py              # Live integration tests
│   └── sota_eval_scenario.yaml  # 20-task autonomous benchmark
├── pyproject.toml
└── cell.bat            # Windows convenience wrapper
```

---

## Testing

Tests make **live LLM calls** — set your `OPENAI_API_KEY` first.

```bash
# Unit tests only (no LLM)
python -m pytest tests/test_app.py::test_memory_functions

# Live integration: verify the LLM calls the remember tool
python -m pytest tests/test_app.py::test_cell_remember_tool_live

# Full 20-task autonomous SOTA benchmark (slow, ~4 min)
python -m pytest tests/test_app.py::test_sota_eval_scenario_live
```

The SOTA benchmark (`tests/sota_eval_scenario.yaml`) instructs Cell to autonomously create files, write Python, run pytest, read JSON configs, use `replace_in_file`, call `remember`, and produce a `SUCCESS.txt` — all in one continuous agentic run.

---

## Logs

All LLM completions, tool calls, and errors are logged to:

```
~/.cell/cell.log
```

Memory is persisted at:

```
~/.cell/context.md
```

---

## Philosophy

Cell is built against the "feature fallacy" in AI CLIs — the idea that adding more abstractions makes an agent more capable. Instead:

- **One loop** — stream → parse → tool call → repeat
- **One config** — `~/.cell/mcp.json` for everything external
- **One memory** — a markdown file, human-readable and editable
- **Zero framework** — no LangChain, no CrewAI, no AutoGen

Extend Cell by adding MCP servers, not by modifying Cell.
