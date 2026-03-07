import asyncio
import os
import json
import argparse
import subprocess
from pathlib import Path
from contextlib import AsyncExitStack

from dotenv import load_dotenv
load_dotenv()

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog
from litellm import acompletion, stream_chunk_builder

from cell.mcp_loader import load_mcp_servers
import logging

LOG_FILE = Path.home() / ".cell" / "cell.log"
logging.basicConfig(filename=str(LOG_FILE), level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

MEMORY_FILE = Path.home() / ".cell" / "context.md"

def load_memory() -> str:
    if not MEMORY_FILE.exists():
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_FILE.write_text("# Cell Persistent Memory\n")
    return MEMORY_FILE.read_text()

def save_memory(fact: str) -> str:
    with open(MEMORY_FILE, "a") as f:
        f.write(f"\n- {fact}")
    return f"Fact remembered: {fact}"

def search_memory(query: str) -> str:
    """Search memory for lines matching a query (case-insensitive)."""
    hits = [l for l in MEMORY_FILE.read_text().splitlines() if query.lower() in l.lower()]
    return "\n".join(hits) if hits else "No matching memories found."

def get_env_context() -> str:
    """Collect cwd, OS, git status for injection into system prompt."""
    cwd = os.getcwd()
    git_info = ""
    try:
        branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD", shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output("git status --short", shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        git_info = f"\nGit branch: {branch}\nGit status:\n{status or '(clean)'}"
    except Exception:
        pass
    return f"OS: {os.name} | CWD: {cwd}{git_info}"

TOOLS = [
    {"type": "function", "function": {
        "name": "remember",
        "description": "Save a fact to long-term memory. Use this for important project details, decisions, credentials, or anything you'll need in future sessions.",
        "parameters": {"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]}
    }},
    {"type": "function", "function": {
        "name": "search_memory",
        "description": "Search your long-term memory for relevant facts. Use this at the start of a task to recall context from previous sessions.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    }},
]

def build_system_prompt() -> str:
    return (
        "You are CELL, an autonomous AI coding and operations agent running in a terminal.\n\n"
        "## Core Directive\n"
        "Use your tools immediately and extensively. Do NOT describe what you will do — just do it. "
        "Chain multiple tool calls in sequence to complete tasks fully. Never stop halfway.\n\n"
        "## Tool Use Rules\n"
        "- **Always read before editing**: use read_file to inspect a file before modifying it.\n"
        "- **Prefer replace_in_file over write_file** for targeted edits to existing files.\n"
        "- **Verify after writing**: read the file back to confirm correctness.\n"
        "- **Run commands to validate**: after creating code, run it with run_command to confirm it works.\n"
        "- **Chain tool calls**: complete the entire task in one continuous run — do not wait for user confirmation between steps.\n"
        "- **Use remember** to store important facts, decisions, and project context for future sessions.\n"
        "- **Use search_memory** at the start of any task to recall relevant context from past sessions.\n"
        "- **Handle errors**: if a command fails, read the output, reason about the fix, and retry.\n"
        "- **Explore the filesystem**: use run_command with `dir` (Windows) or `ls -la` to understand project structure before acting.\n\n"
        "## Coding Standards\n"
        "- Write complete, working, production-quality code — no stubs, no placeholders, no TODOs.\n"
        "- Prefer simple, minimal implementations. Avoid unnecessary abstractions.\n"
        "- Use standard library where possible before reaching for third-party packages.\n\n"
        "## Autonomy\n"
        "You have full autonomy to explore, create, modify, and run anything on this machine. "
        "When given a goal, decompose it into steps and execute ALL of them without asking for permission. "
        "Only stop when the task is fully complete and verified.\n\n"
        f"## Environment\n{get_env_context()}\n\n"
        f"## Persistent Memory\n{load_memory()}"
    )

class CellApp(App):
    """A minimalist terminal interface for Cell using Textual."""

    CSS = "Screen {background: $surface;} Input {dock: bottom; margin: 1; border: solid $primary;} RichLog {height: 1fr; padding: 1; background: $surface; border: solid $secondary;}"
    BINDINGS = [("ctrl+c", "quit", "Quit"), ("ctrl+l", "clear", "Clear")]

    def __init__(self, initial_prompt: str = None):
        super().__init__()
        self.initial_prompt = initial_prompt
        self.messages = [{"role": "system", "content": build_system_prompt()}]
        self.model = os.getenv("CELL_MODEL", "openai/gpt-5.2-codex")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="chat_log", highlight=True, markup=True)
        yield Input(placeholder="Ask Cell...", id="chat_input")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#chat_input").focus()
        log = self.query_one("#chat_log")
        log.write("[bold green]CELL[/bold green] initialized. Loading extensions...")

        self.mcp_stack = AsyncExitStack()
        self.mcp_sessions, self.mcp_tools = await load_mcp_servers(self.mcp_stack)
        loaded_servers = list(self.mcp_sessions.keys())
        if loaded_servers:
            log.write(f"[dim]Loaded MCP servers: {', '.join(loaded_servers)}[/dim]")
        log.write(f"[dim]{get_env_context()}[/dim]")
        log.write("Ready for signals.")

        if self.initial_prompt:
            log.write(f"\n[bold blue]User:[/bold blue] {self.initial_prompt}")
            self.messages.append({"role": "user", "content": self.initial_prompt})
            self.run_worker(self.process_llm())

    async def on_unmount(self) -> None:
        pass  # AsyncExitStack.aclose() must not be called from a different task

    def action_clear(self) -> None:
        self.query_one("#chat_log").clear()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if not event.value.strip():
            return
        user_input = event.value
        self.query_one("#chat_input").value = ""
        log = self.query_one("#chat_log")
        log.write(f"\n[bold blue]User:[/bold blue] {user_input}")
        self.messages.append({"role": "user", "content": user_input})
        logging.info(f"User Input: {user_input}")
        self.run_worker(self.process_llm())

    async def process_llm(self) -> None:
        """The core vascular loop (Reasoning & Tool-Use)."""
        log = self.query_one("#chat_log")
        try:
            while True:
                # Context trimming: keep system prompt + last 40 messages (more headroom for tool chains)
                if len(self.messages) > 41:
                    self.messages = [self.messages[0]] + self.messages[-40:]

                all_tools = TOOLS + getattr(self, "mcp_tools", [])
                logging.debug("Sending completion request to model: " + self.model)

                response_stream = await acompletion(
                    model=self.model,
                    messages=self.messages,
                    tools=all_tools if all_tools else None,
                    stream=True
                )

                chunks = []
                buffer = ""
                log.write("\n")

                async for chunk in response_stream:
                    chunks.append(chunk)
                    delta = chunk.choices[0].delta
                    if delta.content:
                        buffer += delta.content
                        if "\n" in buffer:
                            parts = buffer.split("\n")
                            for part in parts[:-1]:
                                log.write(f"[bold magenta]Cell:[/bold magenta] {part}")
                            buffer = parts[-1]

                if buffer:
                    log.write(f"[bold magenta]Cell:[/bold magenta] {buffer}")

                response_message = stream_chunk_builder(chunks, messages=self.messages).choices[0].message

                if response_message.content:
                    logging.info(f"Assistant Response: {response_message.content}")
                    self.messages.append({"role": "assistant", "content": response_message.content})

                if not response_message.tool_calls:
                    break

                self.messages.append(response_message.model_dump())

                for tool_call in response_message.tool_calls:
                    logging.info(f"Tool Call: {tool_call.function.name} args={tool_call.function.arguments}")
                    try:
                        args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    except Exception as e:
                        result = f"Error parsing arguments: {e}\nRaw: {tool_call.function.arguments}"
                        log.write(f"[bold red]Tool Error:[/bold red] {result}")
                        self.messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": result})
                        continue

                    name = tool_call.function.name
                    log.write(f"[dim]→ {name}({', '.join(f'{k}={repr(v)[:60]}' for k, v in args.items())})[/dim]")

                    if name == "remember":
                        result = save_memory(args.get("fact", ""))
                    elif name == "search_memory":
                        result = search_memory(args.get("query", ""))
                    elif "__" in name:
                        server_name, actual_tool_name = name.split("__", 1)
                        session = self.mcp_sessions.get(server_name)
                        if session:
                            try:
                                mcp_response = await session.call_tool(actual_tool_name, arguments=args)
                                result = "\n".join(item.text for item in mcp_response.content if item.type == "text")
                            except Exception as e:
                                result = f"Error: {e}"
                        else:
                            result = f"Server '{server_name}' not found."
                    else:
                        result = f"Unknown tool: {name}"

                    logging.info(f"Tool {name} → {result[:200]}")
                    self.messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": name, "content": result})

        except Exception as e:
            logging.error(f"Error in Vascular Loop: {e}", exc_info=True)
            log.write(f"[bold red]Error:[/bold red] {e}")

def main():
    parser = argparse.ArgumentParser(description="Cell — autonomous AI agent for the terminal")
    parser.add_argument("prompt", nargs="*", help="Initial prompt")
    args = parser.parse_args()
    CellApp(" ".join(args.prompt) if args.prompt else None).run()

if __name__ == "__main__":
    main()
