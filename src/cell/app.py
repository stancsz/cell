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
from textual.widgets import Header, Footer, Input, RichLog, TextArea, Label
from textual.binding import Binding
import litellm
from litellm import acompletion, stream_chunk_builder

litellm.suppress_debug_info = True
litellm.set_verbose = False

from cell.mcp_loader import load_mcp_servers
import logging

LOG_FILE = Path.home() / ".cell" / "cell.log"
logging.basicConfig(
    filename=str(LOG_FILE), 
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    force=True
)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

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

ASCII_ART = r"""
[bold cyan] █▀▀ █▀▀ █   █   [/bold cyan] [white]CELL[/white] [dim]v1.0[/dim]
[bold cyan] █   █▀▀ █   █   [/bold cyan] [dim]───────[/dim]
[bold cyan] ▀▀▀ ▀▀▀ ▀▀▀ ▀▀▀ [/bold cyan] [bold white]READY[/bold white]
"""

class ChatInput(TextArea):
    """Multiline input that submits on Enter and newlines on Shift+Enter."""
    BINDINGS = [
        Binding("enter", "submit", "Submit Message", show=True, priority=True),
        Binding("shift+enter", "newline", "Newline", show=True, priority=True),
    ]

    def action_submit(self) -> None:
        text = self.text.strip()
        if text:
            # Forward the text to the parent app and clear
            self.app.submit_chat_input(text)
            self.text = ""

    def action_newline(self) -> None:
        self.insert("\n")

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
    is_eval = os.getenv("CELL_EVAL_MODE", "0") == "1"

    completion_protocol = (
        "3. Your LAST action must always be a tool call (write_file, run_command, or read_file), never plain text.\n"
        "   Outputting only text means you have stopped working — only do this after physically verifying all deliverables exist.\n"
        "4. If pytest or any command times out, retry with a shorter subset or debug the failure — never give up.\n"
        "5. WHEN YOU ARE 100% FINISHED, use write_file to create a file named `.eval_done` in the current directory containing the word 'DONE'. This is how you signal completion.\n\n"
    ) if is_eval else (
        "3. When you are finished, or if you need to ask the user a clarifying question, simply communicate it using plain text. Do not force tool usage if it's unnecessary.\n"
        "4. If a command times out, retry it or debug it.\n\n"
    )

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
        "## Completion Protocol — CRITICAL\n"
        "Before considering ANY task done, you MUST:\n"
        "1. Run `dir` (or `ls -la`) on every output directory to confirm every required file actually exists on disk.\n"
        "2. If a required file is missing, create it immediately — do not skip it.\n"
        f"{completion_protocol}"
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

    CSS = """
    Screen {
        background: $background;
    }
    RichLog {
        height: 1fr;
        padding: 0 1;
        background: $background;
        border: none;
    }
    #status_label {
        color: $accent;
        padding: 0 2;
        text-style: italic;
    }
    ChatInput {
        dock: bottom;
        height: auto;
        max-height: 40%;
        margin: 1 2;
        border: round $primary;
        background: $surface;
    }
    """
    BINDINGS = [("ctrl+c", "quit", "Quit"), ("ctrl+l", "clear", "Clear")]

    def __init__(self, initial_prompt: str = None):
        super().__init__()
        self.initial_prompt = initial_prompt
        self.messages = [{"role": "system", "content": build_system_prompt()}]
        self.model = os.getenv("CELL_MODEL", "openai/gpt-5.2-codex")
        # The original task is pinned here so context compaction can always
        # restore it verbatim, even after multiple compaction cycles.
        self._original_task: dict | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="chat_log", highlight=True, markup=True)
        yield Label("", id="status_label")
        yield ChatInput(id="chat_input", text="", language="markdown")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#chat_input").focus()
        log = self.query_one("#chat_log")
        log.write(ASCII_ART)
        log.write("[bold green]CELL[/bold green] initialized. Loading extensions...")

        self.mcp_stack = AsyncExitStack()
        self.mcp_sessions, self.mcp_tools = await load_mcp_servers(self.mcp_stack)
        loaded_servers = list(self.mcp_sessions.keys())
        if loaded_servers:
            log.write(f"[dim]Loaded MCP servers: {', '.join(loaded_servers)}[/dim]")
        log.write(f"[dim]{get_env_context()}[/dim]")
        log.write("Ready for signals.")

        if self.initial_prompt:
            log.write(f"[bold blue]User:[/bold blue] {self.initial_prompt}")
            task_msg = {"role": "user", "content": self.initial_prompt}
            self.messages.append(task_msg)
            self._original_task = task_msg
            self.run_worker(self.process_llm())

    async def on_unmount(self) -> None:
        pass  # AsyncExitStack.aclose() must not be called from a different task

    def action_clear(self) -> None:
        self.query_one("#chat_log").clear()

    def submit_chat_input(self, user_input: str) -> None:
        self.query_one("#chat_input").text = ""
        log = self.query_one("#chat_log")
        # Pretty print
        log.write(f"[bold blue]User:[/bold blue] {user_input}")
        task_msg = {"role": "user", "content": user_input}
        self.messages.append(task_msg)
        # Pin the first user message as the original task so compaction never loses it.
        if self._original_task is None:
            self._original_task = task_msg
        logging.info(f"User Input: {user_input}")
        self.run_worker(self.process_llm())

    async def compact_context(self) -> None:
        """Summarize completed work into a digest so the agent never forgets its task.

        Strategy:
          [0]  system prompt          — always kept verbatim
          [1]  original user task     — always kept verbatim (the spec to satisfy)
          [2]  <DIGEST>               — LLM-written summary of what has been done so far
          [-20:]  recent messages     — kept verbatim for immediate tool context
        """
        log = self.query_one("#chat_log")
        log.write("[dim]⟳ Compacting context…[/dim]")
        logging.info("Context compaction triggered.")

        # Everything between the original task and the recent tail is summarised.
        tail_keep = 20
        # Find the original task: use the pinned reference, fall back to messages[1].
        original_task = self._original_task or self.messages[1]
        middle = self.messages[2 : len(self.messages) - tail_keep]
        recent  = self.messages[-tail_keep:]

        if not middle:
            return  # nothing old enough to compact

        # Build a summarisation prompt using the same model.
        summarise_messages = [
            {
                "role": "system",
                "content": (
                    "You are a context compactor for an autonomous coding agent. "
                    "You will be given a sequence of messages showing the agent's work so far. "
                    "Produce a compact, bullet-point PROGRESS DIGEST that captures:\n"
                    "  • Which files have been created or modified (with paths).\n"
                    "  • Which commands were run and their key outcomes.\n"
                    "  • Which sub-tasks are complete, and which are still outstanding.\n"
                    "  • Any errors encountered and how they were resolved.\n"
                    "Be dense and precise. Do NOT omit file paths or outstanding steps. "
                    "Maximum 400 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Here is the agent's message history to summarise:\n\n"
                    + "\n".join(
                        f"[{m['role'].upper()}] "
                        + (
                            m.get("content") or
                            str([tc.get("function", {}).get("name") for tc in m.get("tool_calls", [])])
                        )[:400]
                        for m in middle
                    )
                ),
            },
        ]

        try:
            digest_response = await acompletion(
                model=self.model,
                messages=summarise_messages,
                stream=False,
            )
            digest_text = digest_response.choices[0].message.content or "(no digest)"
        except Exception as e:
            digest_text = f"(compaction failed: {e})"
            logging.warning(f"Context compaction failed: {e}")

        digest_msg = {
            "role": "user",
            "content": (
                "[CONTEXT COMPACTED — PROGRESS DIGEST]\n"
                "The following summarises what has been accomplished so far. "
                "Your original task instructions above still apply in full — "
                "continue from where you left off.\n\n"
                + digest_text
            ),
        }

        # Rebuild: system + original_task (pinned) + digest + recent tail
        self.messages = [self.messages[0], original_task, digest_msg] + recent
        log.write(f"[dim]✓ Context compacted. Digest: {len(digest_text)} chars. "
                  f"Messages: {len(self.messages)}[/dim]")
        logging.info(f"Context compacted. New length: {len(self.messages)}")

    async def process_llm(self) -> None:
        """The core vascular loop (Reasoning & Tool-Use)."""
        log = self.query_one("#chat_log")
        status_label = self.query_one("#status_label", Label)
        no_tool_retries = 0
        try:
            while True:
                status_label.update("")
                
                # Context compaction: when messages grow too large, summarise the
                # middle history while preserving [system, original_task, recent_20].
                # This ensures the agent never forgets its task, unlike hard truncation.
                max_ctx = int(os.getenv("CELL_MAX_CONTEXT", "80"))
                if len(self.messages) > max_ctx:
                    await self.compact_context()

                all_tools = TOOLS + getattr(self, "mcp_tools", [])
                
                # Fast route check to avoid long thinking on simple conversational inputs
                if self.messages[-1]["role"] == "user" and os.getenv("CELL_EVAL_MODE", "0") != "1":
                    status_label.update("Determining tool requirements...")
                    router_prompt = {
                        "role": "user",
                        "content": (
                            "Analyze my previous message carefully. "
                            "If it requires ANY system action (e.g. running commands, reading/writing files, viewing directories, modifying code, saving/searching memory, testing, etc.), reply with the exact word 'TOOLS'. "
                            "If it is STRICTLY a simple conversational greeting, an acknowledgment (like 'ok' or 'thanks'), or a generic question that requires zero system access or memory operations, reply with the exact word 'CONVERSATION'."
                        )
                    }
                    try:
                        router_resp = await acompletion(
                            model=self.model,
                            messages=[self.messages[-1], router_prompt],
                            max_tokens=10,
                            temperature=0,
                            stream=False
                        )
                        decision = router_resp.choices[0].message.content.strip().upper()
                        if "CONVERSATION" in decision:
                            all_tools = []
                            logging.info("Router decided: CONVERSATION (skipping tools)")
                        else:
                            logging.info(f"Router decided: {decision} (using tools)")
                    except Exception as e:
                        logging.warning(f"Router check failed: {e}")

                logging.debug("Sending completion request to model: " + self.model)

                status_label.update("Thinking...")
                response_stream = await acompletion(
                    model=self.model,
                    messages=self.messages,
                    tools=all_tools if all_tools else None,
                    stream=True
                )

                chunks = []
                buffer = ""
                status_label.update("")
                
                is_first_line = True

                async for chunk in response_stream:
                    chunks.append(chunk)
                    delta = chunk.choices[0].delta
                    if delta.content:
                        buffer += delta.content
                        if "\n" in buffer:
                            parts = buffer.split("\n")
                            for part in parts[:-1]:
                                prefix = "[bold magenta]Cell:[/bold magenta] " if is_first_line else ""
                                log.write(f"{prefix}{part}")
                                is_first_line = False
                            buffer = parts[-1]

                if buffer:
                    prefix = "[bold magenta]Cell:[/bold magenta] " if is_first_line else ""
                    log.write(f"{prefix}{buffer}")

                response_message = stream_chunk_builder(chunks, messages=self.messages).choices[0].message

                if response_message.content:
                    logging.info(f"Assistant Response: {response_message.content}")
                    self.messages.append({"role": "assistant", "content": response_message.content})

                if not response_message.tool_calls:
                    if os.getenv("CELL_EVAL_MODE", "0") != "1":
                        # In normal interactive mode, just stop the loop so the user can read and reply.
                        break

                    no_tool_retries += 1
                    if no_tool_retries >= 3:
                        log.write("[bold red]System:[/bold red] Repeated text-only responses. Pausing for user input.")
                        break

                    # In autonomous eval execution, if the model stops tool-calling
                    # before the task is done, prompt it to continue rather than aborting.
                    log.write(f"[bold red]System:[/bold red] No tool calls detected (Retry {no_tool_retries}/3).")
                    logging.info(f"No tool calls detected. Prompting to continue ({no_tool_retries}/3).")
                    self.messages.append(response_message.model_dump())
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "You replied with plain text and no tool calls. "
                            "If you are finished with the ENTIRE task, create the '.eval_done' "
                            "file using write_file to signal completion. "
                            "Otherwise, you MUST use a tool to continue your work."
                        )
                    })
                    continue

                no_tool_retries = 0
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
                    status_label.update(f"⚙️ Running {name}...")

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
