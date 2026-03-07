import asyncio
import os
import json
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from contextlib import AsyncExitStack

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog
from litellm import acompletion, stream_chunk_builder

from cell.mcp_loader import load_mcp_servers
import logging

LOG_FILE = Path.home() / ".cell" / "cell.log"
logging.basicConfig(filename=str(LOG_FILE), level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

# Memory file path for persistent state without a DB
MEMORY_FILE = Path.home() / ".cell" / "context.md"

def load_memory() -> str:
    """Load persistent memory from the system-injected markdown file."""
    if not MEMORY_FILE.exists():
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_FILE.write_text("# Cell Persistent Memory\n")
    return MEMORY_FILE.read_text()

def save_memory(fact: str) -> str:
    """Save a fact to the memory file."""
    with open(MEMORY_FILE, "a") as f:
        f.write(f"\n- {fact}")
    return f"Fact remembered: {fact}"

# Define baseline tools (This is where native tools reside)
TOOLS = [{"type": "function", "function": {"name": "remember", "description": "Remember a fact in the long-term memory", "parameters": {"type": "object", "properties": {"fact": {"type": "string", "description": "The fact to remember"}}, "required": ["fact"]}}}]

class CellApp(App):
    """A minimalist terminal interface for Cell using Textual."""
    
    CSS = "Screen {background: $surface;} Input {dock: bottom; margin: 1; border: solid $primary;} RichLog {height: 1fr; padding: 1; background: $surface; border: solid $secondary;}"
    
    BINDINGS = [("ctrl+c", "quit", "Quit"), ("ctrl+l", "clear", "Clear")]

    def __init__(self, initial_prompt: str = None):
        super().__init__()
        self.initial_prompt = initial_prompt
        # Initial Vascular loop setup
        self.messages = [
            {
                "role": "system",
                "content": (
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
                    "- **Use remember** to store important facts, decisions, or context for future sessions.\n"
                    "- **Handle errors**: if a command fails, read the output, reason about the fix, and retry.\n\n"
                    "## Coding Standards\n"
                    "- Write complete, working, production-quality code — no stubs or placeholders.\n"
                    "- Prefer simple, minimal implementations. Avoid unnecessary abstractions.\n"
                    "- Use standard library where possible before reaching for third-party packages.\n\n"
                    "## Autonomy\n"
                    "You have full autonomy to explore, create, modify, and run anything on this machine. "
                    "When given a goal, decompose it into steps and execute all of them without asking for permission. "
                    "Only stop when the task is fully complete and verified.\n\n"
                    f"## Persistent Memory\n{load_memory()}"
                )
            }
        ]
        # Use litellm standard format so you can pass gpt-4o, claude-3.5-sonnet, openai/gpt-5.2-codex, etc.
        self.model = os.getenv("CELL_MODEL", "openai/gpt-5.2-codex")

    def compose(self) -> ComposeResult:
        """Create the TUI layout."""
        yield Header(show_clock=True)
        yield RichLog(id="chat_log", highlight=True, markup=True)
        yield Input(placeholder="Ask Cell...", id="chat_input")
        yield Footer()

    async def on_mount(self) -> None:
        """Focus the input and welcome the user upon start."""
        self.query_one("#chat_input").focus()
        log = self.query_one("#chat_log")
        log.write("[bold green]CELL[/bold green] initialized. Loading extensions...")
        
        # Load MCP tools
        self.mcp_stack = AsyncExitStack()
        self.mcp_sessions, self.mcp_tools = await load_mcp_servers(self.mcp_stack)
        loaded_servers = list(self.mcp_sessions.keys())
        if loaded_servers:
            log.write(f"[dim]Loaded MCP servers: {', '.join(loaded_servers)}[/dim]")
        log.write("Ready for signals.")
        
        if self.initial_prompt:
            log.write(f"\n[bold blue]User:[/bold blue] {self.initial_prompt}")
            self.messages.append({"role": "user", "content": self.initial_prompt})
            self.run_worker(self.process_llm())
            
    async def on_unmount(self) -> None:
        """Clean up MCP connections."""
        # Note: Do not call await self.mcp_stack.aclose() here because Textual 
        # executes unmount in a different async task, causing AsyncExitStack to throw a RuntimeError.
        pass

    def action_clear(self) -> None:
        """Clear the chat window."""
        self.query_one("#chat_log").clear()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Triggered when the user hits Enter."""
        if not event.value.strip():
            return
            
        user_input = event.value
        input_widget = self.query_one("#chat_input")
        log = self.query_one("#chat_log")
        
        input_widget.value = ""
        log.write(f"\n[bold blue]User:[/bold blue] {user_input}")
        
        self.messages.append({"role": "user", "content": user_input})
        
        # Fire off completion task
        logging.info(f"User Input: {user_input}")
        self.run_worker(self.process_llm())

    async def process_llm(self) -> None:
        """The core vascular loop (Reasoning & Tool-Use)."""
        log = self.query_one("#chat_log")
        try:
            while True:
                # 0. Context Trimming (Keep System Prompt + Last 20 messages)
                if len(self.messages) > 21:
                    # Index 0 is System, slice off the oldest messages over the limit
                    self.messages = [self.messages[0]] + self.messages[-20:]
                
                # 1. Talk to LLM
                all_tools = TOOLS + getattr(self, "mcp_tools", [])
                logging.debug("Sending completion request to model: " + self.model)
                
                # Stream the completion
                response_stream = await acompletion(
                    model=self.model,
                    messages=self.messages,
                    tools=all_tools if all_tools else None,
                    stream=True
                )
                
                chunks = []
                buffer = ""
                log.write("\n") # New line for new message
                
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
                
                # Store assistant message
                if response_message.content:
                    logging.info(f"Assistant Response: {response_message.content}")
                    self.messages.append({"role": "assistant", "content": response_message.content})
                
                # 2. If no tool use, break out of loop
                if not response_message.tool_calls:
                    break
                    
                # 3. If Tool Use, execute tool and continue loop
                self.messages.append(response_message.model_dump())  # Send the function call record
                
                for tool_call in response_message.tool_calls:
                    logging.info(f"Tool Call Initiated: {tool_call.function.name} with args: {tool_call.function.arguments}")
                    try:
                        args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    except Exception as e:
                        result = f"Error parsing arguments: {str(e)}\nRaw arguments: {tool_call.function.arguments}"
                        log.write(f"[bold red]Tool Error:[/bold red] {result}")
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": result
                        })
                        continue
                        
                    if tool_call.function.name == "remember":
                        log.write(f"[dim]Executing tool: remember('{args.get('fact', '')}')[/dim]")
                        result = save_memory(args.get('fact', ''))
                    elif "__" in tool_call.function.name:
                        server_name, actual_tool_name = tool_call.function.name.split("__", 1)
                        log.write(f"[dim]Executing external: {tool_call.function.name}[/dim]")
                        
                        session = self.mcp_sessions.get(server_name)
                        if session:
                            try:
                                mcp_response = await session.call_tool(actual_tool_name, arguments=args)
                                result = "\n".join(item.text for item in mcp_response.content if item.type == "text")
                            except Exception as e:
                                result = f"Error: {e}"
                        else:
                            result = f"Server {server_name} not found."
                    else:
                        log.write(f"[bold red]Unknown tool:[/bold red] {tool_call.function.name}")
                        result = "Unknown tool"
                        
                    logging.info(f"Tool {tool_call.function.name} output: {result}")
                        
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": result
                    })
                        
        except Exception as e:
            logging.error(f"Error in Vascular Loop: {str(e)}", exc_info=True)
            log.write(f"[bold red]Error in Vascular Loop:[/bold red] {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Cell - A minimalist AI CLI with TUI")
    parser.add_argument("prompt", nargs="*", help="Initial prompt to send to Cell")
    args = parser.parse_args()
    
    initial_prompt = " ".join(args.prompt) if args.prompt else None
    
    app = CellApp(initial_prompt=initial_prompt)
    app.run()

if __name__ == "__main__":
    main()
