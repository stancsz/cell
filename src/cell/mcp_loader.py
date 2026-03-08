import json
import sys
from pathlib import Path
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

CONFIG_FILE = Path.home() / ".cell" / "mcp.json"

async def load_mcp_servers(stack: AsyncExitStack):
    """
    Connect to all MCP servers defined in ~/.cell/mcp.json.
    Returns:
       sessions (dict): Mapping of server_name -> ClientSession
       mcp_tools (list): Litellm formatted tools list
    """
    if not CONFIG_FILE.exists():
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Create a default one pointing to our built-in skills server
        default_config = {
            "mcpServers": {
                "core_skills": {
                    "command": sys.executable,
                    "args": ["-m", "cell.skills.core"]
                }
            }
        }
        CONFIG_FILE.write_text(json.dumps(default_config, indent=2))
        
    config = json.loads(CONFIG_FILE.read_text())
    sessions = {}
    mcp_tools = []
    
    for name, server_config in config.get("mcpServers", {}).items():
        try:
            params = StdioServerParameters(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env")
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            
            # Get tools
            response = await session.list_tools()
            sessions[name] = session
            
            # Format for litellm. The inputSchema is natively a dict
            mcp_tools.extend([{"type": "function", "function": {"name": f"{name}__{t.name}", "description": t.description or "", "parameters": t.inputSchema}} for t in response.tools])
        except Exception as e:
            import logging
            logging.error(f"Failed to load MCP server {name}: {e}")
            
    return sessions, mcp_tools
