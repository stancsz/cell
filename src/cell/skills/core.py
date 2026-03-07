import subprocess
import asyncio
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Cell Core Skills")

@mcp.tool()
def read_file(path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write contents to a file. Overwrites if it exists."""
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

@mcp.tool()
def replace_in_file(path: str, target_text: str, replacement_text: str) -> str:
    """Replace an exact string with a new string in a file. Used for modifying existing files without rewriting them entirely."""
    try:
        with open(path, "r") as f:
            content = f.read()
        
        if target_text not in content:
            return f"Error: target_text not found in {path}"
            
        new_content = content.replace(target_text, replacement_text, 1) # replace first occurrence only
        
        with open(path, "w") as f:
            f.write(new_content)
        return f"Successfully replaced text in {path}"
    except Exception as e:
        return f"Error modifying file: {e}"

@mcp.tool()
async def run_command(command: str) -> str:
    """Run a terminal command and return its output. Automatically times out after 20 seconds to prevent hanging."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
            output = stdout.decode("utf-8", errors="replace")
            
            if proc.returncode != 0:
                return f"Command failed with exit code {proc.returncode}\nOutput:\n{output}"
            return f"STDOUT/STDERR:\n{output}"
            
        except asyncio.TimeoutError:
            proc.kill()
            return f"Error: Command timed out after 20 seconds. It was forcefully killed."
            
    except Exception as e:
        return f"Error running command: {e}"

@mcp.tool()
async def schedule_task(delay_seconds: int, command: str) -> str:
    """Schedule a terminal command to run after a certain delay in seconds."""
    async def task_runner():
        await asyncio.sleep(delay_seconds)
        subprocess.run(command, shell=True)
        
    asyncio.create_task(task_runner())
    return f"Scheduled command '{command}' to run in {delay_seconds} seconds."

if __name__ == "__main__":
    mcp.run()
