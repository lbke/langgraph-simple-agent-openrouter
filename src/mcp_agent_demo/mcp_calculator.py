# This file can be reused outside of LangChain
# Since it uses the MCP standard
from fastmcp import FastMCP

mcp = FastMCP("MyServer")


@mcp.tool
def add_int(a: int, b: int) -> int:
    """Add a and b."""
    return a+b


@mcp.tool
def multiply_int(a: int, b: int) -> int:
    """Multiply a and b."""
    return a*b


if __name__ == "__main__":
    mcp.run()
