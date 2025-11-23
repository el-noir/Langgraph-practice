from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arithmetic-tool")

@mcp.tool("add", description="Add two numbers")
def add(a: float, b: float) -> float:
    return a + b

@mcp.tool("subtract", description="Subtract two numbers")
def subtract(a: float, b: float) -> float:
    return a - b

@mcp.tool("multiply", description="Multiply two numbers")
def multiply(a: float, b: float) -> float:
    return a * b

@mcp.tool("divide", description="Divide two numbers")
def divide(a: float, b: float) -> float:
    if b == 0:
        return "Error: Division by zero"
    return a / b

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
    print("Server started")