# app/mcp_clients/vision_mcp_client.py

from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient


def build_vision_mcp_client() -> MultiServerMCPClient:
    """
    创建 MCP Client。

    这里用 stdio 启动本地 MCP Gateway。
    tool_name_prefix=True 可以避免以后多个 MCP Server 里工具重名。
    """

    server_path = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "mcp_servers"
        / "vision_mcp_server.py"
    )

    client = MultiServerMCPClient(
        {
            "vision": {
                "command": "python",
                "args": [str(server_path)],
                "transport": "stdio",
            }
        },
        tool_name_prefix=True,  # 可以把工具名加上 server 前缀，避免多个 MCP Server 中工具重名
        # handle_tool_errors=True,  # 让 MCP tool 执行错误以 tool message 返回给模型，便于 agent 自我修正
    )

    return client


async def load_vision_mcp_tools():
    """
    从 MCP Server 加载 LangChain-compatible tools。
    """

    client = build_vision_mcp_client()
    tools = await client.get_tools()
    return client, tools
