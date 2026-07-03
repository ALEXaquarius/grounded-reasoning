"""
MCP server — phơi `verify_relation` thành một MCP tool để BẤT KỲ agent tương thích
MCP (Claude, v.v.) dùng ngay: agent tự kiểm chứng suy luận quan hệ trước khi khẳng
định, 0 token, có bằng chứng.

Chạy (cần `pip install mcp`):
    python -m src.agent.mcp_server        # stdio server

Import `mcp` được nạp LƯỜI trong build_server() nên module này import được kể cả khi
chưa cài mcp (tests/CI không cần dependency).
"""
from __future__ import annotations

from src.agent.tool import TOOL_SPEC, verify_relation


def build_server():
    """Dựng FastMCP server phơi tool verify_relation (nạp `mcp` lười)."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("grounded-reasoning")

    @server.tool(name=TOOL_SPEC["name"], description=TOOL_SPEC["description"])
    def verify_relation_tool(
        facts: list[list[str]], subject: str, relation: str, object: str  # noqa: A002
    ) -> dict:
        return verify_relation(facts, subject, relation, object)

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
