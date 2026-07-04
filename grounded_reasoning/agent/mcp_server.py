"""
MCP server — exposes `verify_relation` as an MCP tool so ANY MCP-compatible agent
(Claude, etc.) can use it directly: the agent verifies a relational claim before
asserting it, at 0 tokens, with a proof.

Run (requires `pip install mcp`):
    python -m grounded_reasoning.agent.mcp_server        # stdio server

The `mcp` import is LAZY-loaded inside build_server() so this module still imports
cleanly even without `mcp` installed (tests/CI don't need the dependency).
"""
from __future__ import annotations

from grounded_reasoning.agent.tool import TOOL_SPEC, verify_relation


def build_server():
    """Build a FastMCP server exposing the verify_relation tool (lazy `mcp` import)."""
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
