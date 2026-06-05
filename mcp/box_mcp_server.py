"""
Box MCP configuration.

Box is consumed as a hosted MCP server. The config below is passed straight to
the Anthropic Messages API `mcp_servers` parameter, letting either agent fetch
documents and store artifacts through a managed, authenticated tool.
"""
from __future__ import annotations

import os
from typing import Any, Dict

BOX_MCP_CONFIG: Dict[str, Any] = {
    "type": "url",
    "url": "https://mcp.box.com",
    "name": "box-mcp",
}


def box_mcp_servers() -> list[Dict[str, Any]]:
    """Return the mcp_servers list for an Anthropic API call, or [] if no token."""
    if not os.environ.get("BOX_TOKEN"):
        return []
    return [BOX_MCP_CONFIG]


# Example usage (documentation only):
#
#   from anthropic import Anthropic
#   from mcp.box_mcp_server import box_mcp_servers
#
#   client = Anthropic()
#   response = client.messages.create(
#       model="claude-sonnet-4-20250514",
#       max_tokens=1024,
#       messages=[{"role": "user", "content": "Fetch the SDY997 cohort schema doc."}],
#       mcp_servers=box_mcp_servers(),
#   )
