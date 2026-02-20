#!/usr/bin/env bash
# Run the DJ Setlist Creator MCP server (direct venv invocation)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

exec "$SCRIPT_DIR/.venv/bin/python3" -m mcp_dj.mcp_server
