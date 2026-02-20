#!/usr/bin/env bash
# Run the DJ Setlist Creator MCP server for Claude Desktop
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔌 Starting DJ Setlist Creator MCP Server..."
echo "   Connect from Claude Desktop using config below:"
echo ""
echo '  "dj-setlist-creator": {'
echo '    "command": "uv",'
echo "    \"args\": [\"run\", \"--project\", \"$SCRIPT_DIR\", \"python\", \"-m\", \"mcp_dj.mcp_server\"]"
echo '  }'
echo ""

uv sync --quiet 2>/dev/null || true
uv run python -m mcp_dj.mcp_server
