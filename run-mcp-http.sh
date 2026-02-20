#!/usr/bin/env bash
# Run the MCP DJ server over HTTP (SSE transport)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${MCP_HOST:-127.0.0.1}"
PORT="${MCP_PORT:-8000}"

echo "🔌 Starting MCP DJ over HTTP (SSE)..."
echo "   MCP endpoint: http://${HOST}:${PORT}/mcp"
echo "   Override: MCP_HOST=0.0.0.0 MCP_PORT=9000 $0"
echo ""

uv sync --quiet 2>/dev/null || true
uv run python -m mcp_dj.mcp_server --transport sse --host "$HOST" --port "$PORT"
