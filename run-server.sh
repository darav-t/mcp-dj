#!/usr/bin/env bash
# Run the DJ Setlist Creator web server
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "⚠️  Warning: ANTHROPIC_API_KEY not set. AI chat will use fallback mode."
  echo "   Export it: export ANTHROPIC_API_KEY=sk-ant-..."
  echo ""
fi

PORT="${SETLIST_PORT:-8888}"
echo "🎛️  Starting DJ Setlist Creator on http://localhost:$PORT"
echo "   Press Ctrl+C to stop."
echo ""

# Install deps if needed
uv sync --quiet 2>/dev/null || true

SETLIST_PORT=$PORT uv run python -m mcp_dj.app
