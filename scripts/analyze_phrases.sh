#!/usr/bin/env bash
# Analyze phrase structure for all tracks in the MCP DJ library.
# Results are cached in .data/phrase_cache/ — already-cached tracks are skipped.
#
# Usage:
#   ./scripts/analyze_phrases.sh              # analyze new tracks only
#   ./scripts/analyze_phrases.sh --force      # re-analyze everything
#   ./scripts/analyze_phrases.sh --limit 50   # analyze first 50 tracks (testing)
#
# The server must be running first:
#   uv run python -m mcp_dj.app

set -euo pipefail

HOST="${MCP_DJ_HOST:-http://localhost:8888}"
FORCE=false
LIMIT=""

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)  FORCE=true; shift ;;
    --limit)  LIMIT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Build JSON body
if [[ -n "$LIMIT" ]]; then
  BODY="{\"force\": $FORCE, \"limit\": $LIMIT}"
else
  BODY="{\"force\": $FORCE}"
fi

echo "=== MCP DJ — Phrase Library Analysis ==="
echo "Server : $HOST"
echo "Force  : $FORCE"
[[ -n "$LIMIT" ]] && echo "Limit  : $LIMIT tracks"
echo ""
echo "Starting... (this can take several hours for a full library)"
echo "Already-cached tracks will be skipped unless --force is set."
echo ""

# Check server is reachable
if ! curl -sf "$HOST/" -o /dev/null; then
  echo "ERROR: Server not reachable at $HOST"
  echo "Start it first with: uv run python -m mcp_dj.app"
  exit 1
fi

# Fire the request (server processes synchronously, so this blocks until done)
RESPONSE=$(curl -sf -X POST "$HOST/api/phrases/analyze-library" \
  -H "Content-Type: application/json" \
  -d "$BODY")

# Pretty-print result
echo "$RESPONSE" | python3 -c "
import json, sys
r = json.load(sys.stdin)
print(f'  Total in library : {r[\"total_tracks_in_library\"]}')
print(f'  Analyzed new     : {r[\"analyzed_new\"]}')
print(f'  Loaded from cache: {r[\"loaded_from_cache\"]}')
print(f'  Skipped (no file): {r[\"skipped_no_file\"]}')
print(f'  Errors           : {r[\"errors\"]}')
print(f'  Cache dir        : {r[\"cache_directory\"]}')
print()
print(r['message'])
"
