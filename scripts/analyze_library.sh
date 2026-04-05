#!/usr/bin/env bash
# Master library analysis script.
#
# Runs in order:
#   1. Essentia audio analysis  (BPM, key, mood, genre, danceability, loudness)
#   2. Phrase detection         (Intro/Up/Chorus/Down/Outro + mix profiles)
#
# Both steps write into library_index.jsonl atomically.
# Already-cached tracks are skipped unless --force is passed.
#
# Usage:
#   ./scripts/analyze_library.sh              # analyze new tracks only
#   ./scripts/analyze_library.sh --force      # re-analyze everything
#   ./scripts/analyze_library.sh --limit 50   # test on first 50 tracks
#
# The server must be running:
#   uv run python -m mcp_dj.app

set -euo pipefail

HOST="${MCP_DJ_HOST:-http://localhost:8888}"
FORCE=false
LIMIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)  FORCE=true; shift ;;
    --limit)  LIMIT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -n "$LIMIT" ]]; then
  BODY="{\"force\": $FORCE, \"limit\": $LIMIT}"
else
  BODY="{\"force\": $FORCE}"
fi

echo "=== MCP DJ — Master Library Analysis ==="
echo "Server : $HOST"
echo "Force  : $FORCE"
[[ -n "$LIMIT" ]] && echo "Limit  : $LIMIT tracks"
echo ""

if ! curl -sf "$HOST/" -o /dev/null; then
  echo "ERROR: Server not reachable at $HOST"
  echo "Start it first with: uv run python -m mcp_dj.app"
  exit 1
fi

# ── Step 1: Essentia (includes phrase analysis as a follow-on step) ──────────
echo "Step 1/1: Essentia + Phrase analysis (this may take a while)…"
RESPONSE=$(curl -sf -X POST "$HOST/api/essentia/analyze-library" \
  -H "Content-Type: application/json" \
  -d "$BODY")

echo "$RESPONSE" | python3 -c "
import json, sys
r = json.load(sys.stdin)
print()
print('── Essentia ──────────────────────────')
print(f'  Analyzed new     : {r.get(\"analyzed_new\", 0)}')
print(f'  Loaded from cache: {r.get(\"loaded_from_cache\", 0)}')
print(f'  Skipped (no file): {r.get(\"skipped_no_file_path\", 0) + r.get(\"skipped_missing_file\", 0)}')
print(f'  Errors           : {r.get(\"errors\", 0)}')
print()
print('── Phrase Detection ──────────────────')
print(f'  Analyzed new     : {r.get(\"phrases_analyzed_new\", 0)}')
print(f'  Loaded from cache: {r.get(\"phrases_from_cache\", 0)}')
print(f'  Errors           : {r.get(\"phrases_errors\", 0)}')
print()
if r.get('library_index_rebuilt'):
    print(f'── Library Index rebuilt: {r.get(\"library_index_total\", 0)} records ──')
print()
print('Done.')
"
