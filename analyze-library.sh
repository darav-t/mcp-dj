#!/usr/bin/env bash
# Analyze all tracks in your Rekordbox library with Essentia,
# then rebuild the centralized library index in one step.
#
# After analysis completes, automatically writes:
#   .data/library_index.jsonl       — merged Rekordbox + Essentia + MIK record per track
#   .data/library_attributes.json  — dynamic tag/genre/BPM attribute summary
#   .data/library_context.md        — human-readable LLM context file
#
# Usage:
#   ./analyze-library.sh                   # analyze new/uncached tracks + rebuild index
#   ./analyze-library.sh --force           # re-analyze all tracks + rebuild index
#   ./analyze-library.sh --workers 4       # use 4 parallel worker processes
#   ./analyze-library.sh --dry-run         # preview without analyzing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

exec uv run python -m mcp_dj.analyze_library "$@"
