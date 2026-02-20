#!/usr/bin/env bash
# Analyze all tracks in your Rekordbox library with Essentia.
#
# Usage:
#   ./analyze-library.sh                   # analyze new/uncached tracks
#   ./analyze-library.sh --force           # re-analyze all tracks
#   ./analyze-library.sh --workers 4        # use 4 parallel worker processes
#   ./analyze-library.sh --dry-run         # preview without analyzing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

exec uv run python -m mcp_dj.analyze_library "$@"
