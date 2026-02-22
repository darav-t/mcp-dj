#!/usr/bin/env bash
# Rebuild the centralized library index from the existing Essentia cache.
# No audio re-analysis is performed — reads .data/essentia_cache/ directly.
#
# Writes:
#   .data/library_index.jsonl       — merged Rekordbox + Essentia + MIK record per track
#   .data/library_attributes.json  — dynamic tag/genre/BPM attribute summary
#   .data/library_context.md        — human-readable LLM context file
#
# Usage:
#   ./build-library-index.sh           # rebuild if index is older than 1 hour
#   ./build-library-index.sh --force   # always rebuild

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

exec uv run build-library-index "$@"
