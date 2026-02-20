#!/usr/bin/env bash
# =============================================================================
# DJ Setlist Creator — Install Script
# =============================================================================
# Installs everything needed to run the project:
#   1. uv          — Python package manager
#   2. Python 3.12 — via uv
#   3. Core deps   — pyrekordbox, pydantic, fastapi, anthropic, fastmcp, etc.
#   4. Essentia    — audio analysis (essentia-tensorflow, optional but recommended)
#   5. ML models   — mood, genre, tagging models in .data/models/ (optional)
#   6. Rekordbox   — database key setup (pyrekordbox)
#   7. .env        — environment config file
#
# Usage:
#   ./install.sh                   # core only
#   ./install.sh --essentia        # + essentia-tensorflow + ML models
#   ./install.sh --essentia --skip-models  # + essentia but skip model download
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

INSTALL_ESSENTIA=false
SKIP_MODELS=false

for arg in "$@"; do
  case "$arg" in
    --essentia)      INSTALL_ESSENTIA=true ;;
    --skip-models)   SKIP_MODELS=true ;;
    --help|-h)
      echo "Usage: ./install.sh [--essentia] [--skip-models]"
      echo ""
      echo "  --essentia      Install essentia-tensorflow + download ML models"
      echo "                  Enables: BPM, key, mood, genre, tagging analysis"
      echo "                  Run: analyze-track /path/to/song.mp3"
      echo ""
      echo "  --skip-models   Install essentia-tensorflow but skip model download"
      echo "                  Download later with: ./download_models.sh"
      exit 0
      ;;
  esac
done

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
error()   { echo -e "${RED}[✗]${NC} $*" >&2; }
header()  { echo -e "\n${BOLD}$*${NC}"; }
divider() { echo "------------------------------------------------------------"; }

# -----------------------------------------------------------------------------
# 1. uv
# -----------------------------------------------------------------------------

header "Step 1 — uv package manager"
divider

if command -v uv &>/dev/null; then
  info "uv already installed: $(uv --version 2>&1 | head -1)"
else
  warn "uv not found — installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
  if command -v uv &>/dev/null; then
    info "uv installed: $(uv --version 2>&1 | head -1)"
  else
    error "uv installation failed."
    error "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
  fi
fi

# -----------------------------------------------------------------------------
# 2. Python 3.12+
# -----------------------------------------------------------------------------

header "Step 2 — Python 3.12+"
divider

if uv python find 3.12 &>/dev/null 2>&1; then
  info "Python 3.12+ available"
else
  warn "Python 3.12 not found — installing via uv..."
  uv python install 3.12
  info "Python 3.12 installed"
fi

# -----------------------------------------------------------------------------
# 3. Core Python dependencies
# -----------------------------------------------------------------------------

header "Step 3 — Core Python packages"
divider

echo "  pyrekordbox  — Rekordbox database access"
echo "  pydantic     — data models"
echo "  fastapi      — web server"
echo "  uvicorn      — ASGI server"
echo "  loguru       — logging"
echo "  anthropic    — Claude API"
echo "  fastmcp      — MCP server for Claude Desktop"
echo ""

uv sync
info "Core packages installed"

# -----------------------------------------------------------------------------
# 4. Essentia (optional — recommended for audio analysis)
# -----------------------------------------------------------------------------

header "Step 4 — Essentia audio analysis"
divider

if [ "$INSTALL_ESSENTIA" = true ]; then
  echo "Installing essentia-tensorflow..."
  echo "  Package size: ~135 MB"
  echo ""

  if uv sync --extra essentia; then
    info "essentia-tensorflow installed"
  else
    # Fallback: add it directly if --extra sync fails
    warn "uv sync --extra failed, trying uv add..."
    if uv add essentia-tensorflow; then
      info "essentia-tensorflow installed"
    else
      error "essentia-tensorflow installation failed."
      warn "Try manually: uv add essentia-tensorflow"
      INSTALL_ESSENTIA=false
    fi
  fi
else
  echo "  Essentia analyzes your songs to extract:"
  echo "    • Accurate BPM + beat confidence"
  echo "    • Key detection (Camelot wheel)"
  echo "    • Danceability scoring"
  echo "    • EBU R128 loudness (LUFS + RMS dB)"
  echo "    • Mood classification (happy/sad/aggressive/relaxed/party)"
  echo "    • Genre detection (Discogs400 — 400 styles)"
  echo "    • Music autotagging (techno, beat, electronic, etc.)"
  echo ""
  echo "  Package size: ~135 MB + ~300 MB ML models"
  echo ""
  read -r -p "  Install Essentia for song analysis? [y/N] " REPLY
  REPLY="${REPLY:-N}"
  if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    INSTALL_ESSENTIA=true
    echo ""
    echo "Installing essentia-tensorflow..."
    echo ""
    if uv sync --extra essentia; then
      info "essentia-tensorflow installed"
    else
      warn "uv sync --extra failed, trying uv add..."
      if uv add essentia-tensorflow; then
        info "essentia-tensorflow installed"
      else
        error "essentia-tensorflow installation failed."
        warn "Try manually: uv add essentia-tensorflow"
        INSTALL_ESSENTIA=false
      fi
    fi
  else
    warn "Skipping Essentia — install later with: ./install.sh --essentia"
  fi
fi

# -----------------------------------------------------------------------------
# 5. ML model files (only if essentia was installed)
# -----------------------------------------------------------------------------

header "Step 5 — Essentia ML models"
divider

MODEL_DIR="$SCRIPT_DIR/.data/models"

if [ "$INSTALL_ESSENTIA" = true ]; then
  if [ "$SKIP_MODELS" = true ]; then
    warn "Skipping model download (--skip-models)"
    echo "  Download later: ./download_models.sh"
  else
    # Count already-downloaded models
    EXISTING=0
    if [ -d "$MODEL_DIR" ]; then
      EXISTING=$(find "$MODEL_DIR" -name "*.pb" 2>/dev/null | wc -l | tr -d ' ')
    fi

    if [ "$EXISTING" -ge 20 ]; then
      info "All ML models already present ($EXISTING .pb files in .data/models/)"
    else
      echo "Downloading ML models to .data/models/ (git-ignored)..."
      echo "  Models: VGGish, EffNet, mood (×5), genre Discogs400, MagnaTagATune tags"
      echo "  Total size: ~300 MB"
      echo ""
      if bash "$SCRIPT_DIR/download_models.sh"; then
        info "ML models downloaded"
      else
        warn "Some models failed to download."
        warn "Re-run later: ./download_models.sh"
      fi
    fi
  fi
else
  warn "Skipping ML models (Essentia not installed)"
  echo "  Download later after installing Essentia: ./download_models.sh"
fi

# -----------------------------------------------------------------------------
# 6. Rekordbox database key
# -----------------------------------------------------------------------------

header "Step 6 — Rekordbox database access"
divider

echo "Checking pyrekordbox configuration..."

if uv run python -c "import pyrekordbox; pyrekordbox.show_config()" 2>/dev/null | grep -q "app_password\|db_key"; then
  info "pyrekordbox already configured"
else
  echo ""
  echo "  pyrekordbox needs the Rekordbox master.db decryption key."
  echo "  This is read automatically from your Rekordbox installation."
  echo "  Rekordbox must be installed on this machine."
  echo ""
  read -r -p "  Run pyrekordbox setup now? [Y/n] " REPLY
  REPLY="${REPLY:-Y}"
  if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    if uv run python -m pyrekordbox setup-db; then
      info "Rekordbox database key configured"
    else
      warn "Setup failed — run manually once Rekordbox is installed:"
      warn "  uv run python -m pyrekordbox setup-db"
    fi
  else
    warn "Skipped — run before starting the server:"
    warn "  uv run python -m pyrekordbox setup-db"
  fi
fi

# -----------------------------------------------------------------------------
# 7. .env file
# -----------------------------------------------------------------------------

header "Step 7 — Environment configuration"
divider

ENV_FILE="$SCRIPT_DIR/.env"
ENV_EXAMPLE="$SCRIPT_DIR/.env.example"

if [ -f "$ENV_FILE" ]; then
  info ".env already exists"
else
  if [ -f "$ENV_EXAMPLE" ]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    info ".env created from .env.example"
  else
    cat > "$ENV_FILE" <<'EOF'
# Anthropic API key — required for AI chat features
# Get one at: https://console.anthropic.com/
ANTHROPIC_API_KEY=

# Web server port (default: 8888)
# SETLIST_PORT=8888

# Optional: path to your Mixed In Key Library.csv for energy ratings
# MIK_CSV_PATH=/path/to/Library.csv

# Optional: override Rekordbox database path (auto-detected by default)
# REKORDBOX_DB_PATH=/path/to/master.db
EOF
    info ".env created"
  fi
  echo ""
  warn "Add your Anthropic API key to .env:"
  echo "  ANTHROPIC_API_KEY=sk-ant-..."
fi

# -----------------------------------------------------------------------------
# .data/ directory (internal, no prompt)
# -----------------------------------------------------------------------------

mkdir -p "$SCRIPT_DIR/.data/models" "$SCRIPT_DIR/.data/essentia_cache"

# -----------------------------------------------------------------------------
# 8. Register MCP server in Claude Code / Codex
# -----------------------------------------------------------------------------

header "Step 8 — Register MCP server in Claude Code"
divider

if command -v claude &>/dev/null; then
  if claude mcp get mcp-dj &>/dev/null 2>&1; then
    info "MCP server 'mcp-dj' already registered in Claude Code"
  else
    echo "  Registering 'mcp-dj' MCP server in Claude Code..."
    echo ""

    # Pick up ANTHROPIC_API_KEY from .env if it looks real
    MCP_ENV_ARGS=()
    if [ -f "$ENV_FILE" ]; then
      STORED_KEY=$(grep -E '^ANTHROPIC_API_KEY=' "$ENV_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'")
      if [ -n "$STORED_KEY" ] && [[ "$STORED_KEY" == sk-ant-* ]]; then
        MCP_ENV_ARGS+=(-e "ANTHROPIC_API_KEY=$STORED_KEY")
      fi
    fi

    read -r -p "  Register for all Claude Code projects (user) or this project only (local)? [U/l] " SCOPE_REPLY
    SCOPE_REPLY="${SCOPE_REPLY:-U}"
    if [[ "$SCOPE_REPLY" =~ ^[Ll]$ ]]; then
      MCP_SCOPE="local"
    else
      MCP_SCOPE="user"
    fi

    if claude mcp add "${MCP_ENV_ARGS[@]}" -s "$MCP_SCOPE" mcp-dj -- \
        uv run --project "$SCRIPT_DIR" python -m mcp_dj.mcp_server; then
      info "MCP server registered in Claude Code (scope: $MCP_SCOPE)"
    else
      warn "Automatic registration failed — add it manually:"
      warn "  claude mcp add -s user mcp-dj -- uv run --project \"$SCRIPT_DIR\" python -m mcp_dj.mcp_server"
    fi
  fi
else
  warn "Claude Code CLI not found — skipping automatic MCP registration"
  echo "  Install Claude Code: https://claude.ai/download"
  echo "  Then register manually:"
  echo "    claude mcp add -s user mcp-dj -- \\"
  echo "      uv run --project \"$SCRIPT_DIR\" python -m mcp_dj.mcp_server"
fi

# -----------------------------------------------------------------------------
# 9. Analyze Rekordbox library (only if essentia was installed)
# -----------------------------------------------------------------------------

if [ "$INSTALL_ESSENTIA" = true ]; then
  header "Step 9 — Analyze Rekordbox library"
  divider
  echo "  Essentia can now analyze every song in your Rekordbox library."
  echo "  This extracts BPM, key, mood, genre, and tags for all your tracks,"
  echo "  making AI setlist recommendations much more accurate."
  echo ""
  echo "  Analysis runs in parallel across multiple CPU cores."
  echo "  Depending on library size this may take a while (~10-15s per track)."
  echo "  You can stop it any time with Ctrl+C and resume later."
  echo ""
  read -r -p "  Analyze your Rekordbox library now? [y/N] " REPLY
  REPLY="${REPLY:-N}"
  if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    # Suggest a worker count based on CPU cores
    CPU_COUNT=$(python3 -c "import os; print(max(1, min(4, (os.cpu_count() or 2) // 2)))" 2>/dev/null || echo "2")
    echo ""
    read -r -p "  Parallel workers? (default: $CPU_COUNT, recommended for your machine) " WORKERS
    WORKERS="${WORKERS:-$CPU_COUNT}"
    echo ""
    bash "$SCRIPT_DIR/analyze-library.sh" --workers "$WORKERS"
  else
    warn "Skipping — run later with: ./analyze-library.sh"
  fi
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo ""
divider
echo -e "${BOLD}Installation complete!${NC}"
divider
echo ""

MCP_NAME=$(python3 -c "
import json, sys
try:
    d = json.load(open('$SCRIPT_DIR/claude_desktop_config.json'))
    k = list(d.get('mcpServers', {}).keys())
    print(k[0] if k else 'mcp-dj')
except:
    print('mcp-dj')
" 2>/dev/null || echo "mcp-dj")

echo "  1. Add your Anthropic API key to .env:"
echo "       ANTHROPIC_API_KEY=sk-ant-..."
echo ""
echo "  2. Start the web UI:"
echo "       ./run-server.sh"
echo "       Open: http://localhost:8888"
echo ""
echo "  3. Connect to Claude Desktop (add to claude_desktop_config.json):"
echo "       Server name: $MCP_NAME"
echo "       See: claude_desktop_config.json"
echo ""
echo "     Or register in Claude Code (if not already done during install):"
echo "       claude mcp add -s user $MCP_NAME -- uv run --project \"$SCRIPT_DIR\" python -m mcp_dj.mcp_server"
echo ""

if [ "$INSTALL_ESSENTIA" = true ]; then
  echo "  4. Analyze a track (BPM, key, mood, genre, tags):"
  echo "       analyze-track /path/to/song.mp3"
  echo "       analyze-track /path/to/song.mp3 --output json"
  echo ""
  if [ "$SKIP_MODELS" = true ]; then
    echo "  5. Download ML models when ready:"
    echo "       ./download_models.sh"
    echo ""
  fi
else
  echo "  4. Enable audio analysis (optional, ~135 MB + ~300 MB models):"
  echo "       ./install.sh --essentia"
  echo ""
fi

divider
