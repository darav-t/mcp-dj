#!/usr/bin/env bash
# =============================================================================
# DJ Setlist Creator — Install Script
# =============================================================================
# Installs all dependencies required to run the project:
#   - uv (Python package manager)
#   - Python 3.12+ (via uv)
#   - All Python packages (via uv sync)
#   - Essentia audio analysis library (optional, pip install essentia)
#   - pyrekordbox Rekordbox database key (required to read your library)
#
# Usage:
#   chmod +x install.sh
#   ./install.sh            # standard install
#   ./install.sh --essentia # also install Essentia audio analysis
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_ESSENTIA=false

# Parse flags
for arg in "$@"; do
  case "$arg" in
    --essentia) INSTALL_ESSENTIA=true ;;
    --help|-h)
      echo "Usage: ./install.sh [--essentia]"
      echo ""
      echo "  --essentia   Also install the Essentia audio analysis library"
      echo "               Enables: analyze-track, analyze_track MCP tool"
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
error()   { echo -e "${RED}[✗]${NC} $*"; }
header()  { echo -e "\n${BOLD}$*${NC}"; }
divider() { echo "------------------------------------------------------------"; }

# -----------------------------------------------------------------------------
# 1. Check for uv
# -----------------------------------------------------------------------------

header "Step 1 — uv package manager"
divider

if command -v uv &>/dev/null; then
  UV_VERSION=$(uv --version 2>&1 | head -1)
  info "uv already installed: $UV_VERSION"
else
  warn "uv not found — installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Add uv to PATH for the rest of this script
  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
  if command -v uv &>/dev/null; then
    info "uv installed: $(uv --version 2>&1 | head -1)"
  else
    error "uv installation failed. Install manually: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
  fi
fi

# -----------------------------------------------------------------------------
# 2. Python 3.12+
# -----------------------------------------------------------------------------

header "Step 2 — Python 3.12+"
divider

cd "$SCRIPT_DIR"

if uv python find 3.12 &>/dev/null 2>&1; then
  PY_VERSION=$(uv python find 3.12 2>/dev/null | xargs python3 --version 2>/dev/null || echo "3.12.x")
  info "Python 3.12+ available"
else
  warn "Python 3.12 not found — installing via uv..."
  uv python install 3.12
  info "Python 3.12 installed"
fi

# -----------------------------------------------------------------------------
# 3. Core Python dependencies
# -----------------------------------------------------------------------------

header "Step 3 — Python packages (uv sync)"
divider

echo "Installing: pyrekordbox, pydantic, fastapi, uvicorn, loguru, anthropic, fastmcp"
uv sync
info "Core packages installed"

# -----------------------------------------------------------------------------
# 4. Essentia (optional)
# -----------------------------------------------------------------------------

header "Step 4 — Essentia audio analysis (optional)"
divider

if [ "$INSTALL_ESSENTIA" = true ]; then
  echo "Installing Essentia..."
  if uv pip install essentia; then
    info "Essentia installed"
    echo "  You can now run: python -m setlist_creator.analyze_track /path/to/song.mp3"
  else
    warn "Essentia installation failed."
    warn "Try manually: pip install essentia"
    warn "See: https://github.com/MTG/essentia"
  fi
else
  warn "Skipping Essentia (run with --essentia to include it)"
  echo "  Essentia enables audio feature extraction: BPM accuracy, key detection,"
  echo "  danceability scoring, and loudness analysis."
  echo "  To install later: pip install essentia"
fi

# -----------------------------------------------------------------------------
# 5. Rekordbox database key (pyrekordbox setup)
# -----------------------------------------------------------------------------

header "Step 5 — Rekordbox database access"
divider

echo "Checking pyrekordbox configuration..."

if uv run python -c "import pyrekordbox; pyrekordbox.show_config()" 2>/dev/null | grep -q "app_password\|db_key"; then
  info "pyrekordbox already configured"
else
  echo ""
  echo "pyrekordbox needs the Rekordbox master.db key to decrypt your library."
  echo "Run the following command to set it up:"
  echo ""
  echo -e "  ${BOLD}uv run python -m pyrekordbox setup-db${NC}"
  echo ""
  echo "This reads the key from your Rekordbox installation automatically."
  echo "Rekordbox must be installed on this machine."
  echo ""
  read -r -p "Run pyrekordbox setup now? [Y/n] " REPLY
  REPLY="${REPLY:-Y}"
  if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    if uv run python -m pyrekordbox setup-db; then
      info "Rekordbox database key configured"
    else
      warn "Setup failed — you may need to run it manually after installing Rekordbox."
      warn "Command: uv run python -m pyrekordbox setup-db"
    fi
  else
    warn "Skipped — run 'uv run python -m pyrekordbox setup-db' before starting the server."
  fi
fi

# -----------------------------------------------------------------------------
# 6. Environment file
# -----------------------------------------------------------------------------

header "Step 6 — Environment configuration"
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
  warn "Add your Anthropic API key to .env to enable AI chat:"
  echo "  ANTHROPIC_API_KEY=sk-ant-..."
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
divider
echo -e "${BOLD}Installation complete!${NC}"
divider
echo ""
echo "Next steps:"
echo ""
echo "  1. Add your API key to .env:"
echo "       ANTHROPIC_API_KEY=sk-ant-..."
echo ""
echo "  2. Start the web UI:"
echo "       ./run-server.sh"
echo "       Open: http://localhost:8888"
echo ""
echo "  3. Or connect to Claude Desktop — add to claude_desktop_config.json:"
echo "       $(cat "$SCRIPT_DIR/claude_desktop_config.json" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); k=list(d.get('mcpServers',{}).keys()); print(k[0] if k else 'see claude_desktop_config.json')" 2>/dev/null || echo "see claude_desktop_config.json")"
echo ""
if [ "$INSTALL_ESSENTIA" = true ]; then
echo "  4. Analyze a track with Essentia:"
echo "       python -m setlist_creator.analyze_track /path/to/song.mp3"
echo ""
else
echo "  4. To enable audio analysis (optional):"
echo "       ./install.sh --essentia"
echo ""
fi
divider
