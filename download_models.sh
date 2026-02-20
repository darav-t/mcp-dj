#!/usr/bin/env bash
# =============================================================================
# Essentia ML Model Downloader
# =============================================================================
# Downloads all TensorFlow models required for ML-based audio analysis:
#   - Mood classifiers: Happy, Sad, Aggressive, Relaxed, Party
#   - Music autotagging: MagnaTagATune tags (techno, dance, beat, etc.)
#   - Genre classification: Discogs400 (House, Techno, etc.)
#
# Models are stored at: ~/.setlist_creator/models/
# Total download size: ~250 MB
#
# Usage:
#   chmod +x download_models.sh
#   ./download_models.sh
# =============================================================================

set -euo pipefail

MODEL_DIR="${HOME}/.setlist_creator/models"
BASE_URL="https://essentia.upf.edu/models"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}[↓]${NC} $*"; }
error()   { echo -e "${RED}[✗]${NC} $*"; }
header()  { echo -e "\n${BOLD}$*${NC}"; }
divider() { echo "------------------------------------------------------------"; }

mkdir -p "$MODEL_DIR"

# -----------------------------------------------------------------------------
# Download helper — skips if file already exists
# -----------------------------------------------------------------------------
download() {
  local url="$1"
  local dest="$MODEL_DIR/$(basename "$url")"
  if [ -f "$dest" ]; then
    info "Already exists: $(basename "$url")"
  else
    warn "Downloading: $(basename "$url")"
    if curl -fsSL --retry 3 --retry-delay 2 -o "$dest" "$url"; then
      info "Downloaded:  $(basename "$url")"
    else
      error "Failed:      $url"
      rm -f "$dest"
      return 1
    fi
  fi
}

# =============================================================================
# 1. Embedding models (required by all classifiers)
# =============================================================================
header "Step 1 — Embedding models"
divider
echo "These extract audio representations that classifiers run on top of."
echo ""

# VGGish — used by mood classifiers and MagnaTagATune
download "${BASE_URL}/feature-extractors/vggish/audioset-vggish-3.pb"
download "${BASE_URL}/feature-extractors/vggish/audioset-vggish-3.json"

# MusiCNN — used by MSD autotagging
download "${BASE_URL}/feature-extractors/musicnn/msd-musicnn-1.pb"
download "${BASE_URL}/feature-extractors/musicnn/msd-musicnn-1.json"

# Discogs-EffNet — used by genre classification
download "${BASE_URL}/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb"
download "${BASE_URL}/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.json"

# =============================================================================
# 2. Mood classifiers
# =============================================================================
header "Step 2 — Mood classifiers (Happy, Sad, Aggressive, Relaxed, Party)"
divider

for MOOD in mood_happy mood_sad mood_aggressive mood_relaxed mood_party; do
  download "${BASE_URL}/classification-heads/${MOOD}/${MOOD}-audioset-vggish-1.pb"
  download "${BASE_URL}/classification-heads/${MOOD}/${MOOD}-audioset-vggish-1.json"
done

# =============================================================================
# 3. Music autotagging — MagnaTagATune (mtt)
# =============================================================================
header "Step 3 — Music autotagging (MagnaTagATune / mtt)"
divider
echo "Tags: techno, female, beat, electronic, dance, vocal, guitar, piano, etc."
echo ""

download "${BASE_URL}/classification-heads/mtt/mtt-discogs-effnet-1.pb"
download "${BASE_URL}/classification-heads/mtt/mtt-discogs-effnet-1.json"

# =============================================================================
# 4. Genre classification — Discogs400
# =============================================================================
header "Step 4 — Genre classification (Discogs400)"
divider
echo "400 music styles: House, Techno, Deep House, Trance, etc."
echo ""

download "${BASE_URL}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb"
download "${BASE_URL}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json"

# =============================================================================
# Summary
# =============================================================================
echo ""
divider
echo -e "${BOLD}All models downloaded to:${NC} $MODEL_DIR"
divider
echo ""
ls -lh "$MODEL_DIR" | awk 'NR>1 {printf "  %-55s %s\n", $NF, $5}'
echo ""
TOTAL=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
echo "  Total size: $TOTAL"
echo ""
echo "Models are ready. Run 'analyze-track' to use them:"
echo "  analyze-track /path/to/song.mp3"
echo ""
