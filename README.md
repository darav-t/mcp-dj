# MCP DJ - DJ Setlist Creator

AI-powered DJ setlist generator with harmonic mixing and energy arc planning, built on your local Rekordbox library.

> [!NOTE]
> AI can't replace human emotions, feelings and vibes on dancefloor however it can help you elevate it, use wisely ;)

---

## Features

- **Harmonic mixing** — Camelot wheel compatibility scoring
- **Energy arc planning** — 5 profiles: `journey`, `build`, `peak`, `chill`, `wave`
- **AI chat interface** — Claude-powered natural language setlist requests
- **MyTag-aware set building** — `build_set_from_prompt` filters candidates using your Rekordbox My Tags before applying harmonic/energy scoring
- **Centralized library index** — JSONL index merging Rekordbox metadata, Essentia audio features, and MIK data; built at startup, updated incrementally during analysis
- **Mixed In Key integration** — Uses your MIK energy ratings when available (optional)
- **Essentia audio analysis** — ML-based BPM, key, mood, genre, and music tagging (optional)
- **Rekordbox export** — Creates playlists directly in your Rekordbox library
- **Claude Code slash commands** — `/build-set`, `/dj-library`, `/export-set` for fast in-editor workflows
- **Two interfaces** — FastAPI web UI + FastMCP server for Claude Desktop

---

## How It Works

### 1. Ask in plain language

Describe the set you want — duration, vibe, genre, energy. The AI translates your request into a scored track selection from your own Rekordbox library.

![Generate a setlist from a natural language prompt](screenshots/sample_set_list_generation_1.png)

---

### 2. Explore what it can do

Not sure where to start? Ask *"what can I do with DJ set list creator"* and it will walk you through every capability — library search, setlist generation, harmonic flow analysis, and Rekordbox export.

![Overview of capabilities](screenshots/what_can_i_do.png)

---

### 3. Dig into the reasoning

Ask follow-up questions like *"how do you determine this is darker sound?"* and the AI explains exactly which signals it used — genre tags, energy profile, Camelot key range — and what its limitations are.

![Explanation of how darker sound is determined](screenshots/how_do_you_define_darker_sound.png)

---

### 4. Export directly to Rekordbox

Once you're happy with the setlist, ask it to create a Rekordbox playlist by name. The playlist appears in your library immediately, ready to use.

![Export setlist as a Rekordbox playlist via chat](screenshots/create_playlist_on_rekordbox.png)

---

### 5. Open it in Rekordbox and play

The exported playlist shows up in Rekordbox with all tracks in order, BPM and key visible, ready to load onto decks.

![Exported playlist open in Rekordbox](screenshots/mcp_rekordbox_playlist.png)

---

## Architecture

```
Your Rekordbox library (read-only)
         │
         ▼
  RekordboxDatabase          ← pyrekordbox / SQLCipher
         │
         ▼
   LibraryIndex              ← JSONL index: Rekordbox + Essentia + MIK merged per track
         │                      Built at startup, updated incrementally during analysis
         ▼
   EnergyResolver            ← Essentia cache → MIK CSV (optional) → BPM heuristic
         │
         ▼
   SetlistEngine             ← Camelot wheel scoring + energy arc planning
         │                      Candidate pool filtered from index via My Tags
    ┌────┴────┐
    ▼         ▼
 FastAPI    FastMCP
 Web UI     Claude Desktop / Claude Code
```

| Module | Role |
|---|---|
| `database.py` | Read-only Rekordbox 6 access via pyrekordbox |
| `library_index.py` | Centralized JSONL index — merges Rekordbox + Essentia + MIK; dynamic attribute scanning |
| `camelot.py` | Camelot wheel — harmonic compatibility scoring |
| `energy.py` | Energy resolution: Essentia cache → MIK CSV → album tag → BPM heuristic |
| `energy_planner.py` | 5 energy arc profiles (journey, build, peak, chill, wave) |
| `setlist_engine.py` | Greedy track selection with harmonic + energy + BPM scoring |
| `essentia_analyzer.py` | ML audio analysis: BPM, key, mood, genre, tagging (optional) |
| `analyze_library.py` | Batch library analysis with parallel workers; incremental index updates |
| `analyze_track.py` | Single-track analysis CLI entry point |
| `ai_integration.py` | Claude API with tool calling, conversation history, fallback mode |
| `app.py` | FastAPI web UI |
| `mcp_server.py` | FastMCP server for Claude Desktop |

---

## MCP Tools

| Tool | Description |
|---|---|
| `build_set_from_prompt` | Natural language → harmonically ordered setlist, using My Tag filtering from the library index |
| `generate_setlist` | Setlist from explicit parameters (genre, BPM, energy profile) |
| `plan_set` | Vibe-based set planning (venue, crowd, time of day) |
| `rebuild_library_index` | Rebuild the JSONL index + dynamic attribute summary |
| `get_library_attributes` | Full dynamic attribute summary: tag hierarchy, per-tag BPM/energy/mood, genre stats, co-occurrence |
| `get_track_full_info` | Full merged record for a track (Rekordbox + Essentia + MIK) |
| `search_library` | Search tracks by query, My Tag, or date range |
| `recommend_next_track` | Harmonic + energy recommendations for the currently playing track |
| `get_track_compatibility` | Compatibility analysis between two specific tracks |
| `analyze_track` | Run Essentia audio analysis on a single file |
| `analyze_library_essentia` | Batch-analyze the full library with Essentia |
| `export_setlist_to_rekordbox` | Create a Rekordbox playlist from a generated setlist |

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Rekordbox 6 installed with a library
- (Optional) [Mixed In Key](https://mixedinkey.com/) with CSV export
- (Optional) Anthropic API key for AI chat
- (Optional) [essentia-tensorflow](https://essentia.upf.edu/) for ML audio analysis

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/darav-t/dj-setlist-creator-mcp.git
cd dj-setlist-creator-mcp

# 2. Run the install script (handles uv, Python, deps, and .env setup)
./install.sh

# For Essentia ML audio analysis (recommended):
./install.sh --essentia

# 3. Start the web UI
./run-server.sh
# Opens at http://localhost:8888
```

> The install script sets up everything including pyrekordbox database keys and your `.env` file. Run `./install.sh --help` for all options.

---

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` and edit:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | _(none)_ | Claude API key. Falls back to rule-based mode if unset. |
| `SETLIST_PORT` | `8888` | Port for the web UI. |
| `MIK_CSV_PATH` | _(none)_ | Path to your Mixed In Key `Library.csv` export. Feature disabled if unset. |
| `REKORDBOX_DB_PATH` | Auto-detected | Override the Pioneer directory path if Rekordbox is in a non-standard location. |

### Essentia audio analysis (optional, recommended)

Essentia provides ML-based per-track analysis: BPM accuracy, key detection, danceability, EBU R128 loudness, mood probabilities (happy, sad, aggressive, relaxed, party), Discogs400 genre scores, and MagnaTagATune music tags.

Results are cached at `.data/essentia_cache/` — each track is only analyzed once. The library index is updated incrementally as each track finishes, so you can build sets while analysis is still running.

```bash
# Install essentia + download ML models (~300 MB, one-time)
./install.sh --essentia

# Analyze your full library (skip already-cached tracks)
./analyze-library.sh

# Re-analyze everything
./analyze-library.sh --force

# Use parallel workers for faster analysis
./analyze-library.sh --workers 4

# Download/update models separately
./download_models.sh
```

### Library index

The library index (`.data/library_index.jsonl`) merges all track data into a single JSONL file used for My Tag filtering and attribute lookups. It is built automatically at server startup (refreshed if older than 1 hour) and rebuilt after any batch analysis run.

To force a full rebuild:

```bash
# Via MCP tool (in Claude Desktop or Claude Code)
rebuild_library_index(force=True)
```

### Mixed In Key (optional)

Energy data from Mixed In Key gives the setlist engine accurate per-track energy ratings. Without Essentia or MIK, energy is inferred from BPM and genre tags.

To enable:
1. In Mixed In Key → export your library as CSV
2. Set `MIK_CSV_PATH=/path/to/Library.csv` in your `.env`

---

## Claude Desktop Integration (MCP)

The MCP server runs via your project's virtualenv (`.venv`). Use the same style for each platform:

1. Open `claude_desktop_config.json` in this repo.
2. Replace `/path/to/mcp_dj` with the **absolute path** to this repo on your machine.
3. Merge the `mcpServers` block into your Claude Desktop config for your OS:
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
4. Restart Claude Desktop.

**macOS/Linux** — entry uses bash and the project venv:

```json
"mcp-dj": {
  "command": "/bin/bash",
  "args": ["-c", "cd '/path/to/mcp_dj' && exec .venv/bin/python3 -m mcp_dj.mcp_server"]
}
```

**Windows** — use `cmd` and the Windows venv path:

```json
"mcp-dj": {
  "command": "cmd",
  "args": ["/c", "cd /d \"C:\\path\\to\\mcp_dj\" && .venv\\Scripts\\python.exe -m mcp_dj.mcp_server"]
}
```

Run `./install.sh` to have this entry added to your Claude Desktop config automatically.

Or run the MCP server manually:

```bash
./run-mcp.sh
```

---

## Claude Code Slash Commands

If you use Claude Code (the CLI), three slash commands are available in `.claude/commands/`:

| Command | What it does |
|---|---|
| `/build-set [prompt]` | Build a DJ set from a natural language prompt — calls `build_set_from_prompt`, formats a tracklist table with energy sparkline, and offers export/adjust actions |
| `/dj-library [query]` | Browse your library — search by tag, genre, artist, or track; show library stats and full MyTag hierarchy |
| `/export-set [name]` | Export the most recently generated set to Rekordbox as a named playlist |

Example:

```
/build-set 90min sunset progressive house, start melodic then build to peak
/dj-library Festival
/export-set Sunset Set Feb 2026
```

---

## Project Structure

```
dj-setlist-creator-mcp/
├── mcp_dj/
│   ├── app.py               # FastAPI web application
│   ├── mcp_server.py        # FastMCP server for Claude Desktop
│   ├── models.py            # Pydantic data models
│   ├── database.py          # Rekordbox database layer (read-only)
│   ├── library_index.py     # Centralized JSONL library index + dynamic attributes
│   ├── camelot.py           # Camelot wheel harmonic mixing engine
│   ├── energy.py            # Energy resolution (Essentia / MIK CSV / BPM heuristic)
│   ├── energy_planner.py    # Energy arc profiles
│   ├── setlist_engine.py    # Core setlist generation algorithm
│   ├── essentia_analyzer.py # ML audio analysis (BPM, key, mood, genre, tags)
│   ├── analyze_library.py   # Batch library analysis with parallel workers
│   ├── analyze_track.py     # Single-track analysis CLI entry point
│   ├── ai_integration.py    # Claude API integration with tool calling
│   └── static/
│       └── index.html       # Web UI
├── .claude/
│   └── commands/
│       ├── build-set.md     # /build-set slash command
│       ├── dj-library.md    # /dj-library slash command
│       └── export-set.md    # /export-set slash command
├── tests/
├── screenshots/
├── .data/                   # Git-ignored: Essentia cache, ML models, library index
│   ├── essentia_cache/      # Per-track analysis JSON cache
│   ├── models/              # ML model files (~300 MB, downloaded once)
│   ├── library_index.jsonl  # Merged track index (Rekordbox + Essentia + MIK)
│   └── library_attributes.json  # Dynamic attribute summary (tags, genres, BPM/energy)
├── install.sh               # Full install script (core + optional Essentia)
├── analyze-library.sh       # Batch-analyze Rekordbox library with Essentia
├── download_models.sh       # Download Essentia ML models
├── run-server.sh            # Start the web UI
├── run-mcp.sh               # Start the MCP server
├── .env.example             # Configuration template
└── pyproject.toml
```

---

## Development

```bash
# Run tests
uv run pytest

# Run on a custom port
SETLIST_PORT=9000 ./run-server.sh
```

---

## License

MIT
