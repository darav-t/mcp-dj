# DJ Setlist Creator

AI-powered DJ setlist generator with harmonic mixing and energy arc planning, built on your local Rekordbox library.

## Features

- **Harmonic mixing** — Camelot wheel compatibility scoring
- **Energy arc planning** — 5 profiles: `journey`, `build`, `peak`, `chill`, `wave`
- **AI chat interface** — Claude-powered natural language setlist requests
- **Mixed In Key integration** — Uses your MIK energy ratings when available
- **Rekordbox export** — Creates playlists directly in your Rekordbox library
- **Two interfaces** — FastAPI web UI + FastMCP server for Claude Desktop

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Rekordbox 6 installed with a library
- (Optional) [Mixed In Key](https://mixedinkey.com/) with CSV export
- (Optional) Anthropic API key for AI chat

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/set_list_creator.git
cd set_list_creator

# 2. Copy and configure environment variables
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY (and optionally MIK_CSV_PATH)

# 3. Install dependencies
uv sync

# 4. Start the web UI
./run-server.sh
# Opens at http://localhost:8888
```

## Configuration

All configuration is done via environment variables. Copy `.env.example` to `.env` and edit:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | _(none)_ | Claude API key. Falls back to rule-based mode if unset. |
| `SETLIST_PORT` | `8888` | Port for the web UI. |
| `MIK_CSV_PATH` | `~/Music/Mixed In Key Data/Library.csv` | Path to your Mixed In Key CSV export. |
| `REKORDBOX_DB_PATH` | Auto-detected | Override Rekordbox Pioneer directory path. |

## Claude Desktop Integration (MCP)

1. Open `claude_desktop_config.json` in this repo.
2. Replace `/path/to/set_list_creator` with the absolute path to where you cloned this repo.
3. Merge the `mcpServers` block into your Claude Desktop config:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
4. Restart Claude Desktop.

Alternatively, run the MCP server manually:

```bash
./run-mcp.sh
```

## Project Structure

```
setlist_creator/
├── app.py            # FastAPI web application
├── mcp_server.py     # FastMCP server for Claude Desktop
├── models.py         # Pydantic data models
├── database.py       # Rekordbox database layer (read-only)
├── camelot.py        # Camelot wheel harmonic mixing engine
├── energy.py         # Energy resolution (MIK CSV / album tag / BPM heuristic)
├── energy_planner.py # Energy arc profiles
├── setlist_engine.py # Core setlist generation algorithm
├── ai_integration.py # Claude API integration with tool calling
└── static/
    └── index.html    # Web UI
```

## Development

```bash
# Run tests
uv run pytest

# Run with a specific port
SETLIST_PORT=9000 ./run-server.sh
```

## License

MIT
