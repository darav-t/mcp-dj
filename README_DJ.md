# MCP DJ — Your AI Set Planning Assistant

> AI can't replace human emotions, feelings and vibes on the dancefloor — however it can help you elevate it. Use wisely.

---

You've spent years building a library. Thousands of tracks, each tagged, analyzed, filed. But when it comes to planning a set — pulling the right tracks, in the right key, at the right energy, for the right crowd — you still do it all by hand.

MCP DJ plugs into your existing Rekordbox library and helps you plan sets through conversation. Tell it what you want in plain language. It searches your actual tracks, scores them harmonically and energetically, and hands you a sequenced setlist — ready to export as a Rekordbox playlist.

Your ears, your taste. It handles the legwork.

---

## What it does for your planning

**Builds harmonically coherent sets from your library**

Every track suggestion is scored against the Camelot wheel. Transitions stay in key, or move one step — no jarring key clashes unless you want them.

**Shapes the energy arc automatically**

Choose how you want the energy to move across the set:

| Profile | What it does |
|---|---|
| `journey` | Classic warm-up → build → peak → cool-down |
| `build` | Continuous climb from low to high energy |
| `peak` | High energy throughout — for a peak-time slot |
| `chill` | Low energy — opening set or late-night wind-down |
| `wave` | Multiple peaks and valleys — keeps the crowd guessing |

**Understands your music, not just metadata**

With Essentia analysis enabled, the AI draws on actual audio features for each track — detected BPM, key accuracy, danceability, loudness, and mood scores (aggressive, happy, sad, relaxed, party). This means better matching beyond what's in Rekordbox tags alone.

**Exports directly into Rekordbox**

When you're happy with the set, ask it to create a playlist by name. It appears in your Rekordbox library immediately — tracks in order, ready to load onto decks.

---

## Talk to it like you plan sets

You don't fill in forms. You describe what you want.

```
"Give me a 90-minute techno set that starts dark and industrial,
builds to a peak around the hour mark, then eases off"
```

```
"Find me 10 tracks in the 128–132 BPM range, minimal and hypnotic,
good for a 2am crowd that's been dancing for hours"
```

```
"Build a warm-up set — nothing above 124 BPM, start around 118,
mix of deep house and minimal, about an hour"
```

```
"What tracks do I have that would work after Drumcode-style peak-time techno
when I want to start bringing the energy down?"
```

```
"I'm playing back to back — show me tracks in 8A or 9A that sit
around 130 BPM so I can hand off cleanly"
```

Ask follow-up questions, refine the selection, swap tracks in and out. It remembers the conversation.

---

## From prompt to playlist

### Describe your set

Tell it the duration, vibe, genre, and energy shape. The AI searches your Rekordbox library and assembles a scored setlist.

![Generate a setlist from a natural language prompt](screenshots/sample_set_list_generation_1.png)

---

### Ask it to explain its choices

Want to understand why a track was included or how it defines a darker sound? Ask. It explains exactly which signals it used — key range, energy profile, genre tags, mood scores — and where its knowledge ends.

![Explanation of how darker sound is determined](screenshots/how_do_you_define_darker_sound.png)

---

### Export to Rekordbox

Once the set is dialled in, ask it to save the playlist. One line.

```
"Save this as a Rekordbox playlist called Friday Night Peak"
```

![Export setlist as a Rekordbox playlist via chat](screenshots/create_playlist_on_rekordbox.png)

---

### Open in Rekordbox and play

The playlist shows up in your library with all tracks in order, BPM and key visible, ready to load.

![Exported playlist open in Rekordbox](screenshots/mcp_rekordbox_playlist.png)

---

## What it works with

**Rekordbox 6** — reads your library directly. No import, no export, no CSV wrangling. Your existing tags, energy ratings, and cue points stay exactly where they are.

**Mixed In Key** — if you use MIK, point it at your Library CSV and it uses your MIK energy ratings for more accurate energy scoring.

**Essentia (optional but recommended)** — runs a one-time audio analysis of your library. Extracts BPM, key, danceability, loudness, and mood from the audio itself. Results are cached — you run it once, it's done.

**Claude Desktop** — if you prefer chatting inside Claude Desktop rather than a browser tab, the MCP server mode drops straight into your existing Claude setup.

---

## Getting started

### 1. Install

```bash
git clone https://github.com/darav-t/dj-setlist-creator-mcp.git
cd dj-setlist-creator-mcp

# Core install (sets up Python, deps, Rekordbox keys, and .env)
./install.sh

# Recommended: include Essentia ML audio analysis
./install.sh --essentia
```

### 2. Analyze your library (if using Essentia)

Run this once after install. It processes every track in your Rekordbox library and caches the results. New tracks added later can be analyzed incrementally.

```bash
./analyze-library.sh

# Faster with parallel workers
./analyze-library.sh --workers 4
```

### 3. Add your Anthropic API key

Edit the `.env` file the installer created and set `ANTHROPIC_API_KEY`. Get a key at [console.anthropic.com](https://console.anthropic.com).

### 4. Start the app

```bash
./run-server.sh
# Opens at http://localhost:8888
```

Or use it inside Claude Desktop — see the [technical README](README.md#claude-desktop-integration-mcp) for setup.

---

## What it knows — and what it doesn't

The AI works entirely from your Rekordbox library. It has no opinion on tracks you don't own, no access to streaming services, and no awareness of what's popular right now.

What it reads per track: title, artist, BPM, key, genre, energy level (from Rekordbox rating, MIK CSV, or Essentia), date added, My Tags, and — with Essentia — mood scores, danceability, loudness, and ML-classified genre.

What it cannot replace: your instinct for reading a room, your feel for when to break the rules, and the human connection between you and the crowd.

---

## Tips for getting the most out of it

**Be specific about energy shape, not just genre.** "Dark techno" is vague. "Dark techno, starts at a 5, peaks at an 8 around 45 minutes, then drops to a 6 to close" gives it something to work with.

**Use it for prep, not prescription.** Generate two or three setlist options for the same slot. Use them as starting points, then reorder by feel on the night.

**Run Essentia before you plan.** The mood and danceability scores make a real difference to how well it understands the character of each track — especially if your genre tags are inconsistent.

**Ask about specific transitions.** "What would work after [track name] if I want to keep the energy but shift the mood lighter?" It can recommend based on harmonic compatibility and mood scores.

**Trust your ears last.** Everything it generates is a suggestion. You know your crowd.

---

## License

MIT
