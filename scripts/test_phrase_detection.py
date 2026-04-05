"""
Test phrase detection on a single track and optionally write cues to Rekordbox.

Usage:
    # Dry-run (analyse only, no DB writes) — pass your audio file or set MCP_DJ_TEST_AUDIO
    python scripts/test_phrase_detection.py --audio /path/to/track.mp3

    # Write cues to Rekordbox (replace any existing memory cues on this track)
    python scripts/test_phrase_detection.py --audio /path/to/track.mp3 --write --replace

    # Or: export MCP_DJ_TEST_AUDIO=/path/to/track.mp3  then omit --audio
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Make sure the package root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_dj.phrase_detector import PhraseDetector, write_phrase_cues

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test track metadata (Sun & Moon — Ilan Bluestone Remix, 130 BPM, ID 75741128)
# Audio path: --audio or env MCP_DJ_TEST_AUDIO (never commit local paths)
# ---------------------------------------------------------------------------
TEST_TRACK = {
    "content_id": "75741128",
    "bpm": 130.0,
    "title": "Sun & Moon (Ilan Bluestone Remix)",
}


def run(
    file_path: str,
    *,
    write: bool = False,
    replace: bool = False,
) -> None:
    track = {**TEST_TRACK, "file_path": file_path}

    print(f"\n{'='*60}")
    print(f"Track : {track['title']}")
    print(f"BPM   : {track['bpm']}")
    print(f"File  : {Path(track['file_path']).name}")
    print(f"{'='*60}\n")

    # --- Phrase detection ---
    detector = PhraseDetector()
    cues = detector.detect(track["file_path"], bpm=track["bpm"])

    print(f"\nDetected {len(cues)} sections:\n")
    print(f"  {'Bar':>4}  {'Time':>8}  {'Label':<14}  Color")
    print(f"  {'-'*4}  {'-'*8}  {'-'*14}  {'-'*8}")
    for c in cues:
        mins, secs = divmod(c.time_ms // 1000, 60)
        print(f"  {c.bar_number:>4}  {mins:02d}:{secs:02d}.{(c.time_ms % 1000):03d}  {c.label:<14}  {c.color}")

    if not write:
        print("\nDry-run — pass --write to persist cues to Rekordbox.\n")
        return

    # --- Write to Rekordbox DB ---
    print("\nConnecting to Rekordbox database …")

    async def _write():
        from mcp_dj.database import RekordboxDatabase
        db = RekordboxDatabase()
        await db.connect()
        ids = write_phrase_cues(
            rb_db          = db.db,
            content_id     = track["content_id"],
            cues           = cues,
            replace_existing = replace,
        )
        print(f"\nWrote {len(ids)} cue rows to Rekordbox (IDs: {ids[:3]}{'…' if len(ids)>3 else ''})")
        print("Open Rekordbox → browse to the track → check Memory Cues tab.\n")

    asyncio.run(_write())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test phrase detection")
    parser.add_argument(
        "--audio",
        metavar="PATH",
        default=os.environ.get("MCP_DJ_TEST_AUDIO"),
        help="Path to audio file (or set MCP_DJ_TEST_AUDIO)",
    )
    parser.add_argument("--write", action="store_true", help="Write cues to Rekordbox DB")
    parser.add_argument("--replace", action="store_true", help="Delete existing memory cues first")
    args = parser.parse_args()
    if not args.audio:
        parser.error("pass --audio PATH or set MCP_DJ_TEST_AUDIO to your test track")
    run(args.audio, write=args.write, replace=args.replace)
