"""
analyze-library — Batch analyze all tracks in the Rekordbox database.

Uses multiprocessing so each worker has its own TensorFlow/Essentia session.
Results are sent back to the main process via a Queue so progress output
is always serialized and clean, even with many parallel workers.

Usage:
    ./analyze-library.sh
    ./analyze-library.sh --force            # re-analyze already cached tracks
    ./analyze-library.sh --workers 4        # parallel processes (default: auto)
    ./analyze-library.sh --dry-run          # preview without analyzing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from multiprocessing import Queue

from loguru import logger

# Silence INFO/DEBUG logs in the main process — only show WARNING+
logger.remove()
logger.add(sys.stderr, level="WARNING")

from .database import RekordboxDatabase
from .essentia_analyzer import (
    ESSENTIA_AVAILABLE,
    analyze_file,
    _cache_path,
)
from .energy import EnergyResolver, MixedInKeyLibrary

# ── terminal helpers ──────────────────────────────────────────────────────────

_TERM_WIDTH = shutil.get_terminal_size((80, 20)).columns

GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
RED    = "\033[0;31m"
CYAN   = "\033[0;36m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
NC     = "\033[0m"

# How many tracks to analyze before flushing the index to disk incrementally.
# Lower = safer against Ctrl+C but slightly more I/O; 10 is a good balance.
_FLUSH_EVERY = 10


class _SingleEssentiaStore:
    """Minimal EssentiaFeatureStore wrapper for one freshly-analyzed track.

    Passed to ``LibraryIndex.update_record()`` so only this track's features
    are merged into its JSONL record without needing a full EssentiaFeatureStore
    over the entire library.
    """

    def __init__(self, file_path: str, features: object) -> None:
        self._fp  = file_path
        self._ess = features

    def get(self, fp: str) -> object:
        return self._ess if fp == self._fp else None


def _load_single_ess(file_path: str) -> object:
    """Load one track's Essentia features from its per-track cache file.

    Returns an ``EssentiaFeatures`` instance or None on any failure.
    Safe to call from the main process while workers are still running
    (each worker writes to its own file path — no races).
    """
    try:
        cache_file = _cache_path(file_path)
        if cache_file.exists():
            from .essentia_analyzer import EssentiaFeatures
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            return EssentiaFeatures(**raw)
    except Exception:
        pass
    return None


def _fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}h {m:02d}m {s:02d}s"


def _draw_progress(done: int, total: int, active: list[str], avg_s: float | None) -> None:
    """Render progress bar + active worker tracks on TWO lines."""
    pct = done / total if total else 0
    bar_width = min(30, _TERM_WIDTH - 45)
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)

    eta_str = ""
    if avg_s and done < total:
        eta_str = f"  ETA {_fmt_eta((total - done) * avg_s)}"

    # Line 1: bar + counts + ETA
    bar_line = (
        f"{CYAN}[{bar}]{NC} "
        f"{BOLD}{done}/{total}{NC} ({pct*100:.0f}%)"
        f"{DIM}{eta_str}{NC}"
    )

    # Line 2: currently running tracks (truncated)
    if active:
        names = ", ".join(Path(p).stem for p in active)
        max_w = _TERM_WIDTH - 12
        if len(names) > max_w:
            names = names[:max_w - 1] + "…"
        active_line = f"{DIM}  ▶ {names}{NC}"
    else:
        active_line = ""

    # Move up one line if we already drew two lines, then overwrite
    sys.stdout.write(f"\033[2K\r{bar_line}\n\033[2K\r{active_line}\033[1A\r")
    sys.stdout.flush()


def _clear_progress() -> None:
    """Erase the two progress lines."""
    sys.stdout.write(f"\033[2K\r\n\033[2K\r\033[1A\r")
    sys.stdout.flush()


# ── worker (runs in a subprocess) ─────────────────────────────────────────────

def _worker(task_queue: Queue, result_queue: Queue, force: bool) -> None:
    """Worker process: pull file paths from task_queue, send results to result_queue."""
    logger.remove()  # silence loguru in subprocesses
    while True:
        file_path = task_queue.get()
        if file_path is None:  # poison pill — shut down
            break
        t0 = time.monotonic()
        try:
            analyze_file(file_path, force=force)
            result_queue.put((file_path, "ok", time.monotonic() - t0))
        except FileNotFoundError:
            result_queue.put((file_path, "missing", 0.0))
        except Exception:
            result_queue.put((file_path, "error", 0.0))


def _is_cached(file_path: str) -> bool:
    return _cache_path(file_path).exists()


async def _load_library() -> tuple[list, list]:
    """Load all tracks + My Tag hierarchy from Rekordbox. Returns (tracks, my_tag_tree)."""
    db = RekordboxDatabase()
    await db.connect()
    tracks = await db.get_all_tracks()
    try:
        my_tags = await db.query_my_tags(limit=500)
    except Exception:
        my_tags = []
    await db.disconnect()
    return tracks, my_tags


def _build_library_index(
    tracks: list,
    my_tag_tree: list,
    mik: object = None,
) -> dict:
    """Rebuild the centralized JSONL library index from freshly analyzed tracks.

    Loads the full Essentia cache for every track and writes three output files:
      .data/library_index.jsonl       — one merged record per track
      .data/library_attributes.json  — dynamic tag/genre/BPM attribute summary
      .data/library_context.md        — human-readable LLM context

    Args:
        tracks:      All TrackWithEnergy instances from Rekordbox.
        my_tag_tree: Raw rows from ``db.query_my_tags()``.
        mik:         Already-loaded MixedInKeyLibrary (skips re-loading).
                     If None, MIK is loaded fresh from the env-configured path.
    """
    from .library_index import LibraryIndex
    from .essentia_analyzer import EssentiaFeatureStore

    if mik is None:
        mik = MixedInKeyLibrary.from_env()
        if mik is not None:
            mik.load()
        EnergyResolver(mik).resolve_all(tracks)
    # If mik was passed in, energy is already resolved — skip re-work.

    essentia_store = EssentiaFeatureStore(tracks)
    idx = LibraryIndex()
    return idx.build(
        tracks=tracks,
        essentia_store=essentia_store,
        mik_library=mik,
        my_tag_tree=my_tag_tree,
    )


def _default_workers() -> int:
    cpus = os.cpu_count() or 2
    return max(1, min(4, cpus // 2))


# ── library scan summary ───────────────────────────────────────────────────────

def _print_scan(tracks: list, already_cached: int, to_analyze: int,
                missing: int, no_path: int, workers: int) -> None:
    total = len(tracks)
    with_path = total - no_path
    print(f"\n{BOLD}Library scan{NC}")
    print("─" * 40)
    print(f"  Total tracks in Rekordbox:  {total}")
    print(f"  Have audio file:            {with_path}")
    if no_path:
        print(f"  No file path:               {no_path}")
    print()
    print(f"  {GREEN}Already analyzed:{NC}           {already_cached}")
    print(f"  {CYAN}To analyze now:{NC}             {to_analyze}")
    if missing:
        print(f"  {YELLOW}Missing on disk:{NC}            {missing}")
    print()
    if to_analyze:
        print(f"  Workers: {workers}  │  Estimated: ~{_fmt_eta(to_analyze * 12 / workers)}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    default_workers = _default_workers()

    parser = argparse.ArgumentParser(
        prog="analyze-library",
        description="Batch-analyze all tracks in your Rekordbox library with Essentia.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-analyze tracks that are already cached")
    parser.add_argument("--workers", type=int, default=default_workers, metavar="N",
                        help=f"Parallel worker processes (default: {default_workers})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be analyzed without doing anything")
    args = parser.parse_args()

    if not ESSENTIA_AVAILABLE:
        print("ERROR: Essentia is not installed.\n\n  Run: ./install.sh --essentia\n",
              file=sys.stderr)
        sys.exit(1)

    # ── load & scan library ───────────────────────────────────────────────────
    print("Scanning Rekordbox library…", end="", flush=True)
    try:
        tracks, my_tag_tree = asyncio.run(_load_library())
    except Exception as e:
        print(f"\nERROR: Could not load Rekordbox database: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"\r{' ' * 40}\r", end="")  # clear the scanning line

    with_path  = [t for t in tracks if getattr(t, "file_path", None)]
    no_path    = len(tracks) - len(with_path)

    if args.force:
        pending       = with_path
        already_cached = 0
    else:
        pending        = [t for t in with_path if not _is_cached(t.file_path)]
        already_cached = len(with_path) - len(pending)

    missing    = [t for t in pending if not Path(t.file_path).exists()]
    to_analyze = [t for t in pending if Path(t.file_path).exists()]
    workers    = min(args.workers, max(1, len(to_analyze)))

    _print_scan(tracks, already_cached, len(to_analyze), len(missing), no_path, workers)

    if missing:
        print(f"{YELLOW}Missing files (skipping):{NC}")
        for t in missing[:10]:
            print(f"  {t.file_path}")
        if len(missing) > 10:
            print(f"  … and {len(missing) - 10} more")
        print()

    if args.dry_run:
        print("Dry run — no analysis performed.")
        return

    if not to_analyze:
        print("Nothing to analyze — all tracks already cached.")
        print(f"\n{BOLD}Building library index…{NC}", flush=True)
        try:
            idx_stats = _build_library_index(tracks, my_tag_tree)
            print(
                f"  {GREEN}✓{NC} {idx_stats['total']} tracks indexed  "
                f"({idx_stats['with_essentia']} with Essentia, "
                f"{idx_stats['with_mik']} with MIK)"
            )
            print(f"  Index:      {idx_stats.get('index_path', '.data/library_index.jsonl')}")
        except Exception as e:
            print(f"  {YELLOW}Warning: library index build failed:{NC} {e}")
        return

    # ── incremental index setup ───────────────────────────────────────────────
    # Resolve energy and load MIK once upfront so per-track index updates
    # don't need to re-load them, and so the final _build_library_index()
    # call can reuse the same MIK instance.
    from .library_index import LibraryIndex

    mik_incr = MixedInKeyLibrary.from_env()
    if mik_incr is not None:
        mik_incr.load()
    EnergyResolver(mik_incr).resolve_all(tracks)

    # file_path → track lookup for incremental updates
    fp_to_track: dict[str, object] = {
        t.file_path: t
        for t in tracks
        if getattr(t, "file_path", None)
    }

    # Load any existing index records so previous sessions' data is preserved
    # in the incremental writes (tracks not re-analyzed keep their old record).
    idx_incr = LibraryIndex()
    idx_incr.load_from_disk()
    indexed_at  = datetime.now(timezone.utc).isoformat()
    _since_flush = 0  # count of "ok" tracks since last flush

    # ── parallel analysis via queue ───────────────────────────────────────────
    total      = len(to_analyze)
    ctx        = multiprocessing.get_context("spawn")
    task_q: Queue  = ctx.Queue()
    result_q: Queue = ctx.Queue()

    # Fill task queue
    for t in to_analyze:
        task_q.put(t.file_path)
    # Poison pills — one per worker
    for _ in range(workers):
        task_q.put(None)

    # Track which files are currently being processed
    active: dict[str, float] = {}  # file_path → start_time

    # Start workers
    procs = [
        ctx.Process(target=_worker, args=(task_q, result_q, args.force), daemon=True)
        for _ in range(workers)
    ]
    for p in procs:
        p.start()

    stats: dict[str, int] = {"ok": 0, "error": 0}
    durations: list[float] = []
    wall_start = time.monotonic()
    done = 0

    # Prime: mark first N tasks as active (best-effort — we don't know exact start times)
    for t in to_analyze[:workers]:
        active[t.file_path] = time.monotonic()

    # Draw initial bar (two lines — leave room)
    print()
    _draw_progress(0, total, list(active.keys()), None)

    while done < total:
        try:
            file_path, status, duration = result_q.get(timeout=60)
        except Exception:
            # Check if all workers died unexpectedly
            if all(not p.is_alive() for p in procs):
                break
            continue

        name = Path(file_path).name
        done += 1
        active.pop(file_path, None)

        if status == "ok":
            stats["ok"] += 1
            durations.append(duration)

            # ── incremental index update ──────────────────────────────────────
            # Each worker writes its Essentia result to its own cache file
            # (separate paths — no races). The main process is the sole writer
            # of the JSONL index, so no locking is needed.
            track = fp_to_track.get(file_path)
            if track is not None:
                ess = _load_single_ess(file_path)
                store = _SingleEssentiaStore(file_path, ess) if ess else None
                idx_incr.update_record(
                    track,
                    essentia_store=store,
                    mik_library=mik_incr,
                    indexed_at=indexed_at,
                )
                _since_flush += 1
                if _since_flush >= _FLUSH_EVERY:
                    idx_incr.flush_to_disk()
                    _since_flush = 0
        else:
            stats[status] = stats.get(status, 0) + 1

        avg_s = sum(durations) / len(durations) if durations else None

        # Print the completed track above the progress bar
        _clear_progress()
        if status == "ok":
            print(f"  {GREEN}✓{NC} {name}  {DIM}({duration:.1f}s){NC}")
        elif status == "error":
            print(f"  {RED}✗{NC} {name}")
        # (missing — already reported upfront, skip noise)

        # Mark next queued item as active (approximate)
        _next_idx = done + workers - 1
        if _next_idx < total:
            active[to_analyze[_next_idx].file_path] = time.monotonic()

        _draw_progress(done, total, list(active.keys()), avg_s)

    _clear_progress()

    for p in procs:
        p.join(timeout=5)

    # Flush any remaining in-memory updates that didn't hit the _FLUSH_EVERY threshold
    if _since_flush > 0:
        idx_incr.flush_to_disk()

    wall = time.monotonic() - wall_start
    avg_str = f"  Avg/track:  {sum(durations)/len(durations):.1f}s\n" if durations else ""

    print(f"\n{BOLD}{'─' * 50}{NC}")
    print(f"{GREEN}Done{NC} in {_fmt_eta(wall)}")
    print(f"  Analyzed:   {stats.get('ok', 0)}")
    print(f"  Errors:     {stats.get('error', 0)}")
    print(f"  Cached:     {already_cached} (skipped)")
    print(avg_str, end="")

    # ── rebuild library index ─────────────────────────────────────────────────
    print(f"\n{BOLD}Building library index…{NC}", flush=True)
    try:
        idx_stats = _build_library_index(tracks, my_tag_tree, mik=mik_incr)
        print(
            f"  {GREEN}✓{NC} {idx_stats['total']} tracks indexed  "
            f"({idx_stats['with_essentia']} with Essentia, "
            f"{idx_stats['with_mik']} with MIK)"
        )
        print(f"  Index:      {idx_stats.get('index_path', '.data/library_index.jsonl')}")
    except Exception as e:
        print(f"  {YELLOW}Warning: library index build failed:{NC} {e}")


def _build_index_cli_main() -> None:
    """Entry point for ``build-library-index``.

    Rebuilds .data/library_index.jsonl (+ attributes + context MD) from the
    existing Essentia cache — no audio re-analysis is performed.

    Usage:
        build-library-index
        build-library-index --force   # force rebuild even if index is fresh
    """
    parser = argparse.ArgumentParser(
        prog="build-library-index",
        description=(
            "Rebuild the centralized library index from the existing Essentia cache.\n"
            "Writes .data/library_index.jsonl, library_attributes.json, "
            "and library_context.md."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the index was written within the last hour",
    )
    args = parser.parse_args()

    from .library_index import LibraryIndex

    idx = LibraryIndex()
    if not args.force and idx.is_fresh():
        print("Library index is up-to-date (use --force to rebuild).")
        return

    print("Loading Rekordbox library…", end="", flush=True)
    try:
        tracks, my_tag_tree = asyncio.run(_load_library())
    except Exception as e:
        print(f"\nERROR: Could not load Rekordbox database: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"\r{' ' * 40}\r", end="")

    print(f"{BOLD}Building library index…{NC}", flush=True)
    try:
        stats = _build_library_index(tracks, my_tag_tree)
        print(
            f"  {GREEN}✓{NC} {stats['total']} tracks indexed  "
            f"({stats['with_essentia']} with Essentia, "
            f"{stats['with_mik']} with MIK)"
        )
        print(f"  Index:      {stats.get('index_path', '.data/library_index.jsonl')}")
        print(f"  Attributes: {stats.get('attributes_path', '.data/library_attributes.json')}")
        print(f"  Context:    {stats.get('context_path', '.data/library_context.md')}")
    except Exception as e:
        print(f"  {RED}ERROR:{NC} {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
