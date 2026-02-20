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
import multiprocessing
import os
import shutil
import sys
import time
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

# ── terminal helpers ──────────────────────────────────────────────────────────

_TERM_WIDTH = shutil.get_terminal_size((80, 20)).columns

GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
RED    = "\033[0;31m"
CYAN   = "\033[0;36m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
NC     = "\033[0m"


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


async def _get_tracks() -> list:
    db = RekordboxDatabase()
    await db.connect()
    return await db.get_all_tracks()


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
        tracks = asyncio.run(_get_tracks())
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
        print("Nothing to analyze.")
        return

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
        idx = done + workers - 1
        if idx < total:
            active[to_analyze[idx].file_path] = time.monotonic()

        _draw_progress(done, total, list(active.keys()), avg_s)

    _clear_progress()

    for p in procs:
        p.join(timeout=5)

    wall = time.monotonic() - wall_start
    avg_str = f"  Avg/track:  {sum(durations)/len(durations):.1f}s\n" if durations else ""

    print(f"\n{BOLD}{'─' * 50}{NC}")
    print(f"{GREEN}Done{NC} in {_fmt_eta(wall)}")
    print(f"  Analyzed:   {stats.get('ok', 0)}")
    print(f"  Errors:     {stats.get('error', 0)}")
    print(f"  Cached:     {already_cached} (skipped)")
    print(avg_str, end="")


if __name__ == "__main__":
    main()
