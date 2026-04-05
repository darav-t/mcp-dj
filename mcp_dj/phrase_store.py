"""
PhraseStore — phrase detection storage backed by library_index.jsonl.

Phrase data (sections + mix profile) is stored directly on each track's
record in library_index.jsonl under the keys:
    record["phrases"]     — list of {time_ms, label, bar_number, color}
    record["mix_profile"] — DJ mix parameters (intro length, drop, outro, blend)

The LibraryIndex is the single source of truth.  A lightweight file cache
under .data/phrase_cache/ is used only during batch analysis (so results
survive mid-run crashes before the final index flush).

DJ Mixing Pattern Reference (implemented here):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Standard Blend (most common — 32-64 bars for techno/prog house)    │
  │                                                                     │
  │  Outgoing:  [INTRO][UP][UP][CHORUS][DOWN][DOWN][UP][CHORUS][OUTRO]  │
  │  Incoming:                              [INTRO][UP][UP]←mix starts  │
  │                                                                     │
  │  • Incoming starts at outgoing's outro_start_ms                     │
  │  • Both tracks play for blend_bars simultaneously                   │
  │  • Bass swap at phrase boundary (8-bar window inside blend)         │
  │                                                                     │
  │  Drop-to-Drop (peak time, short intros):                            │
  │  • Align incoming first_chorus_ms with outgoing's second chorus     │
  │                                                                     │
  │  Breakdown Entry (melodic techno / prog house teaser):              │
  │  • Start incoming at its first_down_ms to preview melody            │
  └─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

from .phrase_detector import (
    MixProfile,
    PhraseCue,
    PhraseDetector,
    derive_mix_profile,
)

if TYPE_CHECKING:
    from .library_index import LibraryIndex

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / ".data"
PHRASE_CACHE_DIR = _DATA_DIR / "phrase_cache"


# ---------------------------------------------------------------------------
# PhraseStore
# ---------------------------------------------------------------------------

class PhraseStore:
    """
    Read/write phrase analysis results backed by the library index.

    Primary storage: library_index.jsonl (via LibraryIndex.update_phrase_data)
    Fallback cache:  .data/phrase_cache/{track_id}.json  (for mid-run resilience)

    When a LibraryIndex is attached, all reads go to the index first.
    File cache is written on every save so partial batch runs aren't lost.
    """

    def __init__(
        self,
        library_index: Optional["LibraryIndex"] = None,
        cache_dir: Path = PHRASE_CACHE_DIR,
    ):
        self._index     = library_index
        self._dir       = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def attach_index(self, library_index: "LibraryIndex") -> None:
        """Attach (or replace) the library index after construction."""
        self._index = library_index

    # ------------------------------------------------------------------
    # Existence check
    # ------------------------------------------------------------------

    def has(self, track_id: str) -> bool:
        """Return True if phrase data exists in index OR file cache."""
        if self._index is not None:
            rec = self._index._by_id.get(str(track_id))
            if rec and "phrases" in rec:
                return True
        return (self._dir / f"{track_id}.json").exists()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, track_id: str) -> Optional[dict]:
        """Return raw dict {phrases, mix_profile} or None."""
        # 1. Try library index first
        if self._index is not None:
            rec = self._index._by_id.get(str(track_id))
            if rec and "phrases" in rec:
                return {
                    "track_id":   track_id,
                    "phrases":    rec["phrases"],
                    "mix_profile": rec.get("mix_profile", {}),
                    "bpm":        rec.get("bpm", 0),
                    "duration_s": rec.get("length_seconds", 0),
                }

        # 2. Fall back to file cache
        path = self._dir / f"{track_id}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception as e:
                logger.warning("PhraseStore: corrupt cache for %s: %s", track_id, e)
        return None

    def get_mix_profile(self, track_id: str) -> Optional[MixProfile]:
        """Return MixProfile if available, else None."""
        data = self.get(track_id)
        if not data or "mix_profile" not in data:
            return None
        mp = data["mix_profile"]
        try:
            return MixProfile(**mp)
        except TypeError:
            return None

    def get_phrases(self, track_id: str) -> list[PhraseCue]:
        """Return list of PhraseCue objects if cached, else []."""
        data = self.get(track_id)
        if not data or "phrases" not in data:
            return []
        return [
            PhraseCue(
                time_ms    = p["time_ms"],
                label      = p["label"],
                color      = p.get("color", "none"),
                bar_number = p.get("bar_number", 1),
            )
            for p in data["phrases"]
        ]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(
        self,
        track_id: str,
        cues: list[PhraseCue],
        bpm: float,
        duration_s: float,
    ) -> MixProfile:
        """
        Persist phrase cues + derived MixProfile.

        Writes to BOTH:
          1. Library index (in-memory; caller must flush_to_disk() periodically)
          2. File cache (immediately, for mid-run crash resilience)

        Returns the derived MixProfile.
        """
        mix_profile = derive_mix_profile(cues, bpm, duration_s)

        phrases_list = [
            {
                "time_ms":    c.time_ms,
                "label":      c.label,
                "bar_number": c.bar_number,
                "color":      c.color,
            }
            for c in cues
        ]
        mp_dict = {
            "intro_end_ms":         mix_profile.intro_end_ms,
            "pre_drop_ms":          mix_profile.pre_drop_ms,
            "first_chorus_ms":      mix_profile.first_chorus_ms,
            "first_down_ms":        mix_profile.first_down_ms,
            "second_chorus_ms":     mix_profile.second_chorus_ms,
            "outro_start_ms":       mix_profile.outro_start_ms,
            "pre_drop_bars":        mix_profile.pre_drop_bars,
            "outro_bars":           mix_profile.outro_bars,
            "bar_duration_ms":      mix_profile.bar_duration_ms,
            "mix_in_ms":            mix_profile.mix_in_ms,
            "mix_out_ms":           mix_profile.mix_out_ms,
            "blend_bars":           mix_profile.blend_bars,
            "blend_ms":             mix_profile.blend_ms,
            "preferred_transition": mix_profile.preferred_transition,
        }

        # 1. Update library index in-memory
        if self._index is not None:
            self._index.update_phrase_data(track_id, phrases_list, mp_dict)

        # 2. Write file cache (immediate, crash-safe)
        record: dict[str, Any] = {
            "track_id":    track_id,
            "bpm":         bpm,
            "duration_s":  duration_s,
            "phrases":     phrases_list,
            "mix_profile": mp_dict,
        }
        (self._dir / f"{track_id}.json").write_text(json.dumps(record, indent=2))

        return mix_profile

    def count(self) -> int:
        """Count tracks with phrase data (from index if attached, else file cache)."""
        if self._index is not None:
            return sum(
                1 for r in self._index._by_id.values()
                if "phrases" in r
            )
        return sum(1 for _ in self._dir.glob("*.json"))


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

def analyze_library_phrases(
    tracks: list,
    phrase_store: PhraseStore,
    library_index: Optional["LibraryIndex"] = None,
    force: bool = False,
    flush_every: int = 10,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Run phrase detection on all library tracks.

    For each track:
      1. Detects sections (Intro/Up/Chorus/Down/Outro) and MixProfile
      2. Stores results via phrase_store.save() → library index + file cache
      3. Flushes library_index to disk every `flush_every` tracks

    Parameters
    ----------
    tracks           : List of track objects (.id, .file_path, .bpm, .length)
    phrase_store     : PhraseStore instance (should have library_index attached)
    library_index    : LibraryIndex for flushing (can also be via phrase_store)
    force            : Re-analyze even if data already exists
    flush_every      : Flush library_index to disk after this many analyzed tracks
    progress_callback: Optional callable(track_id, index, total, status)
                       status: "cached" | "analyzed" | "skipped_no_file" | "error:<msg>"
    """
    # Allow library_index passed directly (convenience)
    if library_index is not None and phrase_store._index is None:
        phrase_store.attach_index(library_index)
    _index = phrase_store._index or library_index

    detector   = PhraseDetector()
    total      = len(tracks)
    analyzed   = 0
    cached     = 0
    errors     = 0
    skipped    = 0
    _since_flush = 0

    for i, track in enumerate(tracks):
        track_id = str(track.id)

        if not force and phrase_store.has(track_id):
            cached += 1
            if progress_callback:
                progress_callback(track_id, i, total, "cached")
            continue

        fp = getattr(track, "file_path", None)
        if not fp or not os.path.exists(fp):
            skipped += 1
            if progress_callback:
                progress_callback(track_id, i, total, "skipped_no_file")
            continue

        bpm   = float(getattr(track, "bpm",    0) or 128.0)
        dur_s = float(getattr(track, "length", 0) or 300)

        try:
            cues = detector.detect(fp, bpm=bpm)
            phrase_store.save(track_id, cues, bpm=bpm, duration_s=dur_s)
            analyzed    += 1
            _since_flush += 1

            # Flush library index periodically
            if _index is not None and _since_flush >= flush_every:
                _index.flush_to_disk()
                _since_flush = 0
                logger.info(
                    "Phrase analysis flush: %d analyzed so far (%d/%d tracks)",
                    analyzed, i + 1, total,
                )

            if progress_callback:
                progress_callback(track_id, i, total, "analyzed")

        except Exception as e:
            err_msg = str(e)
            logger.error(
                "Phrase analysis failed for %s (%s): %s",
                getattr(track, "title", track_id), track_id, err_msg,
            )
            errors += 1
            if progress_callback:
                progress_callback(track_id, i, total, f"error:{err_msg[:80]}")

    # Final flush
    if _index is not None and _since_flush > 0:
        _index.flush_to_disk()
        logger.info("Final phrase flush: %d records on disk", len(_index._by_id))

    logger.info(
        "analyze_library_phrases complete: %d analyzed, %d cached, %d errors, %d skipped",
        analyzed, cached, errors, skipped,
    )
    return {
        "total":           total,
        "analyzed":        analyzed,
        "cached":          cached,
        "errors":          errors,
        "skipped_no_file": skipped,
    }


# ---------------------------------------------------------------------------
# Transition planning
# ---------------------------------------------------------------------------

def plan_transition(
    outgoing: MixProfile,
    incoming: MixProfile,
    set_phase: float = 0.5,
) -> dict:
    """
    Given two MixProfiles, compute the optimal transition plan.

    Parameters
    ----------
    outgoing    : MixProfile of the currently playing track
    incoming    : MixProfile of the track about to be mixed in
    set_phase   : Position in the set (0.0 = warmup, 1.0 = end)

    Returns
    -------
    dict with transition_type, mix points, overlap_bars, bass_swap_ms, notes.
    """
    outgoing_runway = outgoing.outro_bars
    incoming_runway = incoming.pre_drop_bars

    possible_blend = min(outgoing_runway, incoming_runway, 64)
    possible_blend = max(possible_blend, 8)

    peak_time       = 0.45 <= set_phase <= 0.75
    both_have_chorus = outgoing.first_chorus_ms > 0 and incoming.first_chorus_ms > 0

    if (peak_time and incoming.pre_drop_bars < 16 and both_have_chorus
            and outgoing.second_chorus_ms > 0):
        transition_type = "drop_to_drop"
    elif (incoming.first_down_ms > 0
          and outgoing.outro_bars >= 32
          and incoming.pre_drop_bars >= 48):
        transition_type = "breakdown_entry"
    else:
        transition_type = "blend"

    if transition_type == "blend":
        overlap_bars        = possible_blend
        outgoing_mix_out_ms = outgoing.outro_start_ms
        incoming_mix_in_ms  = 0

    elif transition_type == "drop_to_drop":
        target_drop_ms      = outgoing.second_chorus_ms or outgoing.outro_start_ms
        incoming_mix_in_ms  = 0
        outgoing_mix_out_ms = target_drop_ms
        overlap_bars        = max(8, int(incoming.first_chorus_ms / incoming.bar_duration_ms))

    else:  # breakdown_entry
        overlap_bars        = outgoing.outro_bars
        outgoing_mix_out_ms = outgoing.outro_start_ms
        incoming_mix_in_ms  = incoming.first_down_ms

    overlap_ms   = int(overlap_bars * outgoing.bar_duration_ms)
    bass_swap_ms = outgoing_mix_out_ms + int(8 * outgoing.bar_duration_ms)

    out_min, out_sec = divmod(outgoing_mix_out_ms // 1000, 60)
    notes = {
        "blend":           f"{overlap_bars}-bar blend — outgoing exits at {out_min}:{out_sec:02d} ({incoming.pre_drop_bars}bar intro)",
        "drop_to_drop":    f"Drop-to-drop — align drops, {overlap_bars}-bar runway",
        "breakdown_entry": f"Breakdown entry — incoming teases from {incoming.first_down_ms//1000}s",
    }[transition_type]

    return {
        "transition_type":     transition_type,
        "outgoing_mix_out_ms": outgoing_mix_out_ms,
        "incoming_mix_in_ms":  incoming_mix_in_ms,
        "overlap_bars":        overlap_bars,
        "overlap_ms":          overlap_ms,
        "bass_swap_ms":        bass_swap_ms,
        "notes":               notes,
    }
