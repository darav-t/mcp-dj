"""
Library Index — Centralized JSONL track database for LLM grep and vector search.

Merges Rekordbox, Essentia, and Mixed In Key data into a single JSONL file at
.data/library_index.jsonl — one complete JSON object per line, one line per track.

Each record contains every field from every data source plus a ``_text`` field
that is a grep-optimized summary string.  Claude can search it with the Grep
tool; any vector-store pipeline can ingest it directly (each line = one document).

A companion ``library_attributes.json`` stores a full statistical summary of the
library (tag counts, BPM ranges per tag/genre, mood distribution, etc.) built
dynamically from the actual data — no hardcoded values anywhere.

A human-readable ``library_context.md`` is also written for direct LLM injection.

Usage:
    index = LibraryIndex()
    stats = index.build(tracks, essentia_store, mik_library, my_tag_tree=rows)
    attrs = index.attributes          # dict — pass as context to any LLM
    results = index.search("aggressive techno 130bpm")
    record  = index.get_by_id("12345")
"""

from __future__ import annotations

import json
import statistics
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT     = Path(__file__).resolve().parent.parent
INDEX_PATH     = _REPO_ROOT / ".data" / "library_index.jsonl"
ATTRIBUTES_PATH = _REPO_ROOT / ".data" / "library_attributes.json"
CONTEXT_PATH   = _REPO_ROOT / ".data" / "library_context.md"
_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Attribute scanning
# ---------------------------------------------------------------------------

def build_attributes(
    tracks: list,
    essentia_store: Any = None,
    my_tag_tree: Optional[list] = None,
) -> dict:
    """
    Scan every track and compute a comprehensive attribute summary.

    All values come from the actual library data — nothing is hardcoded.
    The result is written to ``library_attributes.json`` and returned.

    Args:
        tracks:         List of TrackWithEnergy instances.
        essentia_store: Optional EssentiaFeatureStore — supplies mood/genre data.
        my_tag_tree:    Optional raw rows from ``db.query_my_tags()`` — used to
                        reconstruct the My Tag parent/child hierarchy.

    Returns:
        Dict suitable for JSON serialisation and LLM context injection.
    """
    # Accumulators
    tag_bpms:       dict[str, list[float]] = defaultdict(list)
    tag_energies:   dict[str, list[int]]   = defaultdict(list)
    tag_moods:      dict[str, Counter]     = defaultdict(Counter)
    tag_cooccur:    dict[str, Counter]     = defaultdict(Counter)
    tag_genres:     dict[str, Counter]     = defaultdict(Counter)

    genre_bpms:     dict[str, list[float]] = defaultdict(list)
    genre_energies: dict[str, list[int]]   = defaultdict(list)

    all_bpms:     list[float] = []
    all_energies: list[int]   = []
    energy_dist:  Counter     = Counter()
    energy_srcs:  Counter     = Counter()
    keys:         Counter     = Counter()
    genres:       Counter     = Counter()
    artists:      Counter     = Counter()
    colors:       Counter     = Counter()
    dates:        list[str]   = []
    ratings:      list[int]   = []
    play_counts:  list[int]   = []

    mood_global:  Counter = Counter()
    essentia_count = 0

    tag_counts: Counter = Counter()

    # Full feature vector accumulators — mood probabilities, Discogs genres, music tags
    tag_mood_vectors:    dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    global_mood_vectors: dict[str, list]            = defaultdict(list)
    tag_discogs_vectors:    dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    global_discogs_vectors: dict[str, list]            = defaultdict(list)
    tag_music_tag_vectors:    dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    global_music_tag_vectors: dict[str, list]            = defaultdict(list)

    for track in tracks:
        bpm          = float(getattr(track, "bpm", 0) or 0)
        energy       = getattr(track, "energy", None)
        genre        = (getattr(track, "genre", None) or "").strip()
        key          = getattr(track, "key", None)
        artist       = (getattr(track, "artist", None) or "").strip()
        date_added   = getattr(track, "date_added", None)
        energy_src   = getattr(track, "energy_source", "none") or "none"
        color        = getattr(track, "color", None)
        rating       = getattr(track, "rating", 0) or 0
        play_count   = getattr(track, "play_count", 0) or 0
        track_tags   = list(getattr(track, "my_tags", []) or [])

        # Global accumulators
        if bpm > 0:
            all_bpms.append(bpm)
        if energy is not None:
            all_energies.append(energy)
            energy_dist[energy] += 1
        if key:
            keys[key] += 1
        if genre:
            genres[genre] += 1
        if artist:
            artists[artist] += 1
        if date_added:
            dates.append(date_added)
        if color:
            colors[color] += 1
        energy_srcs[energy_src] += 1
        ratings.append(rating)
        if play_count > 0:
            play_counts.append(play_count)

        # Per-tag accumulators
        for tag in track_tags:
            tag_counts[tag] += 1
            if bpm > 0:
                tag_bpms[tag].append(bpm)
            if energy is not None:
                tag_energies[tag].append(energy)
            if genre:
                tag_genres[tag][genre] += 1

        # Tag co-occurrence
        for i, tag_a in enumerate(track_tags):
            for tag_b in track_tags:
                if tag_a != tag_b:
                    tag_cooccur[tag_a][tag_b] += 1

        # Per-genre accumulators
        if genre:
            if bpm > 0:
                genre_bpms[genre].append(bpm)
            if energy is not None:
                genre_energies[genre].append(energy)

        # Essentia: mood, Discogs genre, and music tags feature vectors
        if essentia_store is not None:
            fp = getattr(track, "file_path", None)
            if fp:
                ess = essentia_store.get(fp)
                if ess is not None:
                    essentia_count += 1
                    mood = ess.dominant_mood()
                    if mood:
                        mood_global[mood] += 1
                        for tag in track_tags:
                            tag_moods[tag][mood] += 1

                    # Full mood probability vector
                    mood_vec = {
                        "happy":      ess.mood_happy,
                        "sad":        ess.mood_sad,
                        "aggressive": ess.mood_aggressive,
                        "relaxed":    ess.mood_relaxed,
                        "party":      ess.mood_party,
                    }
                    for m_key, m_val in mood_vec.items():
                        if m_val is not None:
                            global_mood_vectors[m_key].append(m_val)
                            for tag in track_tags:
                                tag_mood_vectors[tag][m_key].append(m_val)

                    # Discogs genre probability vector (strip "Electronic---" prefix)
                    if ess.genre_discogs:
                        for raw_g, g_score in ess.genre_discogs.items():
                            g_name = raw_g.split("---", 1)[-1]
                            global_discogs_vectors[g_name].append(g_score)
                            for tag in track_tags:
                                tag_discogs_vectors[tag][g_name].append(g_score)

                    # Music tags probability vector
                    if ess.music_tags:
                        for mt in ess.music_tags:
                            t_name, t_score = mt["tag"], mt["score"]
                            global_music_tag_vectors[t_name].append(t_score)
                            for tag in track_tags:
                                tag_music_tag_vectors[tag][t_name].append(t_score)

    # -----------------------------------------------------------------------
    # Helper — compute statistics for a list of numbers
    # -----------------------------------------------------------------------
    def _stats(values: list) -> dict:
        if not values:
            return {}
        s = sorted(values)
        n = len(s)
        return {
            "count": n,
            "min":   round(min(s), 1),
            "max":   round(max(s), 1),
            "avg":   round(statistics.mean(s), 1),
            "p25":   round(s[max(0, n // 4 - 1)], 1),
            "p50":   round(s[n // 2], 1),
            "p75":   round(s[min(n - 1, 3 * n // 4)], 1),
        }

    # -----------------------------------------------------------------------
    # My Tag details (per-tag stats, sorted by count descending)
    # -----------------------------------------------------------------------
    my_tags_sorted: dict[str, int] = dict(tag_counts.most_common())

    my_tag_details: dict[str, dict] = {}
    for tag, count in my_tags_sorted.items():
        detail: dict[str, Any] = {"count": count}
        bpm_s = _stats(tag_bpms[tag])
        if bpm_s:
            detail["bpm"] = bpm_s
        energy_s = _stats(tag_energies[tag])
        if energy_s:
            detail["energy"] = energy_s
        if tag_moods[tag]:
            detail["dominant_mood"] = tag_moods[tag].most_common(1)[0][0]
            detail["mood_dist"] = dict(tag_moods[tag].most_common())
        # Averaged mood probability vector across all tracks with this tag
        if tag_mood_vectors[tag]:
            detail["mood_avg"] = {
                m: round(statistics.mean(vals), 3)
                for m, vals in tag_mood_vectors[tag].items()
                if vals
            }
        # Averaged Discogs genre distribution (top 10 by average score)
        if tag_discogs_vectors[tag]:
            genre_avgs = {
                g: round(statistics.mean(vals), 3)
                for g, vals in tag_discogs_vectors[tag].items()
                if vals
            }
            detail["discogs_genres_avg"] = dict(
                sorted(genre_avgs.items(), key=lambda x: x[1], reverse=True)[:10]
            )
        # Averaged music tags distribution (top 10 by average score)
        if tag_music_tag_vectors[tag]:
            music_tag_avgs = {
                t: round(statistics.mean(vals), 3)
                for t, vals in tag_music_tag_vectors[tag].items()
                if vals
            }
            detail["music_tags_avg"] = dict(
                sorted(music_tag_avgs.items(), key=lambda x: x[1], reverse=True)[:10]
            )
        if tag_genres[tag]:
            detail["top_genres"] = [g for g, _ in tag_genres[tag].most_common(5)]
        if tag_cooccur[tag]:
            detail["often_with"] = [
                {"tag": t, "count": c}
                for t, c in tag_cooccur[tag].most_common(5)
            ]
        my_tag_details[tag] = detail

    # -----------------------------------------------------------------------
    # My Tag hierarchy from raw DB rows
    # -----------------------------------------------------------------------
    my_tag_hierarchy: dict[str, list[str]] = {}
    if my_tag_tree:
        id_to_name = {row["ID"]: row["Name"] for row in my_tag_tree}
        for row in my_tag_tree:
            # Attribute==1 → folder/group; Attribute==0 → leaf tag
            if row.get("Attribute") == 1:
                group = row["Name"]
                if group.startswith("---"):
                    continue
                children = [
                    id_to_name[r["ID"]]
                    for r in my_tag_tree
                    if r.get("ParentID") == row["ID"]
                    and r.get("Attribute") == 0
                    and not id_to_name.get(r["ID"], "").startswith("---")
                ]
                if children:
                    my_tag_hierarchy[group] = children

    # -----------------------------------------------------------------------
    # Genre details
    # -----------------------------------------------------------------------
    genre_details: dict[str, dict] = {}
    for genre_name, count in genres.most_common():
        detail = {"count": count}
        bpm_s = _stats(genre_bpms[genre_name])
        if bpm_s:
            detail["bpm"] = bpm_s
        energy_s = _stats(genre_energies[genre_name])
        if energy_s:
            detail["energy"] = energy_s
        genre_details[genre_name] = detail

    # -----------------------------------------------------------------------
    # Assemble final attributes dict
    # -----------------------------------------------------------------------
    attrs: dict[str, Any] = {
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "track_count":       len(tracks),
        "essentia_coverage": essentia_count,

        # My Tags — hierarchy + per-tag stats
        "my_tag_hierarchy": my_tag_hierarchy,
        "my_tags":          my_tags_sorted,        # {tag: count} ordered by count
        "my_tag_details":   my_tag_details,         # {tag: {count,bpm,energy,mood,…}}

        # Genres — count + BPM/energy ranges
        "genres":           dict(genres.most_common()),
        "genre_details":    genre_details,

        # Global distributions
        "keys":             dict(keys.most_common()),
        "bpm":              _stats(all_bpms),
        "energy": {
            **_stats(all_energies),
            "distribution": dict(sorted(energy_dist.items())),
        },
        "ratings": {
            **_stats(ratings),
            "distribution": dict(sorted(Counter(ratings).items())),
        },
        "play_counts":      _stats(play_counts),
        "moods":            dict(mood_global.most_common()),
        # Library-wide averaged mood probability vector
        "mood_avg": (
            {m: round(statistics.mean(vals), 3) for m, vals in global_mood_vectors.items() if vals}
            if global_mood_vectors else {}
        ),
        # Library-wide averaged Discogs genre distribution (top 20)
        "discogs_genres_avg": (
            dict(sorted(
                {g: round(statistics.mean(vals), 3) for g, vals in global_discogs_vectors.items() if vals}.items(),
                key=lambda x: x[1], reverse=True,
            )[:20])
            if global_discogs_vectors else {}
        ),
        # Library-wide averaged music tags distribution (top 20)
        "music_tags_avg": (
            dict(sorted(
                {t: round(statistics.mean(vals), 3) for t, vals in global_music_tag_vectors.items() if vals}.items(),
                key=lambda x: x[1], reverse=True,
            )[:20])
            if global_music_tag_vectors else {}
        ),
        "colors":           dict(colors.most_common()),
        "energy_sources":   dict(energy_srcs.most_common()),
        "date_range": (
            {"min": min(dates), "max": max(dates)} if dates else {}
        ),

        # Top artists (up to 30)
        "top_artists": [
            {"artist": a, "count": c} for a, c in artists.most_common(30)
        ],
    }

    return attrs


# ---------------------------------------------------------------------------
# Human-readable LLM context document
# ---------------------------------------------------------------------------

def _write_context_md(attrs: dict, path: Path) -> None:
    """
    Write a compact Markdown document summarising the library attributes.

    Designed to be injected verbatim into an LLM prompt so the model knows
    exactly what tags, genres, BPM ranges, and moods are available.
    """
    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"{'#' * level} {text}")
        lines.append("")

    def row(*cells: str) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    def divider(n: int) -> str:
        return "| " + " | ".join(["---"] * n) + " |"

    h(1, "Library Context")
    lines += [
        f"**Generated:** {attrs.get('generated_at', '')}  ",
        f"**Tracks:** {attrs.get('track_count', 0):,}  ",
        f"**Essentia analysis:** {attrs.get('essentia_coverage', 0):,} tracks  ",
        "",
    ]

    # ------------------------------------------------------------------
    # My Tags — grouped by hierarchy
    # ------------------------------------------------------------------
    hierarchy    = attrs.get("my_tag_hierarchy", {})
    tag_details  = attrs.get("my_tag_details", {})
    all_tags     = attrs.get("my_tags", {})

    if hierarchy or all_tags:
        h(2, "My Tags")

        def _tag_table(tags: list[str]) -> None:
            lines.append(row("Tag", "Tracks", "BPM range", "Avg E", "Mood", "Often with"))
            lines.append(divider(6))
            for tag in tags:
                d = tag_details.get(tag, {})
                count = d.get("count", all_tags.get(tag, 0))
                bpm_d = d.get("bpm", {})
                bpm_s = (f"{bpm_d['min']:.0f}–{bpm_d['max']:.0f}"
                         if bpm_d else "—")
                e_d   = d.get("energy", {})
                e_s   = f"{e_d['avg']:.1f}" if e_d else "—"
                mood  = d.get("dominant_mood", "—")
                cooc  = d.get("often_with", [])
                cooc_s = ", ".join(c["tag"] for c in cooc[:3]) if cooc else "—"
                lines.append(row(tag, count, bpm_s, e_s, mood, cooc_s))
            lines.append("")

        if hierarchy:
            for group, children in hierarchy.items():
                h(3, group)
                # Only include tags that actually appear in the library
                present = [t for t in children if t in all_tags]
                if present:
                    _tag_table(present)
        else:
            # Flat list fallback
            _tag_table(list(all_tags.keys())[:40])

    # ------------------------------------------------------------------
    # Genres
    # ------------------------------------------------------------------
    genre_details = attrs.get("genre_details", {})
    if genre_details:
        h(2, "Genres")
        lines.append(row("Genre", "Tracks", "BPM range", "BPM avg", "Avg Energy"))
        lines.append(divider(5))
        for gname, gd in list(genre_details.items())[:20]:
            count = gd.get("count", 0)
            bpm_d = gd.get("bpm", {})
            bpm_range = (f"{bpm_d['min']:.0f}–{bpm_d['max']:.0f}" if bpm_d else "—")
            bpm_avg   = f"{bpm_d['avg']:.1f}" if bpm_d else "—"
            e_d   = gd.get("energy", {})
            e_avg = f"{e_d['avg']:.1f}" if e_d else "—"
            lines.append(row(gname, count, bpm_range, bpm_avg, e_avg))
        lines.append("")

    # ------------------------------------------------------------------
    # Global stats
    # ------------------------------------------------------------------
    h(2, "Global Stats")

    bpm_g = attrs.get("bpm", {})
    if bpm_g:
        lines.append(
            f"- **BPM:** {bpm_g.get('min', 0):.0f}–{bpm_g.get('max', 0):.0f} "
            f"(avg {bpm_g.get('avg', 0):.1f}, p25 {bpm_g.get('p25', 0):.0f}, "
            f"p75 {bpm_g.get('p75', 0):.0f})"
        )

    energy_g = attrs.get("energy", {})
    if energy_g:
        dist = energy_g.get("distribution", {})
        dist_s = "  ".join(f"E{k}:{v}" for k, v in sorted(dist.items()))
        lines.append(
            f"- **Energy:** avg {energy_g.get('avg', 0):.1f}/10 | {dist_s}"
        )

    moods = attrs.get("moods", {})
    if moods:
        total_m = sum(moods.values()) or 1
        mood_s = "  ".join(
            f"{m}: {100 * c // total_m}%" for m, c in moods.items()
        )
        lines.append(f"- **Moods (Essentia):** {mood_s}")

    keys_g = attrs.get("keys", {})
    if keys_g:
        top_k = list(keys_g.items())[:10]
        keys_s = "  ".join(f"{k}({v})" for k, v in top_k)
        lines.append(f"- **Top keys:** {keys_s}")

    colors_g = attrs.get("colors", {})
    if colors_g:
        colors_s = "  ".join(f"{c}({v})" for c, v in list(colors_g.items())[:8])
        lines.append(f"- **Colors:** {colors_s}")

    e_srcs = attrs.get("energy_sources", {})
    if e_srcs:
        srcs_s = "  ".join(f"{s}({c})" for s, c in e_srcs.items())
        lines.append(f"- **Energy sources:** {srcs_s}")

    d_range = attrs.get("date_range", {})
    if d_range:
        lines.append(f"- **Date range:** {d_range.get('min', '?')} → {d_range.get('max', '?')}")

    lines.append("")

    # ------------------------------------------------------------------
    # Top artists
    # ------------------------------------------------------------------
    top_artists = attrs.get("top_artists", [])
    if top_artists:
        h(2, "Top Artists")
        artist_s = "  ".join(
            f"{a['artist']} ({a['count']})" for a in top_artists[:20]
        )
        lines.append(artist_s)
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.debug(f"Library context written → {path}")


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------

def _fmt_duration(length_seconds: int) -> str:
    """Convert integer seconds to 'M:SS' string."""
    if not length_seconds:
        return "0:00"
    m, s = divmod(int(length_seconds), 60)
    return f"{m}:{s:02d}"


def _build_record_from_track(
    track: Any,
    essentia_store: Any = None,
    mik_library: Any = None,
    indexed_at: str = "",
) -> dict:
    """
    Build a complete merged record dict for a single track.

    Pure function (no I/O).  Called by LibraryIndex.build() for every track.
    """
    length_sec = getattr(track, "length", 0) or 0
    record: dict[str, Any] = {
        "_schema_version": _SCHEMA_VERSION,
        "id":            str(track.id),
        "title":         track.title,
        "artist":        track.artist,
        "album":         getattr(track, "album", None),
        "genre":         getattr(track, "genre", None),
        "bpm":           float(track.bpm) if track.bpm else 0.0,
        "key":           getattr(track, "key", None),
        "rating":        getattr(track, "rating", 0) or 0,
        "play_count":    getattr(track, "play_count", 0) or 0,
        "length_seconds": length_sec,
        "duration":      _fmt_duration(length_sec),
        "file_path":     getattr(track, "file_path", None),
        "date_added":    getattr(track, "date_added", None),
        "date_modified": getattr(track, "date_modified", None),
        "comments":      getattr(track, "comments", None),
        "color":         getattr(track, "color", None),
        "color_id":      getattr(track, "color_id", None),
        "my_tags":       list(getattr(track, "my_tags", []) or []),
        "energy":        getattr(track, "energy", None),
        "energy_source": getattr(track, "energy_source", "none"),
    }

    # Strip None values; restore mandatory keys
    record = {k: v for k, v in record.items() if v is not None}
    for key in ("id", "title", "artist", "bpm", "rating", "play_count",
                "length_seconds", "duration", "energy_source", "my_tags"):
        if key not in record:
            record[key] = [] if key == "my_tags" else ""

    # Essentia block
    if essentia_store is not None:
        fp = getattr(track, "file_path", None)
        if fp:
            ess = essentia_store.get(fp)
            if ess is not None:
                ess_dict = ess.to_cache_dict()
                ess_dict.pop("file_path", None)
                if ess_dict:
                    record["essentia"] = ess_dict

    # MIK block
    if mik_library is not None:
        mik_data = mik_library.get_energy(getattr(track, "title", ""))
        if mik_data:
            record["mik"] = mik_data

    record["_text"]       = _build_text_field(record)
    record["_indexed_at"] = indexed_at or datetime.now(timezone.utc).isoformat()
    return record


def _build_text_field(record: dict) -> str:
    """
    Build the grep-optimized ``_text`` summary for a record.

    Prefixed tokens (``energy:8``, ``my_tags:Festival``) allow precise grep
    patterns that don't collide with JSON key names.
    """
    parts: list[str] = []

    for field in ("artist", "title", "genre", "key"):
        v = record.get(field, "")
        if v:
            parts.append(str(v))

    bpm = record.get("bpm", 0)
    if bpm:
        parts.append(f"{bpm:.0f}bpm")

    energy = record.get("energy")
    if energy is not None:
        parts.append(f"energy:{energy}")

    rating = record.get("rating", 0)
    if rating:
        parts.append(f"rating:{rating}")

    pc = record.get("play_count", 0)
    if pc:
        parts.append(f"plays:{pc}")

    my_tags = record.get("my_tags", [])
    if my_tags:
        parts.append("tags:" + " ".join(my_tags))
        for tag in my_tags:
            parts.append(f"my_tags:{tag}")

    if record.get("comments"):
        parts.append(f"comments:{record['comments']}")

    if record.get("color"):
        parts.append(f"color:{record['color']}")

    ess = record.get("essentia", {})
    if ess:
        for fld in ("dominant_mood", "dominant_genre", "dominant_tag"):
            v = ess.get(fld)
            if v:
                prefix = fld.split("_")[1]  # mood / genre / tag
                parts.append(f"{prefix}:{v}")
        if ess.get("danceability") is not None:
            parts.append(f"danceability:{ess['danceability']}")
        if ess.get("lufs") is not None:
            parts.append(f"lufs:{ess['lufs']}")
        if ess.get("key_note"):
            parts.append(ess["key_note"])
        for g in (ess.get("genre") or {}):
            parts.append(g)
        for t in (ess.get("tags") or {}):
            parts.append(f"tag:{t}")

    mik = record.get("mik", {})
    if mik:
        if mik.get("collection"):
            parts.append(f"collection:{mik['collection']}")
        if mik.get("energy"):
            parts.append(f"mik_energy:{mik['energy']}")

    if record.get("date_added"):
        parts.append(f"date:{record['date_added']}")

    src = record.get("energy_source", "")
    if src and src != "none":
        parts.append(f"energy_source:{src}")

    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# LibraryIndex
# ---------------------------------------------------------------------------

class LibraryIndex:
    """
    Builds and queries a JSONL index of all tracks, merging Rekordbox,
    Essentia, and Mixed In Key data into one record per track.

    Also builds ``library_attributes.json`` and ``library_context.md``
    from the actual library data — no hardcoded tag names or genre lists.

    The index file is designed for:
    * **LLM grep** — Claude uses the Grep tool against the ``_text`` field.
    * **Vector store ingest** — each line is a self-contained document.
    * **Programmatic lookup** — ``get_by_id()`` uses an in-memory dict.
    * **LLM context** — ``attributes`` property returns the full stats dict.
    """

    def __init__(self, index_path: Path = INDEX_PATH) -> None:
        self._record_path: Path = index_path
        self._by_id: dict[str, dict] = {}
        self._attributes: Optional[dict] = None
        self._built = False
        self._lock = threading.Lock()  # guards _by_id for concurrent update/flush

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        tracks: list,
        essentia_store: Any = None,
        mik_library: Any = None,
        my_tag_tree: Optional[list] = None,
    ) -> dict[str, Any]:
        """
        Build (or rebuild) the JSONL index and attribute files from scratch.

        Args:
            tracks:         List of TrackWithEnergy instances.
            essentia_store: Optional EssentiaFeatureStore for audio features.
            mik_library:    Optional MixedInKeyLibrary for MIK energy data.
            my_tag_tree:    Optional raw rows from db.query_my_tags() — used to
                            build the My Tag group hierarchy in attributes.

        Returns:
            Stats dict with total, with_essentia, with_mik, index_path, built_at.
        """
        self._record_path.parent.mkdir(parents=True, exist_ok=True)
        self._by_id = {}

        indexed_at   = datetime.now(timezone.utc).isoformat()
        total        = 0
        with_essentia = 0
        with_mik     = 0

        # --- Write JSONL index ---
        with self._record_path.open("w", encoding="utf-8") as fh:
            for track in tracks:
                try:
                    record = _build_record_from_track(
                        track,
                        essentia_store=essentia_store,
                        mik_library=mik_library,
                        indexed_at=indexed_at,
                    )
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    self._by_id[record["id"]] = record
                    total += 1
                    if "essentia" in record:
                        with_essentia += 1
                    if "mik" in record:
                        with_mik += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"LibraryIndex: skipped track "
                        f"{getattr(track, 'id', '?')}: {exc}"
                    )

        self._built = True

        # --- Build and write attribute files ---
        self._attributes = build_attributes(
            tracks,
            essentia_store=essentia_store,
            my_tag_tree=my_tag_tree,
        )
        ATTRIBUTES_PATH.write_text(
            json.dumps(self._attributes, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _write_context_md(self._attributes, CONTEXT_PATH)

        stats = {
            "total":         total,
            "with_essentia": with_essentia,
            "with_mik":      with_mik,
            "index_path":    str(self._record_path),
            "attributes_path": str(ATTRIBUTES_PATH),
            "context_path":  str(CONTEXT_PATH),
            "built_at":      indexed_at,
        }
        logger.info(
            f"Library index built: {total} tracks "
            f"({with_essentia} Essentia, {with_mik} MIK) → {self._record_path}"
        )
        return stats

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str = "",
        my_tag: Optional[str] = None,
        genre: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Search the JSONL index using grep-style matching against ``_text``.

        Reads the file line-by-line (streaming — never loads the whole file).
        """
        if not self._record_path.exists():
            return []

        q_lower   = query.lower()  if query   else ""
        mt_lower  = my_tag.lower() if my_tag  else ""
        gen_lower = genre.lower()  if genre   else ""

        results: list[dict] = []
        with self._record_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if q_lower and q_lower not in record.get("_text", "").lower():
                    continue
                if mt_lower:
                    tags_lower = [t.lower() for t in record.get("my_tags", [])]
                    if not any(mt_lower in t for t in tags_lower):
                        continue
                if gen_lower:
                    if gen_lower not in (record.get("genre") or "").lower():
                        continue

                results.append(record)
                if len(results) >= limit:
                    break

        return results

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_by_id(self, track_id: str) -> Optional[dict]:
        """Return the full record for a track by Rekordbox ID."""
        if self._built:
            return self._by_id.get(str(track_id))

        if not self._record_path.exists():
            return None
        with self._record_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("id") == str(track_id):
                        return record
                except json.JSONDecodeError:
                    continue
        return None

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    @property
    def attributes(self) -> Optional[dict]:
        """Dynamically scanned library attributes (None until build() is called)."""
        return self._attributes

    def load_attributes(self) -> Optional[dict]:
        """
        Load ``library_attributes.json`` from disk into ``self._attributes``.

        Called at startup when ``is_fresh()`` is True to avoid a full rebuild
        while still making attribute data available.
        """
        if not ATTRIBUTES_PATH.exists():
            return None
        try:
            self._attributes = json.loads(ATTRIBUTES_PATH.read_text(encoding="utf-8"))
            logger.debug(f"Library attributes loaded from {ATTRIBUTES_PATH}")
            return self._attributes
        except Exception as exc:
            logger.warning(f"Could not load library attributes: {exc}")
            return None

    # ------------------------------------------------------------------
    # Freshness
    # ------------------------------------------------------------------

    def is_fresh(self, max_age_seconds: int = 3600) -> bool:
        """True if both the JSONL and attributes files exist and are recent."""
        for path in (self._record_path, ATTRIBUTES_PATH):
            if not path.exists():
                return False
            age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
            if age >= max_age_seconds:
                return False
        return True

    # ------------------------------------------------------------------
    # Load without rebuild
    # ------------------------------------------------------------------

    def load_from_disk(self) -> int:
        """
        Load the existing JSONL into the in-memory ``_by_id`` dict and load
        ``library_attributes.json`` — without triggering a full rebuild.

        Returns number of records loaded.
        """
        if not self._record_path.exists():
            return 0

        count = 0
        self._by_id = {}
        with self._record_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._by_id[record["id"]] = record
                    count += 1
                except json.JSONDecodeError:
                    continue

        self._built = True
        self.load_attributes()
        logger.debug(f"LibraryIndex: loaded {count} records from {self._record_path}")
        return count

    # ------------------------------------------------------------------
    # Incremental update (called per-track during analysis)
    # ------------------------------------------------------------------

    def update_record(
        self,
        track: Any,
        essentia_store: Any = None,
        mik_library: Any = None,
        indexed_at: str = "",
    ) -> dict:
        """
        Build and store the merged record for a single track in memory.

        Called incrementally from the main process after each Essentia
        analysis worker result arrives.  Single-writer — no locking needed.
        Call ``flush_to_disk()`` periodically to persist the running state.

        Args:
            track:          TrackWithEnergy instance (energy already resolved).
            essentia_store: Store with a ``.get(file_path)`` method returning
                            ``EssentiaFeatures`` for this track, or None.
            mik_library:    MixedInKeyLibrary for MIK energy data, or None.
            indexed_at:     ISO timestamp string; defaults to now.

        Returns:
            The built record dict (also stored in ``_by_id``).
        """
        record = _build_record_from_track(
            track,
            essentia_store=essentia_store,
            mik_library=mik_library,
            indexed_at=indexed_at or datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._by_id[record["id"]] = record
            self._built = True
        return record

    def flush_to_disk(self) -> int:
        """
        Atomically write all in-memory records to the JSONL file.

        Writes to a ``.tmp`` sibling first, then uses ``Path.replace()``
        for an atomic rename — safe against mid-write crashes and Ctrl+C.
        Single-writer; no locking needed when called from the main process.

        Returns:
            Number of records written (0 if nothing in memory).
        """
        with self._lock:
            if not self._by_id:
                return 0
            records_snapshot = list(self._by_id.values())
        self._record_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._record_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for record in records_snapshot:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp.replace(self._record_path)
        return len(records_snapshot)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = f"{len(self._by_id)} records" if self._built else "not loaded"
        return f"LibraryIndex({status}, path={self._record_path})"
