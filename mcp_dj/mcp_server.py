"""
FastMCP Server for Claude Desktop Integration

Provides tools for generating setlists, recommending tracks, and analyzing
harmonic compatibility — callable directly from Claude Desktop.

To connect to Claude Desktop, add to claude_desktop_config.json:
{
  "mcpServers": {
    "dj-setlist-creator": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/set_list_creator", "python", "-m", "mcp_dj.mcp_server"],
      "env": {}
    }
  }
}
"""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastmcp import FastMCP
from loguru import logger

from .database import RekordboxDatabase
from .energy import MixedInKeyLibrary, EnergyResolver
from .camelot import CamelotWheel
from .energy_planner import EnergyPlanner, ENERGY_PROFILES
from .setlist_engine import SetlistEngine
from .models import SetlistRequest

# Optional Essentia integration — gracefully unavailable if not installed
try:
    from .essentia_analyzer import (
        analyze_file as _essentia_analyze_file,
        analyze_library as _essentia_analyze_library,
        EssentiaFeatureStore,
        ESSENTIA_AVAILABLE,
        CACHE_DIR as ESSENTIA_CACHE_DIR,
    )
except ImportError:
    ESSENTIA_AVAILABLE = False
    _essentia_analyze_file = None
    _essentia_analyze_library = None
    EssentiaFeatureStore = None
    ESSENTIA_CACHE_DIR = None

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

mcp = FastMCP("DJ Setlist Creator")

db: Optional[RekordboxDatabase] = None
engine: Optional[SetlistEngine] = None
_initialized = False


async def _ensure_initialized():
    """Lazy-initialize the database and engine on first tool call."""
    global db, engine, _initialized
    if _initialized:
        return

    logger.info("Initializing Setlist Creator MCP server...")
    db = RekordboxDatabase()
    await db.connect()

    # Load Mixed In Key energy data (optional — requires MIK_CSV_PATH env var)
    mik = MixedInKeyLibrary.from_env()
    if mik is not None:
        mik.load()
    resolver = EnergyResolver(mik)

    all_tracks = await db.get_all_tracks()
    resolver.resolve_all(all_tracks)

    # Load Essentia cache into memory for use during scoring (no analysis triggered)
    essentia_store = None
    if EssentiaFeatureStore is not None:
        essentia_store = EssentiaFeatureStore(all_tracks)
        logger.info(f"Essentia cache: {len(essentia_store)} tracks loaded")

    engine = SetlistEngine(
        tracks=all_tracks,
        camelot=CamelotWheel(),
        energy_planner=EnergyPlanner(),
        essentia_store=essentia_store,
    )
    _initialized = True
    logger.info(f"MCP server ready with {len(all_tracks)} tracks")


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def generate_setlist(
    duration_minutes: int = 60,
    genre: Optional[str] = None,
    bpm_min: Optional[float] = None,
    bpm_max: Optional[float] = None,
    energy_profile: str = "journey",
    starting_track_title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a DJ setlist with harmonic mixing and energy arc planning.

    Args:
        duration_minutes: Target set duration in minutes (default 60)
        genre: Genre filter (e.g. 'tech house', 'techno'). Optional.
        bpm_min: Minimum BPM filter. Optional.
        bpm_max: Maximum BPM filter. Optional.
        energy_profile: Energy arc profile - one of: journey, build, peak, chill, wave
            - journey: Classic warm-up, build, peak, cool-down arc
            - build:   Continuous energy build from low to high
            - peak:    High energy throughout (peak-time slot)
            - chill:   Low-energy ambient/warm-up set
            - wave:    Multiple energy waves with peaks and valleys
        starting_track_title: Title of the track to start with. Optional.

    Returns:
        Complete setlist with track list, energy arc, harmonic score, and stats.
    """
    await _ensure_initialized()

    # Find starting track by title if provided
    starting_id = None
    if starting_track_title:
        for t in engine.tracks:
            if starting_track_title.lower() in t.title.lower():
                starting_id = t.id
                break

    request = SetlistRequest(
        prompt=starting_track_title or f"{duration_minutes}min {energy_profile} set",
        duration_minutes=max(10, min(480, duration_minutes)),
        genre=genre,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        energy_profile=energy_profile if energy_profile in ENERGY_PROFILES else "journey",
        starting_track_id=starting_id,
    )

    setlist = engine.generate_setlist(request)

    # Build per-track essentia enrichment if store is available
    essentia_store = engine.essentia_store

    def _essentia_snippet(file_path):
        if not essentia_store:
            return {}
        ess = essentia_store.get(file_path)
        if not ess:
            return {}
        return {
            "essentia_energy": ess.energy_as_1_to_10(),
            "danceability": ess.danceability_as_1_to_10(),
            "dominant_mood": ess.dominant_mood(),
            "top_genre_discogs": ess.top_genre(),
            "lufs": round(ess.integrated_lufs, 1),
        }

    return {
        "setlist_id": setlist.id,
        "name": setlist.name,
        "track_count": setlist.track_count,
        "duration_minutes": round(setlist.total_duration_seconds / 60),
        "avg_bpm": setlist.avg_bpm,
        "bpm_range": {"min": setlist.bpm_range[0], "max": setlist.bpm_range[1]},
        "harmonic_score": setlist.harmonic_score,
        "energy_arc": setlist.energy_arc,
        "genre_distribution": setlist.genre_distribution,
        "essentia_cache_coverage": (
            f"{len(essentia_store)} tracks with audio analysis" if essentia_store else "no essentia cache"
        ),
        "tracks": [
            {
                "position": st.position,
                "artist": st.track.artist,
                "title": st.track.title,
                "bpm": st.track.bpm,
                "key": st.track.key,
                "energy": st.track.energy,
                "genre": st.track.genre,
                "duration": st.track.duration_formatted(),
                "file_path": st.track.file_path,
                "key_relation": st.key_relation,
                "transition_score": st.transition_score,
                "notes": st.notes,
                **_essentia_snippet(st.track.file_path),
            }
            for st in setlist.tracks
        ],
    }


@mcp.tool()
async def recommend_next_track(
    current_track_title: str,
    energy_direction: str = "maintain",
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Recommend what tracks to play after a specific track.

    The AI considers harmonic compatibility (Camelot wheel), energy flow,
    BPM proximity, and genre coherence.

    Args:
        current_track_title: Title (or "Artist - Title") of the currently playing track
        energy_direction: "up" to raise energy, "down" to lower, "maintain" to keep level
        limit: Number of recommendations to return (default 5, max 20)

    Returns:
        List of recommended tracks with scores and reasoning.
    """
    await _ensure_initialized()

    recs = engine.recommend_next(
        current_track_title=current_track_title,
        energy_direction=energy_direction,
        limit=min(limit, 20),
    )

    if not recs:
        return [{"error": f"Track '{current_track_title}' not found in library"}]

    essentia_store = engine.essentia_store

    def _ess_fields(file_path):
        if not essentia_store:
            return {}
        ess = essentia_store.get(file_path)
        if not ess:
            return {}
        return {
            "essentia_energy": ess.energy_as_1_to_10(),
            "danceability": ess.danceability_as_1_to_10(),
            "dominant_mood": ess.dominant_mood(),
            "top_genre_discogs": ess.top_genre(),
            "lufs": round(ess.integrated_lufs, 1),
        }

    return [
        {
            "rank": i + 1,
            "artist": r.track.artist,
            "title": r.track.title,
            "bpm": r.track.bpm,
            "key": r.track.key,
            "energy": r.track.energy,
            "genre": r.track.genre,
            "duration": r.track.duration_formatted(),
            "file_path": r.track.file_path,
            "overall_score": r.score,
            "harmonic_score": r.harmonic_score,
            "energy_score": r.energy_score,
            "bpm_score": r.bpm_score,
            "genre_score": r.genre_score,
            "reason": r.reason,
            **_ess_fields(r.track.file_path),
        }
        for i, r in enumerate(recs)
    ]


@mcp.tool()
async def get_track_compatibility(
    track_a_title: str,
    track_b_title: str,
) -> Dict[str, Any]:
    """
    Check harmonic and energy compatibility between two specific tracks.

    Args:
        track_a_title: Title of the first track (currently playing)
        track_b_title: Title of the second track (potential next track)

    Returns:
        Detailed compatibility analysis.
    """
    await _ensure_initialized()

    camelot = engine.camelot

    # Find both tracks
    track_a = next(
        (t for t in engine.tracks if track_a_title.lower() in t.title.lower()), None
    )
    track_b = next(
        (t for t in engine.tracks if track_b_title.lower() in t.title.lower()), None
    )

    if not track_a:
        return {"error": f"Track '{track_a_title}' not found"}
    if not track_b:
        return {"error": f"Track '{track_b_title}' not found"}

    h_score, rel = camelot.transition_score(track_a.key or "", track_b.key or "")
    bpm_pct_diff = (
        abs(track_b.bpm - track_a.bpm) / track_a.bpm * 100
        if track_a.bpm > 0 else 0
    )

    energy_delta = (track_b.energy or 5) - (track_a.energy or 5)

    verdict = (
        "✅ Excellent mix" if h_score >= 0.85 and bpm_pct_diff <= 4
        else "👍 Good mix" if h_score >= 0.65
        else "⚠️ Acceptable mix" if h_score >= 0.4
        else "❌ Difficult transition"
    )

    return {
        "track_a": {"artist": track_a.artist, "title": track_a.title, "bpm": track_a.bpm, "key": track_a.key, "energy": track_a.energy},
        "track_b": {"artist": track_b.artist, "title": track_b.title, "bpm": track_b.bpm, "key": track_b.key, "energy": track_b.energy},
        "harmonic_score": h_score,
        "key_relationship": rel,
        "bpm_difference": round(track_b.bpm - track_a.bpm, 1),
        "bpm_pct_difference": round(bpm_pct_diff, 1),
        "energy_delta": energy_delta,
        "verdict": verdict,
        "tips": _get_mix_tips(rel, bpm_pct_diff, energy_delta),
    }


@mcp.tool()
async def analyze_energy_flow(
    track_titles: List[str],
) -> Dict[str, Any]:
    """
    Analyze the energy flow of a sequence of tracks.

    Args:
        track_titles: List of track titles in set order

    Returns:
        Energy arc analysis with recommendations for improvement.
    """
    await _ensure_initialized()

    planner = engine.planner
    found_tracks = []

    for title in track_titles:
        track = next(
            (t for t in engine.tracks if title.lower() in t.title.lower()), None
        )
        found_tracks.append(track)

    energies = [(t.energy or 5) if t else 5 for t in found_tracks]
    n = len(energies)

    # Compare to journey profile
    target_journey = [round(planner.get_target_energy(i / max(1, n - 1), "journey")) for i in range(n)]

    issues = []
    for i in range(1, n):
        delta = abs(energies[i] - energies[i - 1])
        if delta > 3:
            issues.append(f"Position {i+1}: Large energy jump ({delta} levels)")
    for i in range(2, n):
        if energies[i] == energies[i-1] == energies[i-2]:
            issues.append(f"Position {i+1}: Plateau detected (3+ tracks at E{energies[i]})")

    return {
        "track_count": n,
        "energy_values": energies,
        "target_journey": target_journey,
        "avg_energy": round(sum(energies) / n, 1) if energies else 0,
        "energy_range": {"min": min(energies), "max": max(energies)} if energies else {},
        "issues": issues,
        "score": round(1.0 - len(issues) / max(n, 1), 2),
        "tracks_found": [
            {"title": t.title, "energy": t.energy, "key": t.key, "bpm": t.bpm, "file_path": t.file_path}
            if t else None
            for t in found_tracks
        ],
    }


@mcp.tool()
async def get_compatible_tracks(
    key: str,
    bpm: Optional[float] = None,
    bpm_tolerance: float = 4.0,
    energy_min: Optional[int] = None,
    energy_max: Optional[int] = None,
    genre: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Find tracks in the library compatible with a given key and BPM.

    Args:
        key: Camelot key (e.g. '8A', '12B')
        bpm: Target BPM. Optional.
        bpm_tolerance: BPM tolerance percentage (default 4%)
        energy_min: Minimum energy level 1-10. Optional.
        energy_max: Maximum energy level 1-10. Optional.
        genre: Genre filter. Optional.
        limit: Max results (default 20)

    Returns:
        List of compatible tracks sorted by harmonic score.
    """
    await _ensure_initialized()

    camelot = engine.camelot
    compatible_keys = camelot.get_compatible_keys(key)

    if not compatible_keys:
        return [{"error": f"Invalid Camelot key: {key}"}]

    results = []
    for track in engine.tracks:
        # Key filter
        if not track.key:
            continue
        tk = track.key.strip().upper()
        if tk not in compatible_keys:
            continue
        h_score = compatible_keys[tk]

        # BPM filter
        if bpm is not None and track.bpm > 0:
            pct_diff = abs(track.bpm - bpm) / bpm * 100
            if pct_diff > bpm_tolerance:
                continue

        # Energy filter
        if energy_min is not None and (track.energy or 0) < energy_min:
            continue
        if energy_max is not None and (track.energy or 10) > energy_max:
            continue

        # Genre filter
        if genre and track.genre and genre.lower() not in track.genre.lower():
            continue

        _, rel = camelot.transition_score(key, track.key)

        entry = {
            "artist": track.artist,
            "title": track.title,
            "bpm": track.bpm,
            "key": track.key,
            "energy": track.energy,
            "genre": track.genre,
            "rating": track.rating,
            "file_path": track.file_path,
            "harmonic_score": h_score,
            "key_relationship": rel,
        }

        # Enrich with essentia cache data when available
        if engine.essentia_store:
            ess = engine.essentia_store.get(track.file_path)
            if ess:
                entry["essentia_energy"] = ess.energy_as_1_to_10()
                entry["danceability"] = ess.danceability_as_1_to_10()
                entry["dominant_mood"] = ess.dominant_mood()
                entry["top_genre_discogs"] = ess.top_genre()
                entry["lufs"] = round(ess.integrated_lufs, 1)

        results.append(entry)

    results.sort(key=lambda x: (-x["harmonic_score"], -x["rating"]))
    return results[:limit]


@mcp.tool()
async def export_setlist_to_rekordbox(
    setlist_name: str,
    track_titles: Optional[List[str]] = None,
    setlist_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a Rekordbox playlist from a setlist.

    Args:
        setlist_name: Name for the new Rekordbox playlist
        track_titles: List of track titles in order. Use if setlist_id is not available.
        setlist_id: ID of a previously generated setlist. If provided, track_titles ignored.

    Returns:
        Result with playlist_id on success.
    """
    await _ensure_initialized()

    track_ids = []

    if setlist_id:
        setlist = engine.get_setlist(setlist_id)
        if setlist:
            track_ids = [st.track.id for st in setlist.tracks]
        else:
            return {"success": False, "error": f"Setlist {setlist_id} not found"}
    elif track_titles:
        for title in track_titles:
            track = next(
                (t for t in engine.tracks if title.lower() in t.title.lower()), None
            )
            if track:
                track_ids.append(track.id)
    else:
        return {"success": False, "error": "Provide either setlist_id or track_titles"}

    if not track_ids:
        return {"success": False, "error": "No tracks found to export"}

    try:
        playlist_id = await db.create_playlist_with_tracks(
            name=setlist_name, track_ids=track_ids
        )
        return {
            "success": True,
            "playlist_id": playlist_id,
            "playlist_name": setlist_name,
            "track_count": len(track_ids),
            "message": f"Created playlist '{setlist_name}' with {len(track_ids)} tracks in Rekordbox",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_library_summary() -> Dict[str, Any]:
    """
    Get a summary of the loaded Rekordbox library.

    Returns:
        Library statistics including track count, BPM range, top genres, and key distribution.
    """
    await _ensure_initialized()
    summary = engine.get_library_summary()
    summary["energy_profiles"] = {
        k: v["description"] for k, v in ENERGY_PROFILES.items()
    }
    return summary


@mcp.tool()
async def search_library(
    query: str = "",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    my_tag: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Search the Rekordbox library for tracks.

    Args:
        query: Search string (matches title, artist, album, genre). Optional.
        date_from: Filter tracks added on or after this date (YYYY-MM-DD). Optional.
        date_to: Filter tracks added on or before this date (YYYY-MM-DD). Optional.
        my_tag: Filter by Rekordbox My Tag label (e.g. 'High Energy'). Optional.
        limit: Maximum results (default 20)

    Returns:
        List of matching tracks with metadata.
    """
    await _ensure_initialized()

    q = query.strip().lower()
    tag_filter = (my_tag or "").strip().lower()
    results = []

    for t in engine.tracks:
        # Text filter
        if q and not (
            q in t.title.lower()
            or q in t.artist.lower()
            or q in (t.genre or "").lower()
            or q in (t.album or "").lower()
        ):
            continue

        # Date filter (lexicographic comparison works for YYYY-MM-DD)
        d = t.date_added or ""
        if date_from and d < date_from:
            continue
        if date_to and d > date_to:
            continue

        # My Tag filter
        if tag_filter and not any(tag_filter in tag.lower() for tag in t.my_tags):
            continue

        results.append({
            "id": t.id,
            "artist": t.artist,
            "title": t.title,
            "bpm": t.bpm,
            "key": t.key,
            "energy": t.energy,
            "energy_source": t.energy_source,
            "genre": t.genre,
            "rating": t.rating,
            "play_count": t.play_count,
            "duration": t.duration_formatted(),
            "date_added": t.date_added,
            "my_tags": t.my_tags,
        })
        if len(results) >= limit:
            break

    return results


# ---------------------------------------------------------------------------
# Raw Database Table Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_db_artists(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdArtist table — all artists in the Rekordbox library.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of artist records with ID, Name, SearchStr.
    """
    await _ensure_initialized()
    return await db.query_artists(limit=limit, offset=offset)


@mcp.tool()
async def get_db_albums(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdAlbum table — all albums in the Rekordbox library.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of album records with ID, Name, AlbumArtistID, ImagePath, Compilation.
    """
    await _ensure_initialized()
    return await db.query_albums(limit=limit, offset=offset)


@mcp.tool()
async def get_db_genres(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdGenre table — all genres in the Rekordbox library.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of genre records with ID and Name.
    """
    await _ensure_initialized()
    return await db.query_genres(limit=limit, offset=offset)


@mcp.tool()
async def get_db_labels(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdLabel table — all record labels in the Rekordbox library.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of label records with ID and Name.
    """
    await _ensure_initialized()
    return await db.query_labels(limit=limit, offset=offset)


@mcp.tool()
async def get_db_keys(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdKey table — all musical keys stored in Rekordbox.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of key records with ID, ScaleName, and Seq (sort order).
    """
    await _ensure_initialized()
    return await db.query_keys(limit=limit, offset=offset)


@mcp.tool()
async def get_db_colors(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdColor table — color labels used to tag tracks in Rekordbox.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of color records with ID, ColorCode, SortKey, and Commnt (name).
    """
    await _ensure_initialized()
    return await db.query_colors(limit=limit, offset=offset)


@mcp.tool()
async def get_db_playlists(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdPlaylist table — all playlists and playlist folders in Rekordbox.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of playlist records with ID, Name, Seq, Attribute (0=playlist, 1=folder, 4=smart), ParentID, ImagePath.
    """
    await _ensure_initialized()
    return await db.query_playlists(limit=limit, offset=offset)


@mcp.tool()
async def get_db_playlist_songs(
    playlist_id: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Query the djmdSongPlaylist table — track-to-playlist membership records.

    Args:
        playlist_id: Filter by a specific playlist ID. Optional.
        limit: Maximum rows to return (default 500)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of records with ID, PlaylistID, ContentID, TrackNo (position in playlist).
    """
    await _ensure_initialized()
    return await db.query_playlist_songs(playlist_id=playlist_id, limit=limit, offset=offset)


@mcp.tool()
async def get_db_history(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdHistory table — DJ play history sessions in Rekordbox.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of history session records with ID, Name, Seq, Attribute, ParentID, DateCreated.
    """
    await _ensure_initialized()
    return await db.query_history(limit=limit, offset=offset)


@mcp.tool()
async def get_db_history_songs(
    history_id: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Query the djmdSongHistory table — tracks played in each history session.

    Args:
        history_id: Filter by a specific history session ID. Optional.
        limit: Maximum rows to return (default 500)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of records with ID, HistoryID, ContentID, TrackNo (play order).
    """
    await _ensure_initialized()
    return await db.query_history_songs(history_id=history_id, limit=limit, offset=offset)


@mcp.tool()
async def get_db_my_tags(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdMyTag table — all My Tag labels defined in Rekordbox.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of My Tag records with ID, Name, Seq, Attribute, ParentID.
    """
    await _ensure_initialized()
    return await db.query_my_tags(limit=limit, offset=offset)


@mcp.tool()
async def get_db_my_tag_songs(
    my_tag_id: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Query the djmdSongMyTag table — track-to-My Tag assignments.

    Args:
        my_tag_id: Filter by a specific My Tag ID. Optional.
        limit: Maximum rows to return (default 500)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of records with ID, MyTagID, ContentID, TrackNo.
    """
    await _ensure_initialized()
    return await db.query_my_tag_songs(my_tag_id=my_tag_id, limit=limit, offset=offset)


@mcp.tool()
async def get_db_cues(
    content_id: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Query the djmdCue table — hot cues, memory cues, and loops stored in Rekordbox.

    Args:
        content_id: Filter by a specific track (ContentID). Optional.
        limit: Maximum rows to return (default 500)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of cue records with ID, ContentID, InMsec, OutMsec, Kind, Color, Comment, BeatLoopSize, ActiveLoop.
    """
    await _ensure_initialized()
    return await db.query_cues(content_id=content_id, limit=limit, offset=offset)


@mcp.tool()
async def get_db_hot_cue_banklists(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdHotCueBanklist table — hot cue bank collections in Rekordbox.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of hot cue banklist records with ID, Name, Seq, Attribute, ParentID, ImagePath.
    """
    await _ensure_initialized()
    return await db.query_hot_cue_banklists(limit=limit, offset=offset)


@mcp.tool()
async def get_db_song_hot_cue_banklists(
    hot_cue_banklist_id: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Query the djmdSongHotCueBanklist table — hot cue bank entries per track.

    Args:
        hot_cue_banklist_id: Filter by a specific hot cue banklist ID. Optional.
        limit: Maximum rows to return (default 500)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of records with ID, HotCueBanklistID, ContentID, TrackNo, Color, Comment.
    """
    await _ensure_initialized()
    return await db.query_song_hot_cue_banklists(hot_cue_banklist_id=hot_cue_banklist_id, limit=limit, offset=offset)


@mcp.tool()
async def get_db_samplers(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdSampler table — sampler banks defined in Rekordbox.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of sampler records with ID, Name, Seq, Attribute, ParentID.
    """
    await _ensure_initialized()
    return await db.query_samplers(limit=limit, offset=offset)


@mcp.tool()
async def get_db_song_samplers(
    sampler_id: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Query the djmdSongSampler table — track assignments to sampler banks.

    Args:
        sampler_id: Filter by a specific sampler ID. Optional.
        limit: Maximum rows to return (default 500)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of records with ID, SamplerID, ContentID, TrackNo.
    """
    await _ensure_initialized()
    return await db.query_song_samplers(sampler_id=sampler_id, limit=limit, offset=offset)


@mcp.tool()
async def get_db_related_tracks(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdRelatedTracks table — related tracks collections in Rekordbox.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of related tracks records with ID, Name, Seq, Attribute, ParentID, Criteria.
    """
    await _ensure_initialized()
    return await db.query_related_tracks(limit=limit, offset=offset)


@mcp.tool()
async def get_db_song_related_tracks(
    related_tracks_id: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Query the djmdSongRelatedTracks table — track assignments to related tracks collections.

    Args:
        related_tracks_id: Filter by a specific related tracks collection ID. Optional.
        limit: Maximum rows to return (default 500)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of records with ID, RelatedTracksID, ContentID, TrackNo.
    """
    await _ensure_initialized()
    return await db.query_song_related_tracks(related_tracks_id=related_tracks_id, limit=limit, offset=offset)


@mcp.tool()
async def get_db_active_censors(
    content_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Query the djmdActiveCensor table — active censor regions (muted sections) on tracks.

    Args:
        content_id: Filter by a specific track (ContentID). Optional.
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of censor records with ID, ContentID, InMsec, OutMsec, Info, ParameterList.
    """
    await _ensure_initialized()
    return await db.query_active_censors(content_id=content_id, limit=limit, offset=offset)


@mcp.tool()
async def get_db_mixer_params(
    content_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Query the djmdMixerParam table — per-track mixer gain and peak values stored by Rekordbox.

    Args:
        content_id: Filter by a specific track (ContentID). Optional.
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of mixer param records with ID, ContentID, GainHigh, GainLow, PeakHigh, PeakLow.
    """
    await _ensure_initialized()
    return await db.query_mixer_params(content_id=content_id, limit=limit, offset=offset)


@mcp.tool()
async def get_db_content_files(
    content_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Query the contentFile table — file location and hash records for tracks.

    Args:
        content_id: Filter by a specific track (ContentID). Optional.
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of records with ID, ContentID, Path, Hash, Size.
    """
    await _ensure_initialized()
    return await db.query_content_files(content_id=content_id, limit=limit, offset=offset)


@mcp.tool()
async def get_db_image_files(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the imageFile table — artwork and image file records in Rekordbox.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of records with ID, TableName, TargetUUID, TargetID, Path, Size.
    """
    await _ensure_initialized()
    return await db.query_image_files(limit=limit, offset=offset)


@mcp.tool()
async def get_db_setting_files(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the settingFile table — Rekordbox settings/export files.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of records with ID, Path, Hash, Size.
    """
    await _ensure_initialized()
    return await db.query_setting_files(limit=limit, offset=offset)


@mcp.tool()
async def get_db_devices(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdDevice table — DJ devices registered in Rekordbox.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of device records with ID, MasterDBID, Name.
    """
    await _ensure_initialized()
    return await db.query_devices(limit=limit, offset=offset)


@mcp.tool()
async def get_db_menu_items(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdMenuItems table — Rekordbox browser menu item definitions.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of menu item records with ID, Class, Name.
    """
    await _ensure_initialized()
    return await db.query_menu_items(limit=limit, offset=offset)


@mcp.tool()
async def get_db_categories(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdCategory table — browser category configuration in Rekordbox.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of category records with ID, MenuItemID, Seq, Disable, InfoOrder.
    """
    await _ensure_initialized()
    return await db.query_categories(limit=limit, offset=offset)


@mcp.tool()
async def get_db_sort(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdSort table — sort order settings for browser columns in Rekordbox.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of sort records with ID, MenuItemID, Seq, Disable.
    """
    await _ensure_initialized()
    return await db.query_sort(limit=limit, offset=offset)


@mcp.tool()
async def get_db_song_tag_list(limit: int = 500, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the djmdSongTagList table — the tag list (Rekordbox tag list collection) track entries.

    Args:
        limit: Maximum rows to return (default 500)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of records with ID, ContentID, TrackNo.
    """
    await _ensure_initialized()
    return await db.query_song_tag_list(limit=limit, offset=offset)


@mcp.tool()
async def get_db_property() -> Dict[str, Any]:
    """
    Query the djmdProperty table — Rekordbox database metadata and version info.

    Returns:
        Record with DBID, DBVersion, BaseDBDrive, CurrentDBDrive, DeviceID.
    """
    await _ensure_initialized()
    return await db.query_db_property()


@mcp.tool()
async def get_db_agent_registry(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Query the agentRegistry table — Rekordbox internal agent/sync registry entries.

    Args:
        limit: Maximum rows to return (default 100)
        offset: Number of rows to skip (for pagination)

    Returns:
        List of registry records with registry_id, id_1, id_2, str_1, str_2.
    """
    await _ensure_initialized()
    return await db.query_agent_registry(limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# Essentia audio analysis tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def analyze_track(
    file_path: str,
    force: bool = False,
) -> Dict[str, Any]:
    """Analyze a single audio file with Essentia to extract BPM, key, danceability,
    loudness, mood scores, genre classification, and music tags. Results are cached
    at .data/essentia_cache/ so the same file is never analyzed twice.

    ML features (mood, genre, tags) require model files — download once with:
      ./download_models.sh

    Args:
        file_path: Absolute path to an audio file (.mp3, .wav, .flac, .aiff, .m4a)
        force: If True, re-analyze even if a cached result already exists

    Returns:
        Analysis results including BPM, Camelot key, danceability, EBU R128
        loudness (LUFS + RMS dBFS), mood probabilities, Discogs400 genre scores,
        and MagnaTagATune music tags.
    """
    if not ESSENTIA_AVAILABLE:
        return {
            "error": "Essentia is not installed",
            "install_instructions": {
                "install": "pip install essentia",
            },
        }

    try:
        features = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _essentia_analyze_file(file_path, force=force)
        )
    except FileNotFoundError:
        return {"error": f"Audio file not found: {file_path}"}
    except Exception as e:
        logger.error(f"Essentia analysis failed for {file_path}: {e}")
        return {"error": f"Analysis failed: {str(e)}"}

    return {
        "file_path": features.file_path,
        "analyzed_at": features.analyzed_at,
        "essentia_version": features.essentia_version,
        "analysis_seconds": features.analysis_duration_seconds,
        "bpm": {
            "value": round(features.bpm_essentia, 2),
            "confidence": round(features.bpm_confidence, 3),
            "beats_count": features.beats_count,
        },
        "key": {
            "camelot": features.key_essentia,
            "raw": f"{features.key_name_raw} {features.key_scale}" if features.key_name_raw else None,
            "strength": round(features.key_strength, 3),
        },
        "danceability": {
            "score": round(features.danceability, 3),
            "scale_1_to_10": features.danceability_as_1_to_10(),
        },
        "loudness": {
            "integrated_lufs": round(features.integrated_lufs, 2),
            "loudness_range_db": round(features.loudness_range_db, 2),
            "rms_db": round(features.rms_db, 2),
            "rms_linear": round(features.rms_energy, 5),
            "scale_1_to_10": features.energy_as_1_to_10(),
        },
        "mood": {
            "happy":      round(features.mood_happy, 3)      if features.mood_happy      is not None else None,
            "sad":        round(features.mood_sad, 3)        if features.mood_sad        is not None else None,
            "aggressive": round(features.mood_aggressive, 3) if features.mood_aggressive is not None else None,
            "relaxed":    round(features.mood_relaxed, 3)    if features.mood_relaxed    is not None else None,
            "party":      round(features.mood_party, 3)      if features.mood_party      is not None else None,
            "dominant":   features.dominant_mood(),
        },
        "genre_discogs": features.genre_discogs,
        "music_tags": features.music_tags,
        "cache_directory": str(ESSENTIA_CACHE_DIR),
    }


@mcp.tool()
async def analyze_library_essentia(
    force: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Batch-analyze all tracks in the loaded Rekordbox library that have an audio
    file path using Essentia. Results are cached to disk; already-analyzed tracks
    are skipped unless force=True.

    This operation is CPU-intensive and may take several minutes for large libraries.
    Run it once upfront; subsequent setlist generation can use the cached features.

    Args:
        force: Re-analyze tracks even if cached results exist (default: False)
        limit: Maximum number of tracks to analyze in this run. Useful for
               incremental analysis. None means process all tracks with a file path.

    Returns:
        Summary with counts of analyzed, cached, skipped, and errored tracks,
        plus the cache directory location.
    """
    await _ensure_initialized()

    if not ESSENTIA_AVAILABLE:
        return {
            "error": "Essentia is not installed",
            "install_instructions": {
                "install": "pip install essentia",
            },
        }

    tracks_to_process = engine.tracks
    if limit is not None:
        tracks_to_process = tracks_to_process[:limit]

    total_in_library = len(engine.tracks)
    logger.info(f"Starting Essentia library analysis: {len(tracks_to_process)} tracks")

    stats = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _essentia_analyze_library(
            tracks=tracks_to_process,
            force=force,
            skip_missing=True,
        ),
    )

    return {
        "total_tracks_in_library": total_in_library,
        "tracks_processed": len(tracks_to_process),
        "analyzed_new": stats["analyzed"],
        "loaded_from_cache": stats["cached"],
        "skipped_no_file_path": stats["skipped_no_path"],
        "skipped_missing_file": stats["skipped_missing_file"],
        "errors": stats["errors"],
        "cache_directory": str(ESSENTIA_CACHE_DIR),
        "message": (
            f"Analysis complete. {stats['analyzed']} new tracks analyzed, "
            f"{stats['cached']} loaded from cache, "
            f"{stats['errors']} errors."
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_mix_tips(rel: str, bpm_pct_diff: float, energy_delta: int) -> List[str]:
    tips = []

    if rel == "incompatible":
        tips.append("Keys are incompatible — consider key-shifting with DJ software")
    elif rel in ("adjacent_up", "adjacent_down"):
        tips.append("Smooth harmonic transition — can mix long overlaps")
    elif rel == "energy_boost":
        tips.append("Energy boost transition — powerful but use sparingly")
    elif rel == "inner_outer":
        tips.append("Major/minor switch — works well for emotional shifts")

    if bpm_pct_diff > 6:
        tips.append(f"Large BPM gap ({bpm_pct_diff:.0f}%) — consider a quick cut or beatgrid anchor")
    elif bpm_pct_diff > 3:
        tips.append(f"Moderate BPM difference ({bpm_pct_diff:.0f}%) — keep overlap short")

    if abs(energy_delta) > 3:
        tips.append(f"Large energy jump ({energy_delta:+d}) — might be jarring; build more gradually")
    elif energy_delta > 1:
        tips.append("Energy is rising — good for building crowd momentum")
    elif energy_delta < -1:
        tips.append("Energy is dropping — use for planned cooldown moments")

    return tips if tips else ["Transition looks clean!"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the MCP server."""
    import argparse

    def handle_shutdown(sig, frame):
        logger.info("Shutting down MCP server...")
        if db:
            asyncio.get_event_loop().run_until_complete(db.disconnect())
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-certfile", default=None)
    parser.add_argument("--ssl-keyfile", default=None)
    args = parser.parse_args()

    logger.info("Starting DJ Setlist Creator MCP Server...")

    if args.transport == "sse":
        uvicorn_config = {}
        if args.ssl_certfile and args.ssl_keyfile:
            uvicorn_config["ssl_certfile"] = args.ssl_certfile
            uvicorn_config["ssl_keyfile"] = args.ssl_keyfile
        mcp.run(
            transport="sse",
            host=args.host,
            port=args.port,
            uvicorn_config=uvicorn_config if uvicorn_config else None,
        )
    else:
        mcp.run()


if __name__ == "__main__":
    main()
