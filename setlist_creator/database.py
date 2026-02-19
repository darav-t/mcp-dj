"""
Rekordbox Database Layer for Setlist Creator

Based on the VST Rekordbox MCP database layer. Provides read-only access
to the rekordbox library for setlist generation.
"""

import os
from pathlib import Path
from typing import Optional, List

from pyrekordbox import Rekordbox6Database
from loguru import logger

from .models import (
    Track,
    TrackWithEnergy,
    COLOR_ID_TO_NAME,
    RATING_TO_STARS,
)


class RekordboxDatabase:
    """
    Read-only interface for the rekordbox database.
    Uses pyrekordbox for SQLCipher decryption.
    """

    def __init__(self) -> None:
        self.db: Optional[Rekordbox6Database] = None
        self.database_path: Optional[Path] = None
        self._connected = False
        self._all_tracks_cache: Optional[List[TrackWithEnergy]] = None

    async def connect(self) -> None:
        """Connect to the rekordbox database (auto-detects location)."""
        try:
            self.database_path = self._detect_database_path()
            logger.info(f"Connecting to rekordbox database at: {self.database_path}")
            self.db = Rekordbox6Database()

            content_list = list(self.db.get_content())
            active = [c for c in content_list if getattr(c, "rb_local_deleted", 0) == 0]
            logger.info(f"Connected. Found {len(active)} active tracks.")
            self._connected = True

        except Exception as e:
            logger.error(f"Failed to connect to rekordbox database: {e}")
            raise RuntimeError(f"Database connection failed: {e}") from e

    def _detect_database_path(self) -> Path:
        if os.name == "nt":
            base = Path.home() / "AppData" / "Roaming" / "Pioneer"
        else:
            base = Path.home() / "Library" / "Pioneer"
        if not base.exists():
            raise FileNotFoundError(f"Rekordbox directory not found at {base}")
        return base

    async def is_connected(self) -> bool:
        return self._connected and self.db is not None

    async def disconnect(self) -> None:
        if self.db:
            try:
                self.db.close()
                logger.info("Database connection closed.")
            except Exception as e:
                logger.warning(f"Error closing database: {e}")
            finally:
                self.db = None
                self._connected = False
                self._all_tracks_cache = None

    def __del__(self) -> None:
        if self.db:
            try:
                self.db.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_all_tracks(self) -> List[TrackWithEnergy]:
        """Load all active tracks into memory. Cached after first call."""
        if self._all_tracks_cache is not None:
            return self._all_tracks_cache

        if not self.db:
            raise RuntimeError("Database not connected")

        all_content = list(self.db.get_content())
        active = [c for c in all_content if getattr(c, "rb_local_deleted", 0) == 0]
        self._all_tracks_cache = [self._content_to_track(c) for c in active]
        logger.info(f"Loaded {len(self._all_tracks_cache)} tracks into memory.")
        return self._all_tracks_cache

    async def search_tracks(self, query: str = "", limit: int = 50) -> List[TrackWithEnergy]:
        """Search tracks by free-text query across title, artist, album, genre."""
        tracks = await self.get_all_tracks()
        q = query.strip().lower()

        if not q:
            return tracks[:limit]

        results = []
        for t in tracks:
            if (q in t.title.lower()
                    or q in t.artist.lower()
                    or q in (t.album or "").lower()
                    or q in (t.genre or "").lower()):
                results.append(t)
                if len(results) >= limit:
                    break
        return results

    async def get_track_by_id(self, track_id: str) -> Optional[TrackWithEnergy]:
        """Get a single track by its database ID."""
        tracks = await self.get_all_tracks()
        for t in tracks:
            if t.id == track_id:
                return t
        return None

    async def get_track_by_title(self, title: str) -> Optional[TrackWithEnergy]:
        """Fuzzy-find a track by title (case-insensitive substring match)."""
        tracks = await self.get_all_tracks()
        title_lower = title.strip().lower()

        # Exact match first
        for t in tracks:
            if t.title.lower() == title_lower:
                return t

        # Substring match
        for t in tracks:
            if title_lower in t.title.lower() or t.title.lower() in title_lower:
                return t

        # Artist - Title match
        for t in tracks:
            full = f"{t.artist} - {t.title}".lower()
            if title_lower in full or full in title_lower:
                return t

        return None

    # ------------------------------------------------------------------
    # Playlist creation (for export)
    # ------------------------------------------------------------------

    async def create_playlist_with_tracks(self, name: str, track_ids: List[str]) -> str:
        """Create a new Rekordbox playlist and add tracks to it."""
        if not self.db:
            raise RuntimeError("Database not connected")

        playlist = self.db.create_playlist(name=name)
        self._force_commit()
        playlist_id = playlist.ID if hasattr(playlist, "ID") else int(str(playlist))

        for track_id in track_ids:
            try:
                self.db.add_to_playlist(int(playlist_id), int(track_id))
            except Exception as e:
                logger.warning(f"Failed to add track {track_id} to playlist: {e}")

        self._force_commit()
        logger.info(f"Created playlist '{name}' with {len(track_ids)} tracks")
        return str(playlist_id)

    def _force_commit(self) -> None:
        """Commit via SQLAlchemy session directly."""
        session = getattr(self.db, "session", None) or getattr(self.db, "_session", None)
        if session is None:
            raise RuntimeError("Cannot find pyrekordbox SQLAlchemy session")
        session.commit()
        try:
            self.db.registry.clear_buffer()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _content_to_track(self, content) -> TrackWithEnergy:
        """Convert a pyrekordbox content object to TrackWithEnergy."""
        bpm_raw = getattr(content, "BPM", 0) or 0
        bpm = float(bpm_raw) / 100.0 if bpm_raw else 0.0

        artist_name = getattr(content, "ArtistName", "") or ""
        if not artist_name and hasattr(content, "Artist"):
            obj = content.Artist
            artist_name = (obj.Name if hasattr(obj, "Name") else str(obj)) if obj else ""

        album_name = getattr(content, "AlbumName", "") or ""
        if not album_name and hasattr(content, "Album"):
            obj = content.Album
            album_name = (obj.Name if hasattr(obj, "Name") else str(obj)) if obj else ""

        genre_name = getattr(content, "GenreName", "") or ""
        if not genre_name and hasattr(content, "Genre"):
            obj = content.Genre
            genre_name = (obj.Name if hasattr(obj, "Name") else str(obj)) if obj else ""

        key_name = getattr(content, "KeyName", "") or ""
        if not key_name and hasattr(content, "Key"):
            obj = content.Key
            key_name = (obj.Name if hasattr(obj, "Name") else str(obj)) if obj else ""

        color_id = getattr(content, "ColorID", 0) or 0
        color_name = COLOR_ID_TO_NAME.get(int(color_id), "none")

        db_rating = getattr(content, "Rating", 0) or 0
        stars = RATING_TO_STARS.get(int(db_rating), 0)

        return TrackWithEnergy(
            id=str(content.ID),
            title=content.Title or "",
            artist=artist_name,
            album=album_name or None,
            genre=genre_name or None,
            bpm=bpm,
            key=key_name or None,
            rating=stars,
            play_count=int(getattr(content, "DJPlayCount", 0) or 0),
            length=int(getattr(content, "Length", 0) or 0),
            file_path=getattr(content, "FolderPath", "") or None,
            date_added=str(getattr(content, "created_at", "") or "") or None,
            date_modified=str(getattr(content, "StockDate", "") or "") or None,
            comments=getattr(content, "Commnt", "") or None,
            color=color_name,
            color_id=int(color_id),
            energy=None,
            energy_source="none",
        )
