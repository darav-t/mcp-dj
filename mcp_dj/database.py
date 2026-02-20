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
    # Raw table query methods
    # ------------------------------------------------------------------

    def _row_to_dict(self, obj, fields: list) -> dict:
        """Convert a SQLAlchemy model object to a dict for the given fields."""
        result = {}
        for f in fields:
            val = getattr(obj, f, None)
            if val is not None:
                result[f] = str(val) if not isinstance(val, (int, float, bool)) else val
            else:
                result[f] = None
        return result

    async def query_artists(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_artist())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        return [{"ID": str(r.ID), "Name": r.Name, "SearchStr": getattr(r, "SearchStr", None)} for r in rows[offset:offset + limit]]

    async def query_albums(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_album())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        return [{"ID": str(r.ID), "Name": r.Name, "AlbumArtistID": str(getattr(r, "AlbumArtistID", None) or ""), "ImagePath": getattr(r, "ImagePath", None), "Compilation": getattr(r, "Compilation", None)} for r in rows[offset:offset + limit]]

    async def query_genres(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_genre())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        return [{"ID": str(r.ID), "Name": r.Name} for r in rows[offset:offset + limit]]

    async def query_labels(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_label())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        return [{"ID": str(r.ID), "Name": r.Name} for r in rows[offset:offset + limit]]

    async def query_keys(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_key())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        return [{"ID": str(r.ID), "ScaleName": r.ScaleName, "Seq": getattr(r, "Seq", None)} for r in rows[offset:offset + limit]]

    async def query_colors(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_color())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        return [{"ID": str(r.ID), "ColorCode": getattr(r, "ColorCode", None), "SortKey": getattr(r, "SortKey", None), "Commnt": getattr(r, "Commnt", None)} for r in rows[offset:offset + limit]]

    async def query_playlists(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_playlist())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        return [{"ID": str(r.ID), "Name": r.Name, "Seq": getattr(r, "Seq", None), "Attribute": getattr(r, "Attribute", None), "ParentID": str(getattr(r, "ParentID", None) or ""), "ImagePath": getattr(r, "ImagePath", None)} for r in rows[offset:offset + limit]]

    async def query_playlist_songs(self, playlist_id: Optional[str] = None, limit: int = 500, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_playlist_songs())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        if playlist_id:
            rows = [r for r in rows if str(r.PlaylistID) == str(playlist_id)]
        return [{"ID": str(r.ID), "PlaylistID": str(r.PlaylistID), "ContentID": str(r.ContentID), "TrackNo": getattr(r, "TrackNo", None)} for r in rows[offset:offset + limit]]

    async def query_history(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_history())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        return [{"ID": str(r.ID), "Name": r.Name, "Seq": getattr(r, "Seq", None), "Attribute": getattr(r, "Attribute", None), "ParentID": str(getattr(r, "ParentID", None) or ""), "DateCreated": str(getattr(r, "DateCreated", None) or "")} for r in rows[offset:offset + limit]]

    async def query_history_songs(self, history_id: Optional[str] = None, limit: int = 500, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_history_songs())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        if history_id:
            rows = [r for r in rows if str(r.HistoryID) == str(history_id)]
        return [{"ID": str(r.ID), "HistoryID": str(r.HistoryID), "ContentID": str(r.ContentID), "TrackNo": getattr(r, "TrackNo", None)} for r in rows[offset:offset + limit]]

    async def query_my_tags(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_my_tag())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        return [{"ID": str(r.ID), "Name": r.Name, "Seq": getattr(r, "Seq", None), "Attribute": getattr(r, "Attribute", None), "ParentID": str(getattr(r, "ParentID", None) or "")} for r in rows[offset:offset + limit]]

    async def query_my_tag_songs(self, my_tag_id: Optional[str] = None, limit: int = 500, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_my_tag_songs())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        if my_tag_id:
            rows = [r for r in rows if str(r.MyTagID) == str(my_tag_id)]
        return [{"ID": str(r.ID), "MyTagID": str(r.MyTagID), "ContentID": str(r.ContentID), "TrackNo": getattr(r, "TrackNo", None)} for r in rows[offset:offset + limit]]

    async def query_cues(self, content_id: Optional[str] = None, limit: int = 500, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_cue())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        if content_id:
            rows = [r for r in rows if str(r.ContentID) == str(content_id)]
        return [{
            "ID": str(r.ID),
            "ContentID": str(r.ContentID),
            "InMsec": getattr(r, "InMsec", None),
            "OutMsec": getattr(r, "OutMsec", None),
            "Kind": getattr(r, "Kind", None),
            "Color": getattr(r, "Color", None),
            "Comment": getattr(r, "Comment", None),
            "BeatLoopSize": getattr(r, "BeatLoopSize", None),
            "ActiveLoop": getattr(r, "ActiveLoop", None),
        } for r in rows[offset:offset + limit]]

    async def query_hot_cue_banklists(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_hot_cue_banklist())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        return [{"ID": str(r.ID), "Name": r.Name, "Seq": getattr(r, "Seq", None), "Attribute": getattr(r, "Attribute", None), "ParentID": str(getattr(r, "ParentID", None) or ""), "ImagePath": getattr(r, "ImagePath", None)} for r in rows[offset:offset + limit]]

    async def query_samplers(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_sampler())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        return [{"ID": str(r.ID), "Name": r.Name, "Seq": getattr(r, "Seq", None), "Attribute": getattr(r, "Attribute", None), "ParentID": str(getattr(r, "ParentID", None) or "")} for r in rows[offset:offset + limit]]

    async def query_content_files(self, content_id: Optional[str] = None, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_content_file())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        if content_id:
            rows = [r for r in rows if str(r.ContentID) == str(content_id)]
        return [{"ID": str(r.ID), "ContentID": str(r.ContentID), "Path": getattr(r, "Path", None), "Hash": getattr(r, "Hash", None), "Size": getattr(r, "Size", None)} for r in rows[offset:offset + limit]]

    async def query_image_files(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_image_file())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        return [{"ID": str(r.ID), "TableName": getattr(r, "TableName", None), "TargetUUID": getattr(r, "TargetUUID", None), "TargetID": getattr(r, "TargetID", None), "Path": getattr(r, "Path", None), "Size": getattr(r, "Size", None)} for r in rows[offset:offset + limit]]

    async def query_setting_files(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_setting_file())
        rows = [r for r in rows if getattr(r, "rb_local_deleted", 0) == 0]
        return [{"ID": str(r.ID), "Path": getattr(r, "Path", None), "Hash": getattr(r, "Hash", None), "Size": getattr(r, "Size", None)} for r in rows[offset:offset + limit]]

    async def query_agent_registry(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        rows = list(self.db.get_agent_registry())
        return [{"registry_id": str(getattr(r, "registry_id", "") or ""), "id_1": getattr(r, "id_1", None), "id_2": getattr(r, "id_2", None), "str_1": getattr(r, "str_1", None), "str_2": getattr(r, "str_2", None)} for r in rows[offset:offset + limit]]

    async def query_db_property(self) -> dict:
        if not self.db:
            raise RuntimeError("Database not connected")
        try:
            from pyrekordbox.masterdb import DjmdProperty
            session = getattr(self.db, "session", None) or getattr(self.db, "_session", None)
            row = session.query(DjmdProperty).first()
            if row is None:
                return {}
            return {"DBID": getattr(row, "DBID", None), "DBVersion": getattr(row, "DBVersion", None), "BaseDBDrive": getattr(row, "BaseDBDrive", None), "CurrentDBDrive": getattr(row, "CurrentDBDrive", None), "DeviceID": getattr(row, "DeviceID", None)}
        except Exception as e:
            return {"error": str(e)}

    async def query_mixer_params(self, content_id: Optional[str] = None, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        try:
            from pyrekordbox.masterdb import DjmdMixerParam
            session = getattr(self.db, "session", None) or getattr(self.db, "_session", None)
            q = session.query(DjmdMixerParam).filter(DjmdMixerParam.rb_local_deleted == 0)
            if content_id:
                q = q.filter(DjmdMixerParam.ContentID == content_id)
            rows = q.offset(offset).limit(limit).all()
            return [{"ID": str(r.ID), "ContentID": str(r.ContentID), "GainHigh": r.GainHigh, "GainLow": r.GainLow, "PeakHigh": r.PeakHigh, "PeakLow": r.PeakLow} for r in rows]
        except Exception as e:
            return [{"error": str(e)}]

    async def query_devices(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        try:
            from pyrekordbox.masterdb import DjmdDevice
            session = getattr(self.db, "session", None) or getattr(self.db, "_session", None)
            rows = session.query(DjmdDevice).filter(DjmdDevice.rb_local_deleted == 0).offset(offset).limit(limit).all()
            return [{"ID": str(r.ID), "MasterDBID": getattr(r, "MasterDBID", None), "Name": r.Name} for r in rows]
        except Exception as e:
            return [{"error": str(e)}]

    async def query_menu_items(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        try:
            from pyrekordbox.masterdb import DjmdMenuItems
            session = getattr(self.db, "session", None) or getattr(self.db, "_session", None)
            rows = session.query(DjmdMenuItems).filter(DjmdMenuItems.rb_local_deleted == 0).offset(offset).limit(limit).all()
            return [{"ID": str(r.ID), "Class": getattr(r, "Class", None), "Name": r.Name} for r in rows]
        except Exception as e:
            return [{"error": str(e)}]

    async def query_categories(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        try:
            from pyrekordbox.masterdb import DjmdCategory
            session = getattr(self.db, "session", None) or getattr(self.db, "_session", None)
            rows = session.query(DjmdCategory).filter(DjmdCategory.rb_local_deleted == 0).offset(offset).limit(limit).all()
            return [{"ID": str(r.ID), "MenuItemID": str(getattr(r, "MenuItemID", None) or ""), "Seq": getattr(r, "Seq", None), "Disable": getattr(r, "Disable", None), "InfoOrder": getattr(r, "InfoOrder", None)} for r in rows]
        except Exception as e:
            return [{"error": str(e)}]

    async def query_sort(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        try:
            from pyrekordbox.masterdb import DjmdSort
            session = getattr(self.db, "session", None) or getattr(self.db, "_session", None)
            rows = session.query(DjmdSort).filter(DjmdSort.rb_local_deleted == 0).offset(offset).limit(limit).all()
            return [{"ID": str(r.ID), "MenuItemID": str(getattr(r, "MenuItemID", None) or ""), "Seq": getattr(r, "Seq", None), "Disable": getattr(r, "Disable", None)} for r in rows]
        except Exception as e:
            return [{"error": str(e)}]

    async def query_related_tracks(self, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        try:
            from pyrekordbox.masterdb import DjmdRelatedTracks
            session = getattr(self.db, "session", None) or getattr(self.db, "_session", None)
            rows = session.query(DjmdRelatedTracks).filter(DjmdRelatedTracks.rb_local_deleted == 0).offset(offset).limit(limit).all()
            return [{"ID": str(r.ID), "Name": r.Name, "Seq": getattr(r, "Seq", None), "Attribute": getattr(r, "Attribute", None), "ParentID": str(getattr(r, "ParentID", None) or ""), "Criteria": getattr(r, "Criteria", None)} for r in rows]
        except Exception as e:
            return [{"error": str(e)}]

    async def query_song_related_tracks(self, related_tracks_id: Optional[str] = None, limit: int = 500, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        try:
            from pyrekordbox.masterdb import DjmdSongRelatedTracks
            session = getattr(self.db, "session", None) or getattr(self.db, "_session", None)
            q = session.query(DjmdSongRelatedTracks).filter(DjmdSongRelatedTracks.rb_local_deleted == 0)
            if related_tracks_id:
                q = q.filter(DjmdSongRelatedTracks.RelatedTracksID == related_tracks_id)
            rows = q.offset(offset).limit(limit).all()
            return [{"ID": str(r.ID), "RelatedTracksID": str(r.RelatedTracksID), "ContentID": str(r.ContentID), "TrackNo": getattr(r, "TrackNo", None)} for r in rows]
        except Exception as e:
            return [{"error": str(e)}]

    async def query_active_censors(self, content_id: Optional[str] = None, limit: int = 100, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        try:
            from pyrekordbox.masterdb import DjmdActiveCensor
            session = getattr(self.db, "session", None) or getattr(self.db, "_session", None)
            q = session.query(DjmdActiveCensor).filter(DjmdActiveCensor.rb_local_deleted == 0)
            if content_id:
                q = q.filter(DjmdActiveCensor.ContentID == content_id)
            rows = q.offset(offset).limit(limit).all()
            return [{"ID": str(r.ID), "ContentID": str(r.ContentID), "InMsec": getattr(r, "InMsec", None), "OutMsec": getattr(r, "OutMsec", None), "Info": getattr(r, "Info", None), "ParameterList": getattr(r, "ParameterList", None)} for r in rows]
        except Exception as e:
            return [{"error": str(e)}]

    async def query_song_samplers(self, sampler_id: Optional[str] = None, limit: int = 500, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        try:
            from pyrekordbox.masterdb import DjmdSongSampler
            session = getattr(self.db, "session", None) or getattr(self.db, "_session", None)
            q = session.query(DjmdSongSampler).filter(DjmdSongSampler.rb_local_deleted == 0)
            if sampler_id:
                q = q.filter(DjmdSongSampler.SamplerID == sampler_id)
            rows = q.offset(offset).limit(limit).all()
            return [{"ID": str(r.ID), "SamplerID": str(r.SamplerID), "ContentID": str(r.ContentID), "TrackNo": getattr(r, "TrackNo", None)} for r in rows]
        except Exception as e:
            return [{"error": str(e)}]

    async def query_song_hot_cue_banklists(self, hot_cue_banklist_id: Optional[str] = None, limit: int = 500, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        try:
            from pyrekordbox.masterdb import DjmdSongHotCueBanklist
            session = getattr(self.db, "session", None) or getattr(self.db, "_session", None)
            q = session.query(DjmdSongHotCueBanklist).filter(DjmdSongHotCueBanklist.rb_local_deleted == 0)
            if hot_cue_banklist_id:
                q = q.filter(DjmdSongHotCueBanklist.HotCueBanklistID == hot_cue_banklist_id)
            rows = q.offset(offset).limit(limit).all()
            return [{"ID": str(r.ID), "HotCueBanklistID": str(r.HotCueBanklistID), "ContentID": str(r.ContentID), "TrackNo": getattr(r, "TrackNo", None), "Color": getattr(r, "Color", None), "Comment": getattr(r, "Comment", None)} for r in rows]
        except Exception as e:
            return [{"error": str(e)}]

    async def query_song_tag_list(self, limit: int = 500, offset: int = 0) -> list:
        if not self.db:
            raise RuntimeError("Database not connected")
        try:
            from pyrekordbox.masterdb import DjmdSongTagList
            session = getattr(self.db, "session", None) or getattr(self.db, "_session", None)
            rows = session.query(DjmdSongTagList).filter(DjmdSongTagList.rb_local_deleted == 0).offset(offset).limit(limit).all()
            return [{"ID": str(r.ID), "ContentID": str(r.ContentID), "TrackNo": getattr(r, "TrackNo", None)} for r in rows]
        except Exception as e:
            return [{"error": str(e)}]

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

        # DateCreated is the user-visible "Date Added" shown in Rekordbox
        date_added = str(getattr(content, "DateCreated", "") or "").strip() or None

        # MyTagNames may contain duplicates (Rekordbox bug); deduplicate preserving order
        raw_tags = getattr(content, "MyTagNames", []) or []
        seen: set = set()
        my_tags = []
        for t in raw_tags:
            if t and t not in seen:
                seen.add(t)
                my_tags.append(t)

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
            date_added=date_added,
            date_modified=str(getattr(content, "StockDate", "") or "") or None,
            comments=getattr(content, "Commnt", "") or None,
            color=color_name,
            color_id=int(color_id),
            my_tags=my_tags,
            energy=None,
            energy_source="none",
        )
