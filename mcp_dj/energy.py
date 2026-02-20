"""
Energy Resolution for DJ Tracks

Resolves energy levels from multiple sources:
1. Album tag (format "Key - BPM - E7 - Genre")
2. Mixed In Key CSV export
3. BPM + genre heuristic inference
"""

import csv
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from loguru import logger

from .models import TrackWithEnergy


# ---------------------------------------------------------------------------
# Mixed In Key CSV reader
# ---------------------------------------------------------------------------

def _configured_mik_csv() -> Optional[Path]:
    """Return the MIK CSV path from env var, or None if not configured."""
    env_path = os.environ.get("MIK_CSV_PATH")
    return Path(env_path) if env_path else None


class MixedInKeyLibrary:
    """
    Reads the Mixed In Key Library.csv and provides energy lookups by title.
    CSV columns: Collection name, File name, Key result, BPM, Energy

    This feature is optional. Set the MIK_CSV_PATH environment variable to
    the path of your Mixed In Key "Library.csv" export to enable it.
    Use MixedInKeyLibrary.from_env() to get an instance only when configured.
    """

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self._index: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    @classmethod
    def from_env(cls) -> Optional["MixedInKeyLibrary"]:
        """Return a MixedInKeyLibrary if MIK_CSV_PATH is set, else None."""
        path = _configured_mik_csv()
        if path is None:
            logger.info("MIK_CSV_PATH not set — Mixed In Key energy data disabled.")
            return None
        return cls(path)

    def load(self) -> None:
        """Parse the CSV and build a lookup index."""
        if not self.csv_path.exists():
            logger.warning(f"Mixed In Key CSV not found at {self.csv_path}, skipping.")
            self._loaded = True
            return

        self._index = {}
        with open(self.csv_path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = self._normalise(row.get("File name", ""))
                if key:
                    self._index[key] = {
                        "collection": row.get("Collection name", "").strip(),
                        "file_name": row.get("File name", "").strip(),
                        "key": row.get("Key result", "").strip(),
                        "bpm": row.get("BPM", "").strip(),
                        "energy": row.get("Energy", "").strip(),
                    }
        self._loaded = True
        logger.info(f"Mixed In Key library loaded: {len(self._index)} tracks")

    @staticmethod
    def _normalise(s: str) -> str:
        return s.strip().lower()

    def get_energy(self, title: str) -> Optional[Dict[str, Any]]:
        """Look up energy for a track by title."""
        if not self._loaded:
            self.load()

        norm_title = self._normalise(title)

        # Exact match
        if norm_title in self._index:
            return self._index[norm_title]

        # Substring match
        for key, row in self._index.items():
            if key in norm_title or norm_title in key:
                return row

        return None


# ---------------------------------------------------------------------------
# Energy Resolver
# ---------------------------------------------------------------------------

class EnergyResolver:
    """Resolves energy levels for tracks using multiple sources."""

    def __init__(self, mik: Optional[MixedInKeyLibrary] = None) -> None:
        self.mik = mik

    def resolve(self, track: TrackWithEnergy) -> TrackWithEnergy:
        """Determine energy for a track, trying sources in priority order."""
        # Priority 1: Album tag (format "Key - BPM - EnergyN - Genre")
        if track.album:
            energy = self._parse_album_tag_energy(track.album)
            if energy is not None:
                track.energy = energy
                track.energy_source = "album_tag"
                return track

        # Priority 2: Mixed In Key CSV
        if self.mik:
            mik_data = self.mik.get_energy(track.title)
            if mik_data and mik_data.get("energy", "").isdigit():
                val = int(mik_data["energy"])
                if 1 <= val <= 10:
                    track.energy = val
                    track.energy_source = "mik"
                    return track

        # Priority 3: Infer from BPM + genre
        track.energy = self._infer_energy(track.bpm, track.genre)
        track.energy_source = "inferred"
        return track

    def resolve_all(self, tracks: List[TrackWithEnergy]) -> List[TrackWithEnergy]:
        """Resolve energy for all tracks."""
        for track in tracks:
            self.resolve(track)
        sources = {}
        for t in tracks:
            sources[t.energy_source] = sources.get(t.energy_source, 0) + 1
        logger.info(f"Energy resolved for {len(tracks)} tracks: {sources}")
        return tracks

    @staticmethod
    def _parse_album_tag_energy(album: str) -> Optional[int]:
        """Extract energy from album tag format 'Key - BPM - E7 - Genre'."""
        parts = album.split(" - ")
        for part in parts:
            part = part.strip()
            if part.startswith("E") and len(part) >= 2 and part[1:].isdigit():
                val = int(part[1:])
                if 1 <= val <= 10:
                    return val
        return None

    @staticmethod
    def _infer_energy(bpm: float, genre: Optional[str]) -> int:
        """Heuristic energy inference from BPM and genre."""
        if bpm <= 0:
            return 5

        # Base energy from BPM
        if bpm < 100:
            base = 3
        elif bpm < 118:
            base = 4
        elif bpm < 125:
            base = 5
        elif bpm < 128:
            base = 6
        elif bpm < 132:
            base = 7
        elif bpm < 140:
            base = 8
        else:
            base = 9

        # Genre modifier
        genre_lower = (genre or "").lower()
        chill_genres = ["ambient", "downtempo", "lounge", "chill", "deep"]
        high_genres = ["techno", "hard", "drum", "bass", "trance", "psytrance"]

        if any(g in genre_lower for g in chill_genres):
            base = max(1, base - 2)
        elif any(g in genre_lower for g in high_genres):
            base = min(10, base + 1)

        return max(1, min(10, base))
