"""
Data Models for DJ Setlist Creator

Extends the VST Track model with setlist-specific types for harmonic mixing,
energy planning, and AI-powered generation.
"""

from typing import Optional, List, Dict, Tuple, Set
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Rekordbox constants (from VST models)
# ---------------------------------------------------------------------------

COLOR_NAME_TO_ID: dict[str, int] = {
    "none": 0, "pink": 1, "red": 2, "orange": 3, "yellow": 4,
    "green": 5, "aqua": 6, "blue": 7, "purple": 8,
}
COLOR_ID_TO_NAME: dict[int, str] = {v: k for k, v in COLOR_NAME_TO_ID.items()}

STARS_TO_RATING: dict[int, int] = {0: 0, 1: 51, 2: 102, 3: 153, 4: 204, 5: 255}
RATING_TO_STARS: dict[int, int] = {v: k for k, v in STARS_TO_RATING.items()}


# ---------------------------------------------------------------------------
# Track models
# ---------------------------------------------------------------------------

class Track(BaseModel):
    """Rekordbox track with metadata."""

    id: str = Field(..., description="Unique track identifier")
    title: str = Field(..., description="Track title")
    artist: str = Field(..., description="Track artist")
    album: Optional[str] = Field(None, description="Album name")
    genre: Optional[str] = Field(None, description="Musical genre")
    bpm: float = Field(0.0, description="Beats per minute")
    key: Optional[str] = Field(None, description="Musical key (e.g. '5A', '12B')")
    rating: int = Field(0, ge=0, le=5, description="Track rating (0-5 stars)")
    play_count: int = Field(0, ge=0, description="Number of times played")
    length: int = Field(0, ge=0, description="Track length in seconds")
    file_path: Optional[str] = Field(None, description="Path to audio file")
    date_added: Optional[str] = Field(None, description="Date track was added to Rekordbox (YYYY-MM-DD)")
    date_modified: Optional[str] = Field(None, description="Date track was last modified")
    comments: Optional[str] = Field(None, description="Track comments")
    color: Optional[str] = Field(None, description="Color label name")
    color_id: Optional[int] = Field(None, description="Color label ID (0-8)")
    my_tags: List[str] = Field(default_factory=list, description="Rekordbox 'My Tag' labels")

    def duration_formatted(self) -> str:
        if self.length <= 0:
            return "0:00"
        minutes = self.length // 60
        seconds = self.length % 60
        return f"{minutes}:{seconds:02d}"


class TrackWithEnergy(Track):
    """Track enriched with energy data for setlist planning."""

    energy: Optional[int] = Field(None, ge=1, le=10, description="Energy level 1-10")
    energy_source: str = Field("none", description="Source: mik, album_tag, inferred, manual")


# ---------------------------------------------------------------------------
# Setlist models
# ---------------------------------------------------------------------------

class SetlistTrack(BaseModel):
    """A track placed in a setlist with transition metadata."""

    position: int = Field(..., ge=1, description="1-based position in set")
    track: TrackWithEnergy
    transition_score: float = Field(0.0, description="Harmonic compatibility 0-1")
    bpm_delta: float = Field(0.0, description="BPM change from previous")
    key_relation: str = Field("", description="same, adjacent, inner_outer, energy_boost, incompatible")
    energy_delta: int = Field(0, description="Energy change from previous")
    notes: str = Field("", description="Transition notes")


class Setlist(BaseModel):
    """Complete generated setlist."""

    id: str
    name: str
    created_at: str
    total_duration_seconds: int = 0
    track_count: int = 0
    tracks: List[SetlistTrack] = Field(default_factory=list)
    avg_bpm: float = 0.0
    bpm_range: Tuple[float, float] = (0.0, 0.0)
    energy_arc: List[int] = Field(default_factory=list)
    genre_distribution: Dict[str, int] = Field(default_factory=dict)
    harmonic_score: float = 0.0
    generation_prompt: str = ""


class SetlistRequest(BaseModel):
    """User request for setlist generation."""

    prompt: str = ""
    duration_minutes: int = Field(60, ge=10, le=480)
    genre: Optional[str] = None
    bpm_min: Optional[float] = None
    bpm_max: Optional[float] = None
    energy_profile: str = Field("journey", description="journey, build, peak, chill, wave")
    starting_track_id: Optional[str] = None
    excluded_track_ids: List[str] = Field(default_factory=list)


class NextTrackRecommendation(BaseModel):
    """Recommendation for what to play next."""

    track: TrackWithEnergy
    score: float = Field(0.0, description="Overall recommendation score 0-1")
    harmonic_score: float = 0.0
    energy_score: float = 0.0
    bpm_score: float = 0.0
    genre_score: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Chat models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """Chat message for the AI interface."""

    role: str = Field(..., description="user or assistant")
    content: str = ""
    setlist: Optional[Setlist] = None
    recommendations: Optional[List[NextTrackRecommendation]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
