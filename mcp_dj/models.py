"""
Data Models for MCP DJ

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
# Essentia audio analysis features
# ---------------------------------------------------------------------------

class EssentiaFeatures(BaseModel):
    """
    Audio features extracted by Essentia analysis.
    Stored as a JSON cache entry at .data/essentia_cache/<sha256>.json
    """

    # Identity
    file_path: str = Field(..., description="Absolute path to the analyzed audio file")

    # BPM / Rhythm
    bpm_essentia: float = Field(0.0, description="BPM detected by Essentia RhythmExtractor2013")
    bpm_confidence: float = Field(0.0, ge=0.0, description="Beat tracking confidence (raw Essentia value, typically 0-5+)")
    beats_count: int = Field(0, description="Number of beats detected")

    # Key / Harmony
    key_essentia: Optional[str] = Field(
        None, description="Camelot key detected by Essentia (e.g. '8A', '5B')"
    )
    key_name_raw: Optional[str] = Field(None, description="Raw key name from Essentia (e.g. 'C', 'F#')")
    key_scale: Optional[str] = Field(None, description="Scale from Essentia: 'major' or 'minor'")
    key_strength: float = Field(0.0, ge=0.0, le=1.0, description="Key detection confidence 0-1")

    # Danceability
    danceability: float = Field(0.0, ge=0.0, description="Danceability score (raw Essentia value, typically 0-3+)")
    dfa: float = Field(0.0, description="Detrended Fluctuation Analysis exponent")

    # Loudness / Energy
    integrated_lufs: float = Field(0.0, description="EBU R128 integrated loudness in LUFS")
    loudness_range_db: float = Field(0.0, description="EBU R128 loudness range in dB")
    rms_db: float = Field(0.0, description="RMS loudness in dBFS (e.g. -10.2 dB)")
    rms_energy: float = Field(0.0, ge=0.0, description="RMS signal energy (linear, 0-1)")

    # Mood classifiers (0-1 probability, from VGGish + TensorFlow models)
    mood_happy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Happy mood probability 0-1")
    mood_sad: Optional[float] = Field(None, ge=0.0, le=1.0, description="Sad mood probability 0-1")
    mood_aggressive: Optional[float] = Field(None, ge=0.0, le=1.0, description="Aggressive mood probability 0-1")
    mood_relaxed: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relaxed mood probability 0-1")
    mood_party: Optional[float] = Field(None, ge=0.0, le=1.0, description="Party mood probability 0-1")

    # Genre classification (Discogs400 — top genres with confidence scores)
    genre_discogs: Optional[dict] = Field(None, description="Top Discogs400 genres with confidence scores e.g. {'House': 0.54, 'Deep House': 0.13}")

    # Music autotagging (MagnaTagATune — tags above threshold)
    music_tags: Optional[list] = Field(None, description="MagnaTagATune tags above 0.1 threshold, sorted by score e.g. [{'tag': 'techno', 'score': 0.33}]")

    # Internal metadata (not written to AI-facing cache)
    analyzed_at: str = Field("", description="ISO-8601 UTC timestamp of analysis")
    essentia_version: Optional[str] = Field(None, description="Essentia library version used")
    analysis_duration_seconds: float = Field(0.0, description="Time taken to analyze in seconds")

    def to_cache_dict(self) -> dict:
        """Return a compact, AI-readable dict for JSON cache storage.

        - Strips internal metadata (analyzed_at, essentia_version, duration)
        - Rounds floats to 2 decimal places
        - Strips 'Electronic---' prefix from genre keys
        - Flattens music_tags from [{tag, score}] to {tag: score}
        - Adds derived energy (1-10) and danceability_score (1-10)
        """
        def _r(v: float | None, n: int = 2) -> float | None:
            return round(v, n) if v is not None else None

        # Strip "Electronic---" category prefix from Discogs genres
        genre: dict | None = None
        if self.genre_discogs:
            genre = {
                k.split("---", 1)[-1]: round(v, 3)
                for k, v in self.genre_discogs.items()
            }

        # Flatten [{tag, score}] → {tag: score}
        tags: dict | None = None
        if self.music_tags:
            tags = {t["tag"]: round(t["score"], 3) for t in self.music_tags}

        dominant_genre = max(genre, key=lambda g: genre[g]) if genre else None
        dominant_tag = max(tags, key=lambda t: tags[t]) if tags else None

        d: dict = {
            "file_path": self.file_path,
            "bpm": _r(self.bpm_essentia),
            "key": self.key_essentia,
            "key_note": (
                f"{self.key_name_raw} {self.key_scale}" if self.key_name_raw else None
            ),
            "key_strength": _r(self.key_strength),
            "energy": self.energy_as_1_to_10(),
            # Store raw Essentia danceability (0-3+ range) so load_cached_features
            # can restore it without loss of precision.  "danceability" keeps the
            # 1-10 scaled value for display / grep.
            "danceability_raw": _r(self.danceability, 3),
            "danceability": self.danceability_as_1_to_10(),
            "lufs": _r(self.integrated_lufs),
            "dominant_mood": self.dominant_mood(),
            "dominant_genre": dominant_genre,
            "dominant_tag": dominant_tag,
            "mood": {
                "happy":      _r(self.mood_happy),
                "sad":        _r(self.mood_sad),
                "aggressive": _r(self.mood_aggressive),
                "relaxed":    _r(self.mood_relaxed),
                "party":      _r(self.mood_party),
            } if any(v is not None for v in [
                self.mood_happy, self.mood_sad, self.mood_aggressive,
                self.mood_relaxed, self.mood_party,
            ]) else None,
            "genre": genre,
            "tags": tags,
        }
        # Remove None values to keep the file lean
        return {k: v for k, v in d.items() if v is not None}

    def energy_as_1_to_10(self) -> int:
        """Convert integrated LUFS loudness to 1-10 scale. -20 LUFS → 1, -3 LUFS → 10."""
        lufs = max(-20.0, min(-3.0, self.integrated_lufs))
        normalized = (lufs - (-20.0)) / 17.0
        return max(1, min(10, round(1 + normalized * 9)))

    def danceability_as_1_to_10(self) -> int:
        """Convert Essentia danceability (0–3+ range) to 1-10 scale."""
        # Essentia danceability: 0 = not danceable, ~3 = very danceable
        normalized = min(1.0, self.danceability / 3.0)
        return max(1, min(10, round(1 + normalized * 9)))

    def dominant_mood(self) -> Optional[str]:
        """Return the highest-scoring mood label, or None if no mood data."""
        moods = {
            "happy": self.mood_happy,
            "sad": self.mood_sad,
            "aggressive": self.mood_aggressive,
            "relaxed": self.mood_relaxed,
            "party": self.mood_party,
        }
        available = {k: v for k, v in moods.items() if v is not None}
        if not available:
            return None
        return max(available, key=lambda k: available[k])

    def top_genre(self) -> Optional[str]:
        """Return the single highest-confidence Discogs genre, or None."""
        if not self.genre_discogs:
            return None
        return max(self.genre_discogs, key=lambda g: self.genre_discogs[g])


# ---------------------------------------------------------------------------
# LLM-structured set building inputs
# ---------------------------------------------------------------------------

class EnergyCurvePoint(BaseModel):
    """A single point on a custom energy curve generated by the LLM."""

    position: float = Field(..., ge=0.0, le=1.0, description="Position in set 0.0=start, 1.0=end")
    energy: int = Field(..., ge=1, le=10, description="Target energy level 1-10")


class GenrePhase(BaseModel):
    """A genre target for a portion of the set, enabling mixed-genre sets."""

    start: float = Field(..., ge=0.0, le=1.0, description="Start position 0.0-1.0")
    end: float = Field(..., ge=0.0, le=1.0, description="End position 0.0-1.0")
    genre: str = Field(..., description="Genre name for this phase")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="How strictly to enforce this genre 0.0-1.0")


class BpmCurvePoint(BaseModel):
    """A target BPM at a specific position in the set."""

    position: float = Field(..., ge=0.0, le=1.0, description="Position in set 0.0-1.0")
    bpm: float = Field(..., ge=60, le=200, description="Target BPM at this position")


class MoodTarget(BaseModel):
    """Target mood characteristics for the set."""

    moods: Dict[str, float] = Field(
        default_factory=dict,
        description="Mood name to weight: happy, sad, aggressive, relaxed, party (each 0-1)",
    )
    descriptors: List[str] = Field(
        default_factory=list,
        description="Freeform mood descriptors: dark, hypnotic, uplifting, emotional, driving, etc.",
    )


# ---------------------------------------------------------------------------
# Setlist models
# ---------------------------------------------------------------------------

class MixPlan(BaseModel):
    """
    DJ transition plan between two consecutive tracks.

    Tells the mixer WHERE and HOW to transition:
      - outgoing_mix_out_ms : when to start fading the outgoing track (its outro)
      - incoming_mix_in_ms  : when to start the incoming track (usually 0)
      - overlap_bars        : how many bars both tracks play simultaneously
      - transition_type     : "blend" | "drop_to_drop" | "breakdown_entry"
      - bass_swap_ms        : recommended bass EQ swap point

    Based on standard DJ mixing patterns for electronic music:
      blend          — 32-64 bar intro/outro overlap (default for techno/prog house)
      drop_to_drop   — align both drops for maximum impact (peak time, short intros)
      breakdown_entry — incoming starts from its breakdown to preview melody
    """
    transition_type:     str   = Field("blend", description="blend | drop_to_drop | breakdown_entry")
    outgoing_mix_out_ms: int   = Field(0, description="When outgoing track should end (ms)")
    incoming_mix_in_ms:  int   = Field(0, description="Where to cue incoming track (ms, usually 0)")
    overlap_bars:        int   = Field(32, description="Bars of simultaneous playback")
    overlap_ms:          int   = Field(0, description="Overlap duration in ms")
    bass_swap_ms:        int   = Field(0, description="Recommended bass swap point in outgoing track")
    notes:               str   = Field("", description="Human-readable transition description")

    # Incoming track's key structure reference points
    incoming_first_chorus_ms:  int = Field(0, description="Incoming track's first drop (ms)")
    incoming_pre_drop_bars:    int = Field(0, description="Incoming track's intro+build length (bars)")
    outgoing_outro_bars:       int = Field(0, description="Outgoing track's outro length (bars)")


class SetlistTrack(BaseModel):
    """A track placed in a setlist with transition metadata."""

    position: int = Field(..., ge=1, description="1-based position in set")
    track: TrackWithEnergy
    transition_score: float = Field(0.0, description="Harmonic compatibility 0-1")
    bpm_delta: float = Field(0.0, description="BPM change from previous")
    key_relation: str = Field("", description="same, adjacent, inner_outer, energy_boost, incompatible")
    energy_delta: int = Field(0, description="Energy change from previous")
    notes: str = Field("", description="Transition notes")
    mix_plan: Optional["MixPlan"] = Field(None, description="DJ transition plan (populated when phrase data available)")


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
    energy_profile: str = Field("journey", description="Named profile fallback when energy_curve not provided")
    starting_track_id: Optional[str] = None
    excluded_track_ids: List[str] = Field(default_factory=list)

    # --- LLM-structured parameters (all optional for backward compat) ---
    energy_curve: Optional[List[EnergyCurvePoint]] = Field(
        None,
        description="Custom energy curve from LLM. Overrides energy_profile when present.",
    )
    genre_phases: Optional[List[GenrePhase]] = Field(
        None,
        description="Genre targets at different set positions. Overrides single 'genre' field.",
    )
    bpm_curve: Optional[List[BpmCurvePoint]] = Field(
        None,
        description="BPM progression curve. Overrides global bpm_min/bpm_max.",
    )
    mood_target: Optional[MoodTarget] = Field(
        None,
        description="Target mood characteristics for candidate scoring.",
    )
    my_tags: List[str] = Field(
        default_factory=list,
        description="Rekordbox My Tag names to filter candidates by.",
    )


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
