"""
Core Setlist Generation Engine

Combines harmonic mixing (Camelot wheel), energy planning, BPM proximity,
genre coherence, and diversity to generate DJ setlists.
"""

import random
import uuid
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Dict, Tuple

from loguru import logger

from .models import (
    TrackWithEnergy,
    SetlistTrack,
    MixPlan,
    Setlist,
    SetlistRequest,
    NextTrackRecommendation,
    GenrePhase,
    BpmCurvePoint,
    MoodTarget,
)
from .camelot import CamelotWheel
from .energy_planner import EnergyPlanner

# Optional — only present when essentia_analyzer is importable
try:
    from .essentia_analyzer import EssentiaFeatureStore
except ImportError:
    EssentiaFeatureStore = None

# Optional — phrase store for DJ mix planning
try:
    from .phrase_store import PhraseStore, plan_transition
    _HAS_PHRASE_STORE = True
except ImportError:
    PhraseStore = None  # type: ignore
    _HAS_PHRASE_STORE = False


# ---------------------------------------------------------------------------
# Mood descriptor → mood vector mapping (for MoodTarget.descriptors)
# ---------------------------------------------------------------------------
MOOD_DESCRIPTORS: Dict[str, Dict[str, float]] = {
    "dark":        {"aggressive": 0.6, "sad": 0.3, "relaxed": 0.1},
    "hypnotic":    {"relaxed": 0.5, "party": 0.3, "sad": 0.2},
    "uplifting":   {"happy": 0.6, "party": 0.3, "relaxed": 0.1},
    "emotional":   {"sad": 0.4, "happy": 0.3, "relaxed": 0.3},
    "energetic":   {"party": 0.5, "aggressive": 0.3, "happy": 0.2},
    "dreamy":      {"relaxed": 0.6, "sad": 0.2, "happy": 0.2},
    "aggressive":  {"aggressive": 0.8, "party": 0.2},
    "groovy":      {"party": 0.4, "happy": 0.4, "relaxed": 0.2},
    "melancholic": {"sad": 0.6, "relaxed": 0.3, "happy": 0.1},
    "euphoric":    {"happy": 0.5, "party": 0.4, "aggressive": 0.1},
    "minimal":     {"relaxed": 0.5, "sad": 0.3, "party": 0.2},
    "driving":     {"aggressive": 0.4, "party": 0.4, "happy": 0.2},
    "trippy":      {"relaxed": 0.4, "party": 0.3, "sad": 0.3},
    "warm":        {"happy": 0.4, "relaxed": 0.4, "party": 0.2},
    "intense":     {"aggressive": 0.5, "party": 0.4, "happy": 0.1},
    "chill":       {"relaxed": 0.7, "happy": 0.2, "sad": 0.1},
    "funky":       {"happy": 0.5, "party": 0.4, "relaxed": 0.1},
    "deep":        {"relaxed": 0.4, "sad": 0.3, "party": 0.3},
    "heavy":       {"aggressive": 0.7, "party": 0.2, "sad": 0.1},
    "melodic":     {"happy": 0.4, "relaxed": 0.3, "sad": 0.3},
    "raw":         {"aggressive": 0.6, "party": 0.3, "sad": 0.1},
    "atmospheric": {"relaxed": 0.5, "sad": 0.3, "happy": 0.2},
    "bouncy":      {"party": 0.5, "happy": 0.4, "aggressive": 0.1},
    "industrial":  {"aggressive": 0.7, "sad": 0.2, "party": 0.1},
    "acid":        {"aggressive": 0.4, "party": 0.4, "relaxed": 0.2},
}

# ---------------------------------------------------------------------------
# Rekordbox color → energy/mood association
# ---------------------------------------------------------------------------
COLOR_ENERGY_MAP: Dict[str, Dict[str, float]] = {
    "red":    {"energy_center": 8, "aggressive": 0.5, "party": 0.3},
    "orange": {"energy_center": 7, "party": 0.5, "happy": 0.3},
    "yellow": {"energy_center": 6, "happy": 0.6, "party": 0.3},
    "green":  {"energy_center": 5, "party": 0.4, "happy": 0.3, "relaxed": 0.2},
    "aqua":   {"energy_center": 4, "relaxed": 0.4, "happy": 0.3, "party": 0.2},
    "blue":   {"energy_center": 3, "relaxed": 0.6, "sad": 0.2},
    "purple": {"energy_center": 7, "aggressive": 0.4, "sad": 0.3, "party": 0.2},
    "pink":   {"energy_center": 5, "happy": 0.4, "relaxed": 0.3, "party": 0.2},
}

# ---------------------------------------------------------------------------
# Energy source reliability (higher = more trustworthy)
# ---------------------------------------------------------------------------
ENERGY_SOURCE_WEIGHT: Dict[str, float] = {
    "mik": 1.0,
    "manual": 0.9,
    "album_tag": 0.7,
    "inferred": 0.5,
    "none": 0.4,
}


class SetlistEngine:
    """
    Generates DJ setlists using harmonic mixing, energy planning,
    and genre coherence.
    """

    def __init__(
        self,
        tracks: Optional[List[TrackWithEnergy]] = None,
        camelot: Optional[CamelotWheel] = None,
        energy_planner: Optional[EnergyPlanner] = None,
        essentia_store=None,
        phrase_store=None,
    ):
        self.camelot = camelot or CamelotWheel()
        self.planner = energy_planner or EnergyPlanner()
        self.tracks: List[TrackWithEnergy] = []
        self.essentia_store = essentia_store  # EssentiaFeatureStore or None
        self.phrase_store   = phrase_store   # PhraseStore or None

        # Indices for fast lookup
        self.by_key: Dict[str, List[TrackWithEnergy]] = defaultdict(list)
        self.by_genre: Dict[str, List[TrackWithEnergy]] = defaultdict(list)
        self.by_bpm: Dict[int, List[TrackWithEnergy]] = defaultdict(list)
        self._id_lookup: Dict[str, TrackWithEnergy] = {}

        # Stored setlists
        self._setlists: Dict[str, Setlist] = {}

        if tracks:
            self.initialize(tracks)

    def initialize(self, tracks: List[TrackWithEnergy]) -> None:
        """Load tracks and build lookup indices."""
        self.tracks = tracks
        self._build_indices()
        logger.info(
            f"Engine initialized: {len(tracks)} tracks, "
            f"{len(self.by_key)} keys, {len(self.by_genre)} genres"
        )

    def _build_indices(self) -> None:
        """Build lookup indices for fast track selection."""
        self.by_key.clear()
        self.by_genre.clear()
        self.by_bpm.clear()
        self._id_lookup.clear()

        for t in self.tracks:
            self._id_lookup[t.id] = t
            if t.key:
                self.by_key[t.key.upper()].append(t)
            if t.genre:
                self.by_genre[t.genre.lower()].append(t)
            if t.bpm > 0:
                bucket = round(t.bpm)
                self.by_bpm[bucket].append(t)

    # ------------------------------------------------------------------
    # Setlist generation
    # ------------------------------------------------------------------

    def generate_setlist(self, request: SetlistRequest) -> Setlist:
        """
        Generate a complete setlist from a request.

        Algorithm:
        1. Determine target track count from duration
        2. Filter candidates by genre/BPM/My Tag constraints
        3. Select starting track
        4. Greedily select each subsequent track using full metadata scoring
        """
        if not self.tracks:
            raise RuntimeError("Engine not initialized with tracks")

        # Estimate track count: average track ~6 minutes
        avg_track_len = self._avg_track_length()
        target_count = max(3, round((request.duration_minutes * 60) / avg_track_len))

        # Extract custom energy curve from request (LLM-generated)
        energy_curve = self._get_energy_curve(request)

        # Filter candidate pool
        candidates = self._filter_candidates(request)
        if len(candidates) < 3:
            candidates = self.tracks  # Fallback to full library

        # Select starting track
        used_ids = set(request.excluded_track_ids)
        first_track = self._select_starting_track(request, candidates, used_ids)
        used_ids.add(first_track.id)

        # Build setlist greedily
        setlist_tracks: List[SetlistTrack] = [
            SetlistTrack(
                position=1,
                track=first_track,
                transition_score=1.0,
                key_relation="start",
                notes="Opening track",
            )
        ]

        recent_artists: List[str] = [first_track.artist]
        recent_genres: List[str] = [first_track.genre or ""]

        for pos in range(2, target_count + 1):
            current = setlist_tracks[-1].track
            position_pct = (pos - 1) / max(1, target_count - 1)

            # Get energy context — use Essentia-derived energy when available so
            # plateau/jump detection is based on the same energy signal as scoring.
            prev_energy = self._effective_energy(current)
            prev_prev_energy = (
                self._effective_energy(setlist_tracks[-2].track)
                if len(setlist_tracks) >= 2 else None
            )

            # Score all available candidates
            best_track = None
            best_score = -1.0
            best_meta = (0.0, 0.0, "")

            available = [t for t in candidates if t.id not in used_ids]
            if not available:
                available = [t for t in self.tracks if t.id not in used_ids]
            if not available:
                break

            for candidate in available:
                score, h_score, rel = self._score_candidate(
                    candidate=candidate,
                    current_track=current,
                    position_pct=position_pct,
                    profile=request.energy_profile,
                    prev_energy=prev_energy,
                    prev_prev_energy=prev_prev_energy,
                    recent_artists=recent_artists[-5:],
                    recent_genres=recent_genres[-3:],
                    custom_energy_curve=energy_curve,
                    genre_phases=request.genre_phases,
                    bpm_curve=request.bpm_curve,
                    mood_target=request.mood_target,
                )
                if score > best_score:
                    best_score = score
                    best_track = candidate
                    best_meta = (h_score, score, rel)

            if not best_track:
                break

            used_ids.add(best_track.id)

            # Calculate deltas using Essentia energy where available
            bpm_delta = best_track.bpm - current.bpm if current.bpm > 0 else 0.0
            energy_delta = self._effective_energy(best_track) - self._effective_energy(current)

            # Compute DJ mix plan from phrase data (if available)
            mix_plan = self._compute_mix_plan(current, best_track, position_pct)

            setlist_tracks.append(
                SetlistTrack(
                    position=pos,
                    track=best_track,
                    transition_score=best_meta[0],
                    bpm_delta=round(bpm_delta, 1),
                    key_relation=best_meta[2],
                    energy_delta=energy_delta,
                    notes=self._transition_note(best_meta[2], bpm_delta, energy_delta),
                    mix_plan=mix_plan,
                )
            )

            recent_artists.append(best_track.artist)
            recent_genres.append(best_track.genre or "")

        # Build the Setlist model
        setlist = self._build_setlist(setlist_tracks, request)
        self._setlists[setlist.id] = setlist
        return setlist

    def get_setlist(self, setlist_id: str) -> Optional[Setlist]:
        return self._setlists.get(setlist_id)

    # ------------------------------------------------------------------
    # Next track recommendation
    # ------------------------------------------------------------------

    def recommend_next(
        self,
        current_track_id: Optional[str] = None,
        current_track_title: Optional[str] = None,
        energy_direction: str = "maintain",
        position_pct: float = 0.5,
        profile: str = "journey",
        limit: int = 10,
    ) -> List[NextTrackRecommendation]:
        """
        Given a currently playing track, recommend what to play next.
        Supports lookup by ID or title.
        """
        # Find the current track
        current = None
        if current_track_id:
            current = self._id_lookup.get(current_track_id)
        if not current and current_track_title:
            title_lower = current_track_title.strip().lower()
            for t in self.tracks:
                if title_lower in t.title.lower() or t.title.lower() in title_lower:
                    current = t
                    break
                full = f"{t.artist} - {t.title}".lower()
                if title_lower in full:
                    current = t
                    break

        if not current:
            return []

        # Adjust profile based on energy direction
        target_energy = current.energy or 5
        if energy_direction == "up":
            target_energy = min(10, target_energy + 2)
        elif energy_direction == "down":
            target_energy = max(1, target_energy - 2)

        # Essentia features for the current track
        ess_current = self.essentia_store.get(current.file_path) if self.essentia_store else None

        # Score candidates
        scored = []
        for candidate in self.tracks:
            if candidate.id == current.id:
                continue

            h_score, rel = self.camelot.transition_score(
                current.key or "", candidate.key or ""
            )

            # Essentia features for candidate
            ess = self.essentia_store.get(candidate.file_path) if self.essentia_store else None

            # Effective energy: use essentia loudness-derived energy when available
            if ess is not None:
                cand_energy = ess.energy_as_1_to_10()
            else:
                cand_energy = candidate.energy or 5

            # BPM score — prefer essentia BPM (more accurate)
            candidate_bpm = ess.bpm_essentia if (ess and ess.bpm_essentia > 0) else candidate.bpm
            current_bpm = ess_current.bpm_essentia if (ess_current and ess_current.bpm_essentia > 0) else current.bpm
            bpm_score = self._bpm_score(current_bpm, candidate_bpm)

            # Energy score: how close to desired direction
            energy_diff = abs(cand_energy - target_energy)
            energy_score = max(0.0, 1.0 - (energy_diff / 5.0))

            # Genre score — blend Rekordbox + Discogs genre when available
            genre_score = self._genre_score(current.genre, candidate.genre)
            if ess is not None and ess_current is not None:
                discogs_score = self._discogs_genre_score(ess_current.genre_discogs, ess.genre_discogs)
                if discogs_score is not None:
                    genre_score = 0.6 * genre_score + 0.4 * discogs_score

            if ess is not None:
                # Dynamic weights when Essentia data available: danceability and mood
                # become first-class scoring dimensions drawn from reduced base weights.
                danceability_score = ess.danceability_as_1_to_10() / 10.0

                if energy_direction == "up":
                    mood_score = max(
                        ess.mood_party or 0.0,
                        ess.mood_aggressive or 0.0,
                        ess.mood_happy or 0.0,
                    )
                elif energy_direction == "down":
                    mood_score = ess.mood_relaxed or 0.0
                else:
                    mood_score = (
                        0.6 * max(ess.mood_happy or 0.0, ess.mood_party or 0.0)
                        + 0.4 * (ess.mood_relaxed or 0.0)
                    )

                # Track-to-track mood and tags vector similarity (cosine)
                mood_sim = self._mood_vector_similarity(ess_current, ess)
                mood_sim = mood_sim if mood_sim is not None else 0.5
                tags_sim = self._music_tags_similarity(ess_current, ess)
                tags_sim = tags_sim if tags_sim is not None else 0.5

                total = (
                    0.28 * h_score
                    + 0.22 * energy_score
                    + 0.12 * bpm_score
                    + 0.10 * genre_score
                    + 0.08 * mood_sim            # track-to-track mood vector match
                    + 0.06 * danceability_score
                    + 0.04 * mood_score          # position-aware mood target
                    + 0.04 * tags_sim            # track-to-track music tags match
                    + 0.02                       # essentia data availability bonus
                )
            else:
                # No Essentia data: metadata-only weights
                total = (
                    0.35 * h_score
                    + 0.30 * energy_score
                    + 0.20 * bpm_score
                    + 0.15 * genre_score
                )

            reason = self._build_reason(
                candidate, current, h_score, rel, energy_score,
                bpm_score, energy_direction, ess
            )

            scored.append(NextTrackRecommendation(
                track=candidate,
                score=round(total, 3),
                harmonic_score=round(h_score, 3),
                energy_score=round(energy_score, 3),
                bpm_score=round(bpm_score, 3),
                genre_score=round(genre_score, 3),
                reason=reason,
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_candidate(
        self,
        candidate: TrackWithEnergy,
        current_track: TrackWithEnergy,
        position_pct: float,
        profile: str,
        prev_energy: Optional[int],
        prev_prev_energy: Optional[int],
        recent_artists: List[str],
        recent_genres: List[str],
        # --- LLM-structured parameters ---
        custom_energy_curve: Optional[List[Tuple[float, int]]] = None,
        genre_phases: Optional[List[GenrePhase]] = None,
        bpm_curve: Optional[List[BpmCurvePoint]] = None,
        mood_target: Optional[MoodTarget] = None,
    ) -> Tuple[float, float, str]:
        """
        Score a candidate track using ALL available metadata.
        Returns (total_score, harmonic_score, key_relation).

        Scoring dimensions (with Essentia + structured params):
          harmonic (key confidence weighted): 0.24
          energy (custom curve or profile):   0.18
          bpm (proximity + curve blended):    0.10
          genre (phase-aware if phases):      0.09
          mood similarity (track-to-track):   0.07
          mood target (user intent):          0.06
          danceability:                       0.05
          tags similarity:                    0.04
          my_tag similarity:                  0.04
          loudness range similarity:          0.03
          color-energy alignment:             0.03
          diversity:                          0.03
          quality:                            0.02
          essentia bonus:                     0.02
        """
        # Harmonic score
        h_score, rel = self.camelot.transition_score(
            current_track.key or "", candidate.key or ""
        )

        # Essentia features for candidate (and current track for context)
        ess = self.essentia_store.get(candidate.file_path) if self.essentia_store else None
        ess_current = self.essentia_store.get(current_track.file_path) if self.essentia_store else None

        # Key confidence weighting — higher confidence = more reliable harmonic match
        key_confidence = 1.0
        if ess is not None and hasattr(ess, "key_strength") and ess.key_strength > 0:
            # Boost h_score reliability: blend toward 1.0 (full trust) at high confidence
            key_confidence = 0.5 + 0.5 * ess.key_strength

        # Effective energy: Essentia LUFS-derived energy takes priority when available
        cand_energy = ess.energy_as_1_to_10() if ess is not None else (candidate.energy or 5)

        # Energy score: fits the custom curve (LLM) or named profile (fallback)
        energy_score = self.planner.score_energy_placement(
            cand_energy, position_pct, profile, prev_energy, prev_prev_energy,
            custom_curve=custom_energy_curve,
        )

        # Energy source reliability multiplier
        source_weight = ENERGY_SOURCE_WEIGHT.get(candidate.energy_source, 0.4)
        if ess is not None:
            source_weight = 1.0  # Essentia energy is always fully reliable

        # BPM score — use Essentia BPM if available (more accurate than Rekordbox)
        candidate_bpm = ess.bpm_essentia if (ess and ess.bpm_essentia > 0) else candidate.bpm
        current_bpm = ess_current.bpm_essentia if (ess_current and ess_current.bpm_essentia > 0) else current_track.bpm
        bpm_score = self._bpm_score(current_bpm, candidate_bpm)

        # BPM curve: blend proximity score with target curve score
        if bpm_curve:
            bpm_curve_sc = self._bpm_curve_score(candidate_bpm, position_pct, bpm_curve)
            bpm_score = 0.4 * bpm_score + 0.6 * bpm_curve_sc

        # Genre score — blend Rekordbox + Discogs genre when both available
        genre_score = self._genre_score(current_track.genre, candidate.genre)
        if ess is not None and ess_current is not None:
            discogs_score = self._discogs_genre_score(ess_current.genre_discogs, ess.genre_discogs)
            if discogs_score is not None:
                genre_score = 0.6 * genre_score + 0.4 * discogs_score

        # Genre phases: position-aware genre matching
        if genre_phases:
            phase_score = self._genre_phase_score(candidate.genre, position_pct, genre_phases)
            genre_score = 0.3 * genre_score + 0.7 * phase_score

        # My Tag similarity: shared tags between consecutive tracks = smoother flow
        my_tag_sim = self._my_tag_similarity(candidate.my_tags, current_track.my_tags)

        # Loudness range similarity: similar dynamic range = smoother mixing
        loudness_range_sim = 0.5  # neutral default
        if ess is not None and ess_current is not None:
            loudness_range_sim = self._loudness_range_similarity(
                getattr(ess_current, "loudness_range_db", 0),
                getattr(ess, "loudness_range_db", 0),
            )

        # Color-energy alignment
        color_score = 0.5  # neutral default
        if candidate.color and candidate.color != "none":
            target_energy = self.planner.get_target_energy(
                position_pct, profile, custom_energy_curve
            )
            color_score = self._color_energy_score(candidate.color, target_energy)

        # Diversity penalty
        diversity = 1.0
        if candidate.artist in recent_artists:
            diversity *= 0.5
        if (candidate.genre or "").lower() in [g.lower() for g in recent_genres[-3:]]:
            diversity *= 0.9

        # Quality bonus (rating + play count)
        quality = 0.5  # baseline
        if candidate.rating >= 4:
            quality = 0.9
        elif candidate.rating >= 3:
            quality = 0.7
        if candidate.play_count > 5:
            quality = min(1.0, quality + 0.1)

        if ess is not None:
            target_energy_level = self.planner.get_target_energy(
                position_pct, profile, custom_energy_curve
            )

            # Danceability: 0-1 score, normalized from Essentia's 0-3 range
            danceability_score = ess.danceability_as_1_to_10() / 10.0

            # Mood: position-aware scoring
            if target_energy_level >= 7:
                mood_score = max(
                    ess.mood_party or 0.0,
                    ess.mood_aggressive or 0.0,
                    ess.mood_happy or 0.0,
                )
            elif target_energy_level <= 4:
                mood_score = ess.mood_relaxed or 0.0
            else:
                mood_score = (
                    0.6 * max(ess.mood_happy or 0.0, ess.mood_party or 0.0)
                    + 0.4 * (ess.mood_relaxed or 0.0)
                )

            # If user specified a mood target, blend it in
            if mood_target:
                mood_target_sc = self._mood_target_score(ess, mood_target)
                mood_score = 0.3 * mood_score + 0.7 * mood_target_sc

            # Track-to-track mood and tags vector similarity (cosine)
            mood_sim = self._mood_vector_similarity(ess_current, ess)
            mood_sim = mood_sim if mood_sim is not None else 0.5
            tags_sim = self._music_tags_similarity(ess_current, ess)
            tags_sim = tags_sim if tags_sim is not None else 0.5

            # Phrase mix compatibility bonus (0–1):
            # Rewards candidates whose intro length makes for a clean blend with
            # the current track's outro. Tracks with long intros (≥32 bars) get
            # a bonus because they give the DJ more runway to EQ-mix smoothly.
            phrase_score = self._phrase_blend_score(current_track, candidate)

            # Full metadata scoring — all dimensions
            total = (
                0.23 * h_score * key_confidence  # harmonic (key confidence weighted)
                + 0.17 * energy_score * source_weight  # energy (source reliability weighted)
                + 0.10 * bpm_score               # BPM (proximity + curve)
                + 0.09 * genre_score             # genre (phase-aware if phases provided)
                + 0.07 * mood_sim                # track-to-track mood vector match
                + 0.06 * mood_score              # mood target / position-aware
                + 0.05 * danceability_score      # danceability
                + 0.04 * tags_sim                # music tags similarity
                + 0.04 * my_tag_sim              # My Tag overlap
                + 0.03 * loudness_range_sim      # dynamic range matching
                + 0.03 * color_score             # color → energy/mood alignment
                + 0.03 * diversity               # artist/genre repetition penalty
                + 0.03 * phrase_score            # phrase blend compatibility
                + 0.02 * quality                 # rating + play_count
                + 0.01                           # essentia data availability bonus
            )
        else:
            # Phrase mix compatibility bonus (works without Essentia)
            phrase_score = self._phrase_blend_score(current_track, candidate)

            # --- No Essentia data: metadata-only weights
            total = (
                0.29 * h_score
                + 0.24 * energy_score
                + 0.13 * bpm_score
                + 0.10 * genre_score
                + 0.06 * my_tag_sim              # My Tags still available without Essentia
                + 0.04 * phrase_score            # phrase blend compatibility
                + 0.04 * diversity
                + 0.04 * quality
                + 0.04 * color_score             # Color still available without Essentia
            )
            # Blend BPM curve if available (works without Essentia)
            if bpm_curve:
                bpm_curve_sc = self._bpm_curve_score(candidate.bpm, position_pct, bpm_curve)
                total += 0.02 * bpm_curve_sc

        return (total, h_score, rel)

    @staticmethod
    def _bpm_score(current_bpm: float, candidate_bpm: float) -> float:
        """Score BPM proximity. 1.0 if within 2%, 0.0 if >8% away."""
        if current_bpm <= 0 or candidate_bpm <= 0:
            return 0.5
        pct_diff = abs(candidate_bpm - current_bpm) / current_bpm * 100
        return max(0.0, 1.0 - (pct_diff / 8.0))

    @staticmethod
    def _genre_score(current_genre: Optional[str], candidate_genre: Optional[str]) -> float:
        """Score genre similarity."""
        if not current_genre or not candidate_genre:
            return 0.5
        if current_genre.lower() == candidate_genre.lower():
            return 1.0
        # Partial match (e.g., "Tech House" and "House")
        c_words = set(current_genre.lower().split())
        t_words = set(candidate_genre.lower().split())
        if c_words & t_words:
            return 0.7
        return 0.2

    @staticmethod
    def _discogs_genre_score(
        current_discogs: Optional[dict],
        candidate_discogs: Optional[dict],
    ) -> Optional[float]:
        """Score Discogs genre overlap between two tracks using weighted cosine similarity.

        Returns a score in [0, 1] or None if either track lacks Discogs data.
        """
        if not current_discogs or not candidate_discogs:
            return None

        common_genres = set(current_discogs) & set(candidate_discogs)
        if not common_genres:
            return 0.1  # Different genre space

        # Dot product of the two genre vectors over shared genres
        dot = sum(current_discogs[g] * candidate_discogs[g] for g in common_genres)
        mag_a = sum(v ** 2 for v in current_discogs.values()) ** 0.5
        mag_b = sum(v ** 2 for v in candidate_discogs.values()) ** 0.5

        if mag_a == 0 or mag_b == 0:
            return None
        return min(1.0, dot / (mag_a * mag_b))

    @staticmethod
    def _mood_vector_similarity(ess_a, ess_b) -> Optional[float]:
        """Cosine similarity between two tracks' full mood probability vectors.

        Measures how similar the two tracks *feel* across all five mood dimensions
        (happy, sad, aggressive, relaxed, party) rather than just comparing dominant
        labels.  Returns a score in [0, 1] or None if either track lacks mood data.
        """
        if ess_a is None or ess_b is None:
            return None
        moods = ["mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed", "mood_party"]
        vec_a = [getattr(ess_a, m) or 0.0 for m in moods]
        vec_b = [getattr(ess_b, m) or 0.0 for m in moods]
        if not any(vec_a) or not any(vec_b):
            return None
        dot   = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = sum(a ** 2 for a in vec_a) ** 0.5
        mag_b = sum(b ** 2 for b in vec_b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return None
        return min(1.0, dot / (mag_a * mag_b))

    @staticmethod
    def _music_tags_similarity(ess_a, ess_b) -> Optional[float]:
        """Cosine similarity between two tracks' MagnaTagATune music tag vectors.

        Provides a deep genre/texture fingerprint match beyond the Discogs classifier —
        tags like 'techno', 'fast', 'beat', 'electronic' capture fine-grained sonic
        texture that Discogs genres often miss.  Returns a score in [0, 1] or None if
        either track lacks tag data.
        """
        if ess_a is None or ess_b is None:
            return None
        tags_a = {t["tag"]: t["score"] for t in (ess_a.music_tags or [])}
        tags_b = {t["tag"]: t["score"] for t in (ess_b.music_tags or [])}
        if not tags_a or not tags_b:
            return None
        common = set(tags_a) & set(tags_b)
        if not common:
            return 0.1  # Completely different tag spaces
        dot   = sum(tags_a[t] * tags_b[t] for t in common)
        mag_a = sum(v ** 2 for v in tags_a.values()) ** 0.5
        mag_b = sum(v ** 2 for v in tags_b.values()) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return None
        return min(1.0, dot / (mag_a * mag_b))

    # ------------------------------------------------------------------
    # New scoring dimensions (genre phases, BPM curve, mood target, etc.)
    # ------------------------------------------------------------------

    @staticmethod
    def _genre_phase_score(
        candidate_genre: Optional[str],
        position_pct: float,
        genre_phases: List[GenrePhase],
    ) -> float:
        """Score how well a candidate's genre matches the target at this position."""
        if not candidate_genre or not genre_phases:
            return 0.5
        cand_lower = candidate_genre.lower()
        best_match = 0.0

        for phase in genre_phases:
            if phase.start <= position_pct <= phase.end:
                target_lower = phase.genre.lower()
                if target_lower == cand_lower:
                    match = 1.0
                elif target_lower in cand_lower or cand_lower in target_lower:
                    match = 0.7
                else:
                    target_words = set(target_lower.split())
                    cand_words = set(cand_lower.split())
                    if target_words & cand_words:
                        match = 0.5
                    else:
                        match = 0.1
                match *= phase.weight
                best_match = max(best_match, match)

        return best_match if best_match > 0 else 0.5

    @staticmethod
    def _bpm_curve_score(
        candidate_bpm: float,
        position_pct: float,
        bpm_curve: List[BpmCurvePoint],
    ) -> float:
        """Score how well a candidate's BPM matches the target at this position."""
        if candidate_bpm <= 0 or not bpm_curve:
            return 0.5

        sorted_pts = sorted(bpm_curve, key=lambda p: p.position)
        target_bpm = sorted_pts[-1].bpm  # fallback to last point

        for i in range(len(sorted_pts) - 1):
            p1, p2 = sorted_pts[i], sorted_pts[i + 1]
            if p1.position <= position_pct <= p2.position:
                if p2.position == p1.position:
                    target_bpm = p1.bpm
                else:
                    t = (position_pct - p1.position) / (p2.position - p1.position)
                    target_bpm = p1.bpm + t * (p2.bpm - p1.bpm)
                break

        pct_diff = abs(candidate_bpm - target_bpm) / target_bpm * 100
        return max(0.0, 1.0 - (pct_diff / 8.0))

    @staticmethod
    def _mood_target_score(ess, mood_target: MoodTarget) -> float:
        """Score how well a candidate's mood matches the user's target mood."""
        if ess is None or mood_target is None:
            return 0.5

        # Build target mood vector from explicit moods + descriptors
        target_vec: Dict[str, float] = dict(mood_target.moods)
        for desc in mood_target.descriptors:
            desc_lower = desc.lower()
            if desc_lower in MOOD_DESCRIPTORS:
                for mood_name, weight in MOOD_DESCRIPTORS[desc_lower].items():
                    target_vec[mood_name] = target_vec.get(mood_name, 0) + weight

        if not target_vec:
            return 0.5

        # Normalize
        total = sum(target_vec.values())
        if total > 0:
            target_vec = {k: v / total for k, v in target_vec.items()}

        # Build candidate mood vector
        cand_vec = {
            "happy": getattr(ess, "mood_happy", None) or 0.0,
            "sad": getattr(ess, "mood_sad", None) or 0.0,
            "aggressive": getattr(ess, "mood_aggressive", None) or 0.0,
            "relaxed": getattr(ess, "mood_relaxed", None) or 0.0,
            "party": getattr(ess, "mood_party", None) or 0.0,
        }

        # Cosine similarity
        all_moods = set(target_vec) | set(cand_vec)
        dot = sum(target_vec.get(m, 0) * cand_vec.get(m, 0) for m in all_moods)
        mag_t = sum(v ** 2 for v in target_vec.values()) ** 0.5
        mag_c = sum(v ** 2 for v in cand_vec.values()) ** 0.5
        if mag_t == 0 or mag_c == 0:
            return 0.5
        return min(1.0, dot / (mag_t * mag_c))

    @staticmethod
    def _my_tag_similarity(tags_a: List[str], tags_b: List[str]) -> float:
        """Jaccard similarity of My Tag sets — shared tags = smoother flow."""
        if not tags_a or not tags_b:
            return 0.5
        set_a, set_b = set(tags_a), set(tags_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        if union == 0:
            return 0.5
        return intersection / union

    @staticmethod
    def _color_energy_score(color: str, target_energy: float) -> float:
        """Score how well a track's color label aligns with the target energy."""
        color_lower = color.lower() if color else ""
        info = COLOR_ENERGY_MAP.get(color_lower)
        if not info:
            return 0.5
        energy_center = info["energy_center"]
        distance = abs(target_energy - energy_center)
        return max(0.0, 1.0 - (distance / 5.0))

    @staticmethod
    def _loudness_range_similarity(range_a: float, range_b: float) -> float:
        """Score closeness of loudness range (dynamic range) — similar = smoother mixing."""
        if range_a == 0 or range_b == 0:
            return 0.5
        diff = abs(range_a - range_b)
        # 0 dB diff = 1.0, 10 dB diff = 0.0
        return max(0.0, 1.0 - (diff / 10.0))

    @staticmethod
    def _get_energy_curve(request: SetlistRequest) -> Optional[List[Tuple[float, int]]]:
        """Extract custom energy curve from request as raw tuples."""
        if request.energy_curve:
            return [(p.position, p.energy) for p in request.energy_curve]
        return None

    def _avg_track_length(self) -> float:
        """Average track length in seconds across the library."""
        lengths = [t.length for t in self.tracks if t.length > 0]
        if not lengths:
            return 360  # default 6 minutes
        return sum(lengths) / len(lengths)

    # ------------------------------------------------------------------
    # Track selection helpers
    # ------------------------------------------------------------------

    def _select_starting_track(
        self,
        request: SetlistRequest,
        candidates: List[TrackWithEnergy],
        used_ids: set,
    ) -> TrackWithEnergy:
        """Pick the first track based on request constraints."""
        # User specified a starting track
        if request.starting_track_id:
            track = self._id_lookup.get(request.starting_track_id)
            if track:
                return track

        # Opening energy from custom curve or named profile
        energy_curve = self._get_energy_curve(request)
        target_energy = round(self.planner.get_target_energy(
            0.0, request.energy_profile, custom_curve=energy_curve,
        ))

        # Opening genre from first genre phase (if present)
        starting_genre = None
        if request.genre_phases:
            for phase in request.genre_phases:
                if phase.start <= 0.05:
                    starting_genre = phase.genre.lower()
                    break

        available = [t for t in candidates if t.id not in used_ids]
        if not available:
            available = [t for t in self.tracks if t.id not in used_ids]

        def start_score(t: TrackWithEnergy) -> float:
            e_diff = abs(self._effective_energy(t) - target_energy)
            score = -e_diff + (t.rating / 10.0)
            # Strong genre preference for opening if phases specified
            if starting_genre and t.genre and starting_genre in t.genre.lower():
                score += 2.0
            return score

        available.sort(key=start_score, reverse=True)

        # Add some randomness: pick from top 10
        top = available[:10]
        return random.choice(top) if top else available[0]

    def _filter_candidates(self, request: SetlistRequest) -> List[TrackWithEnergy]:
        """Filter tracks by My Tags, genre (phases or single), and BPM constraints."""
        candidates = self.tracks

        # My Tag filtering
        if request.my_tags:
            tag_set = set(request.my_tags)
            tag_filtered = [
                t for t in candidates
                if tag_set & set(t.my_tags)
            ]
            if len(tag_filtered) >= 10:
                candidates = tag_filtered

        # Genre filtering: multi-genre via phases or single genre
        if request.genre_phases:
            all_phase_genres = {phase.genre.lower() for phase in request.genre_phases}
            genre_filtered = [
                t for t in candidates
                if t.genre and any(g in t.genre.lower() for g in all_phase_genres)
            ]
            if len(genre_filtered) >= 10:
                candidates = genre_filtered
        elif request.genre:
            genre_lower = request.genre.lower()
            genre_filtered = [
                t for t in candidates
                if t.genre and genre_lower in t.genre.lower()
            ]
            if len(genre_filtered) >= 10:
                candidates = genre_filtered

        # BPM filtering: use curve bounds if available, else global min/max
        if request.bpm_curve:
            curve_bpms = [p.bpm for p in request.bpm_curve]
            effective_min = min(curve_bpms) * 0.92  # 8% tolerance
            effective_max = max(curve_bpms) * 1.08
            bpm_filtered = [
                t for t in candidates
                if effective_min <= t.bpm <= effective_max
            ]
            if len(bpm_filtered) >= 10:
                candidates = bpm_filtered
        else:
            if request.bpm_min is not None:
                candidates = [t for t in candidates if t.bpm >= request.bpm_min]
            if request.bpm_max is not None:
                candidates = [t for t in candidates if t.bpm <= request.bpm_max]

        return candidates

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _build_setlist(
        self, tracks: List[SetlistTrack], request: SetlistRequest
    ) -> Setlist:
        """Assemble a Setlist model from generated tracks."""
        bpms = [st.track.bpm for st in tracks if st.track.bpm > 0]
        # Use Essentia-derived energy for the arc when available — this is the same
        # signal used during scoring, so the reported arc matches actual selection logic.
        energies = [self._effective_energy(st.track) for st in tracks]
        harmonic_scores = [st.transition_score for st in tracks[1:]]  # skip first

        genre_dist: Dict[str, int] = {}
        total_duration = 0
        for st in tracks:
            total_duration += st.track.length
            g = st.track.genre or "Unknown"
            genre_dist[g] = genre_dist.get(g, 0) + 1

        return Setlist(
            id=str(uuid.uuid4())[:8],
            name=request.prompt[:60] if request.prompt else f"{request.duration_minutes}min set",
            created_at=datetime.now().isoformat(),
            total_duration_seconds=total_duration,
            track_count=len(tracks),
            tracks=tracks,
            avg_bpm=round(sum(bpms) / len(bpms), 1) if bpms else 0.0,
            bpm_range=(min(bpms) if bpms else 0.0, max(bpms) if bpms else 0.0),
            energy_arc=energies,
            genre_distribution=genre_dist,
            harmonic_score=round(
                sum(harmonic_scores) / len(harmonic_scores), 3
            ) if harmonic_scores else 0.0,
            generation_prompt=request.prompt,
        )

    @staticmethod
    def _compute_mix_plan(
        self,
        outgoing: TrackWithEnergy,
        incoming: TrackWithEnergy,
        position_pct: float,
    ) -> Optional[MixPlan]:
        """
        Compute the DJ transition plan between two tracks using phrase data.

        Returns None if phrase data is unavailable for either track.
        Uses standard DJ mixing patterns:
          - blend          : 32-64 bar intro/outro overlap (default)
          - drop_to_drop   : align drops, used at peak time with short intros
          - breakdown_entry: start incoming from its breakdown (melodic teaser)
        """
        if not self.phrase_store:
            return None

        out_profile = self.phrase_store.get_mix_profile(str(outgoing.id))
        in_profile  = self.phrase_store.get_mix_profile(str(incoming.id))

        if not out_profile or not in_profile:
            return None  # No phrase data — fall back to metadata-only transition

        plan = plan_transition(out_profile, in_profile, set_phase=position_pct)

        return MixPlan(
            transition_type      = plan["transition_type"],
            outgoing_mix_out_ms  = plan["outgoing_mix_out_ms"],
            incoming_mix_in_ms   = plan["incoming_mix_in_ms"],
            overlap_bars         = plan["overlap_bars"],
            overlap_ms           = plan["overlap_ms"],
            bass_swap_ms         = plan["bass_swap_ms"],
            notes                = plan["notes"],
            incoming_first_chorus_ms = in_profile.first_chorus_ms,
            incoming_pre_drop_bars   = in_profile.pre_drop_bars,
            outgoing_outro_bars      = out_profile.outro_bars,
        )

    def _phrase_blend_score(
        self,
        current: TrackWithEnergy,
        candidate: TrackWithEnergy,
    ) -> float:
        """
        Score how well the candidate's structure supports a clean mix transition.

        Rewards:
          - Candidate has a long intro (≥32 bars) → DJ has plenty of runway to blend in
          - Current track has a long outro (≥16 bars) → outgoing has space to fade out
          - Both have phrase data (blend is possible at all)
          - Compatible blend lengths (no huge mismatch)

        Returns 0.5 (neutral) when phrase data is unavailable for either track.
        """
        if not self.phrase_store:
            return 0.5

        out_profile = self.phrase_store.get_mix_profile(str(current.id))
        in_profile  = self.phrase_store.get_mix_profile(str(candidate.id))

        if not out_profile or not in_profile:
            return 0.5  # No data — neutral

        # Score incoming intro length (0 = very short, 1 = long DJ-friendly intro)
        pre_drop = in_profile.pre_drop_bars
        if pre_drop >= 64:
            intro_score = 1.0
        elif pre_drop >= 32:
            intro_score = 0.85
        elif pre_drop >= 16:
            intro_score = 0.65
        else:
            intro_score = 0.3   # Short intro — harder to blend, needs drop-to-drop

        # Score outgoing outro length (space to fade out)
        outro = out_profile.outro_bars
        if outro >= 32:
            outro_score = 1.0
        elif outro >= 16:
            outro_score = 0.8
        elif outro >= 8:
            outro_score = 0.6
        else:
            outro_score = 0.3

        # Blend compatibility: ratio of overlap bars each can support
        possible_overlap = min(in_profile.pre_drop_bars, out_profile.outro_bars)
        overlap_score = min(1.0, possible_overlap / 32.0)

        return 0.35 * intro_score + 0.35 * outro_score + 0.30 * overlap_score

    @staticmethod
    def _transition_note(rel: str, bpm_delta: float, energy_delta: int) -> str:
        """Generate a human-readable transition note."""
        parts = []

        rel_desc = {
            "same": "Same key",
            "adjacent_up": "Key +1 (smooth)",
            "adjacent_down": "Key -1 (smooth)",
            "inner_outer": "Major/minor switch",
            "energy_boost": "Energy boost (+7)",
            "diagonal_up": "Diagonal +1",
            "diagonal_down": "Diagonal -1",
            "incompatible": "Key clash - mix carefully",
        }
        parts.append(rel_desc.get(rel, rel))

        if abs(bpm_delta) > 3:
            direction = "up" if bpm_delta > 0 else "down"
            parts.append(f"BPM {direction} {abs(bpm_delta):.0f}")

        if abs(energy_delta) >= 2:
            direction = "up" if energy_delta > 0 else "down"
            parts.append(f"Energy {direction}")

        return ". ".join(parts)

    @staticmethod
    def _build_reason(
        candidate: TrackWithEnergy,
        current: TrackWithEnergy,
        h_score: float,
        rel: str,
        energy_score: float,
        bpm_score: float,
        energy_direction: str,
        ess=None,  # Optional EssentiaFeatures for the candidate
    ) -> str:
        """Build a human-readable recommendation reason."""
        parts = []

        if h_score >= 0.85:
            parts.append(f"Excellent harmonic match ({rel})")
        elif h_score >= 0.6:
            parts.append(f"Good harmonic match ({rel})")
        elif h_score >= 0.3:
            parts.append(f"Acceptable key transition ({rel})")

        cand_energy = ess.energy_as_1_to_10() if ess else (candidate.energy or 5)
        curr_energy = current.energy or 5
        if energy_direction == "up" and cand_energy > curr_energy:
            parts.append("Raises energy as requested")
        elif energy_direction == "down" and cand_energy < curr_energy:
            parts.append("Lowers energy as requested")

        if bpm_score >= 0.9:
            parts.append("Very close BPM")
        elif bpm_score >= 0.7:
            parts.append("Compatible BPM")

        if candidate.genre and current.genre and candidate.genre.lower() == current.genre.lower():
            parts.append(f"Same genre ({candidate.genre})")

        # Essentia-enriched reason snippets
        if ess is not None:
            mood = ess.dominant_mood()
            if mood:
                parts.append(f"Mood: {mood}")
            top_genre = ess.top_genre()
            if top_genre and (not candidate.genre or top_genre.lower() not in candidate.genre.lower()):
                parts.append(f"Discogs: {top_genre}")
            d_score = ess.danceability_as_1_to_10()
            if d_score >= 8:
                parts.append(f"High danceability ({d_score}/10)")

        return ". ".join(parts) if parts else "Compatible track"

    def _effective_energy(self, track: TrackWithEnergy) -> int:
        """Return the best available energy value for a track.

        Priority: Essentia LUFS-derived energy (objective, loudness-based)
        > Rekordbox manual tag (subjective).
        """
        if self.essentia_store:
            ess = self.essentia_store.get(track.file_path)
            if ess is not None:
                return ess.energy_as_1_to_10()
        return track.energy or 5

    # ------------------------------------------------------------------
    # Library stats
    # ------------------------------------------------------------------

    def get_library_summary(self) -> Dict:
        """Get a summary of the loaded library for the AI system prompt."""
        if not self.tracks:
            return {"total": 0}

        bpms = [t.bpm for t in self.tracks if t.bpm > 0]
        energies = [t.energy for t in self.tracks if t.energy is not None]

        genre_counts: Dict[str, int] = {}
        key_counts: Dict[str, int] = {}
        tag_counts: Dict[str, int] = {}
        dates = []
        for t in self.tracks:
            if t.genre:
                genre_counts[t.genre] = genre_counts.get(t.genre, 0) + 1
            if t.key:
                key_counts[t.key] = key_counts.get(t.key, 0) + 1
            for tag in t.my_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            if t.date_added:
                dates.append(t.date_added)

        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)[:12]
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total": len(self.tracks),
            "bpm_min": min(bpms) if bpms else 0,
            "bpm_max": max(bpms) if bpms else 0,
            "bpm_avg": round(sum(bpms) / len(bpms), 1) if bpms else 0,
            "energy_min": min(energies) if energies else 0,
            "energy_max": max(energies) if energies else 0,
            "top_genres": [f"{g} ({c})" for g, c in top_genres],
            "top_keys": [f"{k} ({c})" for k, c in top_keys],
            "key_summary": ", ".join(f"{k}: {c}" for k, c in top_keys[:6]),
            "date_min": min(dates) if dates else "N/A",
            "date_max": max(dates) if dates else "N/A",
            "top_my_tags": [f"{tag} ({c})" for tag, c in top_tags],
        }
