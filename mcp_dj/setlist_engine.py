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
    Setlist,
    SetlistRequest,
    NextTrackRecommendation,
)
from .camelot import CamelotWheel
from .energy_planner import EnergyPlanner

# Optional — only present when essentia_analyzer is importable
try:
    from .essentia_analyzer import EssentiaFeatureStore
except ImportError:
    EssentiaFeatureStore = None


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
    ):
        self.camelot = camelot or CamelotWheel()
        self.planner = energy_planner or EnergyPlanner()
        self.tracks: List[TrackWithEnergy] = []
        self.essentia_store = essentia_store  # EssentiaFeatureStore or None

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
        2. Filter candidates by genre/BPM constraints
        3. Select starting track
        4. Greedily select each subsequent track by scoring candidates
        """
        if not self.tracks:
            raise RuntimeError("Engine not initialized with tracks")

        # Estimate track count: average track ~6 minutes
        avg_track_len = self._avg_track_length()
        target_count = max(3, round((request.duration_minutes * 60) / avg_track_len))

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

            # Get energy context
            prev_energy = current.energy
            prev_prev_energy = (
                setlist_tracks[-2].track.energy if len(setlist_tracks) >= 2 else None
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
                )
                if score > best_score:
                    best_score = score
                    best_track = candidate
                    best_meta = (h_score, score, rel)

            if not best_track:
                break

            used_ids.add(best_track.id)

            # Calculate deltas
            bpm_delta = best_track.bpm - current.bpm if current.bpm > 0 else 0.0
            energy_delta = (
                (best_track.energy or 5) - (current.energy or 5)
            )

            setlist_tracks.append(
                SetlistTrack(
                    position=pos,
                    track=best_track,
                    transition_score=best_meta[0],
                    bpm_delta=round(bpm_delta, 1),
                    key_relation=best_meta[2],
                    energy_delta=energy_delta,
                    notes=self._transition_note(best_meta[2], bpm_delta, energy_delta),
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

            # Danceability bonus from essentia
            danceability_bonus = 0.0
            if ess is not None:
                d_score = ess.danceability_as_1_to_10() / 10.0
                danceability_bonus = 0.03 * d_score

            # Mood bonus: reward tracks whose mood fits the energy direction
            mood_bonus = 0.0
            if ess is not None:
                if energy_direction == "up":
                    mood_bonus = 0.03 * max(
                        ess.mood_party or 0.0,
                        ess.mood_aggressive or 0.0,
                        ess.mood_happy or 0.0,
                    )
                elif energy_direction == "down":
                    mood_bonus = 0.03 * (ess.mood_relaxed or 0.0)

            # Genre score — blend Rekordbox + Discogs genre when available
            genre_score = self._genre_score(current.genre, candidate.genre)
            if ess is not None and ess_current is not None:
                discogs_score = self._discogs_genre_score(ess_current.genre_discogs, ess.genre_discogs)
                if discogs_score is not None:
                    genre_score = 0.6 * genre_score + 0.4 * discogs_score

            # Combined score
            total = (
                0.35 * h_score
                + 0.30 * energy_score
                + 0.20 * bpm_score
                + 0.15 * genre_score
                + danceability_bonus
                + mood_bonus
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
    ) -> Tuple[float, float, str]:
        """
        Score a candidate track. Returns (total_score, harmonic_score, key_relation).

        Weights (without essentia):
          harmonic:  0.35
          energy:    0.30
          bpm:       0.15
          genre:     0.10
          diversity: 0.05
          quality:   0.05

        When essentia cache is available, danceability and mood are folded into
        the energy score and a small bonus is added for Discogs genre matching.
        """
        # Harmonic score
        h_score, rel = self.camelot.transition_score(
            current_track.key or "", candidate.key or ""
        )

        # Essentia features for candidate (and current track for context)
        ess = self.essentia_store.get(candidate.file_path) if self.essentia_store else None
        ess_current = self.essentia_store.get(current_track.file_path) if self.essentia_store else None

        # Effective energy: essentia loudness-derived energy takes priority if available
        if ess is not None:
            cand_energy = ess.energy_as_1_to_10()
        else:
            cand_energy = candidate.energy or 5

        # Energy score (how well the candidate fits the energy arc at this position)
        energy_score = self.planner.score_energy_placement(
            cand_energy, position_pct, profile, prev_energy, prev_prev_energy
        )

        # Danceability bonus from essentia (0.0–1.0 → small boost)
        danceability_bonus = 0.0
        if ess is not None:
            # High danceability tracks score better for party/peak profiles
            d_score = ess.danceability_as_1_to_10() / 10.0
            if profile in ("peak", "build", "wave"):
                danceability_bonus = 0.05 * d_score
            else:
                danceability_bonus = 0.02 * d_score

        # Mood compatibility bonus: prefer energetic moods at peak positions
        mood_bonus = 0.0
        if ess is not None:
            target_energy_level = self.planner.get_target_energy(position_pct, profile)
            if target_energy_level >= 7:
                # High-energy section: prefer party/aggressive moods
                mood_bonus = 0.03 * max(
                    ess.mood_party or 0.0,
                    ess.mood_aggressive or 0.0,
                    ess.mood_happy or 0.0,
                )
            elif target_energy_level <= 4:
                # Low-energy section: prefer relaxed mood
                mood_bonus = 0.03 * (ess.mood_relaxed or 0.0)

        # BPM score — use essentia BPM if available (more accurate than Rekordbox)
        candidate_bpm = ess.bpm_essentia if (ess and ess.bpm_essentia > 0) else candidate.bpm
        current_bpm = ess_current.bpm_essentia if (ess_current and ess_current.bpm_essentia > 0) else current_track.bpm
        bpm_score = self._bpm_score(current_bpm, candidate_bpm)

        # Genre score — augment with Discogs genre data when available
        genre_score = self._genre_score(current_track.genre, candidate.genre)
        if ess is not None and ess_current is not None:
            discogs_score = self._discogs_genre_score(ess_current.genre_discogs, ess.genre_discogs)
            if discogs_score is not None:
                # Blend: 60% Rekordbox genre, 40% Discogs
                genre_score = 0.6 * genre_score + 0.4 * discogs_score

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

        total = (
            0.35 * h_score
            + 0.30 * energy_score
            + 0.15 * bpm_score
            + 0.10 * genre_score
            + 0.05 * diversity
            + 0.05 * quality
            + danceability_bonus
            + mood_bonus
        )

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

        # Find a track matching the profile's opening energy
        target_energy = round(self.planner.get_target_energy(0.0, request.energy_profile))
        available = [t for t in candidates if t.id not in used_ids]

        if not available:
            available = [t for t in self.tracks if t.id not in used_ids]

        # Sort by closeness to target opening energy, then by rating
        def start_score(t: TrackWithEnergy) -> float:
            e_diff = abs((t.energy or 5) - target_energy)
            return -e_diff + (t.rating / 10.0)

        available.sort(key=start_score, reverse=True)

        # Add some randomness: pick from top 10
        top = available[:10]
        return random.choice(top) if top else available[0]

    def _filter_candidates(self, request: SetlistRequest) -> List[TrackWithEnergy]:
        """Filter tracks by genre and BPM constraints from the request."""
        candidates = self.tracks

        if request.genre:
            genre_lower = request.genre.lower()
            genre_filtered = [
                t for t in candidates
                if t.genre and genre_lower in t.genre.lower()
            ]
            if len(genre_filtered) >= 10:
                candidates = genre_filtered

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
        energies = [st.track.energy or 5 for st in tracks]
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
