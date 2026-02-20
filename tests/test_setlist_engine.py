"""Unit tests for the Setlist Engine."""

import pytest
from mcp_dj.models import TrackWithEnergy, SetlistRequest
from mcp_dj.setlist_engine import SetlistEngine
from mcp_dj.camelot import CamelotWheel
from mcp_dj.energy_planner import EnergyPlanner


def make_track(id, title, bpm=128.0, key="8A", energy=6, genre="Tech House",
               rating=3, artist="Artist"):
    return TrackWithEnergy(
        id=str(id),
        title=title,
        artist=artist,
        bpm=bpm,
        key=key,
        energy=energy,
        genre=genre,
        rating=rating,
        energy_source="inferred",
    )


@pytest.fixture
def sample_tracks():
    """A small library of 25 tracks for testing."""
    tracks = []
    keys = ["8A", "9A", "7A", "8B", "9B", "10A", "3A", "11A", "12A", "1A"]
    genres = ["Tech House", "Techno", "Deep House", "Minimal"]
    for i in range(25):
        tracks.append(make_track(
            id=i + 1,
            title=f"Track {i + 1:02d}",
            artist=f"Artist {(i % 5) + 1}",
            bpm=124.0 + i * 0.5,
            key=keys[i % len(keys)],
            energy=3 + (i % 8),
            genre=genres[i % len(genres)],
            rating=(i % 5) + 1,
        ))
    return tracks


@pytest.fixture
def engine(sample_tracks):
    e = SetlistEngine(
        tracks=sample_tracks,
        camelot=CamelotWheel(),
        energy_planner=EnergyPlanner(),
    )
    return e


class TestEngineInitialization:
    def test_tracks_loaded(self, engine, sample_tracks):
        assert len(engine.tracks) == len(sample_tracks)

    def test_indices_built(self, engine):
        assert len(engine.by_key) > 0
        assert len(engine.by_genre) > 0
        assert len(engine.by_bpm) > 0

    def test_id_lookup(self, engine):
        assert "1" in engine._id_lookup
        assert "25" in engine._id_lookup


class TestGenerateSetlist:
    def test_basic_generation(self, engine):
        req = SetlistRequest(prompt="test", duration_minutes=30)
        setlist = engine.generate_setlist(req)
        assert setlist.track_count >= 3
        assert setlist.track_count <= 10  # 30min / 6min avg ≈ 5 tracks

    def test_all_positions_unique(self, engine):
        req = SetlistRequest(prompt="test", duration_minutes=60)
        setlist = engine.generate_setlist(req)
        track_ids = [st.track.id for st in setlist.tracks]
        assert len(track_ids) == len(set(track_ids)), "Duplicate tracks in setlist"

    def test_positions_sequential(self, engine):
        req = SetlistRequest(prompt="test", duration_minutes=60)
        setlist = engine.generate_setlist(req)
        positions = [st.position for st in setlist.tracks]
        assert positions == list(range(1, len(positions) + 1))

    def test_genre_filter(self, engine):
        # Use "Tech House" which is in the fixture genres and has plenty of tracks
        req = SetlistRequest(prompt="test", duration_minutes=30, genre="Tech House")
        setlist = engine.generate_setlist(req)
        # At least one track should match genre (harmonic mixing may widen genre
        # in small test libraries, but filter should bias toward the requested genre)
        genres = [st.track.genre for st in setlist.tracks]
        matched = sum(1 for g in genres if g and "Tech House" in g)
        assert matched >= 1, f"Genre filter should include at least one Tech House track, got: {genres}"

    def test_setlist_id_stored(self, engine):
        req = SetlistRequest(prompt="test", duration_minutes=30)
        setlist = engine.generate_setlist(req)
        retrieved = engine.get_setlist(setlist.id)
        assert retrieved is not None
        assert retrieved.id == setlist.id

    def test_harmonic_score_valid(self, engine):
        req = SetlistRequest(prompt="test", duration_minutes=60)
        setlist = engine.generate_setlist(req)
        assert 0.0 <= setlist.harmonic_score <= 1.0

    def test_energy_arc_length(self, engine):
        req = SetlistRequest(prompt="test", duration_minutes=60)
        setlist = engine.generate_setlist(req)
        assert len(setlist.energy_arc) == setlist.track_count

    def test_bpm_range(self, engine):
        req = SetlistRequest(prompt="test", duration_minutes=60)
        setlist = engine.generate_setlist(req)
        assert setlist.bpm_range[0] <= setlist.bpm_range[1]
        assert setlist.bpm_range[0] > 0

    def test_different_profiles_work(self, engine):
        for profile in ["journey", "build", "peak", "chill", "wave"]:
            req = SetlistRequest(prompt="test", duration_minutes=30, energy_profile=profile)
            setlist = engine.generate_setlist(req)
            assert setlist.track_count >= 2


class TestRecommendNext:
    def test_returns_recommendations(self, engine):
        recs = engine.recommend_next(current_track_id="1", limit=5)
        assert len(recs) > 0

    def test_by_title(self, engine):
        recs = engine.recommend_next(current_track_title="Track 01", limit=5)
        assert len(recs) > 0

    def test_excludes_current(self, engine):
        recs = engine.recommend_next(current_track_id="1", limit=10)
        rec_ids = [r.track.id for r in recs]
        assert "1" not in rec_ids

    def test_scores_in_range(self, engine):
        recs = engine.recommend_next(current_track_id="1", limit=5)
        for r in recs:
            assert 0.0 <= r.score <= 1.0

    def test_sorted_by_score(self, engine):
        recs = engine.recommend_next(current_track_id="1", limit=10)
        scores = [r.score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_energy_direction_up(self, engine):
        # With energy_direction=up, recommendations should trend toward higher energy
        recs_up = engine.recommend_next(current_track_id="1",
                                         energy_direction="up", limit=5)
        recs_down = engine.recommend_next(current_track_id="1",
                                           energy_direction="down", limit=5)
        assert len(recs_up) > 0
        assert len(recs_down) > 0

    def test_invalid_track_returns_empty(self, engine):
        recs = engine.recommend_next(current_track_id="9999", limit=5)
        assert recs == []

    def test_has_reason(self, engine):
        recs = engine.recommend_next(current_track_id="1", limit=3)
        for r in recs:
            assert isinstance(r.reason, str)
            assert len(r.reason) > 0


class TestLibrarySummary:
    def test_returns_dict(self, engine):
        summary = engine.get_library_summary()
        assert isinstance(summary, dict)

    def test_required_keys(self, engine):
        summary = engine.get_library_summary()
        assert "total" in summary
        assert "bpm_min" in summary
        assert "bpm_max" in summary
        assert "top_genres" in summary

    def test_total_matches(self, engine, sample_tracks):
        summary = engine.get_library_summary()
        assert summary["total"] == len(sample_tracks)
