"""Unit tests for the Camelot wheel harmonic mixing engine."""

import pytest
from mcp_dj.camelot import CamelotWheel


@pytest.fixture
def camelot():
    return CamelotWheel()


class TestParseKey:
    def test_valid_keys(self, camelot):
        assert camelot.parse_key("8A") == (8, "A")
        assert camelot.parse_key("12B") == (12, "B")
        assert camelot.parse_key("1A") == (1, "A")
        assert camelot.parse_key("  8A  ") == (8, "A")

    def test_lowercase(self, camelot):
        assert camelot.parse_key("8a") == (8, "A")
        assert camelot.parse_key("12b") == (12, "B")

    def test_invalid_keys(self, camelot):
        assert camelot.parse_key("") is None
        assert camelot.parse_key("13A") is None
        assert camelot.parse_key("0A") is None
        assert camelot.parse_key("8C") is None
        assert camelot.parse_key("ABC") is None


class TestTransitionScore:
    def test_same_key(self, camelot):
        score, rel = camelot.transition_score("8A", "8A")
        assert score == 1.0
        assert rel == "same"

    def test_adjacent_up(self, camelot):
        score, rel = camelot.transition_score("8A", "9A")
        assert score == 0.9
        assert rel == "adjacent_up"

    def test_adjacent_down(self, camelot):
        score, rel = camelot.transition_score("8A", "7A")
        assert score == 0.9
        assert rel == "adjacent_down"

    def test_ring_switch(self, camelot):
        score, rel = camelot.transition_score("8A", "8B")
        assert score == 0.85
        assert rel == "inner_outer"

    def test_energy_boost(self, camelot):
        score, rel = camelot.transition_score("8A", "3A")
        assert score == 0.7
        assert rel == "energy_boost"

    def test_diagonal(self, camelot):
        score, rel = camelot.transition_score("8A", "9B")
        assert score == 0.6
        assert rel == "diagonal_up"

    def test_incompatible(self, camelot):
        score, rel = camelot.transition_score("8A", "4B")
        assert score == 0.1
        assert rel == "incompatible"

    def test_wrap_around(self, camelot):
        # 12A -> 1A should be adjacent_up (wraps)
        score, rel = camelot.transition_score("12A", "1A")
        assert score == 0.9
        assert rel == "adjacent_up"

        # 1A -> 12A should be adjacent_down
        score, rel = camelot.transition_score("1A", "12A")
        assert score == 0.9
        assert rel == "adjacent_down"

    def test_invalid_keys_return_zero(self, camelot):
        score, rel = camelot.transition_score("", "8A")
        assert score == 0.0

    def test_b_ring_adjacent(self, camelot):
        score, rel = camelot.transition_score("8B", "9B")
        assert score == 0.9

    def test_b_ring_switch(self, camelot):
        score, rel = camelot.transition_score("8B", "8A")
        assert score == 0.85
        assert rel == "inner_outer"


class TestGetCompatibleKeys:
    def test_returns_dict(self, camelot):
        keys = camelot.get_compatible_keys("8A")
        assert isinstance(keys, dict)
        assert len(keys) >= 6

    def test_includes_same_key(self, camelot):
        keys = camelot.get_compatible_keys("8A")
        assert "8A" in keys
        assert keys["8A"] == 1.0

    def test_all_scores_valid(self, camelot):
        keys = camelot.get_compatible_keys("8A")
        for k, score in keys.items():
            assert 0.0 <= score <= 1.0

    def test_invalid_key_returns_empty(self, camelot):
        keys = camelot.get_compatible_keys("13X")
        assert keys == {}


class TestFindCompatibleTracks:
    def test_filters_by_key(self, camelot):
        from mcp_dj.models import TrackWithEnergy
        tracks = [
            TrackWithEnergy(id="1", title="A", artist="X", bpm=128, key="8A"),
            TrackWithEnergy(id="2", title="B", artist="Y", bpm=128, key="9A"),  # adjacent
            TrackWithEnergy(id="3", title="C", artist="Z", bpm=128, key="4B"),  # incompatible
        ]
        results = camelot.find_compatible_tracks("8A", tracks, min_score=0.5)
        ids = [t.id for t, _, _ in results]
        assert "1" in ids
        assert "2" in ids
        assert "3" not in ids

    def test_sorted_by_score(self, camelot):
        from mcp_dj.models import TrackWithEnergy
        tracks = [
            TrackWithEnergy(id="1", title="A", artist="X", bpm=128, key="9A"),  # 0.9
            TrackWithEnergy(id="2", title="B", artist="Y", bpm=128, key="8A"),  # 1.0
        ]
        results = camelot.find_compatible_tracks("8A", tracks)
        # Should be sorted: 8A first (1.0), 9A second (0.9)
        assert results[0][0].key == "8A"
        assert results[1][0].key == "9A"
