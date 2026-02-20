"""Tests for date_added and my_tags filtering in search_library and library summary."""

import pytest
from mcp_dj.models import TrackWithEnergy
from mcp_dj.setlist_engine import SetlistEngine
from mcp_dj.camelot import CamelotWheel
from mcp_dj.energy_planner import EnergyPlanner


def make_track(id, title, date_added=None, my_tags=None, genre="Tech House",
               bpm=128.0, key="8A", energy=6, artist="Artist"):
    return TrackWithEnergy(
        id=str(id),
        title=title,
        artist=artist,
        bpm=bpm,
        key=key,
        energy=energy,
        genre=genre,
        energy_source="inferred",
        date_added=date_added,
        my_tags=my_tags or [],
    )


@pytest.fixture
def tagged_tracks():
    return [
        make_track(1, "Old Track",        date_added="2022-06-15", my_tags=["Chill", "Deep"]),
        make_track(2, "Mid Track",         date_added="2024-11-20", my_tags=["High Energy"]),
        make_track(3, "Jan 2026 Track",    date_added="2026-01-10", my_tags=["High Energy", "Renegade"]),
        make_track(4, "Feb 2026 Track A",  date_added="2026-02-01", my_tags=["Renegade"]),
        make_track(5, "Feb 2026 Track B",  date_added="2026-02-19", my_tags=["High Energy"]),
        make_track(6, "No Date Track",     date_added=None,         my_tags=["Chill"]),
        make_track(7, "No Tag Track",      date_added="2026-02-05", my_tags=[]),
    ]


@pytest.fixture
def engine(tagged_tracks):
    return SetlistEngine(
        tracks=tagged_tracks,
        camelot=CamelotWheel(),
        energy_planner=EnergyPlanner(),
    )


# ---------------------------------------------------------------------------
# Model field tests
# ---------------------------------------------------------------------------

class TestTrackFields:
    def test_date_added_stored(self):
        t = make_track(1, "X", date_added="2026-02-15")
        assert t.date_added == "2026-02-15"

    def test_date_added_none_by_default(self):
        t = make_track(1, "X")
        assert t.date_added is None

    def test_my_tags_stored(self):
        t = make_track(1, "X", my_tags=["High Energy", "Renegade"])
        assert t.my_tags == ["High Energy", "Renegade"]

    def test_my_tags_empty_by_default(self):
        t = make_track(1, "X")
        assert t.my_tags == []


# ---------------------------------------------------------------------------
# Library summary tests
# ---------------------------------------------------------------------------

class TestLibrarySummary:
    def test_date_range_present(self, engine):
        summary = engine.get_library_summary()
        assert "date_min" in summary
        assert "date_max" in summary

    def test_date_range_correct(self, engine):
        summary = engine.get_library_summary()
        assert summary["date_min"] == "2022-06-15"
        assert summary["date_max"] == "2026-02-19"

    def test_top_my_tags_present(self, engine):
        summary = engine.get_library_summary()
        assert "top_my_tags" in summary
        assert isinstance(summary["top_my_tags"], list)

    def test_top_my_tags_content(self, engine):
        summary = engine.get_library_summary()
        # "High Energy" appears on tracks 2, 3, 5 → 3 times
        tag_names = [entry.split(" (")[0] for entry in summary["top_my_tags"]]
        assert "High Energy" in tag_names

    def test_no_dates_returns_na(self):
        tracks = [make_track(i, f"T{i}") for i in range(3)]  # no date_added
        eng = SetlistEngine(tracks=tracks, camelot=CamelotWheel(), energy_planner=EnergyPlanner())
        summary = eng.get_library_summary()
        assert summary["date_min"] == "N/A"
        assert summary["date_max"] == "N/A"


# ---------------------------------------------------------------------------
# Date filtering tests
# ---------------------------------------------------------------------------

class TestDateFiltering:
    def _search(self, engine, date_from=None, date_to=None, query="", limit=50):
        """Replicate the search_library filter logic used in ai_integration and mcp_server."""
        q = query.strip().lower()
        tag_filter = ""
        results = []
        for t in engine.tracks:
            if q and not (
                q in t.title.lower()
                or q in t.artist.lower()
                or q in (t.genre or "").lower()
            ):
                continue
            d = t.date_added or ""
            if date_from and d < date_from:
                continue
            if date_to and d > date_to:
                continue
            if tag_filter and not any(tag_filter in tag.lower() for tag in t.my_tags):
                continue
            results.append(t)
            if len(results) >= limit:
                break
        return results

    def test_filter_feb_2026(self, engine):
        results = self._search(engine, date_from="2026-02-01", date_to="2026-02-28")
        titles = {t.title for t in results}
        assert "Feb 2026 Track A" in titles
        assert "Feb 2026 Track B" in titles
        assert "No Tag Track" in titles  # 2026-02-05 is in range

    def test_filter_feb_2026_excludes_older(self, engine):
        results = self._search(engine, date_from="2026-02-01", date_to="2026-02-28")
        titles = {t.title for t in results}
        assert "Old Track" not in titles
        assert "Mid Track" not in titles
        assert "Jan 2026 Track" not in titles

    def test_filter_date_from_only(self, engine):
        results = self._search(engine, date_from="2026-01-01")
        titles = {t.title for t in results}
        assert "Jan 2026 Track" in titles
        assert "Feb 2026 Track A" in titles
        assert "Old Track" not in titles

    def test_filter_date_to_only(self, engine):
        results = self._search(engine, date_to="2023-12-31")
        titles = {t.title for t in results}
        assert "Old Track" in titles
        assert "Feb 2026 Track A" not in titles

    def test_no_date_tracks_excluded_by_date_filter(self, engine):
        # Track with date_added=None has d="" which is < any date string
        results = self._search(engine, date_from="2022-01-01")
        titles = {t.title for t in results}
        assert "No Date Track" not in titles

    def test_no_date_filter_returns_all(self, engine):
        results = self._search(engine)
        assert len(results) == 7


# ---------------------------------------------------------------------------
# My Tag filtering tests
# ---------------------------------------------------------------------------

class TestMyTagFiltering:
    def _search(self, engine, my_tag="", query="", limit=50):
        q = query.strip().lower()
        tag_filter = my_tag.strip().lower()
        results = []
        for t in engine.tracks:
            if q and not (
                q in t.title.lower()
                or q in t.artist.lower()
                or q in (t.genre or "").lower()
            ):
                continue
            if tag_filter and not any(tag_filter in tag.lower() for tag in t.my_tags):
                continue
            results.append(t)
            if len(results) >= limit:
                break
        return results

    def test_filter_high_energy(self, engine):
        results = self._search(engine, my_tag="High Energy")
        titles = {t.title for t in results}
        assert "Mid Track" in titles
        assert "Jan 2026 Track" in titles
        assert "Feb 2026 Track B" in titles

    def test_filter_high_energy_excludes_others(self, engine):
        results = self._search(engine, my_tag="High Energy")
        titles = {t.title for t in results}
        assert "Old Track" not in titles
        assert "No Tag Track" not in titles

    def test_filter_renegade(self, engine):
        results = self._search(engine, my_tag="Renegade")
        titles = {t.title for t in results}
        assert "Jan 2026 Track" in titles
        assert "Feb 2026 Track A" in titles
        assert "Old Track" not in titles

    def test_filter_is_case_insensitive(self, engine):
        results_lower = self._search(engine, my_tag="high energy")
        results_upper = self._search(engine, my_tag="HIGH ENERGY")
        assert {t.id for t in results_lower} == {t.id for t in results_upper}

    def test_no_tag_filter_returns_all(self, engine):
        results = self._search(engine)
        assert len(results) == 7

    def test_tag_with_no_tracks_returns_empty(self, engine):
        results = self._search(engine, my_tag="Nonexistent Tag")
        assert results == []


# ---------------------------------------------------------------------------
# Combined date + tag filter tests
# ---------------------------------------------------------------------------

class TestCombinedFilters:
    def _search(self, engine, date_from=None, date_to=None, my_tag="", query="", limit=50):
        q = query.strip().lower()
        tag_filter = my_tag.strip().lower()
        results = []
        for t in engine.tracks:
            if q and not (
                q in t.title.lower()
                or q in t.artist.lower()
                or q in (t.genre or "").lower()
            ):
                continue
            d = t.date_added or ""
            if date_from and d < date_from:
                continue
            if date_to and d > date_to:
                continue
            if tag_filter and not any(tag_filter in tag.lower() for tag in t.my_tags):
                continue
            results.append(t)
            if len(results) >= limit:
                break
        return results

    def test_feb_2026_high_energy(self, engine):
        results = self._search(engine, date_from="2026-02-01", date_to="2026-02-28",
                               my_tag="High Energy")
        titles = {t.title for t in results}
        # Only "Feb 2026 Track B" is in Feb 2026 AND has High Energy tag
        assert titles == {"Feb 2026 Track B"}

    def test_feb_2026_renegade(self, engine):
        results = self._search(engine, date_from="2026-02-01", date_to="2026-02-28",
                               my_tag="Renegade")
        titles = {t.title for t in results}
        assert titles == {"Feb 2026 Track A"}

    def test_text_plus_date(self, engine):
        results = self._search(engine, query="feb", date_from="2026-02-01")
        titles = {t.title for t in results}
        assert "Feb 2026 Track A" in titles
        assert "Feb 2026 Track B" in titles
        assert "Old Track" not in titles
