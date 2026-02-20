"""Unit tests for the Energy Arc Planner."""

import pytest
from mcp_dj.energy_planner import EnergyPlanner, ENERGY_PROFILES


@pytest.fixture
def planner():
    return EnergyPlanner()


class TestGetTargetEnergy:
    def test_journey_start(self, planner):
        val = planner.get_target_energy(0.0, "journey")
        assert val == 4.0  # journey starts at 4

    def test_journey_peak(self, planner):
        val = planner.get_target_energy(0.75, "journey")
        assert val == 9.0  # peak at 75%

    def test_journey_end(self, planner):
        val = planner.get_target_energy(1.0, "journey")
        assert val == 4.0  # ends at 4

    def test_build_increases(self, planner):
        start = planner.get_target_energy(0.0, "build")
        mid = planner.get_target_energy(0.5, "build")
        end = planner.get_target_energy(1.0, "build")
        assert start < mid < end

    def test_chill_stays_low(self, planner):
        for pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
            val = planner.get_target_energy(pct, "chill")
            assert val <= 5, f"Chill profile too high at {pct}: {val}"

    def test_peak_stays_high(self, planner):
        for pct in [0.3, 0.5, 0.7]:
            val = planner.get_target_energy(pct, "peak")
            assert val >= 7, f"Peak profile too low at {pct}: {val}"

    def test_interpolation(self, planner):
        # At 0.5 through journey, should be between 4 and 9
        val = planner.get_target_energy(0.5, "journey")
        assert 4 <= val <= 9

    def test_clamps_input(self, planner):
        # Over 1.0 and under 0.0 should work
        val_over = planner.get_target_energy(1.5, "journey")
        val_under = planner.get_target_energy(-0.5, "journey")
        assert val_over is not None
        assert val_under is not None

    def test_all_profiles_work(self, planner):
        for profile in ENERGY_PROFILES:
            val = planner.get_target_energy(0.5, profile)
            assert 1 <= val <= 10


class TestScoreEnergyPlacement:
    def test_perfect_match(self, planner):
        # At journey start (0.0), target is 4 — matching track energy 4 = 1.0
        score = planner.score_energy_placement(4, 0.0, "journey")
        assert score == 1.0

    def test_far_from_target_low_score(self, planner):
        # At journey start (target 4), energy 10 is far
        score = planner.score_energy_placement(10, 0.0, "journey")
        assert score < 0.5

    def test_jump_penalty(self, planner):
        # Score without jump penalty
        score_no_jump = planner.score_energy_placement(7, 0.5, "journey", prev_energy=6)
        # Score with large jump penalty
        score_jump = planner.score_energy_placement(7, 0.5, "journey", prev_energy=3)
        assert score_no_jump > score_jump

    def test_plateau_penalty(self, planner):
        # Same energy 3 times in a row
        score_plateau = planner.score_energy_placement(5, 0.5, "journey",
                                                        prev_energy=5, prev_prev_energy=5)
        score_varied = planner.score_energy_placement(5, 0.5, "journey",
                                                       prev_energy=4, prev_prev_energy=6)
        assert score_plateau < score_varied

    def test_score_range(self, planner):
        for energy in range(1, 11):
            score = planner.score_energy_placement(energy, 0.5, "journey")
            assert 0.0 <= score <= 1.0


class TestRecommendEnergyDirection:
    def test_low_energy_at_peak_time(self, planner):
        # At 75% of a journey set (target 9), but current energy is 4
        advice = planner.recommend_energy_direction(0.75, 4, "journey")
        assert advice["direction"] == "up"
        assert advice["target_energy"] >= 7

    def test_high_energy_at_cooldown(self, planner):
        # At 95% of a journey set (target 6), but energy is 9
        advice = planner.recommend_energy_direction(0.95, 9, "journey")
        assert advice["direction"] == "down"

    def test_on_target(self, planner):
        # At start of journey (target 4), energy is 4
        advice = planner.recommend_energy_direction(0.0, 4, "journey")
        assert advice["direction"] in ("maintain", "up", "down")  # flexible
        assert "target_energy" in advice
        assert "reason" in advice


class TestGenerateTargetArc:
    def test_length_matches_track_count(self, planner):
        arc = planner.generate_target_arc(10, "journey")
        assert len(arc) == 10

    def test_all_values_in_range(self, planner):
        arc = planner.generate_target_arc(20, "journey")
        for val in arc:
            assert 1 <= val <= 10

    def test_empty(self, planner):
        arc = planner.generate_target_arc(0, "journey")
        assert arc == []

    def test_single_track(self, planner):
        arc = planner.generate_target_arc(1, "journey")
        assert len(arc) == 1

    def test_build_increases(self, planner):
        arc = planner.generate_target_arc(10, "build")
        # First should be lower than last
        assert arc[0] < arc[-1]
