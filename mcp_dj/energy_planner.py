"""
Energy Arc Planner for DJ Sets

Plans the energy arc of a DJ set using predefined profiles.
Each profile defines a piecewise-linear energy curve that guides
track selection to create compelling crowd experiences.
"""

from typing import Dict, Any, Optional, List, Tuple


# ---------------------------------------------------------------------------
# Energy profiles: list of (position_pct, target_energy) points
# ---------------------------------------------------------------------------

ENERGY_PROFILES: Dict[str, Dict[str, Any]] = {
    "journey": {
        "description": "Classic DJ journey: warm up, build, peak, cool down",
        "curve": [
            (0.0, 4), (0.15, 5), (0.30, 6), (0.50, 7),
            (0.65, 8), (0.75, 9), (0.85, 8), (0.95, 6), (1.0, 4),
        ],
    },
    "build": {
        "description": "Continuous energy build from low to high",
        "curve": [
            (0.0, 3), (0.25, 5), (0.50, 7), (0.75, 8), (1.0, 10),
        ],
    },
    "peak": {
        "description": "High energy throughout, for peak-time slots",
        "curve": [
            (0.0, 7), (0.15, 8), (0.30, 9), (0.70, 9), (0.85, 8), (1.0, 7),
        ],
    },
    "chill": {
        "description": "Low-energy ambient/warm-up set",
        "curve": [
            (0.0, 3), (0.25, 4), (0.50, 5), (0.75, 4), (1.0, 3),
        ],
    },
    "wave": {
        "description": "Multiple energy waves with peaks and valleys",
        "curve": [
            (0.0, 4), (0.15, 7), (0.25, 5), (0.40, 8), (0.50, 6),
            (0.65, 9), (0.75, 7), (0.85, 8), (1.0, 5),
        ],
    },
}


class EnergyPlanner:
    """Plans energy arcs and scores track placements against target curves."""

    def get_target_energy(self, position_pct: float, profile: str = "journey") -> float:
        """
        Interpolate the target energy level at a given position in the set.

        Args:
            position_pct: Position in the set (0.0 = start, 1.0 = end)
            profile: Energy profile name

        Returns:
            Target energy level (1-10, can be fractional)
        """
        position_pct = max(0.0, min(1.0, position_pct))
        curve = ENERGY_PROFILES.get(profile, ENERGY_PROFILES["journey"])["curve"]

        # Find the two nearest points for interpolation
        for i in range(len(curve) - 1):
            p1, e1 = curve[i]
            p2, e2 = curve[i + 1]
            if p1 <= position_pct <= p2:
                # Linear interpolation
                if p2 == p1:
                    return float(e1)
                t = (position_pct - p1) / (p2 - p1)
                return e1 + t * (e2 - e1)

        # Fallback: return last point's energy
        return float(curve[-1][1])

    def score_energy_placement(
        self,
        track_energy: int,
        position_pct: float,
        profile: str = "journey",
        prev_energy: Optional[int] = None,
        prev_prev_energy: Optional[int] = None,
    ) -> float:
        """
        Score how well a track's energy fits at this position.

        Considers:
        1. Distance from target energy curve
        2. Rate of change (avoid huge jumps)
        3. Avoid consecutive same-energy tracks (boring plateaus)
        """
        target = self.get_target_energy(position_pct, profile)
        distance = abs(track_energy - target)
        base_score = max(0.0, 1.0 - (distance / 5.0))

        # Penalize huge jumps (more than 3 levels)
        if prev_energy is not None:
            jump = abs(track_energy - prev_energy)
            if jump > 3:
                base_score *= 0.5
            elif jump > 2:
                base_score *= 0.8

        # Penalize energy plateaus (3+ tracks at same energy)
        if (prev_energy is not None
                and prev_prev_energy is not None
                and track_energy == prev_energy == prev_prev_energy):
            base_score *= 0.7

        return base_score

    def recommend_energy_direction(
        self,
        current_position_pct: float,
        current_energy: int,
        profile: str = "journey",
    ) -> Dict[str, Any]:
        """
        Advise the DJ on energy direction at the current point.

        Returns:
            {"direction": "up"|"down"|"maintain",
             "target_energy": int,
             "reason": str}
        """
        target = self.get_target_energy(current_position_pct, profile)
        target_int = round(target)

        # Check the derivative (direction of the curve)
        delta = 0.05
        future_pct = min(1.0, current_position_pct + delta)
        future_target = self.get_target_energy(future_pct, profile)
        slope = future_target - target

        diff = target_int - current_energy

        if abs(diff) <= 1 and abs(slope) < 0.5:
            return {
                "direction": "maintain",
                "target_energy": target_int,
                "reason": "On target. Maintain current energy level.",
            }
        elif diff > 0 or slope > 0.3:
            return {
                "direction": "up",
                "target_energy": target_int,
                "reason": f"Building energy toward {target_int}. "
                          f"Currently at {current_energy}, target is {target_int}.",
            }
        else:
            return {
                "direction": "down",
                "target_energy": target_int,
                "reason": f"Cooling down toward {target_int}. "
                          f"Currently at {current_energy}, easing to {target_int}.",
            }

    def generate_target_arc(
        self, track_count: int, profile: str = "journey"
    ) -> List[int]:
        """Generate the full target energy arc for a set of N tracks."""
        if track_count <= 0:
            return []
        if track_count == 1:
            return [round(self.get_target_energy(0.5, profile))]

        arc = []
        for i in range(track_count):
            pct = i / (track_count - 1)
            arc.append(round(self.get_target_energy(pct, profile)))
        return arc

    @staticmethod
    def get_profile_names() -> List[str]:
        return list(ENERGY_PROFILES.keys())

    @staticmethod
    def get_profile_description(profile: str) -> str:
        return ENERGY_PROFILES.get(profile, {}).get("description", "Unknown profile")
