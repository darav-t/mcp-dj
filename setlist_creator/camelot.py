"""
Camelot Wheel Harmonic Mixing Engine

Implements the Camelot wheel for scoring harmonic compatibility between tracks.
The wheel has 12 positions (1-12) and 2 rings: A (minor) and B (major).

Compatible transitions:
  same key        (8A -> 8A)   = 1.0
  adjacent +1     (8A -> 9A)   = 0.9
  adjacent -1     (8A -> 7A)   = 0.9
  ring switch     (8A -> 8B)   = 0.85
  energy boost +7 (8A -> 3A)   = 0.7
  diagonal +1     (8A -> 9B)   = 0.6
  diagonal -1     (8A -> 7B)   = 0.6
  incompatible                  = 0.1
"""

import re
from typing import Optional, Tuple, Dict, List

from .models import TrackWithEnergy


# Valid Camelot key pattern: 1-12 followed by A or B
_KEY_PATTERN = re.compile(r"^(\d{1,2})([ABab])$")


class CamelotWheel:
    """Implements Camelot wheel logic for harmonic DJ mixing."""

    @staticmethod
    def parse_key(key: str) -> Optional[Tuple[int, str]]:
        """
        Parse a Camelot key string into (number, letter).
        '8A' -> (8, 'A'), '12B' -> (12, 'B')
        Returns None for invalid keys.
        """
        if not key:
            return None
        match = _KEY_PATTERN.match(key.strip())
        if not match:
            return None
        num = int(match.group(1))
        letter = match.group(2).upper()
        if not (1 <= num <= 12):
            return None
        return (num, letter)

    @staticmethod
    def _wrap(num: int) -> int:
        """Wrap position to 1-12 range."""
        return ((num - 1) % 12) + 1

    def get_compatible_keys(self, key: str) -> Dict[str, float]:
        """
        Return all harmonically compatible keys with their transition scores.
        Returns dict mapping key string -> compatibility score (0.0-1.0).
        """
        parsed = self.parse_key(key)
        if not parsed:
            return {}

        num, letter = parsed
        other = "B" if letter == "A" else "A"
        up = self._wrap(num + 1)
        down = self._wrap(num - 1)
        boost = self._wrap(num + 7)

        return {
            f"{num}{letter}": 1.0,          # Same key
            f"{up}{letter}": 0.9,            # Adjacent up
            f"{down}{letter}": 0.9,          # Adjacent down
            f"{num}{other}": 0.85,           # Ring switch
            f"{boost}{letter}": 0.7,         # Energy boost (+7)
            f"{up}{other}": 0.6,             # Diagonal up
            f"{down}{other}": 0.6,           # Diagonal down
        }

    def transition_score(self, from_key: str, to_key: str) -> Tuple[float, str]:
        """
        Score a key transition.
        Returns (score, relationship_name).
        """
        from_parsed = self.parse_key(from_key)
        to_parsed = self.parse_key(to_key)

        if not from_parsed or not to_parsed:
            return (0.0, "unknown")

        f_num, f_letter = from_parsed
        t_num, t_letter = to_parsed

        # Same key
        if f_num == t_num and f_letter == t_letter:
            return (1.0, "same")

        # Adjacent (same ring)
        if f_letter == t_letter:
            if t_num == self._wrap(f_num + 1):
                return (0.9, "adjacent_up")
            if t_num == self._wrap(f_num - 1):
                return (0.9, "adjacent_down")
            if t_num == self._wrap(f_num + 7):
                return (0.7, "energy_boost")

        # Ring switch (same number)
        if f_num == t_num and f_letter != t_letter:
            return (0.85, "inner_outer")

        # Diagonal (adjacent + ring switch)
        if f_letter != t_letter:
            if t_num == self._wrap(f_num + 1):
                return (0.6, "diagonal_up")
            if t_num == self._wrap(f_num - 1):
                return (0.6, "diagonal_down")

        return (0.1, "incompatible")

    def find_compatible_tracks(
        self,
        current_key: str,
        candidates: List[TrackWithEnergy],
        min_score: float = 0.5,
    ) -> List[Tuple[TrackWithEnergy, float, str]]:
        """
        Find tracks harmonically compatible with the current key.
        Returns list of (track, score, relationship) tuples, sorted by score descending.
        """
        compatible = self.get_compatible_keys(current_key)
        if not compatible:
            return [(t, 0.1, "unknown") for t in candidates if t.key]

        results = []
        for track in candidates:
            if not track.key:
                continue
            key_upper = track.key.strip().upper()
            if key_upper in compatible:
                score = compatible[key_upper]
                if score >= min_score:
                    _, rel = self.transition_score(current_key, track.key)
                    results.append((track, score, rel))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
