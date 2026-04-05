"""
Phrase / Structure Detector for DJ tracks.

Uses a pure numpy + scipy pipeline — no ML frameworks required:
  1. Load audio → mono, 22050 Hz
  2. Compute short-term features per 0.5s frame
  3. Build Self-Similarity Matrix (SSM) from combined features
  4. Apply Foote checkerboard kernel → structural novelty curve
  5. Pick peaks + augment with regular 8-bar grid candidates
  6. Align each boundary to the nearest 4/8/16-bar grid (using known BPM)
  7. Label sections using Rekordbox vocabulary: Intro / Up / Chorus / Down / Outro

Cue color mapping (matches Rekordbox phrase color palette):
  Intro  → green
  Up     → yellow
  Chorus → red
  Down   → blue
  Outro  → purple
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------

COLOR_NAME_TO_RB: dict[str, int] = {
    "none": 0, "pink": 1, "red": 2, "orange": 3,
    "yellow": 4, "green": 5, "aqua": 6, "blue": 7, "purple": 8,
}

SECTION_COLOR: dict[str, str] = {
    "Intro":  "green",
    "Up":     "yellow",
    "Chorus": "red",
    "Down":   "blue",
    "Outro":  "purple",
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PhraseCue:
    time_ms: int          # cue position in milliseconds
    label: str            # section label (e.g. "Chorus", "Intro")
    color: str            # color name ("red", "green", …)
    bar_number: int       # 1-indexed bar number from track start


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class PhraseDetector:
    """
    Detect musical phrase / section boundaries in a DJ track.

    Parameters match Rekordbox's phrase analysis output:
    INTRO → UP → CHORUS → DOWN cycles, ending with OUTRO.
    """

    TARGET_SR = 22_050
    FRAME_LEN = 1.0   # seconds per analysis frame
    HOP_LEN   = 0.5   # seconds hop between frames

    def detect(
        self,
        audio_path: str,
        bpm: Optional[float] = None,
    ) -> List[PhraseCue]:
        """
        Analyse an audio file and return phrase cue points.

        Parameters
        ----------
        audio_path : str
            Path to the audio file (MP3, WAV, AIFF, FLAC, …).
        bpm : float, optional
            Known BPM for bar-grid alignment.

        Returns
        -------
        List[PhraseCue]
            One entry per detected section (including the track start at 0 ms).
        """
        logger.info("PhraseDetector: loading %s", audio_path)
        y, orig_sr = sf.read(audio_path, always_2d=True)

        # Mono mix-down
        y = y.mean(axis=1) if y.ndim > 1 else y

        # Downsample
        if orig_sr != self.TARGET_SR:
            y = _resample(y, orig_sr, self.TARGET_SR)
        sr = self.TARGET_SR

        duration_s = len(y) / sr
        logger.info("  duration=%.1fs  sr=%d", duration_s, sr)

        # 1. Frame-level features
        features = self._compute_features(y, sr)          # (N_frames, D)

        # 2. Self-similarity matrix + Foote novelty
        ssm = _compute_ssm(features)
        # Smaller kernel (4 frames = 2s each side) catches shorter phrase transitions
        novelty = _foote_novelty(ssm, width=4)

        # 3. BPM-aware minimum distance: 8 bars minimum
        bar_dur_s = (4.0 * 60.0 / bpm) if (bpm and bpm > 0) else 8.0
        min_phrase_s = max(8.0, 8 * bar_dur_s)  # at least 8 bars
        min_dist_frames = max(4, int(min_phrase_s / self.HOP_LEN))

        # 4. SSM novelty peaks (lower thresholds to catch more boundaries)
        peaks, _ = find_peaks(
            novelty,
            height=0.03,
            distance=min_dist_frames,
            prominence=0.02,
        )
        boundary_times_s: list[float] = [p * self.HOP_LEN for p in peaks]

        # 5. Augment with regular 8-bar grid candidates
        if bpm and bpm > 0:
            first_beat_s = self._find_first_beat(y, sr)
            # Add every 8-bar boundary as a candidate
            t = first_beat_s + 8 * bar_dur_s  # start from bar 9
            while t < duration_s - bar_dur_s:
                # Only add if no SSM peak is already within 2 bars
                near = any(abs(t - b) < 2 * bar_dur_s for b in boundary_times_s)
                if not near:
                    boundary_times_s.append(t)
                t += 8 * bar_dur_s

        # 6. Bar-grid alignment (snaps all candidates to nearest 4/8/16-bar)
        if bpm and bpm > 0:
            first_beat_s = self._find_first_beat(y, sr)
            boundary_times_s = _align_to_bars(
                boundary_times_s, first_beat_s, bar_dur_s, duration_s
            )
            logger.info(
                "  BPM=%.1f  first_beat=%.2fs  bar_dur=%.2fs",
                bpm, first_beat_s, bar_dur_s,
            )

        # 7. Per-segment features for labeling
        segment_features = _segment_features(y, sr, boundary_times_s, duration_s)

        # 8. Section labels + output
        cues = _label_sections(
            boundary_times_s, segment_features, duration_s, bpm
        )

        logger.info("  detected %d sections", len(cues))
        for c in cues:
            logger.info("    bar%-3d  %6d ms  %-12s (%s)", c.bar_number, c.time_ms, c.label, c.color)

        return cues

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _compute_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Return (N_frames, 6) feature matrix: RMS, centroid, sub-bass, mid, hi, flatness."""
        frame_samples = int(self.FRAME_LEN * sr)
        hop_samples   = int(self.HOP_LEN  * sr)
        n_fft         = min(frame_samples, 2048)

        feature_list: list[list[float]] = []
        pos = 0
        while pos + frame_samples <= len(y):
            frame   = y[pos : pos + frame_samples]
            win     = frame[:n_fft] * np.hanning(n_fft)
            fft_mag = np.abs(np.fft.rfft(win))
            freqs   = np.fft.rfftfreq(n_fft, d=1.0 / sr)
            fft_sum = fft_mag.sum() + 1e-10

            # RMS (log-scaled)
            rms     = float(np.sqrt(np.mean(frame ** 2))) + 1e-10
            log_rms = float(np.log1p(rms * 100))

            # Spectral centroid (normalised to Nyquist)
            centroid = float(np.sum(freqs * fft_mag) / fft_sum) / (sr / 2.0)

            # Sub-bass (20–250 Hz) — kick + bass
            sub_mask   = (freqs >= 20)  & (freqs <= 250)
            sub_energy = float(fft_mag[sub_mask].sum() / fft_sum)

            # Mid (250 Hz–3 kHz) — melody, vocals, synths
            mid_mask   = (freqs >= 250)  & (freqs <= 3000)
            mid_energy = float(fft_mag[mid_mask].sum() / fft_sum)

            # High (3 kHz–8 kHz) — hi-hats, cymbals, air
            hi_mask    = (freqs >= 3000) & (freqs <= 8000)
            hi_energy  = float(fft_mag[hi_mask].sum() / fft_sum)

            # Spectral flatness (log ratio geometric/arithmetic mean)
            # Low flatness = tonal (melody/bass heavy); high flatness = noise-like
            gm = float(np.exp(np.mean(np.log(fft_mag + 1e-10))))
            am = float(np.mean(fft_mag) + 1e-10)
            flatness = float(np.log1p(gm / am))

            feature_list.append([log_rms, centroid, sub_energy, mid_energy, hi_energy, flatness])
            pos += hop_samples

        return np.array(feature_list, dtype=np.float64)

    # ------------------------------------------------------------------
    # First-beat finder
    # ------------------------------------------------------------------

    @staticmethod
    def _find_first_beat(y: np.ndarray, sr: int, look_s: float = 10.0) -> float:
        """Estimate first beat via spectral flux onset in first look_s seconds."""
        hop   = 512
        n_fft = 2048
        end   = min(int(look_s * sr), len(y) - n_fft)

        onsets: list[tuple[float, float]] = []
        prev   = None
        for i in range(0, end, hop):
            frame  = y[i : i + n_fft] * np.hanning(n_fft)
            spec   = np.abs(np.fft.rfft(frame))
            if prev is not None:
                flux = float(np.maximum(spec - prev, 0).sum())
                onsets.append((i / sr, flux))
            prev = spec

        if not onsets:
            return 0.0

        times, fluxes = zip(*onsets)
        flux_arr = np.array(fluxes)
        threshold = float(np.percentile(flux_arr, 75))

        for t, f in onsets:
            if f >= threshold and t > 0.1:
                return t
        return 0.0


# ---------------------------------------------------------------------------
# Pure functions (helpers)
# ---------------------------------------------------------------------------

def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear resampling (adequate for structural analysis)."""
    ratio   = target_sr / orig_sr
    new_len = int(len(y) * ratio)
    return np.interp(
        np.linspace(0, len(y) - 1, new_len),
        np.arange(len(y)),
        y,
    )


def _compute_ssm(features: np.ndarray) -> np.ndarray:
    """Cosine self-similarity matrix from feature rows."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    feat  = features / norms
    return feat @ feat.T


def _foote_novelty(ssm: np.ndarray, width: int = 4) -> np.ndarray:
    """
    Foote checkerboard-kernel novelty function.
    Smaller width = finer time resolution for phrase-level detection.
    """
    N      = ssm.shape[0]
    kernel = np.ones((2 * width, 2 * width))
    kernel[:width, width:]  = -1
    kernel[width:, :width]  = -1

    novelty = np.zeros(N)
    for i in range(width, N - width):
        block      = ssm[i - width : i + width, i - width : i + width]
        novelty[i] = float(np.sum(block * kernel))

    novelty = np.clip(novelty, 0, None)
    if novelty.max() > 0:
        novelty /= novelty.max()
    novelty = gaussian_filter1d(novelty, sigma=1.5)
    return novelty


def _align_to_bars(
    times_s: list[float],
    first_beat_s: float,
    bar_duration_s: float,
    total_s: float,
    quant_options: tuple[int, ...] = (16, 8, 4),
    tolerance_bars: float = 2.5,
) -> list[float]:
    """Snap each boundary time to the nearest N-bar grid point."""
    aligned: list[float] = []
    for t in times_s:
        bars = (t - first_beat_s) / bar_duration_s
        best: Optional[float] = None
        for q in quant_options:
            rounded  = round(bars / q) * q
            snapped  = first_beat_s + rounded * bar_duration_s
            if abs(snapped - t) <= tolerance_bars * bar_duration_s:
                best = snapped
                break
        if best is None:
            rounded = round(bars / 4) * 4
            best    = first_beat_s + rounded * bar_duration_s
        if 0.5 < best < total_s - 0.5:
            aligned.append(best)

    # Deduplicate and sort
    seen: set[float] = set()
    result: list[float] = []
    for t in sorted(aligned):
        key = round(t, 1)
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result


def _segment_features(
    y: np.ndarray,
    sr: int,
    boundaries_s: list[float],
    total_s: float,
) -> list[dict]:
    """Compute RMS, sub-bass ratio, and hi-freq ratio for each segment."""
    edges = [0.0] + list(boundaries_s) + [total_s]
    results: list[dict] = []
    for a, b in zip(edges, edges[1:]):
        chunk = y[int(a * sr) : int(b * sr)]
        if len(chunk) == 0:
            results.append({"rms": 0.0, "sub": 0.0, "hi": 0.0, "mid": 0.0})
            continue

        rms = float(np.sqrt(np.mean(chunk ** 2)))

        # FFT of the full chunk (downsampled for speed)
        # Use up to 65536 samples for the FFT
        chunk_fft = chunk[:min(len(chunk), 65536)]
        n = len(chunk_fft)
        fft_mag = np.abs(np.fft.rfft(chunk_fft * np.hanning(n)))
        freqs   = np.fft.rfftfreq(n, d=1.0 / sr)
        total   = fft_mag.sum() + 1e-10

        sub = float(fft_mag[(freqs >= 20) & (freqs <= 250)].sum() / total)
        mid = float(fft_mag[(freqs >= 250) & (freqs <= 3000)].sum() / total)
        hi  = float(fft_mag[(freqs >= 3000) & (freqs <= 8000)].sum() / total)

        results.append({"rms": rms, "sub": sub, "hi": hi, "mid": mid})
    return results


def _label_sections(
    boundaries_s: list[float],
    seg_features: list[dict],
    total_s: float,
    bpm: Optional[float],
) -> list[PhraseCue]:
    """
    Assign Rekordbox-style section labels: Intro, Up, Chorus, Down, Outro.

    Strategy
    --------
    - Intro : first segment(s) until energy rises significantly
    - Outro : last segment, and any segments starting after 85% of track
    - Chorus: high RMS **and** high sub-bass — the actual drop/peak
    - Down  : low RMS, low sub-bass — breakdown / rest section
    - Up    : everything else — builds, moderate-energy passages

    Using both RMS and sub-bass together prevents buildup sections from
    being mis-classified as Chorus (they have rising RMS but lower sub-bass
    before the kick fully drops).
    """
    if not seg_features:
        return []

    edges    = [0.0] + list(boundaries_s) + [total_s]
    n        = len(edges) - 1
    rms_vals = np.array([f["rms"] for f in seg_features], dtype=np.float64)
    sub_vals = np.array([f["sub"] for f in seg_features], dtype=np.float64)

    # Normalize 0-1
    rms_norm = rms_vals / (rms_vals.max() + 1e-10)
    sub_norm = sub_vals / (sub_vals.max() + 1e-10)

    # Combined score: Chorus requires BOTH high RMS and high sub-bass
    # (distinguishes drop from long melodic intro / build which may have high RMS
    # but lower sub-bass before the kick fully hits)
    chorus_score = (rms_norm * 0.55) + (sub_norm * 0.45)
    chorus_score_norm = chorus_score / (chorus_score.max() + 1e-10)

    # Thresholds: use top 30% as Chorus, bottom 30% as Down
    # Apply only to middle segments to avoid Intro/Outro pulling the distribution
    mid_scores = chorus_score_norm[1:-1] if n > 2 else chorus_score_norm
    chorus_thresh = float(np.percentile(mid_scores, 68)) if len(mid_scores) else 0.68
    down_thresh   = float(np.percentile(mid_scores, 32)) if len(mid_scores) else 0.32

    bar_dur = (4.0 * 60.0 / bpm) if (bpm and bpm > 0) else 8.0

    # First pass
    raw_labels = []
    for i in range(n):
        start       = edges[i]
        start_ratio = start / total_s
        score       = chorus_score_norm[i]

        if i == 0:
            raw_labels.append("Intro")
        elif start_ratio >= 0.85 or i == n - 1:
            raw_labels.append("Outro")
        elif score >= chorus_thresh:
            raw_labels.append("Chorus")
        elif score <= down_thresh:
            raw_labels.append("Down")
        else:
            raw_labels.append("Up")

    # Second pass: extend Intro forward while energy is still low/moderate
    # (catches long melodic intros where no full kick has dropped yet)
    labels = list(raw_labels)
    i = 1
    while i < n and labels[i] not in ("Chorus", "Outro"):
        # If this segment's combined score is < 55% and it comes right after Intro, keep as Intro
        if chorus_score_norm[i] < 0.55 and labels[i - 1] == "Intro":
            labels[i] = "Intro"
        i += 1

    # Third pass: structural rules to match Rekordbox patterns
    for i in range(1, n - 1):
        prev_l = labels[i - 1]
        next_l = labels[i + 1]
        # Single Up sandwiched between Chorus → Chorus
        if labels[i] == "Up" and prev_l == "Chorus" and next_l == "Chorus":
            labels[i] = "Chorus"
        # Single Down sandwiched between Up → Up
        if labels[i] == "Down" and prev_l == "Up" and next_l == "Up":
            labels[i] = "Up"
        # Single Down right after Intro → Up (quiet build start)
        if labels[i] == "Down" and prev_l in ("Intro", "Up") and next_l == "Up":
            labels[i] = "Up"

    # Fourth pass: enforce UP before every CHORUS
    # Rule A: No Chorus can appear before the first Up section has been seen.
    #         (Ensures the intro/build phase is properly labeled before the drop.)
    first_up_seen = False
    for i in range(n):
        if labels[i] == "Up":
            first_up_seen = True
        elif labels[i] == "Chorus" and not first_up_seen:
            labels[i] = "Up"  # Re-label as Up — still in the build phase

    # Rule B: For runs of Down sections leading directly into Chorus (no Up in between),
    #         flip the last 1-2 Down sections to Up (they are the rebuild / re-entry).
    for i in range(1, n):
        if labels[i] == "Chorus":
            if labels[i - 1] in ("Down",):
                j = i - 1
                flip_count = 0
                while j >= 0 and labels[j] == "Down" and flip_count < 2:
                    labels[j] = "Up"
                    flip_count += 1
                    j -= 1

    # Rule C: After the final Chorus, any remaining Up sections are actually Down
    #         (final breakdown / wind-down before Outro).
    last_chorus = max((i for i in range(n) if labels[i] == "Chorus"), default=-1)
    if last_chorus >= 0:
        for i in range(last_chorus + 1, n):
            if labels[i] == "Up":
                labels[i] = "Down"

    # Build cue list
    cues: list[PhraseCue] = []
    for i in range(n):
        label      = labels[i]
        start      = edges[i]
        color      = SECTION_COLOR.get(label, "none")
        bar_number = max(1, int(start / bar_dur) + 1)
        cues.append(PhraseCue(
            time_ms    = int(start * 1000),
            label      = label,
            color      = color,
            bar_number = bar_number,
        ))

    return cues


# ---------------------------------------------------------------------------
# Rekordbox cue writer
# ---------------------------------------------------------------------------

def write_phrase_cues(
    rb_db,
    content_id: str,
    cues: List[PhraseCue],
    replace_existing: bool = False,
) -> list[str]:
    """
    Write PhraseCue objects as memory cues into the Rekordbox database.
    """
    from pyrekordbox.db6.tables import DjmdCue

    content = rb_db.get_content(ID=content_id)
    if content is None:
        raise ValueError(f"No track found with ID={content_id!r}")
    content_uuid = content.UUID

    if replace_existing:
        existing = rb_db.get_cue(ContentID=content_id).all()
        for row in existing:
            if row.Kind == 0:
                rb_db.session.delete(row)
        rb_db.session.flush()

    created_ids: list[str] = []

    for cue in cues:
        in_msec  = cue.time_ms
        in_frame = int(in_msec * 150 / 1000)

        cue_id   = str(_generate_cue_id(content_id, in_msec))
        cue_uuid = str(uuid.uuid4())

        row = DjmdCue(
            ID               = cue_id,
            UUID             = cue_uuid,
            ContentID        = content_id,
            ContentUUID      = content_uuid,
            InMsec           = in_msec,
            InFrame          = in_frame,
            InMpegFrame      = 0,
            InMpegAbs        = 0,
            OutMsec          = None,
            OutFrame         = None,
            OutMpegFrame     = None,
            OutMpegAbs       = None,
            Kind             = 0,
            Color            = 255,
            ColorTableIndex  = 0,
            ActiveLoop       = 0,
            Comment          = cue.label,
            BeatLoopSize     = 0,
            CueMicrosec      = 0,
            InPointSeekInfo  = "",
            OutPointSeekInfo = "",
            rb_data_status       = 0,
            rb_local_data_status = 0,
            rb_local_deleted     = 0,
            rb_local_synced      = 0,
        )
        rb_db.add(row)
        created_ids.append(cue_id)
        logger.info(
            "  wrote cue  id=%-12s  %6d ms  %-12s",
            cue_id, in_msec, cue.label,
        )

    rb_db.session.commit()
    logger.info("Committed %d phrase cues for content_id=%s", len(created_ids), content_id)
    return created_ids


def _generate_cue_id(content_id: str, time_ms: int) -> int:
    """Deterministic uint32 ID from (content_id, time_ms) via FNV-1a hash."""
    data  = f"phrase:{content_id}:{time_ms}".encode()
    h     = 2_166_136_261
    prime = 16_777_619
    for byte in data:
        h = ((h ^ byte) * prime) & 0xFFFFFFFF
    return max(100_000, h)
