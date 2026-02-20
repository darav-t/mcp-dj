"""
Essentia Audio Analysis Module

Analyzes audio files for BPM accuracy, key detection, danceability,
loudness, mood classification, genre detection, and music autotagging.

Installation:
  pip install essentia-tensorflow

Model files (download once with ./download_models.sh):
  .data/models/          (in repo root, git-ignored)

Cache location: .data/essentia_cache/<sha256_of_filepath>.json
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from .models import EssentiaFeatures

# ---------------------------------------------------------------------------
# Optional Essentia import — degrade gracefully if not installed
# ---------------------------------------------------------------------------

try:
    import essentia
    import essentia.standard as es

    ESSENTIA_AVAILABLE = True
    ESSENTIA_VERSION: Optional[str] = essentia.__version__
except ImportError:
    ESSENTIA_AVAILABLE = False
    ESSENTIA_VERSION = None
    logger.debug("Essentia not installed — audio analysis unavailable (optional feature)")

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

# Both live inside the repo under .data/ (git-ignored — large binaries + personal data)
_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _REPO_ROOT / ".data" / "essentia_cache"
MODEL_DIR  = _REPO_ROOT / ".data" / "models"

# ---------------------------------------------------------------------------
# Key conversion: Essentia (standard notation) → Camelot wheel
# ---------------------------------------------------------------------------

# Standard pitch class → Camelot number (minor = A, major = B)
_KEY_TO_CAMELOT_NUMBER: dict[str, int] = {
    "B": 1, "F#": 2, "Gb": 2, "Db": 3, "C#": 3,
    "Ab": 4, "G#": 4, "Eb": 5, "D#": 5, "Bb": 6,
    "A#": 6, "F": 7, "C": 8, "G": 9, "D": 10,
    "A": 11, "E": 12,
}


def _essentia_key_to_camelot(key_name: str, scale: str) -> Optional[str]:
    """Convert Essentia key output to Camelot notation ('8A', '5B', etc.).

    Args:
        key_name: Key name from Essentia e.g. 'C', 'F#', 'Gb'
        scale: 'major' or 'minor'

    Returns:
        Camelot key string e.g. '8A', '5B', or None if key_name unknown
    """
    num = _KEY_TO_CAMELOT_NUMBER.get(key_name)
    if num is None:
        return None
    letter = "A" if scale == "minor" else "B"
    return f"{num}{letter}"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(file_path: str) -> Path:
    return CACHE_DIR / f"{Path(file_path).stem}.json"


def load_cached_features(file_path: str) -> Optional[EssentiaFeatures]:
    """Load cached Essentia features without triggering analysis.

    Supports the compact cache format written by _write_cache.

    Returns:
        EssentiaFeatures if a valid cache entry exists, else None.
    """
    import json
    cache_file = _cache_path(file_path)
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text())
        # Map compact keys back to EssentiaFeatures fields
        mood = data.get("mood") or {}
        tags_dict = data.get("tags")
        music_tags = (
            [{"tag": t, "score": s} for t, s in tags_dict.items()]
            if tags_dict else None
        )
        key_note = data.get("key_note", "")
        parts = key_note.split(" ", 1) if key_note else []
        return EssentiaFeatures(
            file_path=data.get("file_path", file_path),
            bpm_essentia=data.get("bpm", 0.0),
            key_essentia=data.get("key"),
            key_name_raw=parts[0] if parts else None,
            key_scale=parts[1] if len(parts) > 1 else None,
            key_strength=data.get("key_strength", 0.0),
            danceability=data.get("danceability", 0.0),
            integrated_lufs=data.get("lufs", 0.0),
            mood_happy=mood.get("happy"),
            mood_sad=mood.get("sad"),
            mood_aggressive=mood.get("aggressive"),
            mood_relaxed=mood.get("relaxed"),
            mood_party=mood.get("party"),
            genre_discogs=data.get("genre"),
            music_tags=music_tags,
        )
    except Exception as e:
        logger.warning(f"Failed to load essentia cache for {file_path}: {e}")
        return None


def _write_cache(features: EssentiaFeatures, cache_file: Path) -> None:
    """Write compact AI-readable features JSON to disk."""
    import json
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        cache_file.write_text(json.dumps(features.to_cache_dict(), indent=2))
        logger.debug(f"Essentia cache written: {cache_file}")
    except OSError as e:
        logger.warning(f"Failed to write essentia cache: {e}")


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _run_analysis(file_path: str) -> EssentiaFeatures:
    """Execute Essentia algorithms on the audio file.

    Must only be called when ESSENTIA_AVAILABLE is True.
    """
    # Load as mono 44100 Hz
    loader = es.MonoLoader(filename=file_path, sampleRate=44100)
    audio = loader()

    # 1. BPM / Rhythm
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

    # 2. Key detection
    key_extractor = es.KeyExtractor()
    key_name, scale, key_strength = key_extractor(audio)
    camelot_key = _essentia_key_to_camelot(key_name, scale)

    # 3. Danceability
    # dfa is a vector_real (array of DFA exponents per segment) — take the mean
    danceability_algo = es.Danceability(sampleRate=44100)
    danceability, dfa_array = danceability_algo(audio)
    dfa = float(dfa_array.mean()) if len(dfa_array) > 0 else 0.0

    # 4. Loudness (EBU R128) — needs stereo signal; duplicate mono channel
    integrated_lufs = 0.0
    loudness_range = 0.0
    try:
        import numpy as np
        stereo_audio = np.column_stack([audio, audio])
        loudness_algo = es.LoudnessEBUR128(sampleRate=44100)
        _momentary, _short_term, integrated_lufs, loudness_range = loudness_algo(stereo_audio)
        integrated_lufs = float(integrated_lufs)
        loudness_range = float(loudness_range)
    except Exception as e:
        logger.debug(f"LoudnessEBUR128 failed ({e}), falling back to es.Loudness")
        try:
            loudness_db = es.Loudness()(audio)
            integrated_lufs = float(loudness_db)
        except Exception as e2:
            logger.warning(f"Loudness fallback also failed: {e2}")

    # 5. RMS Energy (linear) + RMS in dBFS
    try:
        import numpy as np
        rms_linear = float(np.sqrt(np.mean(audio ** 2)))
        rms_db = float(20.0 * np.log10(max(rms_linear, 1e-10)))
    except Exception as e:
        logger.debug(f"RMS energy calculation failed: {e}")
        rms_linear = 0.0
        rms_db = 0.0

    return EssentiaFeatures(
        file_path=file_path,
        analyzed_at=datetime.now(timezone.utc).isoformat(),
        essentia_version=ESSENTIA_VERSION,
        bpm_essentia=float(bpm),
        bpm_confidence=float(beats_confidence),
        beats_count=int(len(beats)),
        key_essentia=camelot_key,
        key_name_raw=key_name,
        key_scale=scale,
        key_strength=float(key_strength),
        danceability=float(danceability),
        dfa=dfa,
        integrated_lufs=integrated_lufs,
        loudness_range_db=loudness_range,
        rms_db=rms_db,
        rms_energy=rms_linear,
    )


# ---------------------------------------------------------------------------
# ML model inference (mood, genre, autotagging)
# Requires model files in MODEL_DIR — download with ./download_models.sh
# ---------------------------------------------------------------------------

def _model_path(filename: str) -> Optional[Path]:
    """Return model path if it exists, else None."""
    p = MODEL_DIR / filename
    return p if p.exists() else None


def _run_ml_analysis(file_path: str, features: EssentiaFeatures) -> EssentiaFeatures:
    """Run TensorFlow-based ML models on the audio file.

    Requires essentia-tensorflow or essentia with TF support.
    Gracefully skips any model whose files aren't downloaded.
    Returns the features object updated in-place.
    """
    import numpy as np

    # Load audio at 16kHz (required by all ML models)
    try:
        audio_16k = es.MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()
    except Exception as e:
        logger.warning(f"ML analysis: failed to load audio at 16kHz: {e}")
        return features

    # -------------------------------------------------------------------------
    # VGGish embeddings — shared by mood classifiers + MagnaTagATune
    # -------------------------------------------------------------------------
    vggish_pb   = _model_path("audioset-vggish-3.pb")
    vggish_json = _model_path("audioset-vggish-3.json")
    vggish_embeddings = None

    if vggish_pb and vggish_json:
        try:
            vggish_model = es.TensorflowPredictVGGish(
                graphFilename=str(vggish_pb),
                output="model/vggish/embeddings",
            )
            vggish_embeddings = vggish_model(audio_16k)
            logger.debug(f"VGGish embeddings: {vggish_embeddings.shape}")
        except Exception as e:
            logger.debug(f"VGGish embedding failed: {e}")
    else:
        logger.debug("VGGish model not found — skipping mood + tagging. Run ./download_models.sh")

    # -------------------------------------------------------------------------
    # Mood classifiers (each is a 2-class softmax on top of VGGish)
    # -------------------------------------------------------------------------
    if vggish_embeddings is not None:
        mood_models = {
            "mood_happy":      ("mood_happy-audioset-vggish-1.pb",      "mood_happy"),
            "mood_sad":        ("mood_sad-audioset-vggish-1.pb",        "mood_sad"),
            "mood_aggressive": ("mood_aggressive-audioset-vggish-1.pb", "mood_aggressive"),
            "mood_relaxed":    ("mood_relaxed-audioset-vggish-1.pb",    "mood_relaxed"),
            "mood_party":      ("mood_party-audioset-vggish-1.pb",      "mood_party"),
        }
        for field_name, (pb_file, _) in mood_models.items():
            pb = _model_path(pb_file)
            if not pb:
                continue
            try:
                classifier = es.TensorflowPredict2D(
                    graphFilename=str(pb),
                    output="model/Softmax",
                    patchSize=1,
                    batchSize=-1,
                )
                preds = classifier(vggish_embeddings)
                # preds shape: (n_frames, 2) — class 0 is the positive mood
                score = float(preds.mean(axis=0)[0])
                setattr(features, field_name, round(score, 4))
                logger.debug(f"{field_name}: {score:.3f}")
            except Exception as e:
                logger.debug(f"{field_name} inference failed: {e}")

    # -------------------------------------------------------------------------
    # EffNet embeddings — computed once, shared by tagging + genre
    # TensorflowPredictEffnetDiscogs takes raw 16kHz audio directly
    # and uses output node "PartitionedCall" (default) for embeddings
    # -------------------------------------------------------------------------
    effnet_pb = _model_path("discogs-effnet-bs64-1.pb")
    effnet_embeddings = None

    if effnet_pb:
        try:
            effnet_model = es.TensorflowPredictEffnetDiscogs(
                graphFilename=str(effnet_pb),
                output="PartitionedCall:1",  # :1 = 1280-dim embeddings, :0 = 400-class predictions
            )
            effnet_embeddings = effnet_model(audio_16k)
            logger.debug(f"EffNet embeddings: {effnet_embeddings.shape}")
        except Exception as e:
            logger.debug(f"EffNet embedding failed: {e}")
    else:
        logger.debug("EffNet model not found — skipping genre + tagging. Run ./download_models.sh")

    # -------------------------------------------------------------------------
    # Music autotagging — MagnaTagATune (mtt) via EffNet embeddings
    # -------------------------------------------------------------------------
    tagging_pb   = _model_path("mtt-discogs-effnet-1.pb")
    tagging_json = _model_path("mtt-discogs-effnet-1.json")

    if effnet_embeddings is not None and tagging_pb and tagging_json:
        try:
            import json as _json
            with open(tagging_json) as f:
                tag_meta = _json.load(f)
            tag_classes = tag_meta.get("classes", [])

            tag_classifier = es.TensorflowPredict2D(
                graphFilename=str(tagging_pb),
                output="model/Sigmoid",
                patchSize=1,
                batchSize=-1,
            )
            tag_preds = tag_classifier(effnet_embeddings)
            avg_tag_preds = tag_preds.mean(axis=0)

            tags = [
                {"tag": tag_classes[i], "score": round(float(avg_tag_preds[i]), 4)}
                for i in range(len(tag_classes))
                if avg_tag_preds[i] >= 0.1
            ]
            tags.sort(key=lambda x: x["score"], reverse=True)
            features.music_tags = tags if tags else None
            logger.debug(f"Music tags: {[t['tag'] for t in (tags[:5] if tags else [])]}")
        except Exception as e:
            logger.debug(f"Music autotagging failed: {e}")
    elif not (tagging_pb and tagging_json):
        logger.debug("MTT models not found — skipping autotagging. Run ./download_models.sh")

    # -------------------------------------------------------------------------
    # Genre classification — Discogs400 via EffNet embeddings
    # -------------------------------------------------------------------------
    genre_pb   = _model_path("genre_discogs400-discogs-effnet-1.pb")
    genre_json = _model_path("genre_discogs400-discogs-effnet-1.json")

    if effnet_embeddings is not None and genre_pb and genre_json:
        try:
            import json as _json
            with open(genre_json) as f:
                genre_meta = _json.load(f)
            genre_classes = genre_meta.get("classes", [])

            genre_classifier = es.TensorflowPredict2D(
                graphFilename=str(genre_pb),
                input="serving_default_model_Placeholder",
                output="PartitionedCall",
                patchSize=1,
                batchSize=-1,
            )
            genre_preds = genre_classifier(effnet_embeddings)
            avg_genre_preds = genre_preds.mean(axis=0)

            top_indices = np.argsort(avg_genre_preds)[::-1][:10]
            genre_dict = {
                genre_classes[i]: round(float(avg_genre_preds[i]), 4)
                for i in top_indices
                if avg_genre_preds[i] >= 0.01
            }
            features.genre_discogs = genre_dict if genre_dict else None
            logger.debug(f"Top genre: {list(genre_dict.keys())[:3] if genre_dict else 'none'}")
        except Exception as e:
            logger.debug(f"Genre classification failed: {e}")
    elif not (genre_pb and genre_json):
        logger.debug("Discogs400 genre model not found — skipping. Run ./download_models.sh")

    return features


def analyze_file(file_path: str, force: bool = False) -> EssentiaFeatures:
    """Analyze an audio file and return EssentiaFeatures.

    Checks the disk cache first. Writes results to cache after analysis.

    Args:
        file_path: Path to audio file (.mp3, .wav, .flac, .aiff, .m4a).
        force: If True, re-analyze even if a cache entry exists.

    Returns:
        EssentiaFeatures populated with analysis results.

    Raises:
        RuntimeError: If essentia is not installed.
        FileNotFoundError: If the audio file does not exist.
    """
    if not ESSENTIA_AVAILABLE:
        raise RuntimeError(
            "Essentia is not installed. Install with: pip install essentia"
        )

    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Check cache first
    cache_file = _cache_path(str(path))
    if not force and cache_file.exists():
        try:
            cached = EssentiaFeatures.model_validate_json(cache_file.read_text())
            logger.debug(f"Loaded from cache: {file_path}")
            return cached
        except Exception as e:
            logger.warning(f"Cache read failed for {file_path}, re-analyzing: {e}")

    # Run signal analysis
    t0 = time.monotonic()
    logger.info(f"Analyzing: {path.name}")
    features = _run_analysis(str(path))

    # Run ML model inference (mood, genre, tags) — skipped if models not downloaded
    features = _run_ml_analysis(str(path), features)

    features.analysis_duration_seconds = round(time.monotonic() - t0, 2)
    logger.info(
        f"Done ({features.analysis_duration_seconds}s): "
        f"BPM={features.bpm_essentia:.1f} key={features.key_essentia} "
        f"dance={features.danceability:.2f} lufs={features.integrated_lufs:.1f} "
        f"mood={features.dominant_mood() or 'n/a'} genre={features.top_genre() or 'n/a'}"
    )

    _write_cache(features, cache_file)
    return features


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

def analyze_library(
    tracks: list,
    force: bool = False,
    skip_missing: bool = True,
) -> dict[str, Any]:
    """Batch-analyze all tracks that have a file_path.

    Args:
        tracks: List of Track or TrackWithEnergy objects.
        force: Re-analyze even if cached results exist.
        skip_missing: Log a warning for missing files instead of raising.

    Returns:
        Dict with keys: analyzed, cached, skipped_no_path, skipped_missing_file,
        errors, results (dict[track_id -> EssentiaFeatures]).
    """
    results: dict[str, EssentiaFeatures] = {}
    stats: dict[str, Any] = {
        "analyzed": 0,
        "cached": 0,
        "skipped_no_path": 0,
        "skipped_missing_file": 0,
        "errors": 0,
        "results": results,
    }

    for track in tracks:
        fp = getattr(track, "file_path", None)
        if not fp:
            stats["skipped_no_path"] += 1
            continue

        # Check cache without running analysis
        cache_file = _cache_path(fp)
        if not force and cache_file.exists():
            cached = load_cached_features(fp)
            if cached:
                results[track.id] = cached
                stats["cached"] += 1
                continue

        if not Path(fp).exists():
            if skip_missing:
                logger.warning(f"Audio file not found, skipping: {fp}")
                stats["skipped_missing_file"] += 1
                continue
            else:
                raise FileNotFoundError(fp)

        try:
            features = analyze_file(fp, force=force)
            results[track.id] = features
            stats["analyzed"] += 1
        except Exception as e:
            logger.error(f"Analysis failed for {fp}: {e}")
            stats["errors"] += 1

    return stats


# ---------------------------------------------------------------------------
# EssentiaFeatureStore — in-memory index for SetlistEngine integration
# ---------------------------------------------------------------------------

class EssentiaFeatureStore:
    """In-memory index of EssentiaFeatures loaded from disk cache.

    Used by SetlistEngine to look up audio features during scoring
    without triggering analysis at generation time.

    Usage:
        store = EssentiaFeatureStore(tracks)
        features = store.get(track.file_path)  # None if not cached
    """

    def __init__(self, tracks: list) -> None:
        self._store: dict[str, EssentiaFeatures] = {}
        loaded = 0
        for track in tracks:
            fp = getattr(track, "file_path", None)
            if fp:
                cached = load_cached_features(fp)
                if cached:
                    self._store[fp] = cached
                    loaded += 1
        logger.debug(f"EssentiaFeatureStore: loaded {loaded} cached entries")

    def get(self, file_path: Optional[str]) -> Optional[EssentiaFeatures]:
        """Return cached features for a file path, or None."""
        if not file_path:
            return None
        return self._store.get(file_path)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"EssentiaFeatureStore({len(self._store)} tracks)"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """CLI entry point for analyzing a single audio file.

    Usage:
        python -m mcp_dj.analyze_track /path/to/song.mp3
        python -m mcp_dj.analyze_track /path/to/song.mp3 --force
        python -m mcp_dj.analyze_track /path/to/song.mp3 --output json
        analyze-track /path/to/song.mp3  (if installed via uv)
    """
    parser = argparse.ArgumentParser(
        prog="analyze-track",
        description="Analyze an audio file with Essentia and cache the results.",
    )
    parser.add_argument(
        "file_path",
        nargs="+",
        help="Path to audio file (.mp3, .wav, .flac, .aiff, .m4a). Wrap in quotes if it contains spaces.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-analyze even if a cached result already exists",
    )
    parser.add_argument(
        "--output", choices=["pretty", "json"], default="pretty",
        help="Output format (default: pretty)",
    )
    args = parser.parse_args()

    # Rejoin in case the path contained spaces and wasn't quoted
    file_path = " ".join(args.file_path)

    if not ESSENTIA_AVAILABLE:
        print(
            "ERROR: Essentia is not installed.\n"
            "\n"
            "  pip install essentia\n",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        features = analyze_file(file_path, force=args.force)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR during analysis: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output == "json":
        print(features.model_dump_json(indent=2))
    else:
        print()
        print("Essentia Analysis Results")
        print("=" * 50)
        print(f"  File:           {features.file_path}")
        print(f"  Analyzed at:    {features.analyzed_at}")
        print(f"  Essentia:       {features.essentia_version or 'unknown'}")
        print(f"  Analysis time:  {features.analysis_duration_seconds}s")
        print()
        print(f"  BPM:            {features.bpm_essentia:.2f}  "
              f"(confidence: {features.bpm_confidence:.2f}, beats: {features.beats_count})")
        key_display = features.key_essentia or "N/A"
        raw_display = (
            f"{features.key_name_raw} {features.key_scale}" if features.key_name_raw else "N/A"
        )
        print(f"  Key (Camelot):  {key_display}  ({raw_display}, strength: {features.key_strength:.2f})")
        print(f"  Danceability:   {features.danceability:.3f}  "
              f"(1-10 scale: {features.danceability_as_1_to_10()})")
        print(f"  Loudness:       {features.integrated_lufs:.1f} LUFS  "
              f"(range: {features.loudness_range_db:.1f} dB, "
              f"RMS: {features.rms_db:.1f} dBFS, "
              f"1-10 scale: {features.energy_as_1_to_10()})")
        print(f"  RMS Energy:     {features.rms_energy:.4f}")
        print()

        # Mood classifiers
        moods = {
            "Happy":      features.mood_happy,
            "Sad":        features.mood_sad,
            "Aggressive": features.mood_aggressive,
            "Relaxed":    features.mood_relaxed,
            "Party":      features.mood_party,
        }
        if any(v is not None for v in moods.values()):
            print("  Mood:")
            for label, score in moods.items():
                if score is not None:
                    bar = "█" * int(score * 20)
                    print(f"    {label:<12}  {score:.3f}  {bar}")
            dominant = features.dominant_mood()
            if dominant:
                print(f"    → dominant: {dominant}")
            print()

        # Genre
        if features.genre_discogs:
            print("  Genre (Discogs400):")
            for genre, score in list(features.genre_discogs.items())[:5]:
                bar = "█" * int(score * 30)
                print(f"    {genre:<30}  {score:.3f}  {bar}")
            print()

        # Music tags
        if features.music_tags:
            tags_display = ", ".join(
                f"{t['tag']} ({t['score']:.2f})" for t in features.music_tags[:8]
            )
            print(f"  Tags:           {tags_display}")
            print()

        if not any([
            any(v is not None for v in moods.values()),
            features.genre_discogs,
            features.music_tags,
        ]):
            print("  ML models:      not downloaded — run ./download_models.sh")
            print()

        print(f"  Cache:          {_cache_path(file_path)}")
        print()
