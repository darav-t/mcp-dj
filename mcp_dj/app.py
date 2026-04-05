"""
FastAPI Web Application for MCP DJ

Endpoints:
  GET  /                               - Serve the chat UI
  GET  /api/library/stats              - Library statistics
  GET  /api/library/attributes         - Full dynamic library attribute summary
  GET  /api/library/tracks             - Search/list tracks (supports my_tag, date filters)
  GET  /api/library/track/{id}         - Full merged record for a single track
  POST /api/library/rebuild-index      - Rebuild the centralized JSONL library index

  POST /api/chat                       - Chat with AI
  POST /api/chat/clear                 - Clear conversation history

  POST /api/setlist/generate           - Direct setlist generation (structured params)
  POST /api/setlist/plan               - Vibe-based setlist (freeform context → params)
  POST /api/setlist/build              - Prompt-driven setlist (natural language + MyTags)
  GET  /api/setlist/{id}               - Retrieve a generated setlist
  POST /api/setlist/recommend          - Next-track recommendations
  POST /api/setlist/compatibility      - Harmonic compatibility between two tracks
  POST /api/setlist/energy-flow        - Energy-arc analysis for a track sequence
  POST /api/setlist/compatible-tracks  - Find tracks compatible with a given key/BPM

  POST /api/essentia/analyze           - Analyze a single audio file with Essentia
  POST /api/essentia/analyze-library   - Batch-analyze the full library with Essentia

  POST /api/phrases/analyze-library    - Batch phrase detection (section structure + mix profiles)
  GET  /api/track/{id}/phrases         - Get cached phrase data + mix profile for a track
  POST /api/track/{id}/analyze-phrases - Detect phrases on a single track, write Rekordbox cues
"""

import asyncio
import math
import os
import time as _time_module
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from .database import RekordboxDatabase
from .energy import MixedInKeyLibrary, EnergyResolver
from .camelot import CamelotWheel
from .energy_planner import EnergyPlanner, ENERGY_PROFILES
from .setlist_engine import SetlistEngine
from .ai_integration import SetlistAI
from .models import SetlistRequest
from .library_index import LibraryIndex, LibraryIndexFeatureStore
from .intent import parse_set_intent, interpret_vibe
from .phrase_store import PhraseStore, analyze_library_phrases

# Optional Essentia integration
try:
    from .essentia_analyzer import (
        analyze_file as _essentia_analyze_file,
        analyze_library as _essentia_analyze_library,
        EssentiaFeatureStore,
        ESSENTIA_AVAILABLE,
        CACHE_DIR as ESSENTIA_CACHE_DIR,
    )
except ImportError:
    ESSENTIA_AVAILABLE = False
    _essentia_analyze_file = None
    _essentia_analyze_library = None
    EssentiaFeatureStore = None
    ESSENTIA_CACHE_DIR = None

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

db = RekordboxDatabase()
energy_resolver = EnergyResolver()
camelot = CamelotWheel()
planner = EnergyPlanner()
phrase_store = PhraseStore()
engine = SetlistEngine(camelot=camelot, energy_planner=planner, phrase_store=phrase_store)
ai: Optional[SetlistAI] = None
library_index: Optional[LibraryIndex] = None
library_attributes: Optional[Dict[str, Any]] = None   # dynamic — scanned at build time
_mik_library_app: Optional[MixedInKeyLibrary] = None

STATIC_DIR = Path(__file__).parent / "static"
_MAP_CACHE_PATH = Path(__file__).resolve().parent.parent / ".data" / "library_map.json"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global ai, energy_resolver, library_index, library_attributes, _mik_library_app

    # Startup
    await db.connect()

    # Load Mixed In Key energy data (optional — requires MIK_CSV_PATH env var)
    mik = MixedInKeyLibrary.from_env()
    if mik is not None:
        mik.load()
    _mik_library_app = mik
    energy_resolver = EnergyResolver(mik)

    # Load all tracks and resolve energy
    all_tracks = await db.get_all_tracks()
    energy_resolver.resolve_all(all_tracks)

    # Initialize engine
    engine.initialize(all_tracks)

    # Build the centralized library index (merged JSONL for LLM grep + vector search)
    essentia_store_app = None
    if EssentiaFeatureStore is not None:
        essentia_store_app = EssentiaFeatureStore(all_tracks)

    # Fetch My Tag hierarchy for dynamic attribute building (no hardcoded values)
    try:
        my_tag_tree_app = await db.query_my_tags(limit=500)
    except Exception:
        my_tag_tree_app = []

    library_index = LibraryIndex()
    if library_index.is_fresh(max_age_seconds=3600):
        count = library_index.load_from_disk()
        library_attributes = library_index.attributes
        logger.info(f"Library index loaded from disk: {count} records")
    else:
        stats = library_index.build(
            tracks=all_tracks,
            essentia_store=essentia_store_app,
            mik_library=mik,
            my_tag_tree=my_tag_tree_app,
        )
        library_attributes = library_index.attributes
        logger.info(
            f"Library index built: {stats['total']} tracks "
            f"({stats['with_essentia']} with Essentia, {stats['with_mik']} with MIK)"
        )

    # Attach library index to phrase store so phrase data is stored in the JSONL
    phrase_store.attach_index(library_index)

    # Initialize AI — pass library context so Claude can use MyTag-aware set building
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    ai = SetlistAI(
        engine=engine,
        api_key=api_key,
        library_index=library_index,
        library_attributes=library_attributes,
        mik_library=mik,
    )
    if api_key:
        logger.info("Claude API key found. AI chat enabled.")
    else:
        logger.warning("No ANTHROPIC_API_KEY set. Using fallback mode (no AI chat).")

    logger.info(f"Setlist Creator ready. {len(all_tracks)} tracks loaded.")

    yield

    # Shutdown
    await db.disconnect()


app = FastAPI(title="MCP DJ", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes — UI
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Routes — Library
# ---------------------------------------------------------------------------

@app.get("/api/library/stats")
async def library_stats():
    """Library summary stats."""
    summary = engine.get_library_summary()
    summary["energy_profiles"] = {
        k: v["description"] for k, v in ENERGY_PROFILES.items()
    }
    return JSONResponse(summary)


@app.get("/api/library/attributes")
async def get_library_attributes():
    """Full dynamic library attribute summary (tags, genres, BPM/energy distributions, co-occurrence).

    Attributes are built from actual Rekordbox + Essentia + MIK data at startup
    and refreshed whenever the library index is rebuilt.  No values are hardcoded.
    """
    if library_attributes is None:
        raise HTTPException(
            status_code=503,
            detail="Library attributes not available — rebuild the index first.",
        )
    return JSONResponse(library_attributes)


@app.get("/api/library/tracks")
async def library_tracks(
    search: Optional[str] = None,
    my_tag: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 100,
):
    """Search/list tracks from the library.

    Query params:
        search:    Text search (title, artist, genre, album).
        my_tag:    Filter by Rekordbox My Tag label (e.g. 'High Energy').
        date_from: Include only tracks added on/after this date (YYYY-MM-DD).
        date_to:   Include only tracks added on/before this date (YYYY-MM-DD).
        limit:     Maximum results (default 100, capped at 10000).
    """
    q = (search or "").strip().lower()
    tag_filter = (my_tag or "").strip().lower()
    limit = min(limit, 10000)
    results = []

    for t in engine.tracks:
        # Text filter
        if q and not (
            q in t.title.lower()
            or q in t.artist.lower()
            or q in (t.genre or "").lower()
            or q in (t.album or "").lower()
        ):
            continue

        # Date filter (lexicographic comparison works for YYYY-MM-DD strings)
        d = t.date_added or ""
        if date_from and d < date_from:
            continue
        if date_to and d > date_to:
            continue

        # My Tag filter
        if tag_filter and not any(tag_filter in tag.lower() for tag in t.my_tags):
            continue

        results.append({
            "id": t.id,
            "title": t.title,
            "artist": t.artist,
            "genre": t.genre,
            "bpm": t.bpm,
            "key": t.key,
            "energy": t.energy,
            "energy_source": t.energy_source,
            "rating": t.rating,
            "play_count": t.play_count,
            "length": t.length,
            "duration": t.duration_formatted(),
            "color": t.color,
            "date_added": t.date_added,
            "my_tags": t.my_tags,
        })
        if len(results) >= limit:
            break

    return JSONResponse(results)


@app.get("/api/library/track/{track_id}")
async def get_track_full_info(track_id: str):
    """Return the complete merged record for a track from the library index.

    The record contains all Rekordbox metadata, Essentia audio features, and
    Mixed In Key energy data in one structure.

    Path param:
        track_id: Rekordbox track ID (exact) or title substring (case-insensitive).
    """
    if library_index is None:
        raise HTTPException(status_code=503, detail="Library index not initialized")

    # Try exact ID lookup first
    record = library_index.get_by_id(track_id)
    if record:
        return JSONResponse(record)

    # Fall back to title substring search
    results = library_index.search(query=track_id, limit=1)
    if results:
        return JSONResponse(results[0])

    raise HTTPException(status_code=404, detail=f"Track not found: '{track_id}'")


@app.post("/api/library/rebuild-index")
async def rebuild_library_index_endpoint(force: bool = False):
    """Rebuild the centralized JSONL library index on demand.

    Query params:
        force: If true, rebuild even if the index is fresh (< 1 hour old).
    """
    global library_attributes

    if library_index is None:
        raise HTTPException(status_code=503, detail="Library index not initialized")

    if not force and library_index.is_fresh(max_age_seconds=3600):
        return JSONResponse({
            "skipped": True,
            "reason": "Index is fresh (< 1 hour old). Use ?force=true to rebuild.",
            "index_path": str(library_index._record_path),
        })

    essentia_store_rebuild = None
    if EssentiaFeatureStore is not None:
        essentia_store_rebuild = EssentiaFeatureStore(engine.tracks)

    try:
        tag_tree_rebuild = await db.query_my_tags(limit=500)
    except Exception:
        tag_tree_rebuild = []

    stats = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: library_index.build(
            tracks=engine.tracks,
            essentia_store=essentia_store_rebuild,
            mik_library=_mik_library_app,
            my_tag_tree=tag_tree_rebuild,
        ),
    )
    library_attributes = library_index.attributes

    # Refresh engine's essentia store so live set-building immediately uses updated data
    if essentia_store_rebuild is not None:
        engine.essentia_store = essentia_store_rebuild

    # Propagate fresh library context to the AI assistant
    if ai is not None:
        ai.update_library_context(
            library_index=library_index,
            library_attributes=library_attributes,
            mik_library=_mik_library_app,
        )

    return JSONResponse(stats)


# ---------------------------------------------------------------------------
# Routes — Library Map (2D PCA embedding)
# ---------------------------------------------------------------------------

def _camelot_key_to_number(key_str: str) -> int:
    """Convert a Camelot key string ('8A', '12B', …) to an integer 1–24."""
    if not key_str:
        return 1
    s = str(key_str).strip()
    try:
        num    = int(s[:-1])
        letter = s[-1].upper()
        return num if letter == "A" else num + 12
    except (ValueError, IndexError):
        return 1


def _compute_track_map(records: list) -> list:
    """Build a 2-D PCA embedding of library tracks from metadata features.

    Feature vector (weighted):
      • BPM (×2.5)         – most important dimension for DJ mixing
      • Energy (×3.0)      – highest weight; drives the arc
      • Camelot key sin/cos (×1.5) – circular encoding preserves wheel topology
      • Essentia mood: happy, sad, aggressive, relaxed, party (×0.8–1.5)
      • Danceability (×0.8)
      • Genre one-hot top-25 (×2.0)  – creates clear genre clusters

    Returns a list of dicts: track metadata + ``x`` and ``y`` in [-1, 1].
    """
    try:
        import numpy as np
    except ImportError:
        import random; random.seed(42)
        return [
            {**{k: r.get(k) for k in ("id","title","artist","genre","bpm","key","energy","color","my_tags","rating")},
             "x": random.uniform(-1, 1), "y": random.uniform(-1, 1)}
            for r in records
        ]

    # ── 1. Collect top genres for one-hot encoding ────────────────────────────
    genre_counts: dict[str, int] = {}
    for rec in records:
        g = (rec.get("genre") or "").strip()
        if g:
            genre_counts[g] = genre_counts.get(g, 0) + 1

    top_genres = sorted(genre_counts, key=lambda x: -genre_counts[x])[:25]
    genre_to_idx = {g: i for i, g in enumerate(top_genres)}
    n_genres = len(top_genres)

    # ── 2. Build feature rows ─────────────────────────────────────────────────
    rows: list = []
    valid: list = []

    for rec in records:
        bpm    = float(rec.get("bpm") or 0)
        energy = float(rec.get("energy") or 5)

        # Camelot circular key encoding
        kn    = _camelot_key_to_number(rec.get("key") or "1A")
        angle = (kn - 1) / 12.0 * math.pi
        key_sin = math.sin(angle)
        key_cos = math.cos(angle)

        # Essentia mood
        ess        = rec.get("essentia") or {}
        mood       = ess.get("mood") or {}
        happy      = float(mood.get("happy",      0.5))
        sad        = float(mood.get("sad",        0.5))
        aggressive = float(mood.get("aggressive", 0.5))
        relaxed    = float(mood.get("relaxed",    0.5))
        party      = float(mood.get("party",      0.5))
        dance      = float(ess.get("danceability") or 5) / 10.0

        # Genre one-hot
        genre = (rec.get("genre") or "").strip()
        g_vec = [0.0] * n_genres
        if genre in genre_to_idx:
            g_vec[genre_to_idx[genre]] = 1.0

        feat = [
            min(bpm / 180.0, 1.0) * 2.5,   # BPM
            energy / 10.0 * 3.0,            # Energy (highest weight)
            key_sin * 1.5,                  # Key – sin
            key_cos * 1.5,                  # Key – cos
            happy      * 1.0,
            sad        * 0.8,
            aggressive * 1.5,
            relaxed    * 1.0,
            party      * 1.2,
            dance      * 0.8,
        ] + [v * 2.0 for v in g_vec]

        rows.append(feat)
        valid.append(rec)

    if not rows:
        return []

    X = np.array(rows, dtype=np.float64)

    # ── 3. Standardise ────────────────────────────────────────────────────────
    mu    = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    X_std = (X - mu) / sigma

    # ── 4. PCA via full SVD (top 2 components) ────────────────────────────────
    _, _, Vt   = np.linalg.svd(X_std, full_matrices=False)
    coords     = X_std @ Vt[:2].T           # (n, 2)

    # ── 5. Centre at mean, scale by 3×std so the density cluster fills the view;
    #       clip to [-1, 1] so outliers don't push the cloud off-screen.
    for dim in range(2):
        col  = coords[:, dim]
        mean = float(col.mean())
        std  = float(col.std())
        if std > 1e-8:
            coords[:, dim] = (col - mean) / (3.0 * std)
    coords = np.clip(coords, -1.0, 1.0)

    # ── 6. Assemble output ────────────────────────────────────────────────────
    result = []
    for i, rec in enumerate(valid):
        result.append({
            "id":      rec.get("id"),
            "title":   rec.get("title"),
            "artist":  rec.get("artist"),
            "genre":   rec.get("genre") or "",
            "bpm":     rec.get("bpm"),
            "key":     rec.get("key"),
            "energy":  rec.get("energy"),
            "color":   rec.get("color") or "none",
            "my_tags": rec.get("my_tags") or [],
            "rating":  rec.get("rating") or 0,
            "x":       float(coords[i, 0]),
            "y":       float(coords[i, 1]),
        })

    return result


@app.get("/api/library/map")
async def library_map(force: bool = False):
    """2D PCA embedding of all library tracks for scatter-plot visualization.

    Encodes BPM, energy, Camelot key (circular sin/cos), Essentia mood
    (happy/sad/aggressive/relaxed/party), danceability, and genre (one-hot
    top-25) into a feature matrix, then projects to 2-D via PCA.

    Results are cached to ``.data/library_map.json`` and reused for 1 hour.

    Query params:
        force: Recompute even if the cached result is fresh.
    """
    if library_index is None:
        raise HTTPException(status_code=503, detail="Library index not initialized")

    # ── Serve cache if fresh ─────────────────────────────────────────────────
    if not force and _MAP_CACHE_PATH.exists():
        age = _time_module.time() - _MAP_CACHE_PATH.stat().st_mtime
        if age < 3600:
            import json as _json
            with open(_MAP_CACHE_PATH, encoding="utf-8") as fh:
                return JSONResponse(_json.load(fh))

    # ── Read all JSONL records ────────────────────────────────────────────────
    import json as _json
    records: list[dict] = []
    path = library_index._record_path
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(_json.loads(line))
                    except Exception:
                        pass

    # ── Compute embedding in thread executor (CPU-bound) ─────────────────────
    result = await asyncio.get_event_loop().run_in_executor(
        None, _compute_track_map, records
    )

    # ── Cache and return ──────────────────────────────────────────────────────
    _MAP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_MAP_CACHE_PATH, "w", encoding="utf-8") as fh:
        _json.dump(result, fh)

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Routes — Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(body: ChatRequest):
    """Chat with the AI DJ assistant."""
    if not ai:
        raise HTTPException(status_code=503, detail="AI not initialized")

    response = await ai.chat(body.message)
    result = {
        "role": response.role,
        "content": response.content,
        "timestamp": response.timestamp,
    }

    if response.setlist:
        result["setlist"] = response.setlist.model_dump()
    if response.recommendations:
        result["recommendations"] = [r.model_dump() for r in response.recommendations]

    return JSONResponse(result)


@app.post("/api/chat/clear")
async def clear_chat():
    """Clear conversation history."""
    if ai:
        ai.clear_history()
    return {"success": True}


# ---------------------------------------------------------------------------
# Routes — Setlist generation
# ---------------------------------------------------------------------------

@app.post("/api/setlist/generate")
async def generate_setlist(request: SetlistRequest):
    """Generate a setlist directly (structured parameters)."""
    setlist = engine.generate_setlist(request)
    return JSONResponse(setlist.model_dump())


class PlanSetRequest(BaseModel):
    duration_minutes: int = 60
    vibe: str = ""
    situation: str = ""
    venue: str = ""
    crowd_energy: str = ""
    time_of_day: str = ""
    genre_preference: Optional[str] = None


@app.post("/api/setlist/plan")
async def plan_set(body: PlanSetRequest):
    """Build a setlist from a vibe description rather than technical parameters.

    Translates freeform context (vibe, venue, situation, crowd, time) into
    genre/BPM/energy-arc parameters, then generates and returns a full setlist
    alongside an interpretation block showing how the vibe was mapped.
    """
    interpretation = interpret_vibe(
        vibe=body.vibe,
        situation=body.situation,
        venue=body.venue,
        crowd_energy=body.crowd_energy,
        time_of_day=body.time_of_day,
        genre_preference=body.genre_preference or "",
    )

    context_parts = [p for p in [body.vibe, body.situation, body.venue] if p]
    prompt = " | ".join(context_parts) if context_parts else f"{body.duration_minutes}min set"

    request = SetlistRequest(
        prompt=prompt,
        duration_minutes=max(10, min(480, body.duration_minutes)),
        genre=interpretation["genre"],
        bpm_min=interpretation["bpm_min"],
        bpm_max=interpretation["bpm_max"],
        energy_profile=interpretation["energy_profile"],
    )

    setlist = engine.generate_setlist(request)
    essentia_store = engine.essentia_store

    def _ess(file_path):
        if not essentia_store:
            return {}
        ess = essentia_store.get(file_path)
        if not ess:
            return {}
        return {
            "essentia_energy": ess.energy_as_1_to_10(),
            "danceability": ess.danceability_as_1_to_10(),
            "dominant_mood": ess.dominant_mood(),
            "top_genre_discogs": ess.top_genre(),
            "lufs": round(ess.integrated_lufs, 1),
        }

    return JSONResponse({
        "interpretation": {
            "vibe_label": interpretation["vibe_label"],
            "genre": interpretation["genre"],
            "bpm_range": f"{interpretation['bpm_min']:.0f}–{interpretation['bpm_max']:.0f}",
            "energy_profile": interpretation["energy_profile"],
            "reasoning": interpretation["reasoning"],
        },
        "setlist_id": setlist.id,
        "name": setlist.name,
        "track_count": setlist.track_count,
        "duration_minutes": round(setlist.total_duration_seconds / 60),
        "avg_bpm": setlist.avg_bpm,
        "bpm_range": {"min": setlist.bpm_range[0], "max": setlist.bpm_range[1]},
        "harmonic_score": setlist.harmonic_score,
        "energy_arc": setlist.energy_arc,
        "genre_distribution": setlist.genre_distribution,
        "tracks": [
            {
                "position": st.position,
                "artist": st.track.artist,
                "title": st.track.title,
                "bpm": st.track.bpm,
                "key": st.track.key,
                "energy": st.track.energy,
                "genre": st.track.genre,
                "duration": st.track.duration_formatted(),
                "key_relation": st.key_relation,
                "transition_score": st.transition_score,
                "notes": st.notes,
                **_ess(st.track.file_path),
            }
            for st in setlist.tracks
        ],
    })


class BuildSetRequest(BaseModel):
    prompt: str
    duration_minutes: int = 60


@app.post("/api/setlist/build")
async def build_set_from_prompt(body: BuildSetRequest):
    """Build a complete DJ set from a single free-form natural language prompt.

    Queries the library index using Rekordbox My Tags detected in the prompt,
    then applies harmonic mixing and energy-arc planning.

    Examples:
      "60 minute progressive house sunset set"
      "dark underground techno bangers for afters, build to peak"
      "festival main stage with famous EDM and vocals, 90 minutes"
      "wedding set — happy, famous, danceable tracks"
    """
    if library_index is None:
        raise HTTPException(status_code=503, detail="Library index not initialized")

    # 1. Parse intent — fully dynamic using scanned library attributes
    intent = parse_set_intent(body.prompt, attrs=library_attributes)
    duration_minutes = intent["duration_minutes"] or body.duration_minutes
    duration_minutes = max(10, min(480, duration_minutes))

    # 2. Query library index for tracks matching detected My Tags
    candidate_ids: set = set()
    tag_coverage: Dict[str, int] = {}

    if intent["my_tags"]:
        for tag in intent["my_tags"]:
            matches = library_index.search(my_tag=tag, limit=500)
            tag_coverage[tag] = len(matches)
            for rec in matches:
                candidate_ids.add(rec["id"])

    # 3. Map candidate IDs → TrackWithEnergy objects
    if candidate_ids:
        candidate_tracks = [t for t in engine.tracks if str(t.id) in candidate_ids]
    else:
        candidate_tracks = []

    # Fallback: if too few MyTag matches, widen to all tracks
    used_fallback = len(candidate_tracks) < 10
    if used_fallback:
        candidate_tracks = engine.tracks

    # 4. Build a temporary engine scoped to the candidate pool.
    #    Use the library index as the feature source (mood/genre/tags vectors
    #    already in-memory from JSONL — no extra disk I/O).
    from .setlist_engine import SetlistEngine as _SetlistEngine
    from .camelot import CamelotWheel as _CamelotWheel
    from .energy_planner import EnergyPlanner as _EnergyPlanner

    ess_store = (
        LibraryIndexFeatureStore(library_index)
        if library_index is not None and len(library_index._by_id) > 0
        else engine.essentia_store
    )

    temp_engine = _SetlistEngine(
        tracks=candidate_tracks,
        camelot=_CamelotWheel(),
        energy_planner=_EnergyPlanner(),
        essentia_store=ess_store,
    )

    # 5. Generate setlist
    request = SetlistRequest(
        prompt=body.prompt,
        duration_minutes=duration_minutes,
        genre=intent["genre"],
        bpm_min=intent["bpm_min"],
        bpm_max=intent["bpm_max"],
        energy_profile=(
            intent["energy_profile"]
            if intent["energy_profile"] in ENERGY_PROFILES
            else "journey"
        ),
    )
    setlist = temp_engine.generate_setlist(request)

    # 6. Register setlist in the main engine so export works
    engine._setlists[setlist.id] = setlist

    # 7. Per-track Essentia enrichment
    main_ess = engine.essentia_store

    def _ess(file_path: Optional[str]) -> Dict:
        if not main_ess or not file_path:
            return {}
        ess = main_ess.get(file_path)
        if not ess:
            return {}
        out: Dict = {
            "essentia_energy": ess.energy_as_1_to_10(),
            "danceability": ess.danceability_as_1_to_10(),
            "lufs": round(ess.integrated_lufs, 1),
        }
        mood: Dict = {}
        for key, val in [
            ("happy",      ess.mood_happy),
            ("sad",        ess.mood_sad),
            ("aggressive", ess.mood_aggressive),
            ("relaxed",    ess.mood_relaxed),
            ("party",      ess.mood_party),
        ]:
            if val is not None:
                mood[key] = round(val, 2)
        if mood:
            out["mood"] = mood
            out["dominant_mood"] = max(mood, key=mood.get)
        if ess.genre_discogs:
            sorted_genres = sorted(ess.genre_discogs.items(), key=lambda x: x[1], reverse=True)
            out["top_genres"] = {g: round(s, 3) for g, s in sorted_genres[:3]}
            out["top_genre_discogs"] = sorted_genres[0][0] if sorted_genres else None
        if ess.music_tags:
            sorted_tags = sorted(ess.music_tags, key=lambda t: t["score"], reverse=True)
            out["top_tags"] = {t["tag"]: round(t["score"], 3) for t in sorted_tags[:5]}
        return out

    return JSONResponse({
        "setlist_id": setlist.id,
        "name": setlist.name,
        "prompt": body.prompt,
        "intent": {
            "my_tags_detected": intent["my_tags"],
            "tag_coverage": tag_coverage,
            "candidate_pool": (
                len(candidate_tracks)
                if not used_fallback
                else f"{len(candidate_tracks)} (all tracks — no My Tag matches, used genre/BPM fallback)"
            ),
            "genre": intent["genre"],
            "bpm_range": f"{intent['bpm_min']:.0f}–{intent['bpm_max']:.0f}",
            "energy_profile": intent["energy_profile"],
            "reasoning": intent["reasoning"],
        },
        "track_count": setlist.track_count,
        "duration_minutes": round(setlist.total_duration_seconds / 60),
        "avg_bpm": setlist.avg_bpm,
        "bpm_range": {"min": setlist.bpm_range[0], "max": setlist.bpm_range[1]},
        "harmonic_score": setlist.harmonic_score,
        "energy_arc": setlist.energy_arc,
        "genre_distribution": setlist.genre_distribution,
        "tracks": [
            {
                "position": st.position,
                "artist": st.track.artist,
                "title": st.track.title,
                "bpm": st.track.bpm,
                "key": st.track.key,
                "energy": st.track.energy,
                "genre": st.track.genre,
                "my_tags": st.track.my_tags,
                "duration": st.track.duration_formatted(),
                "key_relation": st.key_relation,
                "transition_score": st.transition_score,
                "notes": st.notes,
                **_ess(st.track.file_path),
            }
            for st in setlist.tracks
        ],
    })


@app.get("/api/setlist/{setlist_id}")
async def get_setlist(setlist_id: str):
    """Retrieve a previously generated setlist."""
    setlist = engine.get_setlist(setlist_id)
    if not setlist:
        raise HTTPException(status_code=404, detail="Setlist not found")
    return JSONResponse(setlist.model_dump())


# ---------------------------------------------------------------------------
# Routes — Recommendations & analysis
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    track_id: Optional[str] = None
    track_title: Optional[str] = None
    energy_direction: str = "maintain"
    limit: int = 10


@app.post("/api/setlist/recommend")
async def recommend_next(body: RecommendRequest):
    """Get next-track recommendations based on harmonic compatibility and energy flow."""
    recs = engine.recommend_next(
        current_track_id=body.track_id,
        current_track_title=body.track_title,
        energy_direction=body.energy_direction,
        limit=body.limit,
    )
    return JSONResponse([r.model_dump() for r in recs])


class CompatibilityRequest(BaseModel):
    track_a_title: str
    track_b_title: str


@app.post("/api/setlist/compatibility")
async def get_track_compatibility(body: CompatibilityRequest):
    """Check harmonic and energy compatibility between two tracks."""
    track_a = next(
        (t for t in engine.tracks if body.track_a_title.lower() in t.title.lower()), None
    )
    track_b = next(
        (t for t in engine.tracks if body.track_b_title.lower() in t.title.lower()), None
    )

    if not track_a:
        raise HTTPException(status_code=404, detail=f"Track not found: '{body.track_a_title}'")
    if not track_b:
        raise HTTPException(status_code=404, detail=f"Track not found: '{body.track_b_title}'")

    h_score, rel = camelot.transition_score(track_a.key or "", track_b.key or "")
    bpm_pct_diff = (
        abs(track_b.bpm - track_a.bpm) / track_a.bpm * 100
        if track_a.bpm > 0 else 0
    )
    energy_delta = (track_b.energy or 5) - (track_a.energy or 5)

    verdict = (
        "Excellent mix" if h_score >= 0.85 and bpm_pct_diff <= 4
        else "Good mix" if h_score >= 0.65
        else "Acceptable mix" if h_score >= 0.4
        else "Difficult transition"
    )

    return JSONResponse({
        "track_a": {
            "artist": track_a.artist, "title": track_a.title,
            "bpm": track_a.bpm, "key": track_a.key, "energy": track_a.energy,
        },
        "track_b": {
            "artist": track_b.artist, "title": track_b.title,
            "bpm": track_b.bpm, "key": track_b.key, "energy": track_b.energy,
        },
        "harmonic_score": h_score,
        "key_relationship": rel,
        "bpm_difference": round(track_b.bpm - track_a.bpm, 1),
        "bpm_pct_difference": round(bpm_pct_diff, 1),
        "energy_delta": energy_delta,
        "verdict": verdict,
    })


class EnergyFlowRequest(BaseModel):
    track_titles: List[str]


@app.post("/api/setlist/energy-flow")
async def analyze_energy_flow(body: EnergyFlowRequest):
    """Analyze the energy flow of a sequence of tracks."""
    found_tracks = [
        next((t for t in engine.tracks if title.lower() in t.title.lower()), None)
        for title in body.track_titles
    ]

    energies = [(t.energy or 5) if t else 5 for t in found_tracks]
    n = len(energies)

    target_journey = [
        round(planner.get_target_energy(i / max(1, n - 1), "journey"))
        for i in range(n)
    ]

    issues = []
    for i in range(1, n):
        delta = abs(energies[i] - energies[i - 1])
        if delta > 3:
            issues.append(f"Position {i+1}: Large energy jump ({delta} levels)")
    for i in range(2, n):
        if energies[i] == energies[i - 1] == energies[i - 2]:
            issues.append(f"Position {i+1}: Plateau detected (3+ tracks at E{energies[i]})")

    return JSONResponse({
        "track_count": n,
        "energy_values": energies,
        "target_journey": target_journey,
        "avg_energy": round(sum(energies) / n, 1) if energies else 0,
        "energy_range": {"min": min(energies), "max": max(energies)} if energies else {},
        "issues": issues,
        "score": round(1.0 - len(issues) / max(n, 1), 2),
        "tracks_found": [
            {
                "title": t.title, "energy": t.energy,
                "key": t.key, "bpm": t.bpm,
            }
            if t else None
            for t in found_tracks
        ],
    })


class CompatibleTracksRequest(BaseModel):
    key: str
    bpm: Optional[float] = None
    bpm_tolerance: float = 4.0
    energy_min: Optional[int] = None
    energy_max: Optional[int] = None
    genre: Optional[str] = None
    limit: int = 20


@app.post("/api/setlist/compatible-tracks")
async def get_compatible_tracks(body: CompatibleTracksRequest):
    """Find tracks in the library compatible with a given Camelot key and optional BPM."""
    compatible_keys = camelot.get_compatible_keys(body.key)
    if not compatible_keys:
        raise HTTPException(status_code=400, detail=f"Invalid Camelot key: {body.key}")

    results = []
    for track in engine.tracks:
        if not track.key:
            continue
        tk = track.key.strip().upper()
        if tk not in compatible_keys:
            continue
        h_score = compatible_keys[tk]

        if body.bpm is not None and track.bpm > 0:
            pct_diff = abs(track.bpm - body.bpm) / body.bpm * 100
            if pct_diff > body.bpm_tolerance:
                continue

        if body.energy_min is not None and (track.energy or 0) < body.energy_min:
            continue
        if body.energy_max is not None and (track.energy or 10) > body.energy_max:
            continue

        if body.genre and track.genre and body.genre.lower() not in track.genre.lower():
            continue

        _, rel = camelot.transition_score(body.key, track.key)

        entry: Dict[str, Any] = {
            "artist": track.artist,
            "title": track.title,
            "bpm": track.bpm,
            "key": track.key,
            "energy": track.energy,
            "genre": track.genre,
            "rating": track.rating,
            "harmonic_score": h_score,
            "key_relationship": rel,
        }

        if engine.essentia_store:
            ess = engine.essentia_store.get(track.file_path)
            if ess:
                entry["essentia_energy"] = ess.energy_as_1_to_10()
                entry["danceability"] = ess.danceability_as_1_to_10()
                entry["dominant_mood"] = ess.dominant_mood()
                entry["top_genre_discogs"] = ess.top_genre()
                entry["lufs"] = round(ess.integrated_lufs, 1)

        results.append(entry)

    results.sort(key=lambda x: (-x["harmonic_score"], -(x.get("rating") or 0)))
    return JSONResponse(results[:body.limit])


# ---------------------------------------------------------------------------
# Routes — Export
# ---------------------------------------------------------------------------

class ExportRequest(BaseModel):
    setlist_id: str
    playlist_name: str
    folder_name: str = "MCP-DJ Sets"


@app.post("/api/setlist/export")
async def export_to_rekordbox(body: ExportRequest):
    """Export a setlist to a Rekordbox playlist, placed in the given folder."""
    setlist = engine.get_setlist(body.setlist_id)
    if not setlist:
        raise HTTPException(status_code=404, detail="Setlist not found")

    track_ids = [st.track.id for st in setlist.tracks]
    resolved_folder = body.folder_name.strip() if body.folder_name else None
    try:
        playlist_id = await db.create_playlist_with_tracks(
            name=body.playlist_name,
            track_ids=track_ids,
            folder_name=resolved_folder,
        )
        return {
            "success": True,
            "playlist_id": playlist_id,
            "folder_name": resolved_folder,
            "track_count": len(track_ids),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CreateFolderRequest(BaseModel):
    folder_name: str
    parent_folder_name: Optional[str] = None


@app.post("/api/rekordbox/folder")
async def create_rekordbox_folder(body: CreateFolderRequest):
    """Create (or find) a Rekordbox playlist folder."""
    try:
        parent_id = None
        if body.parent_folder_name:
            parent = db.find_folder(body.parent_folder_name.strip())
            if parent is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Parent folder '{body.parent_folder_name}' not found",
                )
            parent_id = str(parent.ID)
        folder_id = await db.create_folder(body.folder_name.strip(), parent_id=parent_id)
        return {"success": True, "folder_id": folder_id, "folder_name": body.folder_name.strip()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Routes — MCP-DJ Sets playlist editing
# ---------------------------------------------------------------------------

_DEFAULT_FOLDER = "MCP-DJ Sets"


@app.get("/api/rekordbox/playlists")
async def list_mcp_playlists(folder_name: str = _DEFAULT_FOLDER):
    """List all playlists in the given folder."""
    playlists = await db.list_playlists_in_folder(folder_name)
    return {"folder_name": folder_name, "playlists": playlists}


@app.get("/api/rekordbox/playlists/{playlist_id}/tracks")
async def get_mcp_playlist_tracks(playlist_id: str, folder_name: str = _DEFAULT_FOLDER):
    """Get the ordered track list for a playlist."""
    try:
        pl = db._get_playlist_in_folder(playlist_id, folder_name)
        tracks = await db.get_playlist_tracks(playlist_id)
        return {"playlist_id": playlist_id, "playlist_name": pl.Name, "tracks": tracks}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


class RenameMcpPlaylistRequest(BaseModel):
    new_name: str
    folder_name: str = _DEFAULT_FOLDER


@app.patch("/api/rekordbox/playlists/{playlist_id}/rename")
async def rename_mcp_playlist(playlist_id: str, body: RenameMcpPlaylistRequest):
    """Rename a playlist."""
    try:
        old_name = await db.rename_mcp_playlist(playlist_id, body.new_name, body.folder_name)
        return {"success": True, "old_name": old_name, "new_name": body.new_name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AddTracksRequest(BaseModel):
    track_ids: List[str]
    position: Optional[int] = None
    folder_name: str = _DEFAULT_FOLDER


@app.post("/api/rekordbox/playlists/{playlist_id}/tracks")
async def add_tracks_to_mcp_playlist(playlist_id: str, body: AddTracksRequest):
    """Add tracks by content ID to a playlist."""
    try:
        new_count = await db.add_tracks_to_mcp_playlist(
            playlist_id, body.track_ids, body.position, body.folder_name
        )
        return {"success": True, "new_track_count": new_count}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RemoveTrackRequest(BaseModel):
    folder_name: str = _DEFAULT_FOLDER


@app.delete("/api/rekordbox/playlists/{playlist_id}/tracks/{position}")
async def remove_track_from_mcp_playlist(
    playlist_id: str, position: int, folder_name: str = _DEFAULT_FOLDER
):
    """Remove the track at the given 1-based position."""
    try:
        info = await db.remove_track_from_mcp_playlist(playlist_id, position, folder_name)
        return {"success": True, **info}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ReorderTrackRequest(BaseModel):
    from_position: int
    to_position: int
    folder_name: str = _DEFAULT_FOLDER


@app.post("/api/rekordbox/playlists/{playlist_id}/tracks/reorder")
async def reorder_track_in_mcp_playlist(playlist_id: str, body: ReorderTrackRequest):
    """Move a track to a new position."""
    try:
        await db.reorder_track_in_mcp_playlist(
            playlist_id, body.from_position, body.to_position, body.folder_name
        )
        return {"success": True, "moved_from": body.from_position, "moved_to": body.to_position}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/rekordbox/playlists/{playlist_id}")
async def delete_mcp_playlist(playlist_id: str, folder_name: str = _DEFAULT_FOLDER):
    """Delete a playlist from the folder."""
    try:
        name = await db.delete_mcp_playlist(playlist_id, folder_name)
        return {"success": True, "deleted_name": name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Routes — Essentia audio analysis
# ---------------------------------------------------------------------------

class AnalyzeTrackRequest(BaseModel):
    file_path: str
    force: bool = False


@app.post("/api/essentia/analyze")
async def analyze_track(body: AnalyzeTrackRequest):
    """Analyze a single audio file with Essentia.

    Extracts BPM, key, danceability, loudness, mood scores, genre classification,
    and music tags. Results are cached so the same file is never analyzed twice.
    """
    if not ESSENTIA_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Essentia is not installed. Run: pip install essentia",
        )

    try:
        features = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _essentia_analyze_file(body.file_path, force=body.force)
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Audio file not found: {body.file_path}")
    except Exception as e:
        logger.error(f"Essentia analysis failed for {body.file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return JSONResponse({
        "file_path": features.file_path,
        "analyzed_at": features.analyzed_at,
        "essentia_version": features.essentia_version,
        "analysis_seconds": features.analysis_duration_seconds,
        "bpm": {
            "value": round(features.bpm_essentia, 2),
            "confidence": round(features.bpm_confidence, 3),
            "beats_count": features.beats_count,
        },
        "key": {
            "camelot": features.key_essentia,
            "raw": f"{features.key_name_raw} {features.key_scale}" if features.key_name_raw else None,
            "strength": round(features.key_strength, 3),
        },
        "danceability": {
            "score": round(features.danceability, 3),
            "scale_1_to_10": features.danceability_as_1_to_10(),
        },
        "loudness": {
            "integrated_lufs": round(features.integrated_lufs, 2),
            "loudness_range_db": round(features.loudness_range_db, 2),
            "rms_db": round(features.rms_db, 2),
            "rms_linear": round(features.rms_energy, 5),
            "scale_1_to_10": features.energy_as_1_to_10(),
        },
        "mood": {
            "happy":      round(features.mood_happy, 3)      if features.mood_happy      is not None else None,
            "sad":        round(features.mood_sad, 3)        if features.mood_sad        is not None else None,
            "aggressive": round(features.mood_aggressive, 3) if features.mood_aggressive is not None else None,
            "relaxed":    round(features.mood_relaxed, 3)    if features.mood_relaxed    is not None else None,
            "party":      round(features.mood_party, 3)      if features.mood_party      is not None else None,
            "dominant":   features.dominant_mood(),
        },
        "genre_discogs": features.genre_discogs,
        "music_tags": features.music_tags,
        "cache_directory": str(ESSENTIA_CACHE_DIR),
    })


class AnalyzeLibraryRequest(BaseModel):
    force: bool = False
    limit: Optional[int] = None


@app.post("/api/essentia/analyze-library")
async def analyze_library_essentia(body: AnalyzeLibraryRequest):
    """Batch-analyze all tracks in the library with Essentia.

    CPU-intensive — may take several minutes for large libraries.  Results are
    cached; already-analyzed tracks are skipped unless force=true.  The library
    index is automatically rebuilt after analysis so new features are immediately
    available for set building.
    """
    global library_attributes

    if not ESSENTIA_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Essentia is not installed. Run: pip install essentia",
        )

    tracks_to_process = engine.tracks
    if body.limit is not None:
        tracks_to_process = tracks_to_process[:body.limit]

    total_in_library = len(engine.tracks)
    logger.info(f"Starting Essentia library analysis: {len(tracks_to_process)} tracks")

    _track_map = {t.id: t for t in tracks_to_process}
    _FLUSH_EVERY = 10
    _flush_counter = [0]

    def _on_track_complete(track_id: str, file_path: str, features: Any) -> None:
        if library_index is None:
            return
        track = _track_map.get(track_id)
        if track is None:
            return

        class _SingleStore:
            def get(self, fp: str):
                return features if fp == file_path else None

        try:
            library_index.update_record(
                track,
                essentia_store=_SingleStore(),
                mik_library=_mik_library_app,
            )
            _flush_counter[0] += 1
            if _flush_counter[0] % _FLUSH_EVERY == 0:
                n = library_index.flush_to_disk()
                logger.info(
                    f"Incremental index flush: {n} records on disk "
                    f"({_flush_counter[0]} analyzed so far)"
                )
        except Exception as exc:
            logger.warning(f"Incremental index update failed for {file_path}: {exc}")

    stats = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _essentia_analyze_library(
            tracks=tracks_to_process,
            force=body.force,
            skip_missing=True,
            on_track_complete=_on_track_complete,
        ),
    )

    # Final flush for any remaining records not on a flush boundary
    if library_index is not None and _flush_counter[0] % _FLUSH_EVERY != 0:
        n = library_index.flush_to_disk()
        logger.info(f"Final incremental flush: {n} records on disk")

    result: Dict[str, Any] = {
        "total_tracks_in_library": total_in_library,
        "tracks_processed": len(tracks_to_process),
        "analyzed_new": stats["analyzed"],
        "loaded_from_cache": stats["cached"],
        "skipped_no_file_path": stats["skipped_no_path"],
        "skipped_missing_file": stats["skipped_missing_file"],
        "errors": stats["errors"],
        "cache_directory": str(ESSENTIA_CACHE_DIR),
        "message": (
            f"Analysis complete. {stats['analyzed']} new tracks analyzed, "
            f"{stats['cached']} loaded from cache, "
            f"{stats['errors']} errors."
        ),
    }

    # Rebuild the library index to incorporate newly analyzed tracks
    if library_index is not None:
        fresh_store = EssentiaFeatureStore(engine.tracks) if EssentiaFeatureStore else None
        try:
            tag_tree = await db.query_my_tags(limit=500)
        except Exception:
            tag_tree = []
        idx_stats = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: library_index.build(
                tracks=engine.tracks,
                essentia_store=fresh_store,
                mik_library=_mik_library_app,
                my_tag_tree=tag_tree,
            ),
        )
        library_attributes = library_index.attributes
        if fresh_store is not None:
            engine.essentia_store = fresh_store

        # Propagate fresh library context to the AI assistant
        if ai is not None:
            ai.update_library_context(
                library_index=library_index,
                library_attributes=library_attributes,
                mik_library=_mik_library_app,
            )

        result["library_index_rebuilt"] = True
        result["library_index_total"] = idx_stats["total"]
        result["library_index_with_essentia"] = idx_stats["with_essentia"]

    # Run phrase analysis on any tracks not yet cached (non-blocking on errors)
    logger.info("Running phrase analysis on new/uncached tracks…")
    phrase_analyzed = 0
    phrase_cached   = 0
    phrase_errors   = 0

    def _phrase_progress(track_id, idx, total, status):
        nonlocal phrase_analyzed, phrase_cached, phrase_errors
        if status == "analyzed":   phrase_analyzed += 1
        elif status == "cached":   phrase_cached   += 1
        elif status.startswith("error"): phrase_errors += 1

    phrase_stats = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: analyze_library_phrases(
            tracks=tracks_to_process,
            phrase_store=phrase_store,
            library_index=library_index,
            force=body.force,
            flush_every=20,
            progress_callback=_phrase_progress,
        ),
    )
    result["phrases_analyzed_new"]  = phrase_stats["analyzed"]
    result["phrases_from_cache"]    = phrase_stats["cached"]
    result["phrases_errors"]        = phrase_stats["errors"]

    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Routes — Phrase detection / mix profile
# ---------------------------------------------------------------------------

class AnalyzePhrasesLibraryRequest(BaseModel):
    force: bool = False
    limit: Optional[int] = None


@app.post("/api/phrases/analyze-library")
async def analyze_phrases_library(body: AnalyzePhrasesLibraryRequest):
    """Batch phrase detection across the full library.

    Runs the phrase detector on every track and caches:
      • Section structure  (Intro / Up / Chorus / Down / Outro)
      • MixProfile         (intro length, first drop, outro, blend_bars)

    The MixProfile data is used by the set-generation algorithm to:
      • Know WHERE to start mixing the next track (outgoing outro)
      • Know HOW LONG to blend (based on incoming intro length)
      • Choose a transition type (blend / drop-to-drop / breakdown entry)

    Results are cached per-track in .data/phrase_cache/{track_id}.json.
    Already-cached tracks are skipped unless force=true.

    CPU-intensive — roughly 10-30 seconds per track depending on length.
    """
    tracks_to_process = engine.tracks
    if body.limit is not None:
        tracks_to_process = tracks_to_process[:body.limit]

    total_in_library = len(engine.tracks)
    analyzed = 0
    cached   = 0
    errors   = 0
    skipped  = 0

    logger.info(
        "Starting phrase library analysis: %d tracks (force=%s)",
        len(tracks_to_process), body.force,
    )

    def _progress(track_id, idx, total, status):
        nonlocal analyzed, cached, errors, skipped
        if status == "analyzed":
            analyzed += 1
        elif status == "cached":
            cached += 1
        elif status == "skipped_no_file":
            skipped += 1
        elif status.startswith("error"):
            errors += 1
        if (idx + 1) % 25 == 0 or idx + 1 == total:
            logger.info(
                "Phrase analysis progress: %d/%d  analyzed=%d cached=%d errors=%d",
                idx + 1, total, analyzed, cached, errors,
            )

    stats = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: analyze_library_phrases(
            tracks=tracks_to_process,
            phrase_store=phrase_store,
            library_index=library_index,
            force=body.force,
            flush_every=20,
            progress_callback=_progress,
        ),
    )

    return JSONResponse({
        "total_tracks_in_library": total_in_library,
        "tracks_processed":        len(tracks_to_process),
        "analyzed_new":            stats["analyzed"],
        "loaded_from_cache":       stats["cached"],
        "skipped_no_file":         stats["skipped_no_file"],
        "errors":                  stats["errors"],
        "cache_directory":         str(phrase_store._dir),
        "message": (
            f"Phrase analysis complete. {stats['analyzed']} new tracks analyzed, "
            f"{stats['cached']} loaded from cache, {stats['errors']} errors."
        ),
    })


@app.get("/api/track/{track_id}/phrases")
async def get_track_phrases(track_id: str):
    """Return cached phrase detection results for a track.

    Returns section structure (Intro/Up/Chorus/Down/Outro) and the derived
    MixProfile with DJ transition parameters.
    """
    data = phrase_store.get(track_id)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No phrase data cached for track {track_id}. Run analyze-phrases first.",
        )
    return JSONResponse(data)


# ---------------------------------------------------------------------------
# Routes — Audio streaming, cue points, phrase analysis
# ---------------------------------------------------------------------------

import mimetypes as _mimetypes


@app.get("/api/audio/{track_id}")
async def stream_audio_file(track_id: str):
    """Stream the raw audio file for a track by its Rekordbox content ID."""
    track = next((t for t in engine.tracks if str(t.id) == track_id), None)
    if not track or not track.file_path:
        raise HTTPException(status_code=404, detail="Track not found")
    p = Path(track.file_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not on disk: {p.name}")
    mime = _mimetypes.guess_type(str(p))[0] or "audio/mpeg"
    return FileResponse(str(p), media_type=mime)


@app.get("/api/track/{track_id}/cues")
async def get_track_cue_points(track_id: str):
    """Return all cue points (memory + hot cues) for a track."""
    if db.db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    rows = db.db.get_cue(ContentID=track_id).all()
    return JSONResponse([
        {
            "id":      c.ID,
            "time_ms": c.InMsec or 0,
            "kind":    c.Kind  or 0,
            "comment": c.Comment or "",
            "color":   c.Color  or -1,
        }
        for c in rows if c.InMsec is not None
    ])


class AnalyzePhrasesRequest(BaseModel):
    replace: bool = False


@app.post("/api/track/{track_id}/analyze-phrases")
async def analyze_track_phrases(track_id: str, body: AnalyzePhrasesRequest = None):
    """Run phrase/structure detection on a track and write results as memory cues."""
    if body is None:
        body = AnalyzePhrasesRequest()
    track = next((t for t in engine.tracks if str(t.id) == track_id), None)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    if not track.file_path or not Path(track.file_path).exists():
        raise HTTPException(status_code=404, detail="Audio file not accessible on disk")

    from .phrase_detector import PhraseDetector, write_phrase_cues
    detector = PhraseDetector()
    cues = await asyncio.get_event_loop().run_in_executor(
        None, lambda: detector.detect(track.file_path, bpm=track.bpm)
    )

    # Save to phrase store (mix profile + sections)
    dur_s = float(track.length or 300)
    mix_profile = phrase_store.save(track_id, cues, bpm=float(track.bpm or 128), duration_s=dur_s)

    # Optionally write as Rekordbox memory cues
    if db.db is not None:
        write_phrase_cues(db.db, track_id, cues, replace_existing=body.replace)

    return JSONResponse({
        "phrases": [
            {"time_ms": c.time_ms, "label": c.label, "color": c.color, "bar_number": c.bar_number}
            for c in cues
        ],
        "mix_profile": {
            "pre_drop_bars":         mix_profile.pre_drop_bars,
            "outro_bars":            mix_profile.outro_bars,
            "blend_bars":            mix_profile.blend_bars,
            "first_chorus_ms":       mix_profile.first_chorus_ms,
            "outro_start_ms":        mix_profile.outro_start_ms,
            "preferred_transition":  mix_profile.preferred_transition,
        },
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    port = int(os.environ.get("SETLIST_PORT", "8888"))
    logger.info(f"Starting MCP DJ on port {port}")
    uvicorn.run(
        "mcp_dj.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
