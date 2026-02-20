"""
Claude API Integration for Chat-Based Setlist Generation

Uses the Anthropic SDK with tool-calling to let Claude invoke the setlist
engine, search the library, and recommend tracks.
"""

import json
import os
from typing import List, Dict, Any, Optional

from loguru import logger

from .models import (
    ChatMessage,
    Setlist,
    SetlistRequest,
    NextTrackRecommendation,
)
from .setlist_engine import SetlistEngine
from .energy_planner import EnergyPlanner, ENERGY_PROFILES


# ---------------------------------------------------------------------------
# Tool definitions for Claude
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "generate_setlist",
        "description": (
            "Generate a DJ setlist with harmonic mixing and energy planning. "
            "Use this when the user asks to create a set, mix, or playlist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "duration_minutes": {
                    "type": "integer",
                    "description": "Target set duration in minutes (default 60)",
                    "default": 60,
                },
                "genre": {
                    "type": "string",
                    "description": "Genre filter (e.g. 'tech house', 'techno'). Optional.",
                },
                "bpm_min": {
                    "type": "number",
                    "description": "Minimum BPM filter. Optional.",
                },
                "bpm_max": {
                    "type": "number",
                    "description": "Maximum BPM filter. Optional.",
                },
                "energy_profile": {
                    "type": "string",
                    "enum": ["journey", "build", "peak", "chill", "wave"],
                    "description": "Energy arc profile (default 'journey')",
                    "default": "journey",
                },
                "starting_track_title": {
                    "type": "string",
                    "description": "Title of the track to start with. Optional.",
                },
            },
        },
    },
    {
        "name": "recommend_next_track",
        "description": (
            "Recommend what to play next after a specific track. "
            "Use when the user asks 'what should I play after X?' or "
            "'give me recommendations based on this track'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "current_track_query": {
                    "type": "string",
                    "description": "Title or artist-title of the currently playing track",
                },
                "energy_direction": {
                    "type": "string",
                    "enum": ["up", "down", "maintain"],
                    "description": "Whether to raise, lower, or maintain energy",
                    "default": "maintain",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of recommendations (default 5)",
                    "default": 5,
                },
            },
            "required": ["current_track_query"],
        },
    },
    {
        "name": "search_library",
        "description": (
            "Search the DJ's Rekordbox library by text query, date added, or My Tag. "
            "Use to look up specific tracks, artists, genres, recently added tracks, "
            "or tracks with specific Rekordbox My Tag labels."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (matches title, artist, album, genre). Optional.",
                },
                "date_from": {
                    "type": "string",
                    "description": "Filter tracks added on or after this date (YYYY-MM-DD). Optional.",
                },
                "date_to": {
                    "type": "string",
                    "description": "Filter tracks added on or before this date (YYYY-MM-DD). Optional.",
                },
                "my_tag": {
                    "type": "string",
                    "description": "Filter by Rekordbox My Tag label (e.g. 'High Energy', 'Renegade'). Optional.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 20)",
                    "default": 20,
                },
            },
        },
    },
    {
        "name": "get_energy_advice",
        "description": (
            "Get advice on energy direction at a point in a set. "
            "Use when the user asks about energy flow or crowd management."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "current_energy": {
                    "type": "integer",
                    "description": "Current energy level (1-10)",
                },
                "position_pct": {
                    "type": "number",
                    "description": "How far through the set (0.0 = start, 1.0 = end)",
                },
                "profile": {
                    "type": "string",
                    "enum": ["journey", "build", "peak", "chill", "wave"],
                    "default": "journey",
                },
            },
            "required": ["current_energy", "position_pct"],
        },
    },
]


class SetlistAI:
    """
    Integrates Claude API for natural language setlist generation.

    Architecture:
    - User sends a chat message
    - We build a system prompt with library context
    - Claude responds, optionally calling tools
    - We execute tool calls against SetlistEngine
    - We return Claude's response + any generated setlist/recommendations
    """

    def __init__(self, engine: SetlistEngine, api_key: Optional[str] = None):
        self.engine = engine
        self.planner = EnergyPlanner()
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._conversation_history: List[Dict[str, Any]] = []
        self._last_setlist: Optional[Setlist] = None
        self._last_recommendations: Optional[List[NextTrackRecommendation]] = None

    def _get_client(self):
        """Lazy-init the Anthropic client."""
        import anthropic
        return anthropic.Anthropic(api_key=self._api_key)

    def _build_system_prompt(self) -> str:
        stats = self.engine.get_library_summary()
        profiles = ", ".join(
            f"{k} ({v['description']})" for k, v in ENERGY_PROFILES.items()
        )
        return f"""You are an expert DJ assistant helping create setlists from the user's Rekordbox library.
You understand harmonic mixing (Camelot wheel), energy flow, and genre coherence.

LIBRARY SUMMARY:
- Total tracks: {stats.get('total', 0)}
- BPM range: {stats.get('bpm_min', 0):.0f} - {stats.get('bpm_max', 0):.0f} (avg {stats.get('bpm_avg', 0):.0f})
- Top genres: {', '.join(stats.get('top_genres', [])[:8])}
- Top keys: {stats.get('key_summary', 'N/A')}
- Energy range: {stats.get('energy_min', 0)} - {stats.get('energy_max', 0)}
- Date range: {stats.get('date_min', 'N/A')} to {stats.get('date_max', 'N/A')}
- My Tags available: {', '.join(stats.get('top_my_tags', [])) or 'none'}

TRACK FIELDS: Each track has title, artist, genre, bpm, key, energy, rating, date_added (YYYY-MM-DD), my_tags (list of Rekordbox My Tag labels).
Use search_library with date_from/date_to to find recently added tracks, and my_tag to filter by My Tag label.

ENERGY PROFILES: {profiles}

HARMONIC MIXING (Camelot Wheel):
- Same key = perfect match
- +1/-1 position = smooth transition
- A/B switch = relative major/minor
- +7 = energy boost

When generating setlists, think about:
1. Harmonic compatibility for smooth transitions
2. Energy arc to create a compelling crowd journey
3. BPM flow with gradual changes
4. Genre coherence within the set
5. The DJ's narrative and overall experience

Always use the tools to generate setlists and search the library. Explain your reasoning.
Keep responses concise but informative. Format track names as "Artist - Title"."""

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def chat(self, user_message: str) -> ChatMessage:
        """Process a user message and return AI response."""
        self._last_setlist = None
        self._last_recommendations = None

        # If no API key, use the engine directly with simple parsing
        if not self._api_key:
            return await self._fallback_chat(user_message)

        self._conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        try:
            client = self._get_client()
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=self._build_system_prompt(),
                tools=TOOLS,
                messages=self._conversation_history,
            )

            # Process the response, handling tool calls
            return await self._process_response(response)

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            # Fallback to direct engine usage
            return await self._fallback_chat(user_message)

    async def _process_response(self, response) -> ChatMessage:
        """Process Claude's response, executing any tool calls."""
        text_parts = []
        tool_results = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                result = await self._execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

        # If there were tool calls, send results back to Claude for final response
        if tool_results:
            # Add assistant's response with tool calls to history
            self._conversation_history.append({
                "role": "assistant",
                "content": response.content,
            })
            # Add tool results
            self._conversation_history.append({
                "role": "user",
                "content": tool_results,
            })

            # Get Claude's final response incorporating tool results
            try:
                client = self._get_client()
                final_response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    system=self._build_system_prompt(),
                    tools=TOOLS,
                    messages=self._conversation_history,
                )

                final_text = ""
                for block in final_response.content:
                    if block.type == "text":
                        final_text += block.text
                    elif block.type == "tool_use":
                        # Handle nested tool calls
                        result = await self._execute_tool(block.name, block.input)

                self._conversation_history.append({
                    "role": "assistant",
                    "content": final_text,
                })

                return ChatMessage(
                    role="assistant",
                    content=final_text,
                    setlist=self._last_setlist,
                    recommendations=self._last_recommendations,
                )
            except Exception as e:
                logger.error(f"Error getting final response: {e}")
                # Return what we have
                text = "\n".join(text_parts) or "Generated setlist successfully."
                return ChatMessage(
                    role="assistant",
                    content=text,
                    setlist=self._last_setlist,
                    recommendations=self._last_recommendations,
                )
        else:
            text = "\n".join(text_parts)
            self._conversation_history.append({
                "role": "assistant",
                "content": text,
            })
            return ChatMessage(
                role="assistant",
                content=text,
                setlist=self._last_setlist,
                recommendations=self._last_recommendations,
            )

    async def _execute_tool(self, tool_name: str, tool_input: Dict) -> Any:
        """Execute a tool call and return the result."""
        logger.info(f"Executing tool: {tool_name} with input: {tool_input}")

        if tool_name == "generate_setlist":
            return await self._tool_generate_setlist(tool_input)
        elif tool_name == "recommend_next_track":
            return await self._tool_recommend_next(tool_input)
        elif tool_name == "search_library":
            return await self._tool_search_library(tool_input)
        elif tool_name == "get_energy_advice":
            return await self._tool_energy_advice(tool_input)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    async def _tool_generate_setlist(self, input: Dict) -> Dict:
        """Handle generate_setlist tool call."""
        # Find starting track by title if provided
        starting_id = None
        if input.get("starting_track_title"):
            for t in self.engine.tracks:
                if input["starting_track_title"].lower() in t.title.lower():
                    starting_id = t.id
                    break

        request = SetlistRequest(
            prompt=input.get("starting_track_title", "AI-generated set"),
            duration_minutes=input.get("duration_minutes", 60),
            genre=input.get("genre"),
            bpm_min=input.get("bpm_min"),
            bpm_max=input.get("bpm_max"),
            energy_profile=input.get("energy_profile", "journey"),
            starting_track_id=starting_id,
        )

        setlist = self.engine.generate_setlist(request)
        self._last_setlist = setlist

        # Return a summary for Claude to describe
        tracks_summary = []
        for st in setlist.tracks:
            tracks_summary.append({
                "position": st.position,
                "artist": st.track.artist,
                "title": st.track.title,
                "bpm": st.track.bpm,
                "key": st.track.key,
                "energy": st.track.energy,
                "genre": st.track.genre,
                "key_relation": st.key_relation,
                "transition_score": st.transition_score,
            })

        return {
            "setlist_id": setlist.id,
            "track_count": setlist.track_count,
            "total_duration_minutes": round(setlist.total_duration_seconds / 60),
            "avg_bpm": setlist.avg_bpm,
            "harmonic_score": setlist.harmonic_score,
            "energy_arc": setlist.energy_arc,
            "tracks": tracks_summary,
        }

    async def _tool_recommend_next(self, input: Dict) -> Dict:
        """Handle recommend_next_track tool call."""
        recs = self.engine.recommend_next(
            current_track_title=input["current_track_query"],
            energy_direction=input.get("energy_direction", "maintain"),
            limit=input.get("limit", 5),
        )
        self._last_recommendations = recs

        return {
            "recommendations": [
                {
                    "artist": r.track.artist,
                    "title": r.track.title,
                    "bpm": r.track.bpm,
                    "key": r.track.key,
                    "energy": r.track.energy,
                    "score": r.score,
                    "reason": r.reason,
                }
                for r in recs
            ]
        }

    async def _tool_search_library(self, input: Dict) -> Dict:
        """Handle search_library tool call."""
        query = input.get("query", "")
        date_from = input.get("date_from", "")
        date_to = input.get("date_to", "")
        my_tag = (input.get("my_tag") or "").strip().lower()
        limit = input.get("limit", 20)

        results = []
        q = query.strip().lower()
        for t in self.engine.tracks:
            # Text filter (skip if query given and nothing matches)
            if q and not (
                q in t.title.lower()
                or q in t.artist.lower()
                or q in (t.genre or "").lower()
            ):
                continue

            # Date filter
            d = t.date_added or ""
            if date_from and d < date_from:
                continue
            if date_to and d > date_to:
                continue

            # My Tag filter
            if my_tag and not any(my_tag in tag.lower() for tag in t.my_tags):
                continue

            results.append({
                "id": t.id,
                "artist": t.artist,
                "title": t.title,
                "bpm": t.bpm,
                "key": t.key,
                "energy": t.energy,
                "genre": t.genre,
                "rating": t.rating,
                "date_added": t.date_added,
                "my_tags": t.my_tags,
            })
            if len(results) >= limit:
                break

        return {"results": results, "count": len(results)}

    async def _tool_energy_advice(self, input: Dict) -> Dict:
        """Handle get_energy_advice tool call."""
        advice = self.planner.recommend_energy_direction(
            current_position_pct=input["position_pct"],
            current_energy=input["current_energy"],
            profile=input.get("profile", "journey"),
        )
        return advice

    # ------------------------------------------------------------------
    # Fallback (no API key)
    # ------------------------------------------------------------------

    async def _fallback_chat(self, message: str) -> ChatMessage:
        """Direct engine usage when Claude API is unavailable."""
        msg_lower = message.lower()

        # Detect intent from the message
        if any(w in msg_lower for w in ["create", "generate", "make", "build", "set"]):
            return await self._fallback_generate(message)
        elif any(w in msg_lower for w in ["next", "after", "recommend", "play after", "follow"]):
            return await self._fallback_recommend(message)
        elif any(w in msg_lower for w in ["search", "find", "look"]):
            return await self._fallback_search(message)
        else:
            return await self._fallback_generate(message)

    async def _fallback_generate(self, message: str) -> ChatMessage:
        """Parse message and generate setlist without Claude API."""
        # Simple parsing: extract duration, genre, BPM hints
        import re

        duration = 60
        dur_match = re.search(r'(\d+)\s*(?:min|minute|hour|hr)', message.lower())
        if dur_match:
            val = int(dur_match.group(1))
            if 'hour' in message.lower() or 'hr' in message.lower():
                duration = val * 60
            else:
                duration = val

        genre = None
        for g in set(t.genre for t in self.engine.tracks if t.genre):
            if g.lower() in message.lower():
                genre = g
                break

        profile = "journey"
        for p in ["build", "peak", "chill", "wave", "journey"]:
            if p in message.lower():
                profile = p
                break

        bpm_min = None
        bpm_max = None
        bpm_match = re.search(r'(\d{2,3})\s*(?:bpm|BPM)', message)
        if bpm_match:
            center = int(bpm_match.group(1))
            bpm_min = center - 5
            bpm_max = center + 10

        request = SetlistRequest(
            prompt=message,
            duration_minutes=duration,
            genre=genre,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            energy_profile=profile,
        )

        setlist = self.engine.generate_setlist(request)
        self._last_setlist = setlist

        # Build response text
        lines = [f"Generated a {setlist.track_count}-track setlist ({round(setlist.total_duration_seconds/60)} minutes):\n"]
        for st in setlist.tracks:
            energy_str = f"E{st.track.energy}" if st.track.energy else "E?"
            lines.append(
                f"  {st.position}. {st.track.artist} - {st.track.title} "
                f"[{st.track.bpm:.0f} BPM, {st.track.key or '?'}, {energy_str}]"
            )
            if st.position > 1 and st.key_relation:
                lines.append(f"     -> {st.notes}")

        lines.append(f"\nHarmonic score: {setlist.harmonic_score:.1%}")
        lines.append(f"BPM range: {setlist.bpm_range[0]:.0f} - {setlist.bpm_range[1]:.0f}")

        return ChatMessage(
            role="assistant",
            content="\n".join(lines),
            setlist=setlist,
        )

    async def _fallback_recommend(self, message: str) -> ChatMessage:
        """Parse message and recommend next tracks without Claude API."""
        # Extract track reference from message
        # Try to find a track name in the message
        import re

        # Remove common filler words
        cleaned = re.sub(
            r'\b(what|should|i|play|after|next|recommend|following|the|track|song)\b',
            '', message.lower()
        ).strip()

        # Search for the track
        track = None
        for t in self.engine.tracks:
            if cleaned and (cleaned in t.title.lower()
                           or cleaned in f"{t.artist} - {t.title}".lower()):
                track = t
                break

        if not track:
            return ChatMessage(
                role="assistant",
                content="I couldn't find that track in your library. Try searching with a more specific title.",
            )

        # Detect energy direction
        energy_dir = "maintain"
        if any(w in message.lower() for w in ["higher", "up", "raise", "increase", "hype"]):
            energy_dir = "up"
        elif any(w in message.lower() for w in ["lower", "down", "cool", "decrease", "calm"]):
            energy_dir = "down"

        recs = self.engine.recommend_next(
            current_track_id=track.id,
            energy_direction=energy_dir,
            limit=5,
        )
        self._last_recommendations = recs

        lines = [f"After **{track.artist} - {track.title}** [{track.bpm:.0f} BPM, {track.key}, E{track.energy}]:\n"]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"  {i}. **{r.track.artist} - {r.track.title}** "
                f"[{r.track.bpm:.0f} BPM, {r.track.key}, E{r.track.energy}] "
                f"(score: {r.score:.0%})"
            )
            lines.append(f"     {r.reason}")

        return ChatMessage(
            role="assistant",
            content="\n".join(lines),
            recommendations=recs,
        )

    async def _fallback_search(self, message: str) -> ChatMessage:
        """Search library without Claude API."""
        import re
        cleaned = re.sub(
            r'\b(search|find|look|for|tracks?|songs?|in|library|my)\b',
            '', message.lower()
        ).strip()

        results = []
        for t in self.engine.tracks:
            if cleaned and (cleaned in t.title.lower()
                           or cleaned in t.artist.lower()
                           or cleaned in (t.genre or "").lower()):
                results.append(t)
                if len(results) >= 20:
                    break

        if not results:
            return ChatMessage(
                role="assistant",
                content=f"No tracks found matching '{cleaned}'.",
            )

        lines = [f"Found {len(results)} tracks matching '{cleaned}':\n"]
        for t in results:
            lines.append(
                f"  - {t.artist} - {t.title} [{t.bpm:.0f} BPM, {t.key}, "
                f"E{t.energy}, {t.genre}]"
            )

        return ChatMessage(role="assistant", content="\n".join(lines))

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history.clear()
        self._last_setlist = None
        self._last_recommendations = None
