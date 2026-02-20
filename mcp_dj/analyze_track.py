"""
CLI shim enabling: python -m mcp_dj.analyze_track /path/to/song.mp3

See mcp_dj.essentia_analyzer for the full implementation.
"""

from .essentia_analyzer import _cli_main

if __name__ == "__main__":
    _cli_main()
