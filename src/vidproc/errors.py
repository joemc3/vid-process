from __future__ import annotations


class VidprocError(Exception):
    """Base class for all vidproc failures."""


class ConfigError(VidprocError):
    """Configuration is missing or invalid."""


class ExternalToolError(VidprocError):
    """An external binary (ffmpeg, ffprobe, whisper-cli) failed."""
