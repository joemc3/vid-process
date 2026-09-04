from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from vidproc.errors import ExternalToolError


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration_s: float
    width: int
    height: int
    fps: Fraction
    video_codec: str
    audio_sample_rate: int
    audio_channels: int


def parse_probe_json(payload: dict[str, Any], path: Path) -> MediaInfo:
    streams = payload.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ExternalToolError(f"no video stream in {path}")
    if audio is None:
        raise ExternalToolError(f"no audio stream in {path}")

    try:
        duration = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExternalToolError(f"no usable duration in {path}") from exc

    try:
        return MediaInfo(
            path=path,
            duration_s=duration,
            width=int(video["width"]),
            height=int(video["height"]),
            # Always read the real rate. Never assume 16 fps.
            fps=Fraction(video["r_frame_rate"]),
            video_codec=str(video.get("codec_name", "")),
            audio_sample_rate=int(audio["sample_rate"]),
            audio_channels=int(audio["channels"]),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        # ffprobe really does emit r_frame_rate "0/0" — it does so for the audio
        # stream of every sample file, and for video on some VFR sources — and
        # Fraction("0/0") is a ZeroDivisionError, which is not a ValueError.
        raise ExternalToolError(f"unusable stream properties in {path}: {exc}") from exc


def probe(path: Path, ffprobe: str = "ffprobe") -> MediaInfo:
    path = Path(path)
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
        "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise ExternalToolError(f"{ffprobe} not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise ExternalToolError(f"ffprobe failed on {path}: {exc.stderr.strip()}") from exc

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ExternalToolError(f"ffprobe returned unparseable JSON for {path}: {exc}") from exc
    return parse_probe_json(payload, path)
