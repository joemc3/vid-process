from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from vidproc.errors import ExternalToolError
from vidproc.probe import parse_probe_json, probe

FFMPEG = shutil.which("ffmpeg")


def test_parse_probe_json_extracts_fields() -> None:
    payload = {
        "format": {"duration": "3374.462993"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "16/1",
            },
            {"codec_type": "audio", "sample_rate": "44100", "channels": 2},
        ],
    }
    info = parse_probe_json(payload, Path("x.mp4"))
    assert info.duration_s == pytest.approx(3374.462993)
    assert (info.width, info.height) == (1920, 1080)
    assert info.fps == Fraction(16, 1)
    assert info.video_codec == "h264"
    assert info.audio_sample_rate == 44100
    assert info.audio_channels == 2


def test_parse_probe_json_rejects_missing_video() -> None:
    payload = {"format": {"duration": "10"}, "streams": [{"codec_type": "audio"}]}
    with pytest.raises(ExternalToolError, match="no video stream"):
        parse_probe_json(payload, Path("x.mp4"))


def test_fps_is_never_assumed() -> None:
    """Spec: follow the source rate, never hardcode 16."""
    payload = {
        "format": {"duration": "10"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 640,
                "height": 480,
                "r_frame_rate": "30000/1001",
            },
            {"codec_type": "audio", "sample_rate": "48000", "channels": 1},
        ],
    }
    assert parse_probe_json(payload, Path("x.mp4")).fps == Fraction(30000, 1001)


def test_zero_frame_rate_surfaces_as_external_tool_error() -> None:
    """ffprobe emits r_frame_rate "0/0" in practice - it does so for the audio
    stream of every sample file - and Fraction("0/0") raises ZeroDivisionError,
    which is not a ValueError and so needs naming explicitly."""
    payload = {
        "format": {"duration": "10"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 1920,
             "height": 1080, "r_frame_rate": "0/0"},
            {"codec_type": "audio", "sample_rate": "44100", "channels": 2},
        ],
    }
    with pytest.raises(ExternalToolError, match="unusable stream properties"):
        parse_probe_json(payload, Path("x.mp4"))


def test_missing_stream_field_surfaces_as_external_tool_error() -> None:
    payload = {
        "format": {"duration": "10"},
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "r_frame_rate": "16/1"},
            {"codec_type": "audio", "sample_rate": "44100", "channels": 2},
        ],
    }
    with pytest.raises(ExternalToolError, match="unusable stream properties"):
        parse_probe_json(payload, Path("x.mp4"))


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_probe_real_file(tmp_path: Path) -> None:
    media = tmp_path / "gen.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=16:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(media),
        ],
        check=True,
    )
    info = probe(media)
    assert info.fps == Fraction(16, 1)
    assert (info.width, info.height) == (320, 240)
    assert info.duration_s == pytest.approx(2.0, abs=0.2)


def test_probe_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ExternalToolError):
        probe(tmp_path / "nope.mp4")
