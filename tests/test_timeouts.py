"""Every external call is bounded.

A stalled ffmpeg, ffprobe or whisper-cli — typically on a slow or
network-mounted file — must surface as ExternalToolError within its timeout
rather than hang the run indefinitely.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from vidproc.asr import WhisperCppASR
from vidproc.audio import envelope_for, extract_pcm
from vidproc.config import AsrConfig, MediaConfig, load_config
from vidproc.errors import ConfigError, ExternalToolError
from vidproc.probe import probe

TIMEOUT_S = 0.5
# Generous enough not to flake on a loaded machine, far short of the stand-in's
# 30s sleep, so a missing timeout fails the test instead of passing slowly.
MUST_RETURN_WITHIN_S = 10.0


@pytest.fixture
def hanging_binary(tmp_path: Path) -> str:
    """A stand-in for any external tool that never finishes.

    `exec` matters: it makes sleep the direct child, so killing the child on
    timeout really ends it rather than orphaning a grandchild that holds the
    output pipes open.
    """
    script = tmp_path / "hangs"
    script.write_text("#!/bin/sh\nexec sleep 30\n")
    script.chmod(0o755)
    return str(script)


@pytest.fixture
def media(tmp_path: Path) -> Path:
    path = tmp_path / "session.mp4"
    path.write_bytes(b"\0" * 1024)
    return path


def test_probe_times_out(hanging_binary: str, media: Path) -> None:
    started = time.monotonic()
    with pytest.raises(ExternalToolError, match="timed out"):
        probe(media, ffprobe=hanging_binary, timeout_s=TIMEOUT_S)
    assert time.monotonic() - started < MUST_RETURN_WITHIN_S


def test_audio_decode_times_out_and_leaves_no_cache_file(
    hanging_binary: str, media: Path, tmp_path: Path
) -> None:
    dest = tmp_path / "cache" / "out.pcm"
    started = time.monotonic()
    with pytest.raises(ExternalToolError, match="timed out"):
        extract_pcm(media, dest, ffmpeg=hanging_binary, timeout_s=TIMEOUT_S)
    assert time.monotonic() - started < MUST_RETURN_WITHIN_S
    assert not dest.exists()
    assert list(dest.parent.glob("*.part")) == []


def test_envelope_for_passes_the_timeout_through(
    hanging_binary: str, media: Path, tmp_path: Path
) -> None:
    started = time.monotonic()
    with pytest.raises(ExternalToolError, match="timed out"):
        envelope_for(media, tmp_path / "cache", ffmpeg=hanging_binary, timeout_s=TIMEOUT_S)
    assert time.monotonic() - started < MUST_RETURN_WITHIN_S


def test_asr_window_extraction_times_out(hanging_binary: str, media: Path) -> None:
    asr = WhisperCppASR(
        AsrConfig(model_path="unused.bin", timeout_s=TIMEOUT_S), ffmpeg=hanging_binary
    )
    started = time.monotonic()
    with pytest.raises(ExternalToolError, match="timed out"):
        asr.transcribe(media, 0.0, 10.0)
    assert time.monotonic() - started < MUST_RETURN_WITHIN_S


def test_whisper_times_out(hanging_binary: str, tmp_path: Path) -> None:
    asr = WhisperCppASR(
        AsrConfig(binary=hanging_binary, model_path="unused.bin", timeout_s=TIMEOUT_S)
    )
    wav = tmp_path / "window.wav"
    wav.write_bytes(b"")
    started = time.monotonic()
    with pytest.raises(ExternalToolError, match="timed out"):
        asr._run_whisper(wav, tmp_path / "out")
    assert time.monotonic() - started < MUST_RETURN_WITHIN_S


def test_timeout_defaults(tmp_path: Path) -> None:
    """Measured on a 56-minute 2025 session on Apple Silicon: ffprobe 0.4s, full
    audio decode 2s, window extraction 0.1s, whisper large-v3 on a 180s window
    4s warm and ~35s cold. Defaults leave wide headroom for a slower machine or
    a network-mounted file while still turning a stall into an error."""
    cfg = load_config(None, working_dir=tmp_path)
    assert cfg.media.probe_timeout_s == 60.0
    assert cfg.media.decode_timeout_s == 600.0
    assert cfg.asr.timeout_s == 300.0


def test_timeouts_are_configurable(tmp_path: Path) -> None:
    path = tmp_path / "cfg.json"
    path.write_text(
        '{"media": {"probe_timeout_s": 5, "decode_timeout_s": 50},'
        ' "asr": {"timeout_s": 30}}'
    )
    cfg = load_config(path, working_dir=tmp_path)
    assert cfg.media.probe_timeout_s == 5
    assert cfg.media.decode_timeout_s == 50
    assert cfg.asr.timeout_s == 30


@pytest.mark.parametrize(
    "section",
    [
        MediaConfig(probe_timeout_s=0.0),
        MediaConfig(decode_timeout_s=-1.0),
        AsrConfig(timeout_s=0.0),
    ],
)
def test_non_positive_timeouts_are_rejected(section: MediaConfig | AsrConfig) -> None:
    with pytest.raises(ConfigError, match="timeout"):
        section.validate()
