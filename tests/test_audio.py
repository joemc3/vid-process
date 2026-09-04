from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vidproc.audio import (
    ENVELOPE_HOP_S,
    ENVELOPE_SAMPLE_RATE,
    Envelope,
    _cache_key,
    envelope_for,
    envelope_from_pcm,
)

FFMPEG = shutil.which("ffmpeg")


def _write_pcm(path: Path, blocks: list[tuple[float, float]], sr: int = 8000) -> None:
    """blocks: list of (amplitude 0-1, duration seconds)."""
    parts = []
    for amp, dur in blocks:
        n = int(sr * dur)
        if amp == 0.0:
            parts.append(np.zeros(n, dtype=np.int16))
        else:
            t = np.arange(n) / sr
            parts.append((np.sin(2 * np.pi * 440 * t) * amp * 32000).astype(np.int16))
    np.concatenate(parts).tofile(path)


def test_envelope_indexing_round_trips() -> None:
    env = Envelope(db=np.zeros(40), hop_s=0.25)
    assert env.duration_s == 10.0
    assert env.time_at(8) == 2.0
    assert env.index_at(2.0) == 8
    assert env.index_at(1_000_000) == 39  # clamped


def test_loud_and_silent_blocks_are_separated(tmp_path: Path) -> None:
    pcm = tmp_path / "a.pcm"
    _write_pcm(pcm, [(0.0, 4.0), (0.5, 4.0)])
    env = envelope_from_pcm(pcm, ENVELOPE_SAMPLE_RATE, ENVELOPE_HOP_S)
    assert len(env.db) == 32
    quiet, loud = env.db[:16], env.db[16:]
    assert loud.mean() - quiet.mean() > 40.0


def test_silence_does_not_produce_negative_infinity(tmp_path: Path) -> None:
    """log10 of exact zero must be floored, not -inf, or percentiles break."""
    pcm = tmp_path / "z.pcm"
    _write_pcm(pcm, [(0.0, 2.0)])
    env = envelope_from_pcm(pcm, ENVELOPE_SAMPLE_RATE, ENVELOPE_HOP_S)
    assert np.isfinite(env.db).all()


def test_trailing_partial_frame_is_dropped(tmp_path: Path) -> None:
    pcm = tmp_path / "p.pcm"
    _write_pcm(pcm, [(0.3, 1.1)])  # 1.1s at 0.25s hop -> 4 whole frames
    env = envelope_from_pcm(pcm, ENVELOPE_SAMPLE_RATE, ENVELOPE_HOP_S)
    assert len(env.db) == 4


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_envelope_for_media_and_cache(tmp_path: Path) -> None:
    media = tmp_path / "m.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:r=16:d=6",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(media),
        ],
        check=True,
    )
    cache = tmp_path / "cache"
    env = envelope_for(media, cache)
    assert env.duration_s == pytest.approx(6.0, abs=0.5)
    assert len(list(cache.glob("m-*.pcm"))) == 1


def test_cache_key_distinguishes_same_stem_different_files(tmp_path: Path) -> None:
    """sample/raw/X.mp4 and sample/processed/X.mp4 share a stem, and the cache
    directory is shared across sessions — a stem-only key would collide."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()
    a = tmp_path / "raw" / "sess.mp4"
    a.write_bytes(b"x" * 100)
    b = tmp_path / "processed" / "sess.mp4"
    b.write_bytes(b"y" * 200)
    assert _cache_key(a) != _cache_key(b)


def test_cache_key_changes_when_the_source_changes(tmp_path: Path) -> None:
    """A re-copied or re-recorded capture reuses its filename; it must not
    reuse the previous envelope."""
    p = tmp_path / "sess.mp4"
    p.write_bytes(b"x" * 100)
    first = _cache_key(p)
    p.write_bytes(b"y" * 250)
    assert _cache_key(p) != first


def test_failed_extraction_leaves_no_cache_file(tmp_path: Path) -> None:
    """A partial .pcm would be reused forever: the cache key is the SOURCE's
    size and mtime, which do not change when extraction fails."""
    from vidproc.audio import extract_pcm
    from vidproc.errors import ExternalToolError

    dest = tmp_path / "out.pcm"
    with pytest.raises(ExternalToolError):
        extract_pcm(tmp_path / "nonexistent.mp4", dest)
    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []
