from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vidproc.errors import ExternalToolError

# Fixed. Every detection threshold in this project is calibrated to these two
# numbers; changing either invalidates all of them.
ENVELOPE_SAMPLE_RATE = 8000
ENVELOPE_HOP_S = 0.25

# Floor for log10 so digital silence yields a finite dB value rather than -inf,
# which would poison percentile and mean calculations downstream.
_EPSILON = 1e-9


@dataclass(frozen=True)
class Envelope:
    db: np.ndarray
    hop_s: float

    @property
    def duration_s(self) -> float:
        return len(self.db) * self.hop_s

    def time_at(self, index: int) -> float:
        return index * self.hop_s

    def index_at(self, seconds: float) -> int:
        return max(0, min(len(self.db) - 1, int(seconds / self.hop_s)))


def extract_pcm(
    src: Path,
    dest: Path,
    sample_rate: int = ENVELOPE_SAMPLE_RATE,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Decode the audio track to mono signed-16 PCM at sample_rate."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", str(dest),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise ExternalToolError(f"{ffmpeg} not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise ExternalToolError(f"audio extraction failed for {src}: {exc.stderr.strip()}") from exc
    return dest


def envelope_from_pcm(
    pcm_path: Path,
    sample_rate: int = ENVELOPE_SAMPLE_RATE,
    hop_s: float = ENVELOPE_HOP_S,
) -> Envelope:
    """RMS envelope in dB over fixed-length frames."""
    samples = np.fromfile(pcm_path, dtype=np.int16).astype(np.float64)
    frame = int(sample_rate * hop_s)
    if frame <= 0:
        raise ValueError(f"hop_s {hop_s} too small for sample rate {sample_rate}")
    usable = len(samples) // frame * frame
    if usable == 0:
        return Envelope(db=np.zeros(0), hop_s=hop_s)
    frames = samples[:usable].reshape(-1, frame)
    rms = np.sqrt((frames**2).mean(axis=1))
    return Envelope(db=20.0 * np.log10(rms + _EPSILON), hop_s=hop_s)


def envelope_for(
    media: Path,
    cache_dir: Path,
    sample_rate: int = ENVELOPE_SAMPLE_RATE,
    hop_s: float = ENVELOPE_HOP_S,
    ffmpeg: str = "ffmpeg",
) -> Envelope:
    """Envelope for a media file, caching the intermediate PCM beside it."""
    media = Path(media)
    pcm = Path(cache_dir) / f"{media.stem}.pcm"
    if not pcm.exists():
        extract_pcm(media, pcm, sample_rate=sample_rate, ffmpeg=ffmpeg)
    return envelope_from_pcm(pcm, sample_rate=sample_rate, hop_s=hop_s)
