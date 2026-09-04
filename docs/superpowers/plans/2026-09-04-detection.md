# Session Boundary Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Given a raw Poster Theater capture, produce a reviewable proposal containing session start and end timecodes, a confidence classification for each, and the transcript evidence behind them.

**Architecture:** A pure-function core (RMS envelope → boundary detection) wrapped by thin adapters over external tools (ffprobe, ffmpeg, whisper-cli) and a pluggable LLM refiner. End detection is energy-only and needs no model. Head detection produces a conservative energy candidate, then optionally refines it against a transcript. Everything is written to a sidecar JSON so later stages and humans can audit it.

**Tech Stack:** Python 3.11+, numpy, httpx, pytest, `uv` for dependency management. External binaries: `ffmpeg`, `ffprobe`, `whisper-cli` (whisper.cpp). Optional: Ollama (local) or OpenRouter (remote) for head refinement.

**Spec:** `docs/superpowers/specs/2026-09-04-session-trim-pipeline-design.md`

This plan implements spec steps 4–5 only. Render, distribute, and ingest hardening are separate plans.

**Two deliberate deviations from the spec, recorded so they aren't mistaken for oversights:**

1. **Spec §6.5 names Parakeet TDT as the preferred ASR; this plan ships whisper.cpp.** Parakeet is not installed and its accuracy on this audio is unverified (spec §12.4). Task 5 defines the `ASR` protocol so a Parakeet backend is a drop-in addition scored against the same fixtures — that swap is follow-up work, not part of this plan. Whisper's hallucination-on-silence risk is neutralised structurally: it is only ever handed windows the energy pass already identified as containing speech.
2. **Spec §10 lists `envelope.npy` and `transcript.json` as sidecars; this plan caches the intermediate PCM instead and embeds the window transcripts in `proposal.json`.** Recomputing an envelope from cached PCM takes well under a second, so a second cache layer earns nothing, and keeping the transcripts inside the proposal means the evidence and the decision it supports cannot drift apart. The audit requirement in §10 is met.

## Global Constraints

- **Python 3.11+.** Use `from __future__ import annotations` in every module.
- **Never commit or push anything under `sample/`.** It holds real conference footage of identifiable presenters. It is gitignored; keep it that way. Test fixtures committed to the repo are envelopes, transcripts, and timecodes — never audio or video.
- **Remote LLM access goes through OpenRouter only.** Never the Anthropic, OpenAI, or any other provider API directly.
- **`min_tail_silence_s` must never exceed 20.0.** PT01's recording stopped 27s after the session ended; a higher value finds no qualifying silence and returns garbage. Enforce in config validation.
- **Envelope parameters are fixed:** 8000 Hz mono PCM, 0.25 s hop. Detection constants are calibrated to these; changing either invalidates every threshold in this plan.
- **Detection defaults** (spec §6.2, §6.4): `floor_percentile=5.0`, `gate_offset_db=8.0`, `min_speech_s=3.0`, `min_tail_silence_s=10.0`, `head_window_s=180.0`, `tail_window_s=60.0`, `head_preroll_s=1.0`, `tail_pad_s=2.0`.
- **Ground truth for regression** (spec §3): PT01 head 958.0 / end 4292.0; PT06 head 611.0 / end 4323.0; PT12 head 290.0 / end 3184.0.
- **The pipeline must never fail a session because a model was unavailable.** Refinement degrades to the energy candidate with the head flagged low-confidence.
- **Never assume 16 fps or any source property.** Probe it.
- Run everything through `uv run` so the environment is reproducible.

---

### Task 1: Project scaffolding and configuration

**Files:**
- Create: `pyproject.toml`
- Create: `src/vidproc/__init__.py`
- Create: `src/vidproc/errors.py`
- Create: `src/vidproc/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Config`, `DetectionConfig`, `RefinerConfig`, `AsrConfig` dataclasses; `load_config(path: Path | None) -> Config`; exceptions `VidprocError`, `ConfigError`, `ExternalToolError`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vidproc.config import DetectionConfig, load_config
from vidproc.errors import ConfigError


def test_defaults_match_spec(tmp_path: Path) -> None:
    cfg = load_config(None, working_dir=tmp_path)
    d = cfg.detection
    assert d.floor_percentile == 5.0
    assert d.gate_offset_db == 8.0
    assert d.min_speech_s == 3.0
    assert d.min_tail_silence_s == 10.0
    assert d.head_window_s == 180.0
    assert d.tail_window_s == 60.0
    assert d.head_preroll_s == 1.0
    assert d.tail_pad_s == 2.0
    assert cfg.refiner.mode == "none"


def test_tail_silence_above_20_is_rejected() -> None:
    """PT01's tail is only 27s; >20s finds no qualifying silence. Spec §6.2."""
    with pytest.raises(ConfigError, match="min_tail_silence_s"):
        DetectionConfig(min_tail_silence_s=30.0).validate()


def test_tail_silence_at_20_is_allowed() -> None:
    DetectionConfig(min_tail_silence_s=20.0).validate()


def test_unknown_refiner_mode_is_rejected() -> None:
    from vidproc.config import RefinerConfig

    with pytest.raises(ConfigError, match="mode"):
        RefinerConfig(mode="anthropic").validate()


def test_non_none_mode_requires_an_explicit_model() -> None:
    """No safe default exists: some Ollama models carry a :cloud suffix and run
    off-machine, so a guessed default could silently defeat 'local'."""
    from vidproc.config import RefinerConfig

    with pytest.raises(ConfigError, match="requires a model name"):
        RefinerConfig(mode="local").validate()


def test_file_overrides_merge_over_defaults(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps(
            {
                "detection": {"gate_offset_db": 12.0},
                "refiner": {"mode": "local", "model": "qwen3.6:35b-a3b"},
            }
        )
    )
    cfg = load_config(p, working_dir=tmp_path)
    assert cfg.detection.gate_offset_db == 12.0
    assert cfg.detection.min_speech_s == 3.0  # untouched default
    assert cfg.refiner.mode == "local"
    assert cfg.refiner.model == "qwen3.6:35b-a3b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vidproc'`

- [ ] **Step 3: Write the implementation**

Create `pyproject.toml`:

```toml
[project]
name = "vidproc"
version = "0.1.0"
description = "Poster Theater session trim pipeline"
requires-python = ">=3.11"
dependencies = ["numpy>=2.0", "httpx>=0.27"]

[project.scripts]
vidproc = "vidproc.cli:main"

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vidproc"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = ["samples: requires real footage under sample/ (deselect with -m 'not samples')"]
```

Create `src/vidproc/__init__.py`:

```python
from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
```

Create `src/vidproc/errors.py`:

```python
from __future__ import annotations


class VidprocError(Exception):
    """Base class for all vidproc failures."""


class ConfigError(VidprocError):
    """Configuration is missing or invalid."""


class ExternalToolError(VidprocError):
    """An external binary (ffmpeg, ffprobe, whisper-cli) failed."""
```

Create `src/vidproc/config.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from vidproc.errors import ConfigError

REFINER_MODES = ("none", "local", "openrouter")

# Hard ceiling: PT01's recording stopped 27s after the session ended. A larger
# requirement finds no qualifying silence and the detector returns garbage.
MAX_TAIL_SILENCE_S = 20.0


@dataclass(frozen=True)
class DetectionConfig:
    floor_percentile: float = 5.0
    gate_offset_db: float = 8.0
    min_speech_s: float = 3.0
    min_tail_silence_s: float = 10.0
    head_window_s: float = 180.0
    tail_window_s: float = 60.0
    head_preroll_s: float = 1.0
    tail_pad_s: float = 2.0

    def validate(self) -> None:
        if not 0.0 < self.floor_percentile < 50.0:
            raise ConfigError(f"floor_percentile must be in (0, 50): {self.floor_percentile}")
        if self.gate_offset_db <= 0.0:
            raise ConfigError(f"gate_offset_db must be positive: {self.gate_offset_db}")
        if self.min_speech_s <= 0.0:
            raise ConfigError(f"min_speech_s must be positive: {self.min_speech_s}")
        if not 0.0 < self.min_tail_silence_s <= MAX_TAIL_SILENCE_S:
            raise ConfigError(
                f"min_tail_silence_s must be in (0, {MAX_TAIL_SILENCE_S}]: "
                f"{self.min_tail_silence_s}. Larger values fail on short tails (spec §6.2)."
            )
        for name in ("head_window_s", "tail_window_s"):
            if getattr(self, name) <= 0.0:
                raise ConfigError(f"{name} must be positive: {getattr(self, name)}")
        for name in ("head_preroll_s", "tail_pad_s"):
            if getattr(self, name) < 0.0:
                raise ConfigError(f"{name} must not be negative: {getattr(self, name)}")


@dataclass(frozen=True)
class RefinerConfig:
    mode: str = "none"
    model: str = ""
    base_url: str = ""
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_s: float = 60.0

    def validate(self) -> None:
        if self.mode not in REFINER_MODES:
            raise ConfigError(f"refiner mode must be one of {REFINER_MODES}: {self.mode!r}")
        if self.mode != "none" and not self.model:
            raise ConfigError(f"refiner mode {self.mode!r} requires a model name")
        if self.timeout_s <= 0.0:
            raise ConfigError(f"timeout_s must be positive: {self.timeout_s}")

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        if self.mode == "local":
            return "http://localhost:11434/v1"
        if self.mode == "openrouter":
            return "https://openrouter.ai/api/v1"
        raise ConfigError(f"no base URL for refiner mode {self.mode!r}")


@dataclass(frozen=True)
class AsrConfig:
    binary: str = "whisper-cli"
    model_path: str = ""
    language: str = "en"

    def validate(self) -> None:
        if not self.binary:
            raise ConfigError("asr.binary must not be empty")


@dataclass(frozen=True)
class Config:
    working_dir: Path
    detection: DetectionConfig = DetectionConfig()
    refiner: RefinerConfig = RefinerConfig()
    asr: AsrConfig = AsrConfig()

    def validate(self) -> None:
        self.detection.validate()
        self.refiner.validate()
        self.asr.validate()


def _merge(base: Any, overrides: dict[str, Any], label: str) -> Any:
    unknown = set(overrides) - {f for f in base.__dataclass_fields__}
    if unknown:
        raise ConfigError(f"unknown {label} keys: {sorted(unknown)}")
    return replace(base, **overrides)


def load_config(path: Path | None, working_dir: Path) -> Config:
    """Load configuration, merging a JSON file over the spec defaults."""
    raw: dict[str, Any] = {}
    if path is not None:
        try:
            raw = json.loads(Path(path).read_text())
        except FileNotFoundError as exc:
            raise ConfigError(f"config file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON in {path}: {exc}") from exc

    cfg = Config(
        working_dir=Path(working_dir),
        detection=_merge(DetectionConfig(), raw.get("detection", {}), "detection"),
        refiner=_merge(RefinerConfig(), raw.get("refiner", {}), "refiner"),
        asr=_merge(AsrConfig(), raw.get("asr", {}), "asr"),
    )
    cfg.validate()
    return cfg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/vidproc/__init__.py src/vidproc/errors.py src/vidproc/config.py tests/test_config.py
git commit -m "feat: project scaffolding and configuration with spec-derived defaults"
```

---

### Task 2: Media probe

**Files:**
- Create: `src/vidproc/probe.py`
- Test: `tests/test_probe.py`

**Interfaces:**
- Consumes: `vidproc.errors.ExternalToolError`
- Produces: `MediaInfo` frozen dataclass with fields `path: Path`, `duration_s: float`, `width: int`, `height: int`, `fps: Fraction`, `video_codec: str`, `audio_sample_rate: int`, `audio_channels: int`; function `probe(path: Path, ffprobe: str = "ffprobe") -> MediaInfo`

- [ ] **Step 1: Write the failing test**

Create `tests/test_probe.py`:

```python
from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from vidproc.errors import ExternalToolError
from vidproc.probe import MediaInfo, parse_probe_json, probe

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_probe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vidproc.probe'`

- [ ] **Step 3: Write the implementation**

Create `src/vidproc/probe.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_probe.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/vidproc/probe.py tests/test_probe.py
git commit -m "feat: ffprobe wrapper reading real source properties"
```

---

### Task 3: Audio envelope

**Files:**
- Create: `src/vidproc/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Consumes: `vidproc.errors.ExternalToolError`
- Produces: constants `ENVELOPE_SAMPLE_RATE = 8000`, `ENVELOPE_HOP_S = 0.25`; `Envelope` frozen dataclass with `db: np.ndarray`, `hop_s: float`, methods `time_at(i) -> float`, `index_at(t) -> int`, property `duration_s`; functions `extract_pcm(src, dest, sample_rate, ffmpeg="ffmpeg") -> Path`, `envelope_from_pcm(pcm_path, sample_rate, hop_s) -> Envelope`, `envelope_for(media, cache_dir, ...) -> Envelope`

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_audio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vidproc.audio'`

- [ ] **Step 3: Write the implementation**

Create `src/vidproc/audio.py`:

```python
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


def _cache_key(media: Path) -> str:
    """Cache key that changes whenever the source does.

    The stem alone is not enough, and the cache directory is shared across
    every session. This project's own layout puts raw/<session>.mp4 and
    processed/<session>.mp4 side by side with identical stems, and a re-copied
    or re-recorded capture reuses its name. Either would silently serve the
    wrong envelope — and since every detection threshold is computed from it,
    the result would be confidently wrong with nothing failing.
    """
    stat = media.stat()
    return f"{media.stem}-{stat.st_size}-{int(stat.st_mtime)}"


def envelope_for(
    media: Path,
    cache_dir: Path,
    sample_rate: int = ENVELOPE_SAMPLE_RATE,
    hop_s: float = ENVELOPE_HOP_S,
    ffmpeg: str = "ffmpeg",
) -> Envelope:
    """Envelope for a media file, caching the intermediate PCM."""
    media = Path(media)
    pcm = Path(cache_dir) / f"{_cache_key(media)}.pcm"
    if not pcm.exists():
        extract_pcm(media, pcm, sample_rate=sample_rate, ffmpeg=ffmpeg)
    return envelope_from_pcm(pcm, sample_rate=sample_rate, hop_s=hop_s)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_audio.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/vidproc/audio.py tests/test_audio.py
git commit -m "feat: RMS envelope extraction at fixed 8kHz/0.25s resolution"
```

---

### Task 4: Boundary detection

This is the core. Every function here is pure — it takes an `Envelope` and returns numbers, with no I/O — so it tests against synthesised envelopes with known structure, including both pathologies observed in the real footage.

**Files:**
- Create: `src/vidproc/detect.py`
- Test: `tests/test_detect.py`

**Interfaces:**
- Consumes: `vidproc.audio.Envelope`, `vidproc.config.DetectionConfig`
- Produces: `Boundaries` frozen dataclass with `noise_floor_db: float`, `gate_db: float`, `head_candidate_s: float`, `end_s: float`, `pre_head_silence_s: float`, `tail_silence_s: float`; functions `noise_floor_db(env, percentile) -> float`, `runs_above(env, gate_db, min_duration_s) -> list[tuple[float, float]]`, `final_silence_start(env, gate_db, min_silence_s) -> tuple[float, float]`, `detect_boundaries(env, cfg) -> Boundaries`

- [ ] **Step 1: Write the failing test**

Create `tests/test_detect.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from vidproc.audio import Envelope
from vidproc.config import DetectionConfig
from vidproc.detect import (
    Boundaries,
    detect_boundaries,
    final_silence_start,
    noise_floor_db,
    runs_above,
)

HOP = 0.25
FLOOR, PLATEAU, SPEECH = 8.0, 12.0, 35.0


def build(segments: list[tuple[float, float]]) -> Envelope:
    """segments: list of (level_db, duration_s) -> Envelope at 0.25s hop."""
    parts = [np.full(int(round(d / HOP)), lvl, dtype=np.float64) for lvl, d in segments]
    return Envelope(db=np.concatenate(parts), hop_s=HOP)


def test_noise_floor_ignores_loud_majority() -> None:
    env = build([(FLOOR, 60.0), (SPEECH, 600.0)])
    assert noise_floor_db(env, 5.0) == pytest.approx(FLOOR, abs=0.5)


def test_runs_above_filters_short_bursts() -> None:
    env = build([(FLOOR, 10.0), (SPEECH, 1.0), (FLOOR, 10.0), (SPEECH, 8.0), (FLOOR, 5.0)])
    runs = runs_above(env, gate_db=16.0, min_duration_s=3.0)
    assert len(runs) == 1
    assert runs[0][0] == pytest.approx(21.0)


def test_runs_above_handles_envelope_starting_and_ending_loud() -> None:
    env = build([(SPEECH, 10.0)])
    runs = runs_above(env, gate_db=16.0, min_duration_s=3.0)
    assert runs == [(0.0, 10.0)]


def test_open_mic_plateau_does_not_trigger() -> None:
    """PT12: 14s of open-mic room tone at ~12 dB precedes real speech."""
    env = build([(FLOOR, 275.0), (PLATEAU, 14.0), (SPEECH, 100.0)])
    b = detect_boundaries(env, DetectionConfig())
    assert b.gate_db > PLATEAU
    assert b.head_candidate_s == pytest.approx(289.0, abs=1.0)


def test_head_candidate_is_early_on_pt01_pathology() -> None:
    """PT01: moderator speaks, pauses, then the real start. Energy must land on
    the FIRST speech (a safe lower bound); the refiner moves it forward."""
    env = build([(FLOOR, 900.0), (SPEECH, 20.0), (PLATEAU, 25.0), (SPEECH, 600.0)])
    b = detect_boundaries(env, DetectionConfig())
    assert b.head_candidate_s == pytest.approx(900.0, abs=1.0)


def test_end_is_start_of_final_long_silence() -> None:
    env = build([(FLOOR, 100.0), (SPEECH, 500.0), (FLOOR, 190.0)])
    b = detect_boundaries(env, DetectionConfig())
    assert b.end_s == pytest.approx(600.0, abs=0.5)
    assert b.tail_silence_s == pytest.approx(190.0, abs=0.5)


def test_end_survives_a_short_27_second_tail() -> None:
    """PT01's recording stopped 27s after the session ended."""
    env = build([(FLOOR, 100.0), (SPEECH, 500.0), (FLOOR, 27.0)])
    b = detect_boundaries(env, DetectionConfig())
    assert b.end_s == pytest.approx(600.0, abs=0.5)


def test_end_ignores_short_mid_session_gaps() -> None:
    env = build([(FLOOR, 60.0), (SPEECH, 200.0), (FLOOR, 12.0), (SPEECH, 200.0), (FLOOR, 100.0)])
    b = detect_boundaries(env, DetectionConfig())
    assert b.end_s == pytest.approx(472.0, abs=0.5)


def test_no_trailing_silence_yields_end_at_duration() -> None:
    """Recording stopped mid-session: trim nothing from the end.

    Also guards the leading-silence trap — the 60s of quiet at the FRONT is a
    long silence run, but it is not the tail and must never be returned as the
    end cut.
    """
    env = build([(FLOOR, 60.0), (SPEECH, 200.0)])
    b = detect_boundaries(env, DetectionConfig())
    assert b.end_s == pytest.approx(env.duration_s, abs=0.5)
    assert b.tail_silence_s == pytest.approx(0.0, abs=0.5)


def test_pre_head_silence_is_measured() -> None:
    env = build([(FLOOR, 300.0), (SPEECH, 100.0), (FLOOR, 100.0)])
    b = detect_boundaries(env, DetectionConfig())
    assert b.pre_head_silence_s == pytest.approx(300.0, abs=1.0)


def test_silent_file_does_not_crash() -> None:
    env = build([(FLOOR, 120.0)])
    b = detect_boundaries(env, DetectionConfig())
    assert isinstance(b, Boundaries)
    assert b.head_candidate_s == 0.0
    assert b.end_s == pytest.approx(env.duration_s, abs=0.5)


def test_final_silence_start_returns_start_and_length() -> None:
    env = build([(SPEECH, 100.0), (FLOOR, 40.0)])
    start, length = final_silence_start(env, gate_db=16.0, min_silence_s=10.0)
    assert start == pytest.approx(100.0, abs=0.5)
    assert length == pytest.approx(40.0, abs=0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vidproc.detect'`

- [ ] **Step 3: Write the implementation**

Create `src/vidproc/detect.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vidproc.audio import Envelope
from vidproc.config import DetectionConfig


@dataclass(frozen=True)
class Boundaries:
    noise_floor_db: float
    gate_db: float
    head_candidate_s: float
    end_s: float
    pre_head_silence_s: float
    tail_silence_s: float


def noise_floor_db(env: Envelope, percentile: float) -> float:
    """Per-file noise floor. Derived, never hardcoded — see spec §6.1."""
    if len(env.db) == 0:
        return 0.0
    return float(np.percentile(env.db, percentile))


def _boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) index ranges where mask is True."""
    if len(mask) == 0 or not mask.any():
        return []
    diff = np.diff(mask.astype(np.int8))
    starts = (np.flatnonzero(diff == 1) + 1).tolist()
    ends = (np.flatnonzero(diff == -1) + 1).tolist()
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))


def runs_above(env: Envelope, gate_db: float, min_duration_s: float) -> list[tuple[float, float]]:
    """Time ranges where the envelope stays above the gate for long enough."""
    return [
        (s * env.hop_s, e * env.hop_s)
        for s, e in _boolean_runs(env.db > gate_db)
        if (e - s) * env.hop_s >= min_duration_s
    ]


def final_silence_start(
    env: Envelope, gate_db: float, min_silence_s: float
) -> tuple[float, float]:
    """(start, length) of the TRAILING silence — the quiet that runs to the end.

    Anchored to the last frame above the gate, not to "the last long quiet
    run". Those differ: a file that opens with five minutes of silence and
    then never stops talking has a long quiet run, but it is at the front and
    is emphatically not the end of the session.

    Returns (duration, 0.0) — trim nothing from the end — when nothing is ever
    above the gate, or when too little silence follows the last audio, meaning
    the recording was stopped while the session was still going.
    """
    loud = env.db > gate_db
    if not loud.any():
        return env.duration_s, 0.0
    last_loud = int(np.flatnonzero(loud)[-1])
    silence_s = (len(loud) - 1 - last_loud) * env.hop_s
    if silence_s < min_silence_s:
        return env.duration_s, 0.0
    return (last_loud + 1) * env.hop_s, silence_s


def detect_boundaries(env: Envelope, cfg: DetectionConfig) -> Boundaries:
    """Energy-only boundary detection.

    The head is a deliberately conservative lower bound: the first sustained
    speech. On clean sessions it is within a second of the human cut; where
    something is said before the session proper (PT01), it lands early and the
    refiner moves it forward. It must never land late.
    """
    floor = noise_floor_db(env, cfg.floor_percentile)
    gate = floor + cfg.gate_offset_db

    speech = runs_above(env, gate, cfg.min_speech_s)
    head = speech[0][0] if speech else 0.0

    end, tail_silence = final_silence_start(env, gate, cfg.min_tail_silence_s)

    # How much quiet immediately precedes the head candidate. A long run means an
    # unambiguous session start; a short one means something was already going on.
    pre_head = 0.0
    if head > 0.0:
        idx = env.index_at(head)
        quiet = env.db[:idx] <= gate
        trailing = 0
        for value in quiet[::-1]:
            if not value:
                break
            trailing += 1
        pre_head = trailing * env.hop_s

    return Boundaries(
        noise_floor_db=floor,
        gate_db=gate,
        head_candidate_s=head,
        end_s=end,
        pre_head_silence_s=pre_head,
        tail_silence_s=tail_silence,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_detect.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/vidproc/detect.py tests/test_detect.py
git commit -m "feat: energy-based boundary detection with PT01/PT12 pathology coverage"
```

---

### Task 5: ASR adapter

**Files:**
- Create: `src/vidproc/asr.py`
- Test: `tests/test_asr.py`

**Interfaces:**
- Consumes: `vidproc.config.AsrConfig`, `vidproc.errors.ExternalToolError`
- Produces: `Word` frozen dataclass with `text: str`, `start_s: float`, `end_s: float`; `ASR` Protocol with `transcribe(media: Path, start_s: float, end_s: float) -> list[Word]`; `WhisperCppASR` class; `parse_whisper_json(payload: dict, offset_s: float) -> list[Word]`; `words_to_lines(words, max_gap_s=0.8, max_words=14) -> list[Line]` where `Line` has `text: str`, `start_s: float`, `end_s: float`

- [ ] **Step 1: Write the failing test**

Create `tests/test_asr.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from vidproc.asr import Line, Word, parse_whisper_json, words_to_lines


def _seg(text: str, start_ms: int, end_ms: int) -> dict:
    return {"offsets": {"from": start_ms, "to": end_ms}, "text": text}


def test_parse_applies_absolute_offset() -> None:
    payload = {"transcription": [_seg(" Good", 500, 900), _seg(" morning", 900, 1400)]}
    words = parse_whisper_json(payload, offset_s=290.0)
    assert [w.text for w in words] == ["Good", "morning"]
    assert words[0].start_s == pytest.approx(290.5)
    assert words[1].end_s == pytest.approx(291.4)


def test_parse_skips_empty_segments() -> None:
    """whisper-cli emits blank leading segments with -ml 1."""
    payload = {"transcription": [_seg("", 0, 320), _seg(" Hello", 320, 800)]}
    assert [w.text for w in parse_whisper_json(payload, 0.0)] == ["Hello"]


def test_parse_empty_transcription_is_empty_list() -> None:
    assert parse_whisper_json({"transcription": []}, 0.0) == []


def test_malformed_segment_surfaces_as_external_tool_error() -> None:
    """A segment with text but no offsets is unusable output from an external
    tool, and must not escape as a raw KeyError."""
    from vidproc.errors import ExternalToolError

    with pytest.raises(ExternalToolError, match="unusable whisper segment"):
        parse_whisper_json({"transcription": [{"text": " Hello"}]}, 0.0)


def test_words_to_lines_splits_on_long_pause() -> None:
    words = [
        Word("Hello", 0.0, 0.4),
        Word("everybody", 0.4, 1.0),
        Word("Welcome", 5.0, 5.5),   # 4s gap
        Word("everyone", 5.5, 6.0),
    ]
    lines = words_to_lines(words, max_gap_s=0.8)
    assert len(lines) == 2
    assert lines[0].text == "Hello everybody"
    assert lines[1].start_s == pytest.approx(5.0)


def test_words_to_lines_caps_line_length() -> None:
    words = [Word(f"w{i}", i * 0.2, i * 0.2 + 0.1) for i in range(30)]
    lines = words_to_lines(words, max_gap_s=5.0, max_words=14)
    assert all(len(line.text.split()) <= 14 for line in lines)
    assert len(lines) == 3


def test_words_to_lines_on_empty_input() -> None:
    assert words_to_lines([], max_gap_s=0.8) == []


def test_line_is_hashable_and_frozen() -> None:
    line = Line(text="x", start_s=1.0, end_s=2.0)
    with pytest.raises(Exception):
        line.text = "y"  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_asr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vidproc.asr'`

- [ ] **Step 3: Write the implementation**

Create `src/vidproc/asr.py`:

```python
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from vidproc.config import AsrConfig
from vidproc.errors import ExternalToolError


@dataclass(frozen=True)
class Word:
    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class Line:
    text: str
    start_s: float
    end_s: float


class ASR(Protocol):
    def transcribe(self, media: Path, start_s: float, end_s: float) -> list[Word]:
        """Transcribe [start_s, end_s) of media, timestamps absolute in source."""
        ...


def parse_whisper_json(payload: dict[str, Any], offset_s: float) -> list[Word]:
    """Parse whisper-cli --output-json, shifting to the source timeline.

    whisper-cli reports millisecond offsets relative to the clip it was given,
    and emits blank-text segments which are dropped.
    """
    words: list[Word] = []
    for segment in payload.get("transcription", []):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        try:
            offsets = segment["offsets"]
            start_s = offset_s + float(offsets["from"]) / 1000.0
            end_s = offset_s + float(offsets["to"]) / 1000.0
        except (KeyError, TypeError, ValueError) as exc:
            # Same contract as probe.py: an external tool's malformed output
            # surfaces as ExternalToolError, never a raw KeyError.
            raise ExternalToolError(f"unusable whisper segment {segment!r}") from exc
        words.append(Word(text=text, start_s=start_s, end_s=end_s))
    return words


def words_to_lines(words: list[Word], max_gap_s: float = 0.8, max_words: int = 14) -> list[Line]:
    """Group words into readable lines, breaking on pauses and length."""
    lines: list[Line] = []
    current: list[Word] = []

    def flush() -> None:
        if current:
            lines.append(
                Line(
                    text=" ".join(w.text for w in current),
                    start_s=current[0].start_s,
                    end_s=current[-1].end_s,
                )
            )
            current.clear()

    for word in words:
        if current and (
            word.start_s - current[-1].end_s > max_gap_s or len(current) >= max_words
        ):
            flush()
        current.append(word)
    flush()
    return lines


class WhisperCppASR:
    """whisper.cpp adapter.

    Only ever given windows the energy pass already identified as containing
    speech, so Whisper's hallucination-on-silence failure mode cannot bite.
    """

    def __init__(self, cfg: AsrConfig, ffmpeg: str = "ffmpeg") -> None:
        if not cfg.model_path:
            raise ExternalToolError("asr.model_path is required for whisper-cpp")
        self._cfg = cfg
        self._ffmpeg = ffmpeg

    def transcribe(self, media: Path, start_s: float, end_s: float) -> list[Word]:
        if end_s <= start_s:
            return []
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            wav = tmpdir / "window.wav"
            self._extract(media, wav, start_s, end_s)
            stem = tmpdir / "out"
            self._run_whisper(wav, stem)
            raw = (stem.with_suffix(".json")).read_text()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ExternalToolError(f"whisper-cli returned unparseable JSON: {exc}") from exc
        return parse_whisper_json(payload, offset_s=start_s)

    def _extract(self, media: Path, wav: Path, start_s: float, end_s: float) -> None:
        cmd = [
            self._ffmpeg, "-y", "-loglevel", "error",
            "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", str(media),
            "-vn", "-ac", "1", "-ar", "16000", str(wav),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise ExternalToolError(f"{self._ffmpeg} not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise ExternalToolError(f"window extraction failed: {exc.stderr.strip()}") from exc

    def _run_whisper(self, wav: Path, output_stem: Path) -> None:
        cmd = [
            self._cfg.binary,
            "-m", self._cfg.model_path,
            "-f", str(wav),
            "-l", self._cfg.language,
            "-ml", "1",            # one segment per token
            "-sow",                # split on word, not sub-token
            "-oj",                 # JSON output
            "-of", str(output_stem),
            "--no-prints",
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise ExternalToolError(f"{self._cfg.binary} not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise ExternalToolError(f"whisper-cli failed: {exc.stderr.strip()}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_asr.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/vidproc/asr.py tests/test_asr.py
git commit -m "feat: whisper.cpp ASR adapter with absolute-timeline word output"
```

---

### Task 6: Head refinement backends

Three modes behind one interface. `local` and `openrouter` differ only in base URL, model, and key, because Ollama and OpenRouter are both OpenAI-compatible.

The model is asked to pick a **line index**, not a timestamp. Indices are trivially validatable; a hallucinated float is not.

**Files:**
- Create: `src/vidproc/refine.py`
- Test: `tests/test_refine.py`

**Interfaces:**
- Consumes: `vidproc.asr.Line`, `vidproc.config.RefinerConfig`
- Produces: `Refinement` frozen dataclass with `start_s: float`, `reason: str`, `skipped: str`, `model_used: str`, `available: bool`; `Refiner` Protocol with `refine(lines, candidate_s) -> Refinement`; `NullRefiner`; `OpenAICompatibleRefiner`; `build_refiner(cfg) -> Refiner`; helpers `format_lines(lines) -> str`, `apply_choice(lines, index, reason, skipped, model) -> Refinement`

- [ ] **Step 1: Write the failing test**

Create `tests/test_refine.py`:

```python
from __future__ import annotations

import httpx
import pytest

from vidproc.asr import Line
from vidproc.config import RefinerConfig
from vidproc.refine import (
    NullRefiner,
    OpenAICompatibleRefiner,
    apply_choice,
    build_refiner,
    format_lines,
)

PT01 = [
    Line("Sorry, we're having a little technical difficulty.", 900.0, 910.0),
    Line("I'll just remind the audience that this session is being recorded.", 910.0, 916.0),
    Line("Hello?", 953.0, 955.0),
    Line("Good morning, everyone. My name is Alex Rivera.", 958.0, 962.0),
]


def test_null_refiner_returns_candidate_and_reports_unavailable() -> None:
    r = NullRefiner().refine(PT01, candidate_s=900.0)
    assert r.start_s == 900.0
    assert r.available is False


def test_format_lines_is_indexed_and_timestamped() -> None:
    text = format_lines(PT01[:2])
    assert text.splitlines()[0].startswith("0\t900.00\t")
    assert "technical difficulty" in text


def test_apply_choice_maps_index_to_line_start() -> None:
    r = apply_choice(PT01, 3, "first presenter begins", "apology, housekeeping", "m")
    assert r.start_s == 958.0
    assert r.available is True


def test_apply_choice_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="out of range"):
        apply_choice(PT01, 99, "x", "y", "m")


def test_build_refiner_none_mode_gives_null() -> None:
    assert isinstance(build_refiner(RefinerConfig(mode="none")), NullRefiner)


def _refiner(handler) -> OpenAICompatibleRefiner:
    cfg = RefinerConfig(mode="local", model="test-model", base_url="http://x/v1")
    transport = httpx.MockTransport(handler)
    return OpenAICompatibleRefiner(cfg, client=httpx.Client(transport=transport))


def test_remote_refiner_picks_the_line_the_model_chose() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = '{"start_line": 3, "reason": "first presenter", "skipped": "apology"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    r = _refiner(handler).refine(PT01, candidate_s=900.0)
    assert r.start_s == 958.0
    assert r.available is True
    assert "presenter" in r.reason


def test_connection_error_degrades_to_candidate() -> None:
    """Spec: never fail a session because a model was unavailable."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    r = _refiner(handler).refine(PT01, candidate_s=900.0)
    assert r.start_s == 900.0
    assert r.available is False


def test_http_error_degrades_to_candidate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    assert _refiner(handler).refine(PT01, candidate_s=900.0).available is False


def test_malformed_json_degrades_to_candidate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    assert _refiner(handler).refine(PT01, candidate_s=900.0).available is False


def test_out_of_range_index_from_model_degrades_to_candidate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = '{"start_line": 42, "reason": "r", "skipped": "s"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    assert _refiner(handler).refine(PT01, candidate_s=900.0).available is False


def test_empty_choices_list_degrades_to_candidate() -> None:
    """A 200 response carrying no choices must degrade, not raise. IndexError is
    a LookupError, not a KeyError, so catching KeyError alone misses it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    assert _refiner(handler).refine(PT01, candidate_s=900.0).available is False


def test_empty_lines_returns_candidate_without_calling_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("model must not be called with no transcript")

    r = _refiner(handler).refine([], candidate_s=900.0)
    assert r.start_s == 900.0
    assert r.available is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_refine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vidproc.refine'`

- [ ] **Step 3: Write the implementation**

Create `src/vidproc/refine.py`:

```python
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

import httpx

from vidproc.asr import Line
from vidproc.config import RefinerConfig

SYSTEM_PROMPT = """\
You are trimming the opening of a recorded conference session so it can be published.

You are given numbered transcript lines from the very start of a recording, each with a
timestamp in seconds. Choose the line where the published video should begin.

Start at the first line that belongs in the published video. Exclude anything before it:
- microphone checks and level tests ("Hello?", "can you hear me", "I'm pretty loud")
- apologies for technical difficulties or delays
- pre-session logistics addressed to staff rather than the audience

Keep the session's real opening. That is usually a moderator welcoming the audience or
introducing the session, but if the moderator's remarks are entirely mic checks, apologies,
or housekeeping, start at the first presenter instead.

Respond with JSON only: {"start_line": <integer index>, "reason": "<why that line>",
"skipped": "<short description of what you excluded, or empty string>"}
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "start_line": {"type": "integer"},
        "reason": {"type": "string"},
        "skipped": {"type": "string"},
    },
    "required": ["start_line", "reason", "skipped"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Refinement:
    start_s: float
    reason: str
    skipped: str
    model_used: str
    available: bool


class Refiner(Protocol):
    def refine(self, lines: list[Line], candidate_s: float) -> Refinement:
        ...


def format_lines(lines: list[Line]) -> str:
    return "\n".join(f"{i}\t{line.start_s:.2f}\t{line.text}" for i, line in enumerate(lines))


def apply_choice(
    lines: list[Line], index: int, reason: str, skipped: str, model: str
) -> Refinement:
    if not 0 <= index < len(lines):
        raise ValueError(f"start_line {index} out of range (0..{len(lines) - 1})")
    return Refinement(
        start_s=lines[index].start_s,
        reason=reason,
        skipped=skipped,
        model_used=model,
        available=True,
    )


def _degraded(candidate_s: float, why: str) -> Refinement:
    return Refinement(
        start_s=candidate_s, reason=why, skipped="", model_used="", available=False
    )


class NullRefiner:
    """Mode 'none'. Proposes the energy candidate; head is always reviewed."""

    def refine(self, lines: list[Line], candidate_s: float) -> Refinement:
        return _degraded(candidate_s, "no refinement model configured")


class OpenAICompatibleRefiner:
    """Ollama (local) and OpenRouter (remote) share this wire format."""

    def __init__(self, cfg: RefinerConfig, client: httpx.Client | None = None) -> None:
        self._cfg = cfg
        self._client = client
        self._url = f"{cfg.resolved_base_url()}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self._cfg.api_key_env, "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def refine(self, lines: list[Line], candidate_s: float) -> Refinement:
        if not lines:
            return _degraded(candidate_s, "no transcript available to refine against")

        body = {
            "model": self._cfg.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_lines(lines)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "session_start",
                    "strict": True,
                    "schema": RESPONSE_SCHEMA,
                },
            },
        }

        client = self._client or httpx.Client(timeout=self._cfg.timeout_s)
        try:
            response = client.post(self._url, json=body, headers=self._headers())
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return apply_choice(
                lines,
                int(parsed["start_line"]),
                str(parsed.get("reason", "")),
                str(parsed.get("skipped", "")),
                self._cfg.model,
            )
        except httpx.HTTPError as exc:
            return _degraded(candidate_s, f"refiner unreachable: {type(exc).__name__}")
        except (LookupError, ValueError, TypeError) as exc:
            # LookupError, not KeyError: a 200 response carrying {"choices": []}
            # raises IndexError on choices[0], and IndexError is a LookupError.
            # Catching KeyError alone lets it escape and breaks the guarantee
            # that refinement never fails a session.
            return _degraded(candidate_s, f"refiner returned unusable output: {exc}")
        finally:
            if self._client is None:
                client.close()


def build_refiner(cfg: RefinerConfig) -> Refiner:
    if cfg.mode == "none":
        return NullRefiner()
    return OpenAICompatibleRefiner(cfg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_refine.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add src/vidproc/refine.py tests/test_refine.py
git commit -m "feat: pluggable head refinement (none/local/openrouter) with mandatory degradation"
```

---

### Task 7: Confidence scoring

**Files:**
- Create: `src/vidproc/confidence.py`
- Test: `tests/test_confidence.py`

**Interfaces:**
- Consumes: `vidproc.asr.Line`, `vidproc.detect.Boundaries`, `vidproc.refine.Refinement`
- Produces: constants `MAX_HEAD_GAP_S = 10.0`, `MIN_PRE_HEAD_SILENCE_S = 5.0`, `MIN_TAIL_SILENCE_FOR_CONFIDENCE_S = 20.0`, `CLOSING_PHRASES`; `Confidence` frozen dataclass with `head_confident: bool`, `end_confident: bool`, `head_reasons: tuple[str, ...]`, `end_reasons: tuple[str, ...]`; functions `has_closing_language(lines) -> bool`, `score(boundaries, refinement, tail_lines) -> Confidence`

- [ ] **Step 1: Write the failing test**

Create `tests/test_confidence.py`:

```python
from __future__ import annotations

from vidproc.asr import Line
from vidproc.confidence import has_closing_language, score
from vidproc.detect import Boundaries
from vidproc.refine import Refinement

CLOSING = [Line("Thank you, guys, for attending our session. Thank you.", 4288.0, 4292.0)]
NOT_CLOSING = [Line("and the p-value was under 0.05 for that cohort.", 4288.0, 4292.0)]


def boundaries(head: float = 289.5, pre: float = 300.0, tail: float = 190.0) -> Boundaries:
    return Boundaries(
        noise_floor_db=8.0, gate_db=16.0, head_candidate_s=head,
        end_s=3184.0, pre_head_silence_s=pre, tail_silence_s=tail,
    )


def refinement(start: float, available: bool = True) -> Refinement:
    return Refinement(start_s=start, reason="r", skipped="", model_used="m", available=available)


def test_pt12_shape_is_confident() -> None:
    """gap 0.5s, clean silence before."""
    c = score(boundaries(head=289.5), refinement(290.0), CLOSING)
    assert c.head_confident and c.end_confident


def test_pt06_shape_is_confident() -> None:
    """gap 7.0s is inside the 10s tolerance."""
    c = score(boundaries(head=604.0), refinement(611.0), CLOSING)
    assert c.head_confident


def test_pt01_shape_flags_head_on_gap() -> None:
    """gap 87s — the bumpy session must not auto-approve."""
    c = score(boundaries(head=871.0), refinement(958.0), CLOSING)
    assert not c.head_confident
    assert any("gap" in r for r in c.head_reasons)


def test_unavailable_refiner_flags_head() -> None:
    c = score(boundaries(head=289.5), refinement(289.5, available=False), CLOSING)
    assert not c.head_confident


def test_short_pre_head_silence_flags_head() -> None:
    c = score(boundaries(head=289.5, pre=2.0), refinement(290.0), CLOSING)
    assert not c.head_confident


def test_pt01_short_tail_still_confident_at_27s() -> None:
    c = score(boundaries(tail=27.0), refinement(290.0), CLOSING)
    assert c.end_confident


def test_tail_below_20s_flags_end() -> None:
    c = score(boundaries(tail=12.0), refinement(290.0), CLOSING)
    assert not c.end_confident


def test_missing_closing_language_flags_end() -> None:
    c = score(boundaries(), refinement(290.0), NOT_CLOSING)
    assert not c.end_confident


def test_closing_language_detection_is_case_insensitive() -> None:
    assert has_closing_language([Line("THANKS, EVERYBODY, FOR JOINING US.", 0.0, 1.0)])
    assert has_closing_language([Line("I think that's the end of our session, right?", 0.0, 1.0)])
    assert not has_closing_language([Line("the odds ratio was 1.4", 0.0, 1.0)])


def test_skipped_content_is_recorded_but_does_not_flag() -> None:
    """PT06 legitimately skips a mic check and must still pass."""
    r = Refinement(start_s=611.0, reason="welcome", skipped="mic check", model_used="m", available=True)
    c = score(boundaries(head=604.0), r, CLOSING)
    assert c.head_confident
    assert any("skipped" in reason for reason in c.head_reasons)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_confidence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vidproc.confidence'`

- [ ] **Step 3: Write the implementation**

Create `src/vidproc/confidence.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from vidproc.asr import Line
from vidproc.detect import Boundaries
from vidproc.refine import Refinement

# Thresholds fitted to the three known sessions (spec §6.6). Provisional —
# re-fit once more sessions exist.
MAX_HEAD_GAP_S = 10.0
MIN_PRE_HEAD_SILENCE_S = 5.0
MIN_TAIL_SILENCE_FOR_CONFIDENCE_S = 20.0

CLOSING_PHRASES = (
    "thank you for attending",
    "thanks for attending",
    "for attending our session",
    "thank you all",
    "thanks everybody",
    "thanks, everybody",
    "thank you everybody",
    "for joining us",
    "thank you for joining",
    "thanks for joining",
    "end of our session",
    "end of the session",
    "look forward to",
    "great discussion",
)


@dataclass(frozen=True)
class Confidence:
    head_confident: bool
    end_confident: bool
    head_reasons: tuple[str, ...]
    end_reasons: tuple[str, ...]


def has_closing_language(lines: list[Line]) -> bool:
    haystack = " ".join(line.text for line in lines).lower()
    return any(phrase in haystack for phrase in CLOSING_PHRASES)


def score(
    boundaries: Boundaries, refinement: Refinement, tail_lines: list[Line]
) -> Confidence:
    head_reasons: list[str] = []
    end_reasons: list[str] = []

    gap = abs(refinement.start_s - boundaries.head_candidate_s)
    head_ok = True

    if not refinement.available:
        head_ok = False
        head_reasons.append("no refinement available; energy candidate used unverified")
    elif gap > MAX_HEAD_GAP_S:
        head_ok = False
        head_reasons.append(
            f"gap of {gap:.1f}s between energy candidate and model choice "
            f"exceeds {MAX_HEAD_GAP_S:.0f}s"
        )
    else:
        head_reasons.append(f"energy and model agree within {gap:.1f}s")

    if boundaries.pre_head_silence_s < MIN_PRE_HEAD_SILENCE_S:
        head_ok = False
        head_reasons.append(
            f"only {boundaries.pre_head_silence_s:.1f}s of silence before the start "
            f"(want {MIN_PRE_HEAD_SILENCE_S:.0f}s)"
        )

    # Recorded for review context only. PT06 legitimately skips a mic check and
    # must still pass, so this never flags on its own.
    if refinement.available and refinement.skipped:
        head_reasons.append(f"skipped before start: {refinement.skipped}")

    end_ok = True
    if boundaries.tail_silence_s < MIN_TAIL_SILENCE_FOR_CONFIDENCE_S:
        end_ok = False
        end_reasons.append(
            f"final silence only {boundaries.tail_silence_s:.1f}s "
            f"(want {MIN_TAIL_SILENCE_FOR_CONFIDENCE_S:.0f}s)"
        )
    else:
        end_reasons.append(f"final silence {boundaries.tail_silence_s:.1f}s")

    if has_closing_language(tail_lines):
        end_reasons.append("closing language found near the end")
    else:
        end_ok = False
        end_reasons.append("no closing language found near the end")

    return Confidence(
        head_confident=head_ok,
        end_confident=end_ok,
        head_reasons=tuple(head_reasons),
        end_reasons=tuple(end_reasons),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_confidence.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/vidproc/confidence.py tests/test_confidence.py
git commit -m "feat: confidence scoring calibrated to the three known sessions"
```

---

### Task 8: Proposal assembly and CLI

**Files:**
- Create: `src/vidproc/proposal.py`
- Create: `src/vidproc/analyze.py`
- Create: `src/vidproc/cli.py`
- Test: `tests/test_analyze.py`

**Interfaces:**
- Consumes: everything above
- Produces: `Proposal` frozen dataclass with `session: str`, `source: str`, `duration_s: float`, `fps: str`, `start_s: float`, `end_s: float`, `noise_floor_db: float`, `gate_db: float`, `head_candidate_s: float`, `head_reason: str`, `head_skipped: str`, `model_used: str`, `head_confident: bool`, `end_confident: bool`, `head_reasons: tuple[str, ...]`, `end_reasons: tuple[str, ...]`, `head_lines: tuple[dict, ...]`, `tail_lines: tuple[dict, ...]`; `Proposal.to_json() -> str`; `analyze(media, cfg, asr=None, refiner=None) -> Proposal`; `main(argv=None) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyze.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vidproc.analyze import analyze
from vidproc.asr import Line, Word
from vidproc.audio import Envelope
from vidproc.config import Config, DetectionConfig
from vidproc.refine import Refinement


class FakeASR:
    def __init__(self, words: list[Word]) -> None:
        self._words = words
        self.calls: list[tuple[float, float]] = []

    def transcribe(self, media: Path, start_s: float, end_s: float) -> list[Word]:
        self.calls.append((start_s, end_s))
        return [w for w in self._words if start_s <= w.start_s < end_s]


class FakeRefiner:
    def __init__(self, start_s: float, available: bool = True) -> None:
        self._start_s = start_s
        self._available = available

    def refine(self, lines: list[Line], candidate_s: float) -> Refinement:
        return Refinement(
            start_s=self._start_s, reason="chosen", skipped="mic check",
            model_used="fake", available=self._available,
        )


@pytest.fixture
def fake_envelope(monkeypatch) -> None:
    """900s silence, 20s speech, 25s plateau, 600s speech, 190s silence."""
    hop = 0.25
    segs = [(8.0, 900.0), (35.0, 20.0), (12.0, 25.0), (35.0, 600.0), (8.0, 190.0)]
    db = np.concatenate([np.full(int(round(d / hop)), lvl) for lvl, d in segs])
    monkeypatch.setattr(
        "vidproc.analyze.envelope_for", lambda *a, **k: Envelope(db=db, hop_s=hop)
    )


@pytest.fixture
def fake_probe(monkeypatch) -> None:
    from fractions import Fraction

    from vidproc.probe import MediaInfo

    monkeypatch.setattr(
        "vidproc.analyze.probe",
        lambda p, **k: MediaInfo(
            path=p, duration_s=1735.0, width=1920, height=1080,
            fps=Fraction(16, 1), video_codec="h264",
            audio_sample_rate=44100, audio_channels=2,
        ),
    )


def make_config(tmp_path: Path) -> Config:
    return Config(working_dir=tmp_path, detection=DetectionConfig())


def test_proposal_uses_refined_head(tmp_path, fake_envelope, fake_probe) -> None:
    words = [
        Word("Hello?", 900.5, 901.0),
        Word("Welcome", 945.2, 946.0),
        Word("Thank", 1530.0, 1530.5),
        Word("you", 1530.5, 1531.0),
        Word("for", 1531.0, 1531.3),
        Word("attending", 1531.3, 1532.0),
    ]
    p = analyze(
        tmp_path / "AAO2025PT01.mp4", make_config(tmp_path),
        asr=FakeASR(words), refiner=FakeRefiner(945.2),
    )
    assert p.session == "AAO2025PT01"
    assert p.head_candidate_s == pytest.approx(900.0, abs=1.0)
    assert p.start_s == pytest.approx(945.2 - 1.0, abs=0.01)  # preroll applied
    assert p.model_used == "fake"


def test_large_gap_flags_head(tmp_path, fake_envelope, fake_probe) -> None:
    p = analyze(
        tmp_path / "s.mp4", make_config(tmp_path),
        asr=FakeASR([]), refiner=FakeRefiner(1000.0),
    )
    assert p.head_confident is False


def test_asr_is_only_given_bounded_windows(tmp_path, fake_envelope, fake_probe) -> None:
    """Never hand the transcriber dead air — that is the hallucination trigger."""
    asr = FakeASR([])
    cfg = make_config(tmp_path)
    analyze(tmp_path / "s.mp4", cfg, asr=asr, refiner=FakeRefiner(905.0))
    head_call, tail_call = asr.calls
    assert head_call[0] >= 900.0 - 1.0
    assert head_call[1] - head_call[0] == pytest.approx(cfg.detection.head_window_s)
    assert tail_call[1] - tail_call[0] == pytest.approx(cfg.detection.tail_window_s)


def test_end_gets_tail_pad(tmp_path, fake_envelope, fake_probe) -> None:
    cfg = make_config(tmp_path)
    p = analyze(tmp_path / "s.mp4", cfg, asr=FakeASR([]), refiner=FakeRefiner(905.0))
    assert p.end_s == pytest.approx(1545.0 + cfg.detection.tail_pad_s, abs=1.0)


def test_start_never_goes_negative(tmp_path, fake_probe, monkeypatch) -> None:
    hop = 0.25
    db = np.concatenate([np.full(int(40 / hop), 35.0), np.full(int(60 / hop), 8.0)])
    monkeypatch.setattr(
        "vidproc.analyze.envelope_for", lambda *a, **k: Envelope(db=db, hop_s=hop)
    )
    p = analyze(
        tmp_path / "s.mp4", make_config(tmp_path),
        asr=FakeASR([]), refiner=FakeRefiner(0.0),
    )
    assert p.start_s >= 0.0


def test_tail_window_stays_bounded_on_a_short_file(tmp_path, monkeypatch) -> None:
    """A file shorter than tail_window_s still yields a bounded tail window.

    final_silence_start returns the full duration when there is no adequate
    trailing silence, so end_s can be small. The window must stay non-negative
    and never exceed tail_window_s of audio.
    """
    from fractions import Fraction

    from vidproc.probe import MediaInfo

    hop = 0.25
    db = np.full(int(30 / hop), 35.0)  # 30s, all speech, no trailing silence
    monkeypatch.setattr(
        "vidproc.analyze.envelope_for", lambda *a, **k: Envelope(db=db, hop_s=hop)
    )
    monkeypatch.setattr(
        "vidproc.analyze.probe",
        lambda p, **k: MediaInfo(
            path=p, duration_s=30.0, width=1920, height=1080, fps=Fraction(16, 1),
            video_codec="h264", audio_sample_rate=44100, audio_channels=2,
        ),
    )
    asr = FakeASR([])
    cfg = make_config(tmp_path)
    analyze(tmp_path / "short.mp4", cfg, asr=asr, refiner=FakeRefiner(0.0))
    _, tail_call = asr.calls
    assert tail_call[0] >= 0.0
    assert tail_call[1] - tail_call[0] <= cfg.detection.tail_window_s


def test_proposal_json_round_trips(tmp_path, fake_envelope, fake_probe) -> None:
    p = analyze(
        tmp_path / "s.mp4", make_config(tmp_path),
        asr=FakeASR([]), refiner=FakeRefiner(905.0),
    )
    data = json.loads(p.to_json())
    assert data["start_s"] == pytest.approx(p.start_s)
    assert "head_reasons" in data and isinstance(data["head_reasons"], list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_analyze.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vidproc.analyze'`

- [ ] **Step 3: Write the implementation**

Create `src/vidproc/proposal.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Proposal:
    session: str
    source: str
    duration_s: float
    fps: str
    start_s: float
    end_s: float
    noise_floor_db: float
    gate_db: float
    head_candidate_s: float
    head_reason: str
    head_skipped: str
    model_used: str
    head_confident: bool
    end_confident: bool
    head_reasons: tuple[str, ...]
    end_reasons: tuple[str, ...]
    head_lines: tuple[dict[str, Any], ...]
    tail_lines: tuple[dict[str, Any], ...]

    @property
    def output_duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def needs_review(self) -> bool:
        return not (self.head_confident and self.end_confident)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=list)
```

Create `src/vidproc/analyze.py`:

```python
from __future__ import annotations

from pathlib import Path

from vidproc.asr import ASR, Line, WhisperCppASR, words_to_lines
from vidproc.audio import envelope_for
from vidproc.config import Config
from vidproc.confidence import score
from vidproc.detect import detect_boundaries
from vidproc.probe import probe
from vidproc.proposal import Proposal
from vidproc.refine import Refiner, build_refiner


def _as_dicts(lines: list[Line]) -> tuple[dict[str, object], ...]:
    return tuple(
        {"start_s": round(line.start_s, 2), "end_s": round(line.end_s, 2), "text": line.text}
        for line in lines
    )


def analyze(
    media: Path,
    cfg: Config,
    asr: ASR | None = None,
    refiner: Refiner | None = None,
) -> Proposal:
    """Produce a reviewable proposal for one raw session file."""
    media = Path(media)
    asr = asr if asr is not None else WhisperCppASR(cfg.asr)
    refiner = refiner if refiner is not None else build_refiner(cfg.refiner)

    info = probe(media)
    env = envelope_for(media, cfg.working_dir / "cache")
    bounds = detect_boundaries(env, cfg.detection)

    d = cfg.detection

    # Only ever transcribe bounded windows that energy says contain speech.
    head_words = asr.transcribe(
        media, bounds.head_candidate_s, bounds.head_candidate_s + d.head_window_s
    )
    head_lines = words_to_lines(head_words)

    # Width is min(end_s, tail_window_s) — never more than one window. On a file
    # shorter than a window the tail window is the whole file, which is fine:
    # the invariant exists to stop the transcriber being handed long stretches
    # of dead air, and a file shorter than one window cannot contain any.
    tail_start = max(0.0, bounds.end_s - d.tail_window_s)
    tail_words = asr.transcribe(media, tail_start, bounds.end_s)
    tail_lines = words_to_lines(tail_words)

    refinement = refiner.refine(head_lines, bounds.head_candidate_s)
    confidence = score(bounds, refinement, tail_lines)

    start_s = max(0.0, refinement.start_s - d.head_preroll_s)
    end_s = min(info.duration_s, bounds.end_s + d.tail_pad_s)

    return Proposal(
        session=media.stem,
        source=str(media),
        duration_s=info.duration_s,
        fps=str(info.fps),
        start_s=start_s,
        end_s=end_s,
        noise_floor_db=round(bounds.noise_floor_db, 2),
        gate_db=round(bounds.gate_db, 2),
        head_candidate_s=bounds.head_candidate_s,
        head_reason=refinement.reason,
        head_skipped=refinement.skipped,
        model_used=refinement.model_used,
        head_confident=confidence.head_confident,
        end_confident=confidence.end_confident,
        head_reasons=confidence.head_reasons,
        end_reasons=confidence.end_reasons,
        head_lines=_as_dicts(head_lines),
        tail_lines=_as_dicts(tail_lines),
    )
```

Create `src/vidproc/cli.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vidproc.analyze import analyze
from vidproc.config import load_config
from vidproc.errors import VidprocError


def _format_clock(seconds: float) -> str:
    return f"{int(seconds // 60)}m{seconds % 60:04.1f}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vidproc", description="Propose session cut points.")
    parser.add_argument("media", type=Path, help="raw session file")
    parser.add_argument("-c", "--config", type=Path, default=None)
    parser.add_argument("-w", "--working-dir", type=Path, default=Path("working"))
    parser.add_argument("--json", action="store_true", help="print the proposal as JSON")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, working_dir=args.working_dir)
        proposal = analyze(args.media, cfg)
    except VidprocError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_dir = cfg.working_dir / proposal.session
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "proposal.json").write_text(proposal.to_json())

    if args.json:
        print(proposal.to_json())
        return 0

    print(f"{proposal.session}  ({_format_clock(proposal.duration_s)} raw)")
    print(f"  start  {proposal.start_s:9.2f}s  {_format_clock(proposal.start_s)}"
          f"   {'ok' if proposal.head_confident else 'REVIEW'}")
    for reason in proposal.head_reasons:
        print(f"           - {reason}")
    if proposal.head_lines:
        print(f'           opens on: "{proposal.head_lines[0]["text"]}"')
    print(f"  end    {proposal.end_s:9.2f}s  {_format_clock(proposal.end_s)}"
          f"   {'ok' if proposal.end_confident else 'REVIEW'}")
    for reason in proposal.end_reasons:
        print(f"           - {reason}")
    print(f"  output duration {_format_clock(proposal.output_duration_s)}")
    print(f"  ffmpeg: -ss {proposal.start_s:.2f} -to {proposal.end_s:.2f}")
    print(f"  written to {out_dir / 'proposal.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_analyze.py -v && uv run pytest -v`
Expected: 7 passed in test_analyze; whole suite green

- [ ] **Step 5: Commit**

```bash
git add src/vidproc/proposal.py src/vidproc/analyze.py src/vidproc/cli.py tests/test_analyze.py
git commit -m "feat: proposal assembly and analyze CLI"
```

---

### Task 9: Regression against real footage

Scores the detector against the three hand-made cut points. This is the test that says whether any of it actually works. It skips cleanly when `sample/` is absent, so CI and other machines stay green.

**Files:**
- Create: `tests/test_regression_samples.py`
- Modify: `README.md` (replace wholesale — the existing content documents only the file-copy monitor)

**Interfaces:**
- Consumes: `vidproc.audio.envelope_for`, `vidproc.detect.detect_boundaries`, `vidproc.config.DetectionConfig`
- Produces: nothing consumed by later tasks

- [ ] **Step 1: Write the failing test**

Create `tests/test_regression_samples.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from vidproc.audio import envelope_for
from vidproc.config import DetectionConfig
from vidproc.detect import detect_boundaries

SAMPLES = Path(__file__).resolve().parents[1] / "sample" / "raw"

# Hand-made cut points recovered by envelope cross-correlation (spec §3).
GROUND_TRUTH = {
    "AAO2025PT01": (958.0, 4292.0),
    "AAO2025PT06": (611.0, 4323.0),
    "AAO2025PT12": (290.0, 3184.0),
}

# End detection is energy-only and must be tight (spec §6.2 measured -0.75/-1.25/+0.75).
END_TOLERANCE_S = 2.0

# The head candidate is a lower bound: never later than the human cut, and no
# more than 120s early (PT01's technical-difficulty apology is 87s early).
HEAD_MAX_EARLY_S = 120.0
HEAD_MAX_LATE_S = 0.0

pytestmark = pytest.mark.samples


def _sample(session: str) -> Path:
    path = SAMPLES / f"{session}.mp4"
    if not path.exists():
        pytest.skip(f"{path} not present; sample footage is local-only")
    return path


@pytest.mark.parametrize("session", sorted(GROUND_TRUTH))
def test_end_detection_within_tolerance(session: str, tmp_path: Path) -> None:
    media = _sample(session)
    _, truth_end = GROUND_TRUTH[session]
    env = envelope_for(media, tmp_path)
    bounds = detect_boundaries(env, DetectionConfig())
    error = bounds.end_s - truth_end
    assert abs(error) <= END_TOLERANCE_S, f"{session} end off by {error:+.2f}s"


@pytest.mark.parametrize("session", sorted(GROUND_TRUTH))
def test_head_candidate_is_a_safe_lower_bound(session: str, tmp_path: Path) -> None:
    media = _sample(session)
    truth_head, _ = GROUND_TRUTH[session]
    env = envelope_for(media, tmp_path)
    bounds = detect_boundaries(env, DetectionConfig())
    error = bounds.head_candidate_s - truth_head
    assert error <= HEAD_MAX_LATE_S, f"{session} head candidate is {error:+.2f}s LATE"
    assert error >= -HEAD_MAX_EARLY_S, f"{session} head candidate is {error:+.2f}s early"


@pytest.mark.parametrize("session", sorted(GROUND_TRUTH))
def test_noise_floor_is_in_the_observed_band(session: str, tmp_path: Path) -> None:
    """Measured 7.4-8.1 dB across all three. A wide drift means the capture
    chain changed and the thresholds need re-fitting."""
    env = envelope_for(_sample(session), tmp_path)
    bounds = detect_boundaries(env, DetectionConfig())
    assert 5.0 <= bounds.noise_floor_db <= 12.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_regression_samples.py -v`
Expected: FAIL — assertions run against real footage before the detector is tuned, or SKIP if `sample/` is absent. If every case skips, the sample footage is missing and this task cannot be verified; stop and say so rather than marking it done.

- [ ] **Step 3: Confirm the detector meets the bounds**

No new implementation should be required — Tasks 3 and 4 already implement this. If a case fails, the defect is in `detect.py`, not in the test. Do not loosen the tolerances to make it pass; report the failure with the measured error.

Print the actual numbers while iterating:

```bash
uv run python - <<'PY'
from pathlib import Path
from vidproc.audio import envelope_for
from vidproc.config import DetectionConfig
from vidproc.detect import detect_boundaries

TRUTH = {"AAO2025PT01": (958.0, 4292.0), "AAO2025PT06": (611.0, 4323.0), "AAO2025PT12": (290.0, 3184.0)}
cache = Path("working/cache")
for session, (th, te) in TRUTH.items():
    media = Path("sample/raw") / f"{session}.mp4"
    if not media.exists():
        print(f"{session}: missing"); continue
    b = detect_boundaries(envelope_for(media, cache), DetectionConfig())
    print(f"{session}  floor={b.noise_floor_db:5.1f}  gate={b.gate_db:5.1f}  "
          f"head={b.head_candidate_s:8.2f} ({b.head_candidate_s - th:+7.2f})  "
          f"end={b.end_s:8.2f} ({b.end_s - te:+6.2f})")
PY
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -v`
Expected: all green. Then confirm the suite is clean without footage: `uv run pytest -m "not samples" -v`

- [ ] **Step 5: Replace the README**

The existing `README.md` documents only the file-copy monitor and is now misleading. Replace it entirely with:

````markdown
# vidproc

Finds where a recorded conference session actually starts and ends, so the raw capture can be
trimmed without scrubbing through it by hand.

Given a raw Poster Theater capture, `vidproc` proposes a start and end timecode, says how
confident it is in each, and shows the transcript around both cuts as evidence.

## How it works

The capture is silent until the sound board goes live, so the end of a session is found by
energy alone — the start of the final silence — and lands within about a second of a
hand-made cut. The start is harder: the first speech is often a mic check or an apology for
technical difficulties, so energy gives a conservative lower bound and an optional language
model picks the first line that belongs in the published video.

## Install

```bash
uv sync
```

External binaries, not installed by `uv`:

| Tool | Needed for |
| --- | --- |
| `ffprobe` | reading source properties |
| `ffmpeg` | audio extraction |
| `whisper-cli` (whisper.cpp) | transcribing the windows around each cut |

On macOS: `brew install ffmpeg whisper-cpp`. A whisper model file is also required —
set `asr.model_path` to something like `~/models/ggml-large-v3.bin`.

## Usage

```bash
uv run vidproc sample/raw/AAO2025PT12.mp4 --working-dir working
uv run vidproc sample/raw/AAO2025PT12.mp4 --json      # machine-readable
```

The proposal is written to `<working-dir>/<session>/proposal.json`. The terminal output ends
with a ready-to-paste `-ss` / `-to` pair, and marks either boundary `REVIEW` when confidence
is low.

## Configuration

Pass a JSON file with `-c`. Any key may be omitted; defaults are shown.

```json
{
  "detection": {
    "floor_percentile": 5.0,
    "gate_offset_db": 8.0,
    "min_speech_s": 3.0,
    "min_tail_silence_s": 10.0,
    "head_window_s": 180.0,
    "tail_window_s": 60.0,
    "head_preroll_s": 1.0,
    "tail_pad_s": 2.0
  },
  "refiner": { "mode": "none", "model": "", "base_url": "", "api_key_env": "OPENROUTER_API_KEY" },
  "asr": { "binary": "whisper-cli", "model_path": "", "language": "en" }
}
```

`min_tail_silence_s` is capped at 20.0. One observed session was stopped 27 seconds after it
ended; a larger requirement finds no qualifying silence and the detector returns garbage.

## Refiner modes

| Mode | Behaviour |
| --- | --- |
| `none` | No model. The energy candidate is proposed and the start is always flagged for review. |
| `local` | Ollama at `http://localhost:11434/v1`. Nothing leaves the machine. |
| `openrouter` | OpenRouter at `https://openrouter.ai/api/v1`, key from `$OPENROUTER_API_KEY`. |

Remote access goes through OpenRouter only, never a provider API directly.

**Models listed by `ollama list` with a `:cloud` suffix run on Ollama's servers, not yours.**
Do not configure one as `local` if the point is that no content leaves the machine.

If the configured backend is unreachable — no network, Ollama not running, missing key —
the run degrades to the energy candidate and flags the start for review. A session is never
failed because a model was unavailable.

## Tests

```bash
uv run pytest                      # everything
uv run pytest -m "not samples"     # skip tests needing real footage
```

## sample/

`sample/` holds real conference recordings of identifiable presenters, kept for regression
testing. It is gitignored and **must never be committed, pushed, or uploaded anywhere.**
Fixtures that do get committed are derived artifacts only — timecodes and transcripts, never
media.
````

- [ ] **Step 6: Commit**

```bash
git add tests/test_regression_samples.py README.md
git commit -m "test: regression against three hand-cut sessions; rewrite README"
```

---

## Verification

Before considering this plan complete:

```bash
uv run pytest -v                        # everything, including real footage
uv run pytest -m "not samples" -v       # must also pass with no footage present
git status --short                      # sample/ must NOT appear
git check-ignore -v sample/             # must report the .gitignore rule
uv run vidproc sample/raw/AAO2025PT12.mp4 --working-dir working
```

The last command should report a start near 289s, an end near 3186s, and — with a refiner configured — quote the opening line. PT01 should print `REVIEW` on its head.
