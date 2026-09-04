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

The proposal is written to `<working-dir>/<session>/proposal.json`. The terminal output includes a
ready-to-paste `-ss` / `-to` pair, and marks either boundary `REVIEW` when confidence is low.

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
  "refiner": { "mode": "none", "model": "", "base_url": "", "api_key_env": "OPENROUTER_API_KEY", "timeout_s": 60.0 },
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
