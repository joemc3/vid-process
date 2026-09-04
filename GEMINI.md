# GEMINI.md

**See `README.md` for the operator guide and `CLAUDE.md` for full agent guidance.** This file is a
pointer so that agents loading only `GEMINI.md` are not misled by stale content.

## Project overview

`vidproc` finds where a recorded conference session actually starts and ends, so a raw capture can
be trimmed without a human scrubbing through it. It proposes a start timecode, an end timecode, a
confidence classification for each, and the transcript around both cuts as evidence. Rendering,
copying, and uploading remain manual.

The end cut is found by audio energy alone and needs no model. The start cut is harder — the first
speech is often a mic check or an apology for technical difficulties — so an energy pass gives a
conservative lower bound and an optional language model refines it against a transcript.

`video_monitor.py` is a separate legacy file-copy monitor, not yet migrated, with an unrelated
`config.json` schema.

## Technologies

Python 3.11+, numpy, httpx, pytest, managed with `uv`. External binaries: `ffmpeg`, `ffprobe`,
`whisper-cli` (whisper.cpp). Optionally Ollama (local) or OpenRouter (remote) for refinement.

## Building and running

```bash
uv sync
uv run vidproc <file> -c config.json -w working
uv run pytest
```

## Hard constraints

- Never commit, push, or upload anything under `sample/` or `working/` — real recordings of
  identifiable presenters, and audio and transcripts derived from them. Both are gitignored.
- Remote LLM access goes through OpenRouter only.
- Never assume a source property — probe it. These captures are 16 fps.
- Detection thresholds and the ground truth in `tests/test_regression_samples.py` were measured
  against real footage. Do not loosen them to make a test pass.
