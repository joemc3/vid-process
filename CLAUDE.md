# CLAUDE.md

Guidance for Claude Code working in this repository.

**Read `README.md` first** — it is the operator guide and the source of truth for prerequisites,
setup, usage, and known limitations. This file covers only what an agent needs beyond that.

## What this repo contains

Two separate tools at different stages of maturity.

**`vidproc`** (`src/vidproc/`) — the active project. Given a raw conference recording it proposes
a start and end timecode, a confidence classification for each, and the transcript around both
cuts as evidence. This replaces the manual VLC-scrubbing step of a larger post-production
workflow.

**`video_monitor.py`** — a legacy standalone script that watches a capture share and copies
finished recordings off it. Still in use, not yet migrated, and it has known defects (an unbounded
hang when a file stalls below the minimum size, and single-file blocking that stops the whole
queue). Its `config.json` schema is **unrelated** to vidproc's. Don't conflate the two.

## Hard constraints

These are not style preferences. Violating them causes real harm.

- **`sample/` and `working/` must never be committed, pushed, or uploaded anywhere.** `sample/`
  holds real conference recordings of identifiable presenters discussing clinical work, kept for
  regression testing. `working/` fills with decoded audio and transcript excerpts derived from it.
  Both are gitignored — keep it that way. Committed fixtures are derived values only: timecodes,
  transcripts, envelopes. Never media.
- **Remote LLM access goes through OpenRouter only**, never a provider API directly.
- **A model listed by `ollama list` with a `:cloud` suffix is not local** — it runs on Ollama's
  servers. Never treat one as satisfying "nothing leaves the machine".
- **Never assume a source property — probe it.** These captures are 16 fps, which is unusual. A
  hand-written pipeline previously assumed 29.97 and silently inflated every output by 87% more
  frames.

## Architecture

A pure-function core wrapped by thin adapters over external tools.

```
probe.py    ffprobe wrapper — real source properties, never assumed
audio.py    RMS envelope, fixed 8 kHz / 0.25 s, content-addressed PCM cache
detect.py   PURE. Boundary detection over an envelope. No I/O at all.
asr.py      ASR protocol + whisper.cpp adapter. Bounded windows only.
refine.py   Refiner protocol + none/local/openrouter backends
confidence.py  PURE. Scores each boundary against measured thresholds.
analyze.py  Wires the above into one proposal
cli.py      Entry point, writes proposal.json
```

`detect.py` and `confidence.py` do no I/O by design — that is what lets them be tested against
synthesised envelopes with no media files. Keep it that way; it is not incidental.

`ASR` and `Refiner` are real seams, injected into `analyze()` and faked in tests, so a different
engine can be swapped in without touching detection logic.

## Numbers you must not casually change

Every threshold in this project was measured against three real sessions whose hand-made cut
points are known. They are not guesses, and "tidying" them silently breaks the detector.

- Envelope resolution 8000 Hz / 0.25 s — every threshold is calibrated to these.
- `min_tail_silence_s` capped at 20.0 — one session's recording stopped 27 s after it ended.
- Confidence thresholds: gap ≤ 10.0 s, pre-head silence ≥ 5.0 s, tail silence ≥ 20.0 s.
- The ground truth and tolerances in `tests/test_regression_samples.py`. If a change makes that
  file fail, the change is wrong — do not loosen the bounds to make it pass.

The head candidate must always be a **conservative lower bound**: earlier than the true cut is
correct and expected, later is a defect.

## Commands

```bash
uv sync                            # set up
uv run pytest                      # full suite (needs sample/ for 9 of them)
uv run pytest -m "not samples"     # suite without real footage — must also pass
uv run ruff check                  # lint
uv run vidproc <file> -c cfg.json -w working
```

## Conventions

- Python 3.11+, `from __future__ import annotations` at the top of every module.
- Errors from external tools surface as `vidproc.errors.ExternalToolError`, never a raw
  `KeyError`, `JSONDecodeError`, `ZeroDivisionError`, or `OSError`.
- Refinement must never raise. If a model is unreachable it returns the energy candidate with
  `available=False` and the session is flagged for review. A session is never failed because a
  model was unavailable.
- Never hand the transcriber a window the energy pass has not identified as containing speech.
  Whisper's documented failure mode is hallucinating during non-speech, and these recordings open
  with 5–16 minutes of it.

## Design documents

- `docs/superpowers/specs/2026-09-04-session-trim-pipeline-design.md` — the binding design, with
  the measured evidence behind every number and a list of stated assumptions.
- `docs/superpowers/plans/2026-09-04-detection.md` — the implementation plan for what is built.

## Known gaps

Documented in README under *Known limitations*, plus:

- No external call has a timeout — four subprocess invocations. Fix them in one consistent pass.
- No `test_cli.py`. `cli.main()` itself is still untested end to end; the `opens on:` line it
  prints is covered indirectly by `tests/test_proposal.py`.
