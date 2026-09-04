# AGENTS.md

Instructions for coding agents working in this repository.

**`README.md` is the operator guide. `CLAUDE.md` is the full agent guidance.** Read those first —
this file is a summary so that agents which only load `AGENTS.md` are not misled.

## What this repo contains

- **`vidproc`** (`src/vidproc/`) — the active project. Finds the start and end cut points of a
  recorded conference session and proposes them for human confirmation, with transcript evidence.
- **`video_monitor.py`** — a legacy standalone file-copy monitor, not yet migrated. Its
  `config.json` schema is unrelated to vidproc's.

## Hard constraints

- **Never commit, push, or upload anything under `sample/` or `working/`.** `sample/` holds real
  conference recordings of identifiable presenters; `working/` fills with decoded audio and
  transcript excerpts derived from them. Both are gitignored — keep it that way.
- **Remote LLM access goes through OpenRouter only**, never a provider API directly.
- **Never assume a source property — probe it.** These captures are 16 fps, not 29.97.
- **Do not change measured constants.** Detection thresholds, envelope resolution (8000 Hz /
  0.25 s), and the ground truth in `tests/test_regression_samples.py` were measured against real
  footage. If a change makes that file fail, the change is wrong — do not loosen the bounds.

## Commands

```bash
uv sync                            # set up
uv run pytest                      # full suite
uv run pytest -m "not samples"     # without real footage — must also pass
uv run ruff check                  # lint
```

## Code style

- Python 3.11+, `from __future__ import annotations` at the top of every module.
- Type hints on public functions. Prefer `Path` over `str` for filesystem paths.
- `snake_case` functions and variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
- Google-style docstrings. Explain *why*, not *what* — especially for any measured constant.
- Errors from external tools surface as `vidproc.errors.ExternalToolError`, never a raw `KeyError`,
  `JSONDecodeError`, `ZeroDivisionError`, or `OSError`.
- `cli.py` prints to stdout by design; nothing else should.
