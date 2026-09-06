# vidproc

Finds where a recorded conference session actually starts and ends, so a raw capture can be
trimmed without scrubbing through it by hand in VLC.

Point it at a raw Poster Theater capture and it gives you a start timecode, an end timecode, how
confident it is in each, and the transcript around both cuts so you can check its work:

```
AAO2025PT12  (56m14.5s raw)
  start     288.50s  4m48.5s   ok
           - energy and model agree within 0.0s
           opens on: "All right. Good morning, everyone. My name is..."
  end      3186.75s  53m06.8s   ok
           - final silence 189.5s
           - closing language found near the end
  output duration 48m18.2s
  ffmpeg: -ss 288.50 -to 3186.75
```

You take that `-ss` / `-to` pair and render with it. **The render is not automated yet** — see
[Rendering the trimmed file](#rendering-the-trimmed-file).

## What it does and doesn't do

**Does:** finds the two cut points for one session file, and tells you how much to trust each one.

**Doesn't:** render, resize, copy files anywhere, upload to the CDN or the host, or touch the
meeting guide. Those are still manual. It also doesn't split a session into per-presenter clips —
one session in, one pair of timecodes out.

## Prerequisites

Everything below assumes macOS on Apple Silicon.

### 1. Homebrew

If `brew --version` fails, install it from [brew.sh](https://brew.sh).

### 2. uv

Manages Python and the project's dependencies, so you don't need to install Python yourself.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your terminal and check that `uv --version` works.

### 3. ffmpeg and whisper.cpp

```bash
brew install ffmpeg whisper-cpp
```

Verify: `ffprobe -version` and `whisper-cli --help` should both produce output.

### 4. A whisper model

whisper.cpp needs a model file, which is a separate download. Start with the small one:

```bash
mkdir -p ~/models
curl -L -o ~/models/ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

That is **144 MB**, and it is enough. On the three 2025 sessions it produced *identical* cut
points to the 3.1 GB model, in about a third of the time. It reliably transcribes the things that
matter here — "good morning", "welcome to this session", "I'm pretty loud, okay" — even though it
mangles proper nouns — it turned one presenter's name into a different name entirely, and *poster
theater* into "social theater". Wrong names don't change where the cut goes.

If you want transcripts accurate enough to reuse for anything else, get the large model instead —
but it is **3.1 GB**, so don't pull it over conference wifi on the morning of an event:

```bash
curl -L -o ~/models/ggml-large-v3.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
```

### 5. A language model — optional, but it improves the start cut

The end cut needs no model at all and is accurate without one. The start cut is harder, because
the first speech in a recording is often a mic check or an apology for technical difficulties, and
a model reading the transcript can skip past those. You have three choices — see
[Refiner modes](#refiner-modes). If you skip this entirely everything still works; the start is
just always flagged for you to confirm.

**For a local model** (nothing leaves the machine), install [Ollama](https://ollama.com) and pull
one:

```bash
ollama pull qwen3.6:35b-a3b
```

**For a remote model**, get an OpenRouter key and export it:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

## Setup

```bash
cd vid-process
uv sync
```

That creates the environment and installs dependencies. Check it worked:

```bash
uv run vidproc --help
```

## Quick start

Make a config file. This one uses the small whisper model and a local language model:

```bash
cat > myconfig.json <<'EOF'
{
  "asr":     { "model_path": "/Users/YOURNAME/models/ggml-base.en.bin" },
  "refiner": { "mode": "local", "model": "qwen3.6:35b-a3b" }
}
EOF
```

Use the full path to your home directory — `~` is not expanded inside a JSON file.

Then run it:

```bash
uv run vidproc /path/to/AAO2026PT01.mp4 -c myconfig.json -w working
```

A ~60 minute session takes about 40 seconds with the small whisper model, or two to three minutes
with the large one — most of it transcription either way. The proposal is also written to
`working/<session>/proposal.json` if you want it machine-readable, and `--json` prints that
instead of the human-readable summary.

## Reading the output

Each boundary is marked `ok` or `REVIEW`.

**`ok`** means both of that boundary's confidence checks passed. **`REVIEW`** means at least one
did not, and the reason is printed underneath — for example `only 0.5s of silence before the
start` or `no refinement available; energy candidate used unverified`.

**Always read the `opens on:` line.** It shows the first words of what will be published, and it
is the fastest way to catch a wrong start. If it reads like a mic check — *"Hello everybody. Oh
yeah, I'm pretty loud"* — the start is too early and you should move it forward yourself.

`REVIEW` does not mean the number is wrong, and `ok` does not guarantee it is right. Treat both as
a proposal to check, not an answer, until you have watched the tool get it right enough times to
trust it.

## Rendering the trimmed file

Take the `-ss` / `-to` pair from the output:

```bash
ffmpeg -y -ss 288.50 -to 3186.75 -i raw/AAO2026PT01.mp4 \
  -vf "scale=-2:'min(720,ih)',format=yuv420p" \
  -c:v libx264 -preset slower -crf 22 -level 3.1 -movflags +faststart \
  -c:a aac -b:a 80k processed/AAO2026PT01.mp4
```

Then check the result in VLC — the first few seconds, the last few seconds, and a spot in the
middle — before it goes anywhere.

This differs from the command used in 2025, in four ways that all matter:

- **No `-r`.** These captures are 16 fps. The old command forced 29.850746 fps, duplicating frames
  to inflate a 16 fps source: 87% more frames encoded for a 19% larger file, and no visual
  difference. Let the source rate pass through.
- **No `-pass 1`.** It does nothing alongside `-crf`, and it writes a ~34 MB
  `ffmpeg2pass-0.log.mbtree` into whatever directory you ran from.
- **No `-profile:v high422`.** It was silently overridden anyway — the output was already
  High/4:2:0.
- **`-ss` and `-to` go before `-i`.** Still frame-accurate, and it skips decoding the part you are
  cutting away.

## Configuration

Pass a JSON file with `-c`. Every key is optional; defaults are shown.

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
  "refiner": {
    "mode": "none",
    "model": "",
    "base_url": "",
    "api_key_env": "OPENROUTER_API_KEY",
    "timeout_s": 60.0
  },
  "asr": { "binary": "whisper-cli", "model_path": "", "language": "en" }
}
```

The two you are most likely to change:

- **`head_preroll_s`** — how long before the first word to start. At 1.0s the 2025 sessions keep
  their opening word; the hand-made cuts clipped it.
- **`tail_pad_s`** — how long after the last audio to end.

`min_tail_silence_s` is capped at 20.0 and should not be raised. One 2025 session had its
recording stopped 27 seconds after the session ended; require more silence than exists and the
detector finds none and returns nonsense.

## Refiner modes

| Mode | Behaviour |
| --- | --- |
| `none` | No model. The energy candidate is proposed and the start is **always** flagged for review. |
| `local` | Ollama at `http://localhost:11434/v1`. Nothing leaves the machine. |
| `openrouter` | OpenRouter at `https://openrouter.ai/api/v1`, key read from `$OPENROUTER_API_KEY`. |

Remote access goes through OpenRouter only, never a provider API directly.

**A model listed by `ollama list` with a `:cloud` suffix runs on Ollama's servers, not yours.** Do
not configure one under `local` if the point is that nothing leaves the machine. The tool rejects a
`local` config whose `base_url` points anywhere but loopback, but it cannot tell a cloud-backed
model *name* from a local one — that part is on you.

If the model is unreachable — no network at the venue, Ollama not running, a missing key — the run
still completes. It falls back to the energy candidate, marks the start `REVIEW`, and says why. A
session is never failed because a model was unavailable.

## Known limitations

Measured against three 2025 sessions whose hand-made cut points are known.

**End detection is reliable.** Within about a second on all three, and it needs no model.

**Start detection is better than it was, but still worth checking.** Of the three: two land
within about a second of the hand-made cut, and one is flagged for review because the recording
has almost no silence before it starts.

The case that used to be wrong by 8 seconds *without* being flagged is fixed. The mic check and
the welcome had landed in the same transcript line, so the correct point was not in the set of
options the model could choose from at all, and nothing looked anomalous. Lines now break on
sentence endings as well as on pauses, which keeps the opening selectable.

Three sessions is a small sample and the model is still choosing from a transcript. Read the
`opens on:` line on every session, and treat every proposal as something to review.

The detector also assumes the recording contains a decent stretch of dead air, which holds when
the sound board is not feeding the capture before the session starts. A recording joined
mid-session may produce a start that is too *late* — the one error the design otherwise avoids. It
will usually be flagged, but check it.

## Troubleshooting

**`asr.model_path is required for whisper-cpp`** — your config has no `model_path`, or the path is
wrong. Use an absolute path, not `~`.

**`whisper-cli not found on PATH`** — `brew install whisper-cpp`.

**`ffprobe not found on PATH`** — `brew install ffmpeg`.

**`refiner mode 'local' must not point off-machine`** — your `base_url` under `local` is not
loopback. Either fix it or switch to `openrouter`.

**Start marked `REVIEW` with "no refinement available"** — the model was unreachable. Check that
`ollama list` responds, or that `$OPENROUTER_API_KEY` is set. Not fatal: the timecodes are still
usable, you just have to check the start yourself.

**A run seems to hang** — no external call has a timeout yet. If ffmpeg or whisper stalls on a
slow or network-mounted file, Ctrl-C is safe: partial cache files are cleaned up rather than
reused. Copy the raw to a local disk first; it is faster anyway.

## Tests

```bash
uv run pytest                      # everything
uv run pytest -m "not samples"     # skip the tests that need real footage
```

The `samples` tests score the detector against the three 2025 sessions and skip cleanly when
`sample/` is absent, so the suite stays green on a machine without the footage.

## Also in this repo

`video_monitor.py`, `filecheck.md`, and `config.example.json` / `config.AAO25.json` belong to an
older file-copy monitor that watches the capture share and copies finished recordings off it. That
is a separate tool with a **separate and unrelated `config.json` format** — don't confuse its
config with vidproc's. It has not been migrated yet.

## sample/

`sample/` holds real conference recordings of identifiable presenters, kept for regression
testing. It is gitignored and **must never be committed, pushed, or uploaded anywhere.** So is
`working/`, which fills with decoded audio and transcript excerpts derived from it. Fixtures that
do get committed are derived values only — timecodes and transcripts, never media.
