# Poster Theater Session Trim Pipeline — Design

Date: 2026-09-04
Status: Draft for review
Scope: Steps 1–8 of the Poster Theater recording workflow. Steps 9–12 explicitly excluded.

## 1. Purpose

Replace the manual VLC-scrubbing step in the Poster Theater post-production workflow with an
automated detect → propose → confirm → render → distribute pipeline.

Today, producing one publishable session requires a human to open the raw capture in VLC, hunt for
the second at which the session actually begins, hunt for the second at which it ends, do arithmetic,
paste those numbers into a hand-written ffmpeg command, and spot-check the result. That is the
bottleneck. Everything else in the workflow is either already automated or is a file copy.

The pipeline must earn trust incrementally: it proposes cut points and a human approves them, with
the review burden shrinking as measured accuracy justifies it.

## 2. Scope

In scope — workflow steps 1–8:

| Step | Description | Current state |
| --- | --- | --- |
| 1 | Copy raw from Surface share to backup drive `raw/` | `video_monitor.py`, needs hardening |
| 2 | Copy raw from backup drive to local working directory | Manual / optional |
| 3 | Watch-and-copy automation | `video_monitor.py` |
| 4–5 | Find session start and end timecodes | **Manual. This is the bottleneck.** |
| 6 | Render trimmed + resized output | Hand-pasted ffmpeg command |
| 7 | QC entry/exit points and spot-check | Manual in VLC |
| 8 | Copy processed file to backup drive `processed/` | Manual |

Out of scope — steps 9–12. FTP upload to CDN, FTP upload to host, the MediaObjects Updater, and
meeting-guide verification remain manual. These involve credentials, a strict ordering constraint
(CDN before host), and a web admin UI. Automating them buys little and adds failure modes. They may
become a second sub-project once steps 1–8 are trusted.

Also out of scope: splitting a session into per-presenter clips. One session produces one file.

### 2.1 Proof-of-concept framing

This build targets AAO Poster Theater specifically, but it is a proof of concept for a pipeline that
will later handle other conferences. There is no schedule pressure and no acceptable version of
"working badly but sooner" — correctness governs, and the next AAO meeting is not a constraint on
scope or quality.

The consequence is not to build multi-conference machinery now. It is to keep the seams in the right
places, so that generalising later is additive rather than a rewrite. Concretely, these are
event-specific and must be data or injected dependencies rather than constants in code:

| Event-specific | Why it will differ elsewhere |
| --- | --- |
| Paths, share names, drive layout | Different venue, different rig |
| Session filename conventions (`AAO2025PT01.mp4`) | Each conference names its own |
| The head-cut rule and its prompt (§6.4) | Depends on session format; poster theater's moderator-then-presenters shape is not universal |
| Detection thresholds and pads (§6.2, §6.4) | Depends on the capture chain and how the audio is mixed |
| Render profile — resolution, CRF, audio bitrate (§7) | Depends on the destination platform |
| Source frame rate | 16 fps is this capture device; follow the source, never assume |

Equally, these are deliberately **not** built now: per-conference configuration management, a profile
registry, multi-event orchestration, or any abstraction whose only current caller is AAO. One
conference's worth of concrete behaviour, cleanly separated, is a better foundation for the second
conference than a generalisation guessed at from a single example.

## 3. Evidence base

This design is derived from three raw/processed pairs from AAO 2025 Poster Theater, held locally in
`sample/` (gitignored, must never be committed or pushed).

Ground truth was recovered by cross-correlating the RMS envelope of each processed file against its
raw counterpart at 0.25s resolution. All three aligned with correlation ≥ 0.968 and a sharp peak, and
all three are self-consistent (`end − head` equals the processed duration exactly).

| Session | Raw duration | Head cut | End cut | Tail dropped | Correlation |
| --- | --- | --- | --- | --- | --- |
| PT01 | 4319.07s (71:59) | 958.0s | 4292.0s | 27.0s | 0.9899 |
| PT06 | 4606.26s (76:46) | 611.0s | 4323.0s | 283.2s | 0.9679 |
| PT12 | 3374.46s (56:14) | 290.0s | 3184.0s | 190.2s | 0.9885 |

Round numbers confirm these were produced by hand-typed `-ss` / `-to` values.

### 3.1 Source characteristics

Consistent across all three raws:

- Container MP4, video H.264 High profile, 1920×1080, **16 fps** (constant, not 29.97)
- Video bitrate ~98 kbps, audio AAC-LC 44.1 kHz stereo ~130 kbps, total ~235 kbps
- Video stream `start_time` 0.125s, audio `start_time` 0.0s — a 125 ms A/V offset present in source

The 16 fps figure is important and is addressed in §7.

### 3.2 Audio characteristics

Because Freeman supplies a single pre-mixed feed and the board is not live before the session, dead
air in these captures is near-digital silence rather than room tone.

| Session | Noise floor (p5) | Body p50 | Dropped tail p50 |
| --- | --- | --- | --- |
| PT01 | 7.4 dB | 29.8 dB | 7.9 dB |
| PT06 | 7.5 dB | 39.7 dB | 8.4 dB |
| PT12 | 8.1 dB | 34.8 dB | 8.3 dB |

The noise floor is remarkably stable (7.4–8.1 dB). Session body level varies by ~10 dB between
sessions, reflecting day-to-day differences in how the mix was set. Speech peaks reach 40–50 dB.

There is a distinct intermediate state — an open microphone in a quiet room — sitting around
11–13 dB, above the noise floor but well below speech. On PT12 this runs from 275.5s to 289.2s. A
detection gate set too low fires on this and lands ~14s early.

### 3.3 What the human is actually deciding

The three head cuts, read as content rather than signal:

| Session | Skipped | Started on |
| --- | --- | --- |
| PT01 | "Sorry, we're having a little technical difficulty", housekeeping, "Hello?" mic test | "Good morning, everyone. My name is Alex Rivera." |
| PT06 | "Hello, everybody, oh yeah, I'm pretty loud, okay" | "Welcome to this session on podium posters." |
| PT12 | nothing | "[All right,] good morning everyone." |

The operative rule is **start at the first line that belongs in the published video**. Mic checks,
technical apologies, and pre-session logistics are excluded. This is a semantic judgment about
content, not a threshold on a signal — which is why the previous attempt to find it with energy
detection alone stalled.

Note that in PT01 the moderator's introduction was itself dropped, because it consisted entirely of a
technical-difficulty apology and housekeeping. The rule is not "start at the moderator".

The three tails are uniform: closing remarks run directly into the board being cut.

- PT01: "Thank you, guys, for attending our session. Thank you." → silence
- PT06: "Thanks, everybody, for joining us. Look forward to the next year." → silence
- PT12: "Great discussion. Thank you." → silence

**No applause occurs in any of the three sessions.** See §6.3.

## 4. Architecture

Five components with narrow interfaces, each independently testable.

```
  ingest  ──▶  analyze  ──▶  review  ──▶  render  ──▶  distribute
    │            │             │            │             │
 raw file    proposal      decision      output       backup drive
             (JSON)        (JSON)        (mp4)        processed/
```

Each stage reads and writes files on disk and records its result in a per-session sidecar JSON. A
crash or interruption resumes from the last completed stage rather than restarting. The sidecar is
the unit of state; there is no database.

| Component | Input | Output | Depends on |
| --- | --- | --- | --- |
| `ingest` | Surface share, backup drive | raw file in working dir | filesystem only |
| `analyze` | raw file | proposal JSON (cut points + confidence + evidence) | ffmpeg, ASR, LLM |
| `review` | proposal JSON | decision JSON (approved/adjusted cut points) | policy setting, human |
| `render` | raw file + decision JSON | processed mp4 | ffmpeg |
| `distribute` | processed mp4 | file on backup drive `processed/` | filesystem, Parallels bridge |

`analyze` is the only component with model dependencies. `render` and `distribute` are deterministic
and pure. This boundary matters: it means the risky, probabilistic part of the system produces a
reviewable artifact, and everything downstream of the human decision is mechanical.

## 5. Ingest (steps 1–3)

The existing `video_monitor.py` covers step 1 in outline but has defects that must be fixed:

1. **Unbounded hang.** `wait_for_stable_file_size` returns `False` only if the file is absent at
   entry. A recording that dies below `min_file_size` and stops growing loops forever, and because
   the main loop is single-threaded and blocking, the whole monitor dies with it. Every subsequent
   recording that day goes unprocessed. Needs a wall-clock timeout after which the file is marked
   failed and the loop moves on.
2. **Serialized processing.** While waiting out a stability window on one file, no other file is even
   discovered. Concurrent room captures queue behind each other.
3. **Completion inferred from filename.** A copy that fails halfway leaves a truncated file in the
   destination, which permanently marks that session complete. Copies must go to a temp name and be
   renamed atomically on success, with size verification.

Flow, revised per §3 of the operational doc:

1. Watch the Surface share (`/Volumes/Captures/AAO_RAW`) for new `*.mp4`.
2. On stability, copy to the backup drive `DA_Captures/AAO_25/raw/` — atomic, verified.
3. Copy from the **backup drive** to the local working directory, not from the share again.

Step 3 reads from the backup drive rather than the network share because the raw is already safely
there and there are two identical drives carrying it. The share is read exactly once.

The local working copy is not optional. The raw is read at least four times — once for envelope
analysis, once for transcription, once per render, and again for QC — and the backup drive is reached
through the Parallels bridge. Paying that cost once and serving everything after from local NVMe is
strictly better.

**Constraint:** the backup drive is NTFS and is exposed to macOS via Parallels, not natively mounted.
Every read and write to it depends on the VM running with the drive shared. `ingest` and `distribute`
must check for the mount and fail with a clear message rather than a filesystem error.

## 6. Analyze (steps 4–5)

### 6.1 Envelope

Decode audio to 8 kHz mono, compute RMS in 0.25s frames, convert to dB. On a ~70 minute session this
takes roughly two seconds. Derive a per-file noise floor as the 5th percentile of the envelope and set
the detection gate at **floor + 8 dB**. Measured gates were 15.4, 15.5, and 16.1 dB — comfortably above
the 11–13 dB open-mic plateau and far below speech.

The gate is derived per file rather than hardcoded. The noise floor was stable across these three
samples, but three samples from one event is not enough to fix a constant.

### 6.2 End detection — energy alone

Find the **start of the final contiguous silence run of at least 10 seconds**. Measured against ground
truth:

| Session | Detected | Actual | Error |
| --- | --- | --- | --- |
| PT01 | 4291.25s | 4292.0s | −0.75s |
| PT06 | 4321.75s | 4323.0s | −1.25s |
| PT12 | 3184.75s | 3184.0s | +0.75s |

All within 1.25 seconds. No model required.

**The minimum silence length must not exceed 20 seconds.** PT01's recording was stopped 27 seconds
after the session ended; a 30-second requirement fails to find any qualifying silence and the
detector returns garbage. 10s is the chosen default.

Apply a configurable pad after the detected point (default 2.0s) plus an audio fade-out (default
1.0s), per the operator's stated preference.

A short window before the detected end (default 60s) is also transcribed, solely to supply the
closing-language signal in §6.6 and the review context in §8. It does not affect the detected
timecode.

### 6.3 Applause

Not implemented. Zero occurrences across three sessions. The end rule keys on silence, and applause
is loud, so if applause does occur the detected end already falls after it. The pad-and-fade in §6.2
covers the case. If a future sample shows applause being truncated, revisit — do not build this
speculatively.

### 6.4 Head detection — energy, then semantics

Energy alone is insufficient. Best achievable with a pure threshold, swept across 20 parameter
combinations:

| Session | Energy candidate | Actual | Error |
| --- | --- | --- | --- |
| PT12 | 289.50s | 290.0s | −0.50s |
| PT06 | 604.00s | 611.0s | −7.00s |
| PT01 | 871.00s | 958.0s | **−87.00s** |

PT01 cannot be fixed by tuning, because what the detector trips on is genuine speech — the moderator
apologizing for technical difficulties. No threshold distinguishes that from the session opening.

Two-stage approach:

**Stage 1 — candidate.** Gate at floor + 8 dB, first run of sustained audio ≥ 3.0s. With these
settings the candidate is earlier than the true cut on all three samples (−87.0, −7.0, −0.5), giving a
safe lower bound.

**Stage 2 — refinement.** Transcribe from the candidate forward for a bounded window (default 180s)
with word-level timestamps. Pass the transcript to a language model with the rule from §3.3: identify
the first line that belongs in the published video, skipping mic checks, technical apologies, and
pre-session logistics. Return the timestamp of that line, and the reason for skipping anything before
it.

Refinement is a **pluggable backend selected by config**, with three modes:

| Mode | Behaviour |
| --- | --- |
| `local` | Ollama on `http://localhost:11434/v1`. Nothing leaves the machine. |
| `openrouter` | OpenRouter on `https://openrouter.ai/api/v1`, key from `OPENROUTER_API_KEY`. |
| `none` | No model. Stage 1's candidate is proposed as-is and the head is always flagged for review. |

Remote access is via OpenRouter only — never a provider API directly. Ollama and OpenRouter are both
OpenAI-compatible, so `local` and `openrouter` are one client class differing in base URL, model, and
key; `none` is a null implementation of the same interface.

Caution when configuring `local`: models listed by `ollama list` with a `:cloud` suffix (e.g.
`deepseek-v3.2:cloud`) execute on Ollama's servers, not this machine, and must not be used where the
point of `local` is that no content leaves. Genuinely local candidates present on this machine include
`qwen3.6:35b-a3b`, `gemma4:31b-it-q8_0`, and `gpt-oss:120b`.

Only the head window is ever sent to a model. The end-detection closing-language signal (§6.6) is
plain keyword matching, computed locally in every mode.

**Degradation is mandatory, not optional.** If the configured backend is unreachable — no network at a
venue, Ollama not running, missing key — refinement returns the Stage 1 candidate with the head marked
low-confidence, and the session is flagged for review. The pipeline must never fail a session because a
model was unavailable.

Applied to the three samples this selects 958s, 611s, and 289.5s — matching the human within 0.5s.

Apply a configurable pre-roll before the chosen word (default 1.0s) so the first syllable is not
clipped. Note that the PT12 cut at 290.0 landed between "All right" and "good morning everyone",
clipping "All right"; a 1.0s pre-roll from onset at 289.5 would retain it.

Only the window is transcribed, not the session. This keeps a session's analysis in the seconds-to-
tens-of-seconds range rather than minutes.

### 6.5 ASR selection

Requirement: word-level timestamps, English, local, fast.

`whisper.cpp` is installed (1.8.2; 1.9.2 available) with `ggml-large-v3.bin` present. It works, but
**`--vad` defaults to false**, and Whisper's documented failure mode is hallucinating text during
non-speech. Captures here open with 5–16 minutes of near-silence, which is precisely the trigger. Any
previous attempt that fed whole files to Whisper without VAD would have produced confident nonsense in
the dead air. This is the most likely explanation for the 2025 attempt being abandoned.

Two mitigations, both applied:

1. Never transcribe dead air. Stage 2 only ever sees audio from the energy candidate forward.
2. Enable VAD explicitly if whisper.cpp is used.

Preferred: **Parakeet TDT 0.6b v3 via `parakeet-mlx`**. Lower WER than Whisper large-v3 on English
(6.32% vs 7.44%), roughly an hour of audio in ~53 seconds on Apple Silicon, native word timestamps.
Not yet installed — see §12.

The ASR layer must sit behind an interface with a single method (audio window in, timestamped words
out) so the engine can be swapped without touching detection logic.

Cloud ASR was evaluated and rejected for the default path: AssemblyAI at $0.15–0.21/hr and Deepgram at
$0.46/hr make cost irrelevant (~$2–6 per event), but a pipeline that needs the network is a pipeline
that can fail at a conference venue on the busiest morning.

### 6.6 Confidence and flagging

`analyze` emits a confidence classification per boundary, used by the review policy in §8.

Head signals:

- **Gap** between energy candidate and model selection. Measured: 0.5s (PT12), 7.0s (PT06), 87.0s
  (PT01). Small gap indicates the two independent methods agree.
- **Skip reason present.** If the model had to discard content to reach its choice, more judgment was
  exercised.
- **Clean silence before the choice.** Silence immediately preceding the selected line indicates an
  unambiguous session start.

End signals:

- **Length of the final silence run.** Longer is more confident. PT01's 27s is the shortest observed.
- **Closing language present** in the final ~60s ("thank you for attending", "that's the end of our
  session", "thanks everybody for joining").

Initial thresholds, chosen to reproduce the desired outcome on the three known samples:

- **Head is confident** when the gap is ≤ 10s *and* there are ≥ 5s below the gate immediately before
  the energy candidate. Measured: PT12 (gap 0.5s, quiet before) confident; PT06 (gap 7.0s, quiet
  before) confident; PT01 (gap 87.0s) flagged.
- **End is confident** when the final silence run is ≥ 20s *and* closing language appears in the
  transcribed tail window. All three qualify — PT01's 27s silence is the narrowest margin observed.

A skip reason is recorded as review context but does not by itself flag, because PT06 legitimately
skips a mic check and should still pass. Length of skipped material is captured so that the
relationship can be re-examined once more sessions exist.

These thresholds are fitted to three samples and should be treated as provisional. Because the
default policy is `all` (§8), a mis-set threshold costs a needless review rather than a bad render,
and every proposal-versus-decision pair is logged so they can be re-fitted from real usage.

## 7. Render (step 6)

Current command, from the operational doc:

```
ffmpeg -y -i ./raw/AAO2025PT14.mp4 -r 29.850746 -ss 157 -to 3110 \
  -vf "scale=trunc(oh*a/2)*2:'min(720,ih)',format=yuv420p" \
  -c:v libx264 -preset slower -crf 22 -profile:v high422 -level 3.1 \
  -movflags faststart -c:a aac -b:a 80k -pass 1 -strict -2 ./processed/AAO2025PT14.mp4
```

Four defects, all verified empirically:

1. **`-r 29.850746` upsamples a 16 fps source.** All three raws are 16 fps. This duplicates frames to
   inflate the output to 29.85 fps. Measured on a 300-second excerpt: 8,955 frames and 7.31 MB with
   the current command, versus 4,800 frames and 6.13 MB without it — 87% more frames encoded for a
   19% larger file, at `-preset slower`, for no visual benefit. Remove it and let the source rate
   pass through.
2. **`-pass 1` is meaningless alongside `-crf`** — CRF is single-pass, no second pass ever runs. It
   also writes `ffmpeg2pass-0.log` and a ~34 MB `ffmpeg2pass-0.log.mbtree` into the current working
   directory on every run. Remove, along with `-strict -2`.
3. **`-profile:v high422` is silently overridden.** With `format=yuv420p` in the chain, x264 reports
   `profile High, level 3.1, 4:2:0, 8-bit` and ffprobe confirms `profile=High` in the output. Harmless
   but misleading; had it taken effect it would have produced 4:2:2 output that breaks hardware
   decoders and HLS clients. Remove.
4. **`-ss` after `-i` forces output seeking**, decoding everything before the start point. Input
   seeking is frame-accurate in current ffmpeg — verified byte-identical durations both ways — so
   moving `-ss` and `-to` before `-i` is a free saving.

Corrected form:

```
ffmpeg -y -ss <start> -to <end> -i <raw> \
  -vf "scale=-2:'min(720,ih)',format=yuv420p" \
  -c:v libx264 -preset slower -crf 22 -level 3.1 -movflags +faststart \
  -c:a aac -b:a 80k <output>
```

`-to` is absolute on the input timeline when paired with `-ss` — verified — so existing `-ss`/`-to`
arithmetic carries over unchanged. `scale=-2:'min(720,ih)'` reproduces the existing behaviour
(1920×1080 → 1280×720) while guaranteeing an even width.

Audio fade-out per §6.2 is added as an `afade` filter. A matching fade-in at the head is available but
defaults off, pending operator preference.

Rendering is deterministic given a decision JSON, so a re-render after an adjusted cut point is a
single idempotent command.

## 8. Review (step 7) and policy

`review` presents each proposal with the evidence needed to approve it at a glance:

- Proposed start and end timecodes and resulting duration
- The transcript around each boundary, with the selected line marked
- The model's stated reason for skipping anything before the start
- Confidence classification per boundary
- Short preview clips extracted around each cut point for playback

Approving is confirmation; overriding is entering a corrected timecode, which feeds `render` directly.

**Review policy** is a setting, not a hardcoded behaviour, because trust must be earned:

| Mode | Behaviour |
| --- | --- |
| `all` | Every session is reviewed. **Default.** |
| `flagged` | Only low-confidence sessions stop for review; confident ones proceed automatically. |
| `none` | Fully automatic. Available but not recommended until accuracy is established. |

Under `flagged`, PT06 and PT12 would flow through and PT01 would stop — which is the desired
behaviour on these samples. The mode is exposed as a CLI flag and persisted in config.

Every session records its proposal, decision, and any human override regardless of mode, so that
accuracy can be measured over time and the decision to relax the policy rests on data.

## 9. Distribute (step 8)

Copy the rendered file to the backup drive `DA_Captures/AAO_25/processed/`. Atomic write to a temp
name, verify size, rename. Fail loudly if the Parallels-mounted drive is unavailable rather than
writing to a path that silently is not the drive.

The pipeline stops here. Steps 9–12 remain manual.

## 10. State and file layout

```
working/
  <session>/
    raw.mp4              # local working copy
    envelope.npy         # cached RMS envelope
    proposal.json        # analyze output: cut points, confidence, evidence
    decision.json        # review output: final cut points, who decided, overrides
    transcript.json      # window transcripts retained for audit
    processed.mp4        # render output
```

The sidecars make each stage resumable and make the system auditable after the fact — when a cut is
wrong, the transcript and the model's reasoning are still on disk.

## 11. Testing

- **Regression against ground truth.** The three sample pairs are a fixture set. Detection changes are
  scored as error in seconds against the known human cut points. Current baseline: end within 1.25s on
  all three; head within 0.5s on two and requiring semantic refinement on the third.
- **Envelope and detection are pure functions** over an audio envelope, so they test without media
  files — synthesise envelopes with known structure, including the PT01 pathology (speech, gap, speech)
  and the PT12 open-mic plateau.
- **Render is verified by probing output**: duration, frame count, resolution, profile, and absence of
  stray pass-log files.
- **Ingest failure modes get explicit tests**: a file that stalls below minimum size must time out and
  not block the queue; an interrupted copy must not mark a session complete.
- Sample media stays local and gitignored; fixtures committed to the repo are envelopes and
  transcripts, never video.

## 12. Assumptions and open questions

Flagged explicitly, because the evidence base is three sessions from a single event, single venue,
single capture rig.

1. **Noise floor stability.** 7.4–8.1 dB across three files. The gate is derived per file to reduce
   exposure, but if a future capture has a genuinely noisy floor the floor+8 dB offset may need
   revisiting.
2. **No applause.** Zero occurrences in three sessions. §6.3 is a deliberate omission, not an oversight.
3. **16 fps.** Constant across three raws. If a room is rigged differently the render must follow the
   source rate rather than assume 16.
4. **`parakeet-mlx` is not yet installed** and its accuracy on this audio is unverified. The ASR
   interface exists so that whisper.cpp with `--vad` remains a working fallback.
9. **Local refinement quality is unverified.** Whether a local model handles the PT01 judgment
   ("skip the technical-difficulty apology") as reliably as a frontier model is exactly what the
   three-mode design exists to measure. The regression fixture in §11 scores each mode against the
   same three known cut points, so the choice can be made on data rather than preference.
5. **Head-cut rule generality.** Derived from three examples. It correctly explains all three, but
   "first line that belongs in the published video" is a judgment that may have edge cases not present
   in this sample — a session with no moderator, a session joined in progress, two sessions in one file.
6. **125 ms A/V offset** exists in the source (video `start_time` 0.125, audio 0.0). It has not caused
   observable problems and ffmpeg handles it during trim, but it should be confirmed in rendered output.
7. **Parallels dependency** for the NTFS backup drive is an operational precondition the pipeline
   cannot satisfy itself.
8. **Pre-roll preference.** Default 1.0s before the selected word is proposed but not confirmed by the
   operator; last year's PT12 cut clipped "All right" and it is not established whether that was
   intentional.

## 13. Explicitly not building

- Per-presenter segmentation. One session, one file.
- Applause detection (§6.3).
- FTP upload to CDN or host, MediaObjects Updater, meeting-guide verification (steps 9–12).
- Full-session transcription in the critical path. Optional, and only if captions become a requirement.
- A general-purpose video editor. This pipeline trims two ends and resizes.
