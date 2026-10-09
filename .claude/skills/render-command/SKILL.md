---
name: render-command
description: Turn a session ID plus hand-found start and end times (e.g. "PT01, 200, 30:13") into the operator's ffmpeg render command for AAO 2026 and run it. Use whenever Joe gives a poster theater session and cut points.
---

# Render a session from manual cut points

During AAO 2026 Joe finds each session's start and end by hand and sends them as
`<session>, <start>, <end>`, for example `PT01, 200, 30:13`. Build the render command, run it,
and report back.

## Input

- **Session**: `PT01`, `PT14`, and so on. Keep the two-digit form (`PT1` becomes `PT01`).
- **Start and end**: each one can be given in any of these forms:
  - plain seconds: `200`, `849.5`
  - minutes and seconds: `30:13` → 30 × 60 + 13 = 1813
  - hours, minutes and seconds: `1:02:05` → 3600 + 120 + 5 = 3725
  - words, such as "30 minutes 13 seconds", mean the same as `30:13`

## The command

Runs from `~/tmp/aao_26`, which contains `raw/` and `processed/`:

```bash
ffmpeg -y -i ./raw/AAO2026<SESSION>.mp4 -r 29.850746 -ss <START> -to <END> -vf "scale=trunc(oh*a/2)*2:'min(720,ih)',format=yuv420p" -c:v libx264 -preset slower -crf 22 -profile:v high422 -level 3.1 -movflags faststart -c:a aac -b:a 80k -pass 1 -strict -2 ./processed/AAO2026<SESSION>.mp4
```

**Substitute only the session name in the two paths, `-ss` and `-to`.** Every other argument is
the operator's command from the vault runbook *Poster Theater Recording Setup*, used verbatim.
`-r 29.850746` is deliberate. Never remove, reorder or "fix" anything.

## Steps

1. **Convert.** Show the conversion for every value that wasn't plain seconds, e.g.
   `30:13 → 30 × 60 + 13 = 1813`. If end ≤ start, or a value doesn't parse, stop and ask.
2. **Check the input.** `ls ~/tmp/aao_26/raw`. The name `AAO2026<SESSION>.mp4` follows 2025's
   `AAO2025PT14.mp4`. If the file isn't there, stop and say what is there; if real names differ,
   use them and tell Joe to update this skill. Get the source length with
   `ffprobe -v error -show_entries format=duration -of csv=p=0 <file>` and stop if END is past it.
3. **Check the output.** If `~/tmp/aao_26/processed/AAO2026<SESSION>.mp4` already exists, ask
   before running: `-y` overwrites it without asking.
4. **Show and run.** Show the command in a bash block, then run it with `cd ~/tmp/aao_26 && <command>`
   as a background Bash task (`run_in_background`), since `-preset slower` on an hour of video
   takes a while. Tell Joe it has started; don't poll.
5. **Verify when it finishes.** Report the exit status. On failure, show the tail of the output.
   On success, ffprobe the output's duration and compare it to END − START. Say whether it matches
   within about a second, and remind Joe to check the start and end in VLC.

Keep replies short: conversion, command, "started"; then the result when it finishes.
