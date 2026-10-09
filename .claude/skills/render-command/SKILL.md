---
name: render-command
description: Turn a session ID plus hand-found start and end times (e.g. "PT01, 200, 30:13") into the operator's ffmpeg render command for AAO 2026. Use whenever Joe gives a poster theater session and cut points and wants the command to run.
---

# Render command from manual cut points

During AAO 2026 Joe finds each session's start and end by hand and sends them as
`<session>, <start>, <end>`, for example `PT01, 200, 30:13`. Reply with one ready-to-paste
ffmpeg command.

## Input

- **Session**: `PT01`, `PT14`, and so on. Keep the two-digit form (`PT1` becomes `PT01`).
- **Start and end**: each one can be given in any of these forms:
  - plain seconds: `200`, `849.5`
  - minutes and seconds: `30:13` → 30 × 60 + 13 = 1813
  - hours, minutes and seconds: `1:02:05` → 3600 + 120 + 5 = 3725
  - words, such as "30 minutes 13 seconds", mean the same as `30:13`

## Output

1. Show the conversion for every value that wasn't plain seconds, so Joe can check the arithmetic,
   e.g. `30:13 → 30 × 60 + 13 = 1813`.
2. Give the command on one line, in a bash code block, to run from `~/tmp/aao_26`
   (which contains `raw/` and `processed/`):

```bash
ffmpeg -y -i ./raw/AAO2026<SESSION>.mp4 -r 29.850746 -ss <START> -to <END> -vf "scale=trunc(oh*a/2)*2:'min(720,ih)',format=yuv420p" -c:v libx264 -preset slower -crf 22 -profile:v high422 -level 3.1 -movflags faststart -c:a aac -b:a 80k -pass 1 -strict -2 ./processed/AAO2026<SESSION>.mp4
```

## Rules

- **Substitute only the session name in the two paths, `-ss` and `-to`.** Every other argument is
  the operator's command from the vault runbook *Poster Theater Recording Setup*, used verbatim.
  `-r 29.850746` is deliberate. Never remove, reorder or "fix" anything.
- If end ≤ start, or a value doesn't parse, say so and ask. Don't guess.
- The filename pattern `AAO2026<SESSION>.mp4` follows 2025's `AAO2025PT14.mp4`. Before the first
  command of a session, `ls ~/tmp/aao_26/raw` to confirm the file exists under that name. If the
  real names differ, use them and tell Joe to update this skill.
- Keep the reply short: the conversion line(s) and the command. Nothing else unless something is off.
