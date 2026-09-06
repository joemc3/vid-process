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


SENTENCE_END = (".", "?", "!")

# Tokens that end in "." without ending a sentence. whisper is run with -ml 1
# so every token is its own segment, and breaking after one of these puts a
# mid-sentence split into the transcript the operator reads as evidence.
ABBREVIATIONS = frozenset(
    {
        "dr.", "drs.", "mr.", "mrs.", "ms.", "prof.", "st.", "jr.", "sr.",
        "vs.", "etc.", "e.g.", "i.e.", "no.", "fig.", "approx.",
    }
)


def ends_sentence(text: str) -> bool:
    if not text.endswith(SENTENCE_END):
        return False
    if text.lower() in ABBREVIATIONS:
        return False
    # A lone initial -- "J." in "J. Okonkwo" -- is part of a name.
    return not (len(text) == 2 and text[0].isalpha() and text[1] == ".")


def words_to_lines(words: list[Word], max_gap_s: float = 0.8, max_words: int = 14) -> list[Line]:
    """Group words into readable lines, breaking on sentences, pauses and length.

    refine() can only propose a start that is a line boundary, so a line that
    runs a mic check into the session opening makes the correct cut point
    unselectable. Breaking on sentence-final punctuation as well as on pauses
    keeps the opening reachable when the speaker does not pause before it.
    """
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
            word.start_s - current[-1].end_s > max_gap_s
            or len(current) >= max_words
            or ends_sentence(current[-1].text)
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
