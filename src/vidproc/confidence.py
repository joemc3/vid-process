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
