from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vidproc.audio import Envelope
from vidproc.config import DetectionConfig


@dataclass(frozen=True)
class Boundaries:
    noise_floor_db: float
    gate_db: float
    head_candidate_s: float
    end_s: float
    pre_head_silence_s: float
    tail_silence_s: float


def noise_floor_db(env: Envelope, percentile: float) -> float:
    """Per-file noise floor. Derived, never hardcoded — see spec §6.1."""
    if len(env.db) == 0:
        return 0.0
    return float(np.percentile(env.db, percentile))


def _boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) index ranges where mask is True."""
    if len(mask) == 0 or not mask.any():
        return []
    diff = np.diff(mask.astype(np.int8))
    starts = (np.flatnonzero(diff == 1) + 1).tolist()
    ends = (np.flatnonzero(diff == -1) + 1).tolist()
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))


def runs_above(env: Envelope, gate_db: float, min_duration_s: float) -> list[tuple[float, float]]:
    """Time ranges where the envelope stays above the gate for long enough."""
    return [
        (s * env.hop_s, e * env.hop_s)
        for s, e in _boolean_runs(env.db > gate_db)
        if (e - s) * env.hop_s >= min_duration_s
    ]


def final_silence_start(
    env: Envelope, gate_db: float, min_silence_s: float
) -> tuple[float, float]:
    """(start, length) of the TRAILING silence — the quiet that runs to the end.

    Anchored to the last frame above the gate, not to "the last long quiet
    run". Those differ: a file that opens with five minutes of silence and
    then never stops talking has a long quiet run, but it is at the front and
    is emphatically not the end of the session.

    Returns (duration, 0.0) — trim nothing from the end — when nothing is ever
    above the gate, or when too little silence follows the last audio, meaning
    the recording was stopped while the session was still going.
    """
    loud = env.db > gate_db
    if not loud.any():
        return env.duration_s, 0.0
    last_loud = int(np.flatnonzero(loud)[-1])
    silence_s = (len(loud) - 1 - last_loud) * env.hop_s
    if silence_s < min_silence_s:
        return env.duration_s, 0.0
    return (last_loud + 1) * env.hop_s, silence_s


def detect_boundaries(env: Envelope, cfg: DetectionConfig) -> Boundaries:
    """Energy-only boundary detection.

    The head is a deliberately conservative lower bound: the first sustained
    speech. On clean sessions it is within a second of the human cut; where
    something is said before the session proper (PT01), it lands early and the
    refiner moves it forward. It must never land late.
    """
    floor = noise_floor_db(env, cfg.floor_percentile)
    gate = floor + cfg.gate_offset_db

    speech = runs_above(env, gate, cfg.min_speech_s)
    head = speech[0][0] if speech else 0.0

    end, tail_silence = final_silence_start(env, gate, cfg.min_tail_silence_s)

    # How much quiet immediately precedes the head candidate. A long run means an
    # unambiguous session start; a short one means something was already going on.
    pre_head = 0.0
    if head > 0.0:
        idx = env.index_at(head)
        quiet = env.db[:idx] <= gate
        trailing = 0
        for value in quiet[::-1]:
            if not value:
                break
            trailing += 1
        pre_head = trailing * env.hop_s

    return Boundaries(
        noise_floor_db=floor,
        gate_db=gate,
        head_candidate_s=head,
        end_s=end,
        pre_head_silence_s=pre_head,
        tail_silence_s=tail_silence,
    )
