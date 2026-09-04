from __future__ import annotations

import numpy as np
import pytest

from vidproc.audio import Envelope
from vidproc.config import DetectionConfig
from vidproc.detect import (
    Boundaries,
    detect_boundaries,
    final_silence_start,
    noise_floor_db,
    runs_above,
)

HOP = 0.25
FLOOR, PLATEAU, SPEECH = 8.0, 12.0, 35.0


def build(segments: list[tuple[float, float]]) -> Envelope:
    """segments: list of (level_db, duration_s) -> Envelope at 0.25s hop."""
    parts = [np.full(round(d / HOP), lvl, dtype=np.float64) for lvl, d in segments]
    return Envelope(db=np.concatenate(parts), hop_s=HOP)


def test_noise_floor_ignores_loud_majority() -> None:
    env = build([(FLOOR, 60.0), (SPEECH, 600.0)])
    assert noise_floor_db(env, 5.0) == pytest.approx(FLOOR, abs=0.5)


def test_runs_above_filters_short_bursts() -> None:
    env = build([(FLOOR, 10.0), (SPEECH, 1.0), (FLOOR, 10.0), (SPEECH, 8.0), (FLOOR, 5.0)])
    runs = runs_above(env, gate_db=16.0, min_duration_s=3.0)
    assert len(runs) == 1
    assert runs[0][0] == pytest.approx(21.0)


def test_runs_above_handles_envelope_starting_and_ending_loud() -> None:
    env = build([(SPEECH, 10.0)])
    runs = runs_above(env, gate_db=16.0, min_duration_s=3.0)
    assert runs == [(0.0, 10.0)]


def test_open_mic_plateau_does_not_trigger() -> None:
    """PT12: 14s of open-mic room tone at ~12 dB precedes real speech."""
    env = build([(FLOOR, 275.0), (PLATEAU, 14.0), (SPEECH, 100.0)])
    b = detect_boundaries(env, DetectionConfig())
    assert b.gate_db > PLATEAU
    assert b.head_candidate_s == pytest.approx(289.0, abs=1.0)


def test_head_candidate_is_early_on_pt01_pathology() -> None:
    """PT01: moderator speaks, pauses, then the real start. Energy must land on
    the FIRST speech (a safe lower bound); the refiner moves it forward."""
    env = build([(FLOOR, 900.0), (SPEECH, 20.0), (PLATEAU, 25.0), (SPEECH, 600.0)])
    b = detect_boundaries(env, DetectionConfig())
    assert b.head_candidate_s == pytest.approx(900.0, abs=1.0)


def test_end_is_start_of_final_long_silence() -> None:
    env = build([(FLOOR, 100.0), (SPEECH, 500.0), (FLOOR, 190.0)])
    b = detect_boundaries(env, DetectionConfig())
    assert b.end_s == pytest.approx(600.0, abs=0.5)
    assert b.tail_silence_s == pytest.approx(190.0, abs=0.5)


def test_end_survives_a_short_27_second_tail() -> None:
    """PT01's recording stopped 27s after the session ended."""
    env = build([(FLOOR, 100.0), (SPEECH, 500.0), (FLOOR, 27.0)])
    b = detect_boundaries(env, DetectionConfig())
    assert b.end_s == pytest.approx(600.0, abs=0.5)


def test_end_ignores_short_mid_session_gaps() -> None:
    env = build([(FLOOR, 60.0), (SPEECH, 200.0), (FLOOR, 12.0), (SPEECH, 200.0), (FLOOR, 100.0)])
    b = detect_boundaries(env, DetectionConfig())
    assert b.end_s == pytest.approx(472.0, abs=0.5)


def test_no_trailing_silence_yields_end_at_duration() -> None:
    """Recording stopped mid-session: trim nothing from the end.

    Also guards the leading-silence trap — the 60s of quiet at the FRONT is a
    long silence run, but it is not the tail and must never be returned as the
    end cut.
    """
    env = build([(FLOOR, 60.0), (SPEECH, 200.0)])
    b = detect_boundaries(env, DetectionConfig())
    assert b.end_s == pytest.approx(env.duration_s, abs=0.5)
    assert b.tail_silence_s == pytest.approx(0.0, abs=0.5)


def test_pre_head_silence_is_measured() -> None:
    env = build([(FLOOR, 300.0), (SPEECH, 100.0), (FLOOR, 100.0)])
    b = detect_boundaries(env, DetectionConfig())
    assert b.pre_head_silence_s == pytest.approx(300.0, abs=1.0)


def test_silent_file_does_not_crash() -> None:
    env = build([(FLOOR, 120.0)])
    b = detect_boundaries(env, DetectionConfig())
    assert isinstance(b, Boundaries)
    assert b.head_candidate_s == 0.0
    assert b.end_s == pytest.approx(env.duration_s, abs=0.5)


def test_final_silence_start_returns_start_and_length() -> None:
    env = build([(SPEECH, 100.0), (FLOOR, 40.0)])
    start, length = final_silence_start(env, gate_db=16.0, min_silence_s=10.0)
    assert start == pytest.approx(100.0, abs=0.5)
    assert length == pytest.approx(40.0, abs=0.5)
