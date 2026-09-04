from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vidproc.analyze import analyze
from vidproc.asr import Line, Word
from vidproc.audio import Envelope
from vidproc.config import Config, DetectionConfig
from vidproc.refine import Refinement


class FakeASR:
    def __init__(self, words: list[Word]) -> None:
        self._words = words
        self.calls: list[tuple[float, float]] = []

    def transcribe(self, media: Path, start_s: float, end_s: float) -> list[Word]:
        self.calls.append((start_s, end_s))
        return [w for w in self._words if start_s <= w.start_s < end_s]


class FakeRefiner:
    def __init__(self, start_s: float, available: bool = True) -> None:
        self._start_s = start_s
        self._available = available

    def refine(self, lines: list[Line], candidate_s: float) -> Refinement:
        return Refinement(
            start_s=self._start_s, reason="chosen", skipped="mic check",
            model_used="fake", available=self._available,
        )


@pytest.fixture
def fake_envelope(monkeypatch) -> None:
    """900s silence, 20s speech, 25s plateau, 600s speech, 190s silence."""
    hop = 0.25
    segs = [(8.0, 900.0), (35.0, 20.0), (12.0, 25.0), (35.0, 600.0), (8.0, 190.0)]
    db = np.concatenate([np.full(int(round(d / hop)), lvl) for lvl, d in segs])
    monkeypatch.setattr(
        "vidproc.analyze.envelope_for", lambda *a, **k: Envelope(db=db, hop_s=hop)
    )


@pytest.fixture
def fake_probe(monkeypatch) -> None:
    from fractions import Fraction

    from vidproc.probe import MediaInfo

    monkeypatch.setattr(
        "vidproc.analyze.probe",
        lambda p, **k: MediaInfo(
            path=p, duration_s=1735.0, width=1920, height=1080,
            fps=Fraction(16, 1), video_codec="h264",
            audio_sample_rate=44100, audio_channels=2,
        ),
    )


def make_config(tmp_path: Path) -> Config:
    return Config(working_dir=tmp_path, detection=DetectionConfig())


def test_proposal_uses_refined_head(tmp_path, fake_envelope, fake_probe) -> None:
    words = [
        Word("Hello?", 900.5, 901.0),
        Word("Welcome", 945.2, 946.0),
        Word("Thank", 1530.0, 1530.5),
        Word("you", 1530.5, 1531.0),
        Word("for", 1531.0, 1531.3),
        Word("attending", 1531.3, 1532.0),
    ]
    p = analyze(
        tmp_path / "AAO2025PT01.mp4", make_config(tmp_path),
        asr=FakeASR(words), refiner=FakeRefiner(945.2),
    )
    assert p.session == "AAO2025PT01"
    assert p.head_candidate_s == pytest.approx(900.0, abs=1.0)
    assert p.start_s == pytest.approx(945.2 - 1.0, abs=0.01)  # preroll applied
    assert p.model_used == "fake"


def test_large_gap_flags_head(tmp_path, fake_envelope, fake_probe) -> None:
    p = analyze(
        tmp_path / "s.mp4", make_config(tmp_path),
        asr=FakeASR([]), refiner=FakeRefiner(1000.0),
    )
    assert p.head_confident is False


def test_asr_is_only_given_bounded_windows(tmp_path, fake_envelope, fake_probe) -> None:
    """Never hand the transcriber dead air — that is the hallucination trigger."""
    asr = FakeASR([])
    cfg = make_config(tmp_path)
    analyze(tmp_path / "s.mp4", cfg, asr=asr, refiner=FakeRefiner(905.0))
    head_call, tail_call = asr.calls
    assert head_call[0] >= 900.0 - 1.0
    assert head_call[1] - head_call[0] == pytest.approx(cfg.detection.head_window_s)
    assert tail_call[1] - tail_call[0] == pytest.approx(cfg.detection.tail_window_s)


def test_end_gets_tail_pad(tmp_path, fake_envelope, fake_probe) -> None:
    cfg = make_config(tmp_path)
    p = analyze(tmp_path / "s.mp4", cfg, asr=FakeASR([]), refiner=FakeRefiner(905.0))
    assert p.end_s == pytest.approx(1545.0 + cfg.detection.tail_pad_s, abs=1.0)


def test_start_never_goes_negative(tmp_path, fake_probe, monkeypatch) -> None:
    hop = 0.25
    db = np.concatenate([np.full(int(40 / hop), 35.0), np.full(int(60 / hop), 8.0)])
    monkeypatch.setattr(
        "vidproc.analyze.envelope_for", lambda *a, **k: Envelope(db=db, hop_s=hop)
    )
    p = analyze(
        tmp_path / "s.mp4", make_config(tmp_path),
        asr=FakeASR([]), refiner=FakeRefiner(0.0),
    )
    assert p.start_s >= 0.0


def test_tail_window_stays_bounded_on_a_short_file(tmp_path, monkeypatch) -> None:
    """A file shorter than tail_window_s still yields a bounded tail window.

    final_silence_start returns the full duration when there is no adequate
    trailing silence, so end_s can be small. The window must stay non-negative
    and never exceed tail_window_s of audio.
    """
    from fractions import Fraction

    from vidproc.probe import MediaInfo

    hop = 0.25
    db = np.full(int(30 / hop), 35.0)  # 30s, all speech, no trailing silence
    monkeypatch.setattr(
        "vidproc.analyze.envelope_for", lambda *a, **k: Envelope(db=db, hop_s=hop)
    )
    monkeypatch.setattr(
        "vidproc.analyze.probe",
        lambda p, **k: MediaInfo(
            path=p, duration_s=30.0, width=1920, height=1080, fps=Fraction(16, 1),
            video_codec="h264", audio_sample_rate=44100, audio_channels=2,
        ),
    )
    asr = FakeASR([])
    cfg = make_config(tmp_path)
    analyze(tmp_path / "short.mp4", cfg, asr=asr, refiner=FakeRefiner(0.0))
    _, tail_call = asr.calls
    assert tail_call[0] >= 0.0
    assert tail_call[1] - tail_call[0] <= cfg.detection.tail_window_s


def test_proposal_json_round_trips(tmp_path, fake_envelope, fake_probe) -> None:
    p = analyze(
        tmp_path / "s.mp4", make_config(tmp_path),
        asr=FakeASR([]), refiner=FakeRefiner(905.0),
    )
    data = json.loads(p.to_json())
    assert data["start_s"] == pytest.approx(p.start_s)
    assert "head_reasons" in data and isinstance(data["head_reasons"], list)
