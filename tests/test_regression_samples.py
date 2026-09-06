from __future__ import annotations

import os
from pathlib import Path

import pytest

from vidproc.asr import Line, WhisperCppASR, words_to_lines
from vidproc.audio import envelope_for
from vidproc.config import AsrConfig, DetectionConfig
from vidproc.detect import detect_boundaries

SAMPLES = Path(__file__).resolve().parents[1] / "sample" / "raw"

# Hand-made cut points recovered by envelope cross-correlation (spec §3).
GROUND_TRUTH = {
    "AAO2025PT01": (958.0, 4292.0),
    "AAO2025PT06": (611.0, 4323.0),
    "AAO2025PT12": (290.0, 3184.0),
}

# End detection is energy-only and must be tight (spec §6.2 measured -0.75/-1.25/+0.75).
END_TOLERANCE_S = 2.0

# The head candidate is a lower bound: never later than the human cut, and no
# more than 120s early (PT01's technical-difficulty apology is 87s early).
HEAD_MAX_EARLY_S = 120.0
HEAD_MAX_LATE_S = 0.0

# refine() returns lines[index].start_s -- it can only ever propose a start that
# words_to_lines already produced. So if no line begins at the human cut, the
# right answer is not in the option set and no model, however good, can pick it.
# This bound is on the option set itself, independent of any model.
#
# Held to END_TOLERANCE_S: the same margin the project already treats as a
# boundary being right. A line that begins later than that clips the opening
# words; the alternative the model actually reaches for is the line before it,
# which opens on the mic check.
HEAD_LINE_TOLERANCE_S = END_TOLERANCE_S

pytestmark = pytest.mark.samples


def _sample(session: str) -> Path:
    path = SAMPLES / f"{session}.mp4"
    if not path.exists():
        pytest.skip(f"{path} not present; sample footage is local-only")
    return path


@pytest.mark.parametrize("session", sorted(GROUND_TRUTH))
def test_end_detection_within_tolerance(session: str, tmp_path: Path) -> None:
    media = _sample(session)
    _, truth_end = GROUND_TRUTH[session]
    env = envelope_for(media, tmp_path)
    bounds = detect_boundaries(env, DetectionConfig())
    error = bounds.end_s - truth_end
    assert abs(error) <= END_TOLERANCE_S, f"{session} end off by {error:+.2f}s"


@pytest.mark.parametrize("session", sorted(GROUND_TRUTH))
def test_head_candidate_is_a_safe_lower_bound(session: str, tmp_path: Path) -> None:
    media = _sample(session)
    truth_head, _ = GROUND_TRUTH[session]
    env = envelope_for(media, tmp_path)
    bounds = detect_boundaries(env, DetectionConfig())
    error = bounds.head_candidate_s - truth_head
    assert error <= HEAD_MAX_LATE_S, f"{session} head candidate is {error:+.2f}s LATE"
    assert error >= -HEAD_MAX_EARLY_S, f"{session} head candidate is {error:+.2f}s early"


@pytest.mark.parametrize("session", sorted(GROUND_TRUTH))
def test_noise_floor_is_in_the_observed_band(session: str, tmp_path: Path) -> None:
    """Measured 7.4-8.1 dB across all three. A wide drift means the capture
    chain changed and the thresholds need re-fitting."""
    env = envelope_for(_sample(session), tmp_path)
    bounds = detect_boundaries(env, DetectionConfig())
    assert 5.0 <= bounds.noise_floor_db <= 12.0


def _asr() -> WhisperCppASR:
    """Whisper adapter using the operator's own model, or skip."""
    override = os.environ.get("VIDPROC_ASR_MODEL")
    candidates = [Path(override)] if override else [
        Path.home() / "models" / name
        for name in ("ggml-base.en.bin", "ggml-large-v3.bin")
    ]
    for path in candidates:
        if path.exists():
            return WhisperCppASR(AsrConfig(model_path=str(path)))
    pytest.skip("no whisper model found; set VIDPROC_ASR_MODEL to one")


def _head_lines(media: Path, tmp_path: Path) -> list[Line]:
    """The lines the refiner would be given, exactly as analyze() builds them."""
    cfg = DetectionConfig()
    env = envelope_for(media, tmp_path)
    bounds = detect_boundaries(env, cfg)
    words = _asr().transcribe(
        media, bounds.head_candidate_s, bounds.head_candidate_s + cfg.head_window_s
    )
    return words_to_lines(words)


@pytest.mark.parametrize("session", sorted(GROUND_TRUTH))
def test_true_start_is_selectable_as_a_line(session: str, tmp_path: Path) -> None:
    """The human cut must be reachable as a line boundary."""
    media = _sample(session)
    truth_head, _ = GROUND_TRUTH[session]
    lines = _head_lines(media, tmp_path)
    assert lines, f"{session}: no head transcript"
    nearest = min(lines, key=lambda line: abs(line.start_s - truth_head))
    error = nearest.start_s - truth_head
    assert abs(error) <= HEAD_LINE_TOLERANCE_S, (
        f"{session}: nearest line starts {error:+.2f}s from the human cut "
        f"({nearest.start_s:.2f}s vs {truth_head:.2f}s): {nearest.text!r}"
    )
