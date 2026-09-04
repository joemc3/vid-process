from __future__ import annotations

from vidproc.asr import Line
from vidproc.confidence import has_closing_language, score
from vidproc.detect import Boundaries
from vidproc.refine import Refinement

CLOSING = [Line("Thank you, guys, for attending our session. Thank you.", 4288.0, 4292.0)]
NOT_CLOSING = [Line("and the p-value was under 0.05 for that cohort.", 4288.0, 4292.0)]


def boundaries(head: float = 289.5, pre: float = 300.0, tail: float = 190.0) -> Boundaries:
    return Boundaries(
        noise_floor_db=8.0, gate_db=16.0, head_candidate_s=head,
        end_s=3184.0, pre_head_silence_s=pre, tail_silence_s=tail,
    )


def refinement(start: float, available: bool = True) -> Refinement:
    return Refinement(start_s=start, reason="r", skipped="", model_used="m", available=available)


def test_pt12_shape_is_confident() -> None:
    """gap 0.5s, clean silence before."""
    c = score(boundaries(head=289.5), refinement(290.0), CLOSING)
    assert c.head_confident and c.end_confident


def test_pt06_shape_is_confident() -> None:
    """gap 7.0s is inside the 10s tolerance."""
    c = score(boundaries(head=604.0), refinement(611.0), CLOSING)
    assert c.head_confident


def test_pt01_shape_flags_head_on_gap() -> None:
    """gap 87s — the bumpy session must not auto-approve."""
    c = score(boundaries(head=871.0), refinement(958.0), CLOSING)
    assert not c.head_confident
    assert any("gap" in r for r in c.head_reasons)


def test_unavailable_refiner_flags_head() -> None:
    c = score(boundaries(head=289.5), refinement(289.5, available=False), CLOSING)
    assert not c.head_confident


def test_short_pre_head_silence_flags_head() -> None:
    c = score(boundaries(head=289.5, pre=2.0), refinement(290.0), CLOSING)
    assert not c.head_confident


def test_pt01_short_tail_still_confident_at_27s() -> None:
    c = score(boundaries(tail=27.0), refinement(290.0), CLOSING)
    assert c.end_confident


def test_tail_below_20s_flags_end() -> None:
    c = score(boundaries(tail=12.0), refinement(290.0), CLOSING)
    assert not c.end_confident


def test_missing_closing_language_flags_end() -> None:
    c = score(boundaries(), refinement(290.0), NOT_CLOSING)
    assert not c.end_confident


def test_closing_language_detection_is_case_insensitive() -> None:
    assert has_closing_language([Line("THANKS, EVERYBODY, FOR JOINING US.", 0.0, 1.0)])
    assert has_closing_language([Line("I think that's the end of our session, right?", 0.0, 1.0)])
    assert not has_closing_language([Line("the odds ratio was 1.4", 0.0, 1.0)])


def test_skipped_content_is_recorded_but_does_not_flag() -> None:
    """PT06 legitimately skips a mic check and must still pass."""
    r = Refinement(start_s=611.0, reason="welcome", skipped="mic check", model_used="m", available=True)
    c = score(boundaries(head=604.0), r, CLOSING)
    assert c.head_confident
    assert any("skipped" in reason for reason in c.head_reasons)
