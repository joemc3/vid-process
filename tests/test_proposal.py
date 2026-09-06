from __future__ import annotations

from vidproc.proposal import Proposal


def _proposal(start_s: float, head_lines: tuple[dict[str, object], ...]) -> Proposal:
    return Proposal(
        session="AAO2025PT06", source="x.mp4", duration_s=4600.0, fps="16/1",
        start_s=start_s, end_s=4323.75, noise_floor_db=7.5, gate_db=15.5,
        head_candidate_s=604.0, head_reason="", head_skipped="", model_used="",
        head_confident=True, end_confident=True, head_reasons=(), end_reasons=(),
        head_lines=head_lines, tail_lines=(),
    )


# Real PT06 head window: the refiner skipped the mic check and started at 611.0,
# which head_preroll_s pulls back to 610.0.
PT06_LINES = (
    {"start_s": 604.0, "end_s": 605.0, "text": "Hello everybody."},
    {"start_s": 605.0, "end_s": 608.0, "text": "Oh yeah, I'm pretty loud."},
    {"start_s": 608.0, "end_s": 611.0, "text": "Okay."},
    {"start_s": 611.0, "end_s": 619.0, "text": "Welcome to this session on podium posters."},
)


def test_opens_on_skips_lines_that_finish_before_the_cut() -> None:
    """The operator's one documented check must show what is actually
    published, not the first line of the transcript window."""
    assert _proposal(610.0, PT06_LINES).opens_on == "Okay."


def test_opens_on_is_the_first_line_when_nothing_is_skipped() -> None:
    assert _proposal(603.0, PT06_LINES).opens_on == "Hello everybody."


def test_opens_on_is_empty_without_a_transcript() -> None:
    assert _proposal(610.0, ()).opens_on == ""


def test_opens_on_falls_back_to_the_last_line_when_the_cut_is_past_them_all() -> None:
    assert _proposal(9999.0, PT06_LINES).opens_on == (
        "Welcome to this session on podium posters."
    )
