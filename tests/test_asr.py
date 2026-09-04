from __future__ import annotations

import pytest

from vidproc.asr import Line, Word, parse_whisper_json, words_to_lines


def _seg(text: str, start_ms: int, end_ms: int) -> dict:
    return {"offsets": {"from": start_ms, "to": end_ms}, "text": text}


def test_parse_applies_absolute_offset() -> None:
    payload = {"transcription": [_seg(" Good", 500, 900), _seg(" morning", 900, 1400)]}
    words = parse_whisper_json(payload, offset_s=290.0)
    assert [w.text for w in words] == ["Good", "morning"]
    assert words[0].start_s == pytest.approx(290.5)
    assert words[1].end_s == pytest.approx(291.4)


def test_parse_skips_empty_segments() -> None:
    """whisper-cli emits blank leading segments with -ml 1."""
    payload = {"transcription": [_seg("", 0, 320), _seg(" Hello", 320, 800)]}
    assert [w.text for w in parse_whisper_json(payload, 0.0)] == ["Hello"]


def test_parse_empty_transcription_is_empty_list() -> None:
    assert parse_whisper_json({"transcription": []}, 0.0) == []


def test_malformed_segment_surfaces_as_external_tool_error() -> None:
    """A segment with text but no offsets is unusable output from an external
    tool, and must not escape as a raw KeyError."""
    from vidproc.errors import ExternalToolError

    with pytest.raises(ExternalToolError, match="unusable whisper segment"):
        parse_whisper_json({"transcription": [{"text": " Hello"}]}, 0.0)


def test_words_to_lines_splits_on_long_pause() -> None:
    words = [
        Word("Hello", 0.0, 0.4),
        Word("everybody", 0.4, 1.0),
        Word("Welcome", 5.0, 5.5),   # 4s gap
        Word("everyone", 5.5, 6.0),
    ]
    lines = words_to_lines(words, max_gap_s=0.8)
    assert len(lines) == 2
    assert lines[0].text == "Hello everybody"
    assert lines[1].start_s == pytest.approx(5.0)


def test_words_to_lines_caps_line_length() -> None:
    words = [Word(f"w{i}", i * 0.2, i * 0.2 + 0.1) for i in range(30)]
    lines = words_to_lines(words, max_gap_s=5.0, max_words=14)
    assert all(len(line.text.split()) <= 14 for line in lines)
    assert len(lines) == 3


def test_words_to_lines_on_empty_input() -> None:
    assert words_to_lines([], max_gap_s=0.8) == []


def test_line_is_hashable_and_frozen() -> None:
    line = Line(text="x", start_s=1.0, end_s=2.0)
    with pytest.raises(Exception):
        line.text = "y"  # type: ignore[misc]
