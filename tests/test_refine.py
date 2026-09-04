from __future__ import annotations

import httpx
import pytest

from vidproc.asr import Line
from vidproc.config import RefinerConfig
from vidproc.refine import (
    NullRefiner,
    OpenAICompatibleRefiner,
    apply_choice,
    build_refiner,
    format_lines,
)

PT01 = [
    Line("Sorry, we're having a little technical difficulty.", 900.0, 910.0),
    Line("I'll just remind the audience that this session is being recorded.", 910.0, 916.0),
    Line("Hello?", 953.0, 955.0),
    Line("Good morning, everyone. My name is Alex Rivera.", 958.0, 962.0),
]


def test_null_refiner_returns_candidate_and_reports_unavailable() -> None:
    r = NullRefiner().refine(PT01, candidate_s=900.0)
    assert r.start_s == 900.0
    assert r.available is False


def test_format_lines_is_indexed_and_timestamped() -> None:
    text = format_lines(PT01[:2])
    assert text.splitlines()[0].startswith("0\t900.00\t")
    assert "technical difficulty" in text


def test_apply_choice_maps_index_to_line_start() -> None:
    r = apply_choice(PT01, 3, "first presenter begins", "apology, housekeeping", "m")
    assert r.start_s == 958.0
    assert r.available is True


def test_apply_choice_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="out of range"):
        apply_choice(PT01, 99, "x", "y", "m")


def test_build_refiner_none_mode_gives_null() -> None:
    assert isinstance(build_refiner(RefinerConfig(mode="none")), NullRefiner)


def _refiner(handler) -> OpenAICompatibleRefiner:
    cfg = RefinerConfig(mode="local", model="test-model", base_url="http://x/v1")
    transport = httpx.MockTransport(handler)
    return OpenAICompatibleRefiner(cfg, client=httpx.Client(transport=transport))


def test_remote_refiner_picks_the_line_the_model_chose() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = '{"start_line": 3, "reason": "first presenter", "skipped": "apology"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    r = _refiner(handler).refine(PT01, candidate_s=900.0)
    assert r.start_s == 958.0
    assert r.available is True
    assert "presenter" in r.reason


def test_connection_error_degrades_to_candidate() -> None:
    """Spec: never fail a session because a model was unavailable."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    r = _refiner(handler).refine(PT01, candidate_s=900.0)
    assert r.start_s == 900.0
    assert r.available is False


def test_http_error_degrades_to_candidate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    assert _refiner(handler).refine(PT01, candidate_s=900.0).available is False


def test_malformed_json_degrades_to_candidate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    assert _refiner(handler).refine(PT01, candidate_s=900.0).available is False


def test_out_of_range_index_from_model_degrades_to_candidate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = '{"start_line": 42, "reason": "r", "skipped": "s"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    assert _refiner(handler).refine(PT01, candidate_s=900.0).available is False


def test_empty_lines_returns_candidate_without_calling_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("model must not be called with no transcript")

    r = _refiner(handler).refine([], candidate_s=900.0)
    assert r.start_s == 900.0
    assert r.available is False
