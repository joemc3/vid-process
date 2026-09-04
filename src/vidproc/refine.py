from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

import httpx

from vidproc.asr import Line
from vidproc.config import RefinerConfig

SYSTEM_PROMPT = """\
You are trimming the opening of a recorded conference session so it can be published.

You are given numbered transcript lines from the very start of a recording, each with a
timestamp in seconds. Choose the line where the published video should begin.

Start at the first line that belongs in the published video. Exclude anything before it:
- microphone checks and level tests ("Hello?", "can you hear me", "I'm pretty loud")
- apologies for technical difficulties or delays
- pre-session logistics addressed to staff rather than the audience

Keep the session's real opening. That is usually a moderator welcoming the audience or
introducing the session, but if the moderator's remarks are entirely mic checks, apologies,
or housekeeping, start at the first presenter instead.

Respond with JSON only: {"start_line": <integer index>, "reason": "<why that line>",
"skipped": "<short description of what you excluded, or empty string>"}
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "start_line": {"type": "integer"},
        "reason": {"type": "string"},
        "skipped": {"type": "string"},
    },
    "required": ["start_line", "reason", "skipped"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Refinement:
    start_s: float
    reason: str
    skipped: str
    model_used: str
    available: bool


class Refiner(Protocol):
    def refine(self, lines: list[Line], candidate_s: float) -> Refinement:
        ...


def format_lines(lines: list[Line]) -> str:
    return "\n".join(f"{i}\t{line.start_s:.2f}\t{line.text}" for i, line in enumerate(lines))


def apply_choice(
    lines: list[Line], index: int, reason: str, skipped: str, model: str
) -> Refinement:
    if not 0 <= index < len(lines):
        raise ValueError(f"start_line {index} out of range (0..{len(lines) - 1})")
    return Refinement(
        start_s=lines[index].start_s,
        reason=reason,
        skipped=skipped,
        model_used=model,
        available=True,
    )


def _degraded(candidate_s: float, why: str) -> Refinement:
    return Refinement(
        start_s=candidate_s, reason=why, skipped="", model_used="", available=False
    )


class NullRefiner:
    """Mode 'none'. Proposes the energy candidate; head is always reviewed."""

    def refine(self, lines: list[Line], candidate_s: float) -> Refinement:
        return _degraded(candidate_s, "no refinement model configured")


class OpenAICompatibleRefiner:
    """Ollama (local) and OpenRouter (remote) share this wire format."""

    def __init__(self, cfg: RefinerConfig, client: httpx.Client | None = None) -> None:
        self._cfg = cfg
        self._client = client
        self._url = f"{cfg.resolved_base_url()}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self._cfg.api_key_env, "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def refine(self, lines: list[Line], candidate_s: float) -> Refinement:
        if not lines:
            return _degraded(candidate_s, "no transcript available to refine against")

        body = {
            "model": self._cfg.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_lines(lines)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "session_start",
                    "strict": True,
                    "schema": RESPONSE_SCHEMA,
                },
            },
        }

        client = self._client or httpx.Client(timeout=self._cfg.timeout_s)
        try:
            response = client.post(self._url, json=body, headers=self._headers())
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return apply_choice(
                lines,
                int(parsed["start_line"]),
                str(parsed.get("reason", "")),
                str(parsed.get("skipped", "")),
                self._cfg.model,
            )
        except httpx.HTTPError as exc:
            return _degraded(candidate_s, f"refiner unreachable: {type(exc).__name__}")
        except (KeyError, ValueError, TypeError) as exc:
            return _degraded(candidate_s, f"refiner returned unusable output: {exc}")
        finally:
            if self._client is None:
                client.close()


def build_refiner(cfg: RefinerConfig) -> Refiner:
    if cfg.mode == "none":
        return NullRefiner()
    return OpenAICompatibleRefiner(cfg)
