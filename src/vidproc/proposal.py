from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Proposal:
    session: str
    source: str
    duration_s: float
    fps: str
    start_s: float
    end_s: float
    noise_floor_db: float
    gate_db: float
    head_candidate_s: float
    head_reason: str
    head_skipped: str
    model_used: str
    head_confident: bool
    end_confident: bool
    head_reasons: tuple[str, ...]
    end_reasons: tuple[str, ...]
    head_lines: tuple[dict[str, Any], ...]
    tail_lines: tuple[dict[str, Any], ...]

    @property
    def output_duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def needs_review(self) -> bool:
        return not (self.head_confident and self.end_confident)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=list)
