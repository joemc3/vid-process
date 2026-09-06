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
    def opens_on(self) -> str:
        """The first transcript line the trimmed output actually contains.

        Not head_lines[0]: the refiner skips mic checks and false starts, so
        the first line of the window is often material the cut excludes.
        Showing it there would advertise content that was deliberately
        dropped, in the one line the operator is told to check.
        """
        for line in self.head_lines:
            if float(line["end_s"]) > self.start_s:
                return str(line["text"])
        return str(self.head_lines[-1]["text"]) if self.head_lines else ""

    @property
    def needs_review(self) -> bool:
        return not (self.head_confident and self.end_confident)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=list)
