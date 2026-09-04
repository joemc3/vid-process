from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from vidproc.errors import ConfigError

REFINER_MODES = ("none", "local", "openrouter")

# Hard ceiling: PT01's recording stopped 27s after the session ended. A larger
# requirement finds no qualifying silence and the detector returns garbage.
MAX_TAIL_SILENCE_S = 20.0


@dataclass(frozen=True)
class DetectionConfig:
    floor_percentile: float = 5.0
    gate_offset_db: float = 8.0
    min_speech_s: float = 3.0
    min_tail_silence_s: float = 10.0
    head_window_s: float = 180.0
    tail_window_s: float = 60.0
    head_preroll_s: float = 1.0
    tail_pad_s: float = 2.0

    def validate(self) -> None:
        if not 0.0 < self.floor_percentile < 50.0:
            raise ConfigError(f"floor_percentile must be in (0, 50): {self.floor_percentile}")
        if self.gate_offset_db <= 0.0:
            raise ConfigError(f"gate_offset_db must be positive: {self.gate_offset_db}")
        if self.min_speech_s <= 0.0:
            raise ConfigError(f"min_speech_s must be positive: {self.min_speech_s}")
        if not 0.0 < self.min_tail_silence_s <= MAX_TAIL_SILENCE_S:
            raise ConfigError(
                f"min_tail_silence_s must be in (0, {MAX_TAIL_SILENCE_S}]: "
                f"{self.min_tail_silence_s}. Larger values fail on short tails (spec §6.2)."
            )
        for name in ("head_window_s", "tail_window_s"):
            if getattr(self, name) <= 0.0:
                raise ConfigError(f"{name} must be positive: {getattr(self, name)}")
        for name in ("head_preroll_s", "tail_pad_s"):
            if getattr(self, name) < 0.0:
                raise ConfigError(f"{name} must not be negative: {getattr(self, name)}")


@dataclass(frozen=True)
class RefinerConfig:
    mode: str = "none"
    model: str = ""
    base_url: str = ""
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_s: float = 60.0

    def validate(self) -> None:
        if self.mode not in REFINER_MODES:
            raise ConfigError(f"refiner mode must be one of {REFINER_MODES}: {self.mode!r}")
        if self.mode != "none" and not self.model:
            raise ConfigError(f"refiner mode {self.mode!r} requires a model name")
        if self.timeout_s <= 0.0:
            raise ConfigError(f"timeout_s must be positive: {self.timeout_s}")

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        if self.mode == "local":
            return "http://localhost:11434/v1"
        if self.mode == "openrouter":
            return "https://openrouter.ai/api/v1"
        raise ConfigError(f"no base URL for refiner mode {self.mode!r}")


@dataclass(frozen=True)
class AsrConfig:
    binary: str = "whisper-cli"
    model_path: str = ""
    language: str = "en"

    def validate(self) -> None:
        if not self.binary:
            raise ConfigError("asr.binary must not be empty")


@dataclass(frozen=True)
class Config:
    working_dir: Path
    detection: DetectionConfig = DetectionConfig()
    refiner: RefinerConfig = RefinerConfig()
    asr: AsrConfig = AsrConfig()

    def validate(self) -> None:
        self.detection.validate()
        self.refiner.validate()
        self.asr.validate()


def _merge(base: Any, overrides: dict[str, Any], label: str) -> Any:
    unknown = set(overrides) - {f for f in base.__dataclass_fields__}
    if unknown:
        raise ConfigError(f"unknown {label} keys: {sorted(unknown)}")
    return replace(base, **overrides)


def load_config(path: Path | None, working_dir: Path) -> Config:
    """Load configuration, merging a JSON file over the spec defaults."""
    raw: dict[str, Any] = {}
    if path is not None:
        try:
            raw = json.loads(Path(path).read_text())
        except FileNotFoundError as exc:
            raise ConfigError(f"config file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON in {path}: {exc}") from exc

    cfg = Config(
        working_dir=Path(working_dir),
        detection=_merge(DetectionConfig(), raw.get("detection", {}), "detection"),
        refiner=_merge(RefinerConfig(), raw.get("refiner", {}), "refiner"),
        asr=_merge(AsrConfig(), raw.get("asr", {}), "asr"),
    )
    cfg.validate()
    return cfg
