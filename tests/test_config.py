from __future__ import annotations

import json
from pathlib import Path

import pytest

from vidproc.config import DetectionConfig, load_config
from vidproc.errors import ConfigError


def test_defaults_match_spec(tmp_path: Path) -> None:
    cfg = load_config(None, working_dir=tmp_path)
    d = cfg.detection
    assert d.floor_percentile == 5.0
    assert d.gate_offset_db == 8.0
    assert d.min_speech_s == 3.0
    assert d.min_tail_silence_s == 10.0
    assert d.head_window_s == 180.0
    assert d.tail_window_s == 60.0
    assert d.head_preroll_s == 1.0
    assert d.tail_pad_s == 2.0
    assert cfg.refiner.mode == "none"


def test_tail_silence_above_20_is_rejected() -> None:
    """PT01's tail is only 27s; >20s finds no qualifying silence. Spec §6.2."""
    with pytest.raises(ConfigError, match="min_tail_silence_s"):
        DetectionConfig(min_tail_silence_s=30.0).validate()


def test_tail_silence_at_20_is_allowed() -> None:
    DetectionConfig(min_tail_silence_s=20.0).validate()


def test_unknown_refiner_mode_is_rejected() -> None:
    from vidproc.config import RefinerConfig

    with pytest.raises(ConfigError, match="mode"):
        RefinerConfig(mode="anthropic").validate()


def test_non_none_mode_requires_an_explicit_model() -> None:
    """No safe default exists: some Ollama models carry a :cloud suffix and run
    off-machine, so a guessed default could silently defeat 'local'."""
    from vidproc.config import RefinerConfig

    with pytest.raises(ConfigError, match="requires a model name"):
        RefinerConfig(mode="local").validate()


def test_file_overrides_merge_over_defaults(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps(
            {
                "detection": {"gate_offset_db": 12.0},
                "refiner": {"mode": "local", "model": "qwen3.6:35b-a3b"},
            }
        )
    )
    cfg = load_config(p, working_dir=tmp_path)
    assert cfg.detection.gate_offset_db == 12.0
    assert cfg.detection.min_speech_s == 3.0  # untouched default
    assert cfg.refiner.mode == "local"
    assert cfg.refiner.model == "qwen3.6:35b-a3b"


def test_null_section_raises_config_error(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text('{"detection": null}')
    with pytest.raises(ConfigError, match="must be an object"):
        load_config(p, working_dir=tmp_path)


def test_wrongly_typed_value_raises_config_error(tmp_path: Path) -> None:
    """A hand-edited typo must produce a message, not a traceback."""
    p = tmp_path / "c.json"
    p.write_text('{"detection": {"gate_offset_db": "loud"}}')
    with pytest.raises(ConfigError, match="invalid value in configuration"):
        load_config(p, working_dir=tmp_path)


def test_local_mode_rejects_a_remote_base_url() -> None:
    """'local' must mean local: a remote base_url under mode 'local' would send
    transcript text off-machine while the operator believes it did not."""
    from vidproc.config import RefinerConfig

    with pytest.raises(ConfigError, match="must not point off-machine"):
        RefinerConfig(mode="local", model="m", base_url="https://example.com/v1").validate()


def test_local_mode_accepts_a_loopback_base_url() -> None:
    from vidproc.config import RefinerConfig

    RefinerConfig(mode="local", model="m", base_url="http://127.0.0.1:11434/v1").validate()
