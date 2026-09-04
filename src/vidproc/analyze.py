from __future__ import annotations

from pathlib import Path

from vidproc.asr import ASR, Line, WhisperCppASR, words_to_lines
from vidproc.audio import envelope_for
from vidproc.config import Config
from vidproc.confidence import score
from vidproc.detect import detect_boundaries
from vidproc.probe import probe
from vidproc.proposal import Proposal
from vidproc.refine import Refiner, build_refiner


def _as_dicts(lines: list[Line]) -> tuple[dict[str, object], ...]:
    return tuple(
        {"start_s": round(line.start_s, 2), "end_s": round(line.end_s, 2), "text": line.text}
        for line in lines
    )


def analyze(
    media: Path,
    cfg: Config,
    asr: ASR | None = None,
    refiner: Refiner | None = None,
) -> Proposal:
    """Produce a reviewable proposal for one raw session file."""
    media = Path(media)
    asr = asr if asr is not None else WhisperCppASR(cfg.asr)
    refiner = refiner if refiner is not None else build_refiner(cfg.refiner)

    info = probe(media)
    env = envelope_for(media, cfg.working_dir / "cache")
    bounds = detect_boundaries(env, cfg.detection)

    d = cfg.detection

    # Only ever transcribe bounded windows that energy says contain speech.
    head_words = asr.transcribe(
        media, bounds.head_candidate_s, bounds.head_candidate_s + d.head_window_s
    )
    head_lines = words_to_lines(head_words)

    tail_start = max(0.0, bounds.end_s - d.tail_window_s)
    tail_words = asr.transcribe(media, tail_start, bounds.end_s)
    tail_lines = words_to_lines(tail_words)

    refinement = refiner.refine(head_lines, bounds.head_candidate_s)
    confidence = score(bounds, refinement, tail_lines)

    start_s = max(0.0, refinement.start_s - d.head_preroll_s)
    end_s = min(info.duration_s, bounds.end_s + d.tail_pad_s)

    return Proposal(
        session=media.stem,
        source=str(media),
        duration_s=info.duration_s,
        fps=str(info.fps),
        start_s=start_s,
        end_s=end_s,
        noise_floor_db=round(bounds.noise_floor_db, 2),
        gate_db=round(bounds.gate_db, 2),
        head_candidate_s=bounds.head_candidate_s,
        head_reason=refinement.reason,
        head_skipped=refinement.skipped,
        model_used=refinement.model_used,
        head_confident=confidence.head_confident,
        end_confident=confidence.end_confident,
        head_reasons=confidence.head_reasons,
        end_reasons=confidence.end_reasons,
        head_lines=_as_dicts(head_lines),
        tail_lines=_as_dicts(tail_lines),
    )
