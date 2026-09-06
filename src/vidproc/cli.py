from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vidproc.analyze import analyze
from vidproc.config import load_config
from vidproc.errors import VidprocError


def _format_clock(seconds: float) -> str:
    return f"{int(seconds // 60)}m{seconds % 60:04.1f}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vidproc", description="Propose session cut points.")
    parser.add_argument("media", type=Path, help="raw session file")
    parser.add_argument("-c", "--config", type=Path, default=None)
    parser.add_argument("-w", "--working-dir", type=Path, default=Path("working"))
    parser.add_argument("--json", action="store_true", help="print the proposal as JSON")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, working_dir=args.working_dir)
        proposal = analyze(args.media, cfg)
    except VidprocError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_dir = cfg.working_dir / proposal.session
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "proposal.json").write_text(proposal.to_json())

    if args.json:
        print(proposal.to_json())
        return 0

    print(f"{proposal.session}  ({_format_clock(proposal.duration_s)} raw)")
    print(f"  start  {proposal.start_s:9.2f}s  {_format_clock(proposal.start_s)}"
          f"   {'ok' if proposal.head_confident else 'REVIEW'}")
    for reason in proposal.head_reasons:
        print(f"           - {reason}")
    if proposal.opens_on:
        print(f'           opens on: "{proposal.opens_on}"')
    print(f"  end    {proposal.end_s:9.2f}s  {_format_clock(proposal.end_s)}"
          f"   {'ok' if proposal.end_confident else 'REVIEW'}")
    for reason in proposal.end_reasons:
        print(f"           - {reason}")
    print(f"  output duration {_format_clock(proposal.output_duration_s)}")
    print(f"  ffmpeg: -ss {proposal.start_s:.2f} -to {proposal.end_s:.2f}")
    print(f"  written to {out_dir / 'proposal.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
