"""Ingest failure modes, spec §5 and §11.

The monitor runs unattended at the venue. Each test here is a way it used to
lose recordings silently: a stalled file hung the whole loop, one file waiting
blocked every other room, and an interrupted copy left a truncated file that
was then treated as done.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import video_monitor
from video_monitor import Monitor

STABILITY_S = 180.0
MIN_SIZE_MB = 0.001  # 1048 bytes, so test files stay tiny
BIG = 4096  # bytes, above the minimum
SMALL = 100  # bytes, below it


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def dirs(tmp_path: Path) -> dict[str, Path]:
    d = {name: tmp_path / name for name in ("share", "backup", "processing")}
    d["share"].mkdir()
    return d


@pytest.fixture
def clock() -> Clock:
    return Clock()


def make_monitor(dirs: dict[str, Path], clock: Clock) -> Monitor:
    config = {
        "imageSource": str(dirs["share"]),
        "backupLocation": str(dirs["backup"]),
        "processingLocation": str(dirs["processing"]),
        "file_pattern": "*.mp4",
        "stability_period_sec": STABILITY_S,
        "check_interval_sec": 5,
        "scan_interval_sec": 30,
        "min_file_size": MIN_SIZE_MB,
    }
    return Monitor(config, clock=clock)


def write(path: Path, size: int) -> None:
    path.write_bytes(b"x" * size)


def settle(monitor: Monitor, clock: Clock) -> None:
    """Observe, wait out the stability period, observe again."""
    monitor.tick()
    clock.advance(STABILITY_S)
    monitor.tick()


def test_stable_file_is_copied_to_backup_and_processing(dirs, clock) -> None:
    write(dirs["share"] / "PT01.mp4", BIG)
    settle(make_monitor(dirs, clock), clock)
    assert (dirs["backup"] / "PT01.mp4").stat().st_size == BIG
    assert (dirs["processing"] / "PT01.mp4").stat().st_size == BIG


def test_file_is_not_copied_before_it_has_been_stable_long_enough(dirs, clock) -> None:
    write(dirs["share"] / "PT01.mp4", BIG)
    monitor = make_monitor(dirs, clock)
    monitor.tick()
    clock.advance(STABILITY_S - 1)
    monitor.tick()
    assert not (dirs["backup"] / "PT01.mp4").exists()


def test_growing_file_restarts_the_stability_clock(dirs, clock) -> None:
    src = dirs["share"] / "PT01.mp4"
    write(src, BIG)
    monitor = make_monitor(dirs, clock)
    monitor.tick()
    clock.advance(STABILITY_S - 1)
    write(src, BIG * 2)  # still recording
    monitor.tick()
    clock.advance(STABILITY_S - 1)
    monitor.tick()
    assert not (dirs["backup"] / "PT01.mp4").exists()
    clock.advance(1)
    monitor.tick()
    assert (dirs["backup"] / "PT01.mp4").stat().st_size == BIG * 2


def test_file_that_stalls_below_minimum_size_is_failed_not_waited_on(dirs, clock) -> None:
    """Spec §5 defect 1: this used to loop forever and stop the monitor."""
    write(dirs["share"] / "PT01.mp4", SMALL)
    monitor = make_monitor(dirs, clock)
    settle(monitor, clock)  # returns at all: no hang
    assert monitor.status("PT01.mp4") == "failed"
    assert not (dirs["backup"] / "PT01.mp4").exists()


def test_stalled_file_does_not_block_another_room(dirs, clock) -> None:
    """Spec §5 defect 2: one file waiting must not stop others being copied."""
    write(dirs["share"] / "PT01.mp4", SMALL)
    write(dirs["share"] / "PT02.mp4", BIG)
    monitor = make_monitor(dirs, clock)
    settle(monitor, clock)
    assert monitor.status("PT01.mp4") == "failed"
    assert (dirs["backup"] / "PT02.mp4").stat().st_size == BIG


def test_recording_still_in_progress_does_not_block_a_finished_one(dirs, clock) -> None:
    live = dirs["share"] / "PT01.mp4"
    write(live, BIG)
    write(dirs["share"] / "PT02.mp4", BIG)
    monitor = make_monitor(dirs, clock)
    monitor.tick()
    clock.advance(STABILITY_S)
    write(live, BIG * 2)
    monitor.tick()
    assert (dirs["backup"] / "PT02.mp4").exists()
    assert not (dirs["backup"] / "PT01.mp4").exists()


def test_failed_file_that_starts_growing_again_is_picked_back_up(dirs, clock) -> None:
    src = dirs["share"] / "PT01.mp4"
    write(src, SMALL)
    monitor = make_monitor(dirs, clock)
    settle(monitor, clock)
    assert monitor.status("PT01.mp4") == "failed"
    write(src, BIG)
    settle(monitor, clock)
    assert monitor.status("PT01.mp4") == "done"
    assert (dirs["backup"] / "PT01.mp4").stat().st_size == BIG


def test_interrupted_copy_does_not_mark_the_session_complete(dirs, clock, monkeypatch) -> None:
    """Spec §5 defect 3: a truncated file under the real name used to count as
    done forever, because completion was inferred from the filename alone."""
    write(dirs["share"] / "PT01.mp4", BIG)
    real_copy2 = video_monitor.shutil.copy2

    def dies_halfway(src, dst):
        Path(dst).write_bytes(b"x" * (BIG // 2))
        raise OSError("share went away")

    monkeypatch.setattr(video_monitor.shutil, "copy2", dies_halfway)
    monitor = make_monitor(dirs, clock)
    settle(monitor, clock)
    assert monitor.status("PT01.mp4") != "done"
    assert not (dirs["backup"] / "PT01.mp4").exists()
    assert list(dirs["backup"].iterdir()) == []

    # The next pass retries and succeeds.
    monkeypatch.setattr(video_monitor.shutil, "copy2", real_copy2)
    monitor.tick()
    assert monitor.status("PT01.mp4") == "done"
    assert (dirs["backup"] / "PT01.mp4").stat().st_size == BIG


def test_copy_in_progress_is_never_visible_under_the_final_name(
    dirs, clock, monkeypatch
) -> None:
    """Cleanup on error is not enough: a power cut or kill -9 mid-copy runs no
    cleanup. Only a temporary name keeps a partial file from passing for a
    finished one."""
    write(dirs["share"] / "PT01.mp4", BIG)
    visible_mid_copy: list[bool] = []
    real_copy2 = video_monitor.shutil.copy2

    def observing(src, dst):
        Path(dst).write_bytes(b"x")
        visible_mid_copy.append((Path(dst).parent / "PT01.mp4").exists())
        return real_copy2(src, dst)

    monkeypatch.setattr(video_monitor.shutil, "copy2", observing)
    settle(make_monitor(dirs, clock), clock)
    assert visible_mid_copy == [False, False]


def test_short_copy_without_an_error_is_rejected(dirs, clock, monkeypatch) -> None:
    write(dirs["share"] / "PT01.mp4", BIG)

    def silently_short(src, dst):
        Path(dst).write_bytes(b"x" * (BIG - 1))

    monkeypatch.setattr(video_monitor.shutil, "copy2", silently_short)
    monitor = make_monitor(dirs, clock)
    settle(monitor, clock)
    assert monitor.status("PT01.mp4") != "done"
    assert list(dirs["backup"].iterdir()) == []


def test_processing_copy_is_taken_from_the_backup_not_the_share(dirs, clock, monkeypatch) -> None:
    """Spec §5 flow: read the share once. The second copy comes from the
    backup drive, which by then holds a verified copy."""
    write(dirs["share"] / "PT01.mp4", BIG)
    sources: list[Path] = []
    real_copy2 = video_monitor.shutil.copy2

    def recording(src, dst):
        sources.append(Path(src).parent)
        return real_copy2(src, dst)

    monkeypatch.setattr(video_monitor.shutil, "copy2", recording)
    settle(make_monitor(dirs, clock), clock)
    assert sources == [dirs["share"], dirs["backup"]]


def test_existing_destination_of_a_different_size_is_never_overwritten(dirs, clock) -> None:
    """A recording restarted under the same name must not silently replace a
    good backup. A human has to decide which one is right."""
    write(dirs["share"] / "PT01.mp4", BIG)
    dirs["backup"].mkdir()
    write(dirs["backup"] / "PT01.mp4", BIG * 3)
    monitor = make_monitor(dirs, clock)
    settle(monitor, clock)
    assert monitor.status("PT01.mp4") == "failed"
    assert (dirs["backup"] / "PT01.mp4").stat().st_size == BIG * 3


def test_already_copied_file_is_recognised_on_restart_without_recopying(
    dirs, clock, monkeypatch
) -> None:
    for name in ("share", "backup", "processing"):
        dirs[name].mkdir(exist_ok=True)
        write(dirs[name] / "PT01.mp4", BIG)

    def must_not_copy(src, dst):
        raise AssertionError("re-copied a file that was already complete")

    monkeypatch.setattr(video_monitor.shutil, "copy2", must_not_copy)
    monitor = make_monitor(dirs, clock)
    monitor.tick()
    assert monitor.status("PT01.mp4") == "done"


def test_file_that_disappears_from_the_share_is_dropped(dirs, clock) -> None:
    src = dirs["share"] / "PT01.mp4"
    write(src, BIG)
    monitor = make_monitor(dirs, clock)
    monitor.tick()
    src.unlink()
    clock.advance(STABILITY_S)
    monitor.tick()
    assert monitor.status("PT01.mp4") is None


def test_pending_reflects_files_still_being_watched(dirs, clock) -> None:
    write(dirs["share"] / "PT01.mp4", BIG)
    monitor = make_monitor(dirs, clock)
    monitor.tick()
    assert monitor.pending
    clock.advance(STABILITY_S)
    monitor.tick()
    assert not monitor.pending
