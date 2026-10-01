#!/usr/bin/env python3
"""
Video File Monitor and Copy Script

Monitors a network share for new video files being written by ffmpeg,
waits for each to finish, then copies it to the backup location and from there
to the processing location. No single file can hold up the others.
"""

import os
import sys
import time
import shutil
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Set, Optional, Dict, Any, Callable
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ================================
# DEFAULT CONFIGURATION
# ================================

DEFAULT_CONFIG = {
    # Source location where ffmpeg writes recording files
    "imageSource": r"\\RECORDING-SERVER\SharedFolder",

    # Location where files are copied for processing
    "processingLocation": r"C:\path\to\processing",

    # Location where files are backed up
    "backupLocation": r"C:\path\to\backup",

    # File pattern to monitor (e.g., "*.mp4", "*.mkv", "recording_*.mp4")
    "file_pattern": "*.mp4",

    # How long file size must remain stable before considering complete (seconds)
    "stability_period_sec": 15,

    # How often to check file size during stability monitoring (seconds)
    "check_interval_sec": 5,

    # How often to scan for new files in the source directory (seconds)
    "scan_interval_sec": 30,

    # Minimum file size in MB (files smaller than this will not be processed)
    "min_file_size": 10,
}

# ================================
# END DEFAULT CONFIGURATION
# ================================


def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from a JSON file. Falls back to defaults if file doesn't exist.

    Args:
        config_file: Path to the configuration JSON file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_file)

    # If config file doesn't exist, create it with defaults
    if not config_path.exists():
        logger.warning(f"Configuration file not found: {config_file}")
        logger.info(f"Creating default configuration file: {config_file}")
        try:
            with open(config_path, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)
            logger.info(f"Please edit {config_file} with your paths and settings, then restart the script.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to create config file: {e}")
            logger.info("Using default configuration values")
            return DEFAULT_CONFIG.copy()

    # Load configuration from file
    try:
        with open(config_path, 'r') as f:
            user_config = json.load(f)

        # Merge with defaults (in case new fields are added in updates)
        config = DEFAULT_CONFIG.copy()
        config.update(user_config)

        logger.info(f"Loaded configuration from: {config_file}")
        return config

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        logger.info("Using default configuration values")
        return DEFAULT_CONFIG.copy()
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        logger.info("Using default configuration values")
        return DEFAULT_CONFIG.copy()


def get_files_in_directory(directory: str, pattern: str = "*") -> Set[str]:
    """
    Get a set of filenames matching the pattern in the specified directory.

    Args:
        directory: Path to the directory to scan
        pattern: Glob pattern for files to match (e.g., "*.mp4")

    Returns:
        Set of filenames (not full paths) found in the directory
    """
    try:
        path = Path(directory)
        if not path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return set()

        files = {f.name for f in path.glob(pattern) if f.is_file()}
        return files
    except Exception as e:
        logger.error(f"Error scanning directory {directory}: {e}")
        return set()


WATCHING = "watching"
DONE = "done"
FAILED = "failed"


class DestinationConflict(Exception):
    """A file of the same name but a different size is already at the destination."""


@dataclass
class _Watch:
    size: int
    changed_at: float
    status: str


def copy_verified(source_file: Path, dest_dir: Path, expected_size: int) -> Path:
    """
    Copy a file into dest_dir so that a partial copy can never pass for a finished one.

    The copy is written under a temporary name, its size checked against
    expected_size, and only then renamed into place. A destination that already
    holds the file at the expected size counts as done; one that holds it at a
    different size is never overwritten.

    Args:
        source_file: Full path to the source file
        dest_dir: Destination directory
        expected_size: Size in bytes the finished copy must have

    Returns:
        Path of the finished copy

    Raises:
        DestinationConflict: The destination holds a different file of the same name
        OSError: The copy failed or came out the wrong size; nothing is left behind
    """
    dest = dest_dir / source_file.name
    if dest.exists():
        existing = dest.stat().st_size
        if existing == expected_size:
            return dest
        raise DestinationConflict(
            f"{dest} already exists at {existing:,} bytes, expected {expected_size:,}"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    part = dest_dir / (source_file.name + ".part")
    logger.info(f"Copying {source_file} -> {dest}")
    try:
        shutil.copy2(str(source_file), str(part))
        copied = part.stat().st_size
        if copied != expected_size:
            raise OSError(f"copy is {copied:,} bytes, expected {expected_size:,}")
        os.replace(str(part), str(dest))
    except BaseException:
        # Includes Ctrl-C: a half-written .part must never be left to be mistaken
        # for anything.
        if part.exists():
            part.unlink()
        raise
    return dest


class Monitor:
    """
    Watches the share and copies each finished recording to backup, then processing.

    Every call to tick() looks at every file once and returns; nothing waits on
    a single file. A file is finished once its size has not changed for
    stability_period_sec. A finished file below min_file_size is marked failed
    rather than waited on forever, and is picked back up if it starts growing.

    The share is read once per file: the processing copy is taken from the
    verified backup copy.
    """

    def __init__(self, config: Dict[str, Any], clock: Callable[[], float] = time.monotonic):
        self._source = Path(config["imageSource"])
        self._backup = Path(config["backupLocation"])
        self._processing = Path(config["processingLocation"])
        self._pattern = config["file_pattern"]
        self._stability_s = float(config["stability_period_sec"])
        self._min_bytes = float(config["min_file_size"]) * 1024 * 1024
        self._clock = clock
        self._files: Dict[str, _Watch] = {}

    def status(self, file_name: str) -> Optional[str]:
        watch = self._files.get(file_name)
        return watch.status if watch else None

    @property
    def pending(self) -> bool:
        return any(w.status == WATCHING for w in self._files.values())

    def tick(self) -> None:
        names = get_files_in_directory(str(self._source), self._pattern)
        for gone in sorted(set(self._files) - names):
            logger.warning(f"No longer on the share, stopped watching: {gone}")
            del self._files[gone]
        for name in sorted(names):
            try:
                self._observe(name)
            except Exception as e:
                # One bad file must never take the others down with it.
                logger.error(f"Unexpected error handling {name}: {e}")

    def _already_copied(self, name: str, size: int) -> bool:
        for dest_dir in (self._backup, self._processing):
            dest = dest_dir / name
            if not dest.exists() or dest.stat().st_size != size:
                return False
        return True

    def _observe(self, name: str) -> None:
        try:
            size = (self._source / name).stat().st_size
        except OSError as e:
            logger.warning(f"Cannot read size of {name}, will retry: {e}")
            return
        now = self._clock()
        watch = self._files.get(name)

        if watch is None:
            if self._already_copied(name, size):
                logger.info(f"Already copied: {name}")
                self._files[name] = _Watch(size, now, DONE)
            else:
                logger.info(f"New file: {name} ({size / (1024 * 1024):.2f} MB)")
                self._files[name] = _Watch(size, now, WATCHING)
            return

        if size != watch.size:
            if watch.status != WATCHING:
                logger.info(f"{name} changed size after being marked {watch.status}; watching again")
            else:
                logger.info(f"{name} size changed: {watch.size:,} -> {size:,} bytes")
            watch.size, watch.changed_at, watch.status = size, now, WATCHING
            return

        if watch.status != WATCHING or now - watch.changed_at < self._stability_s:
            return

        if size < self._min_bytes:
            logger.error(
                f"FAILED: {name} stopped at {size / (1024 * 1024):.2f} MB, below the "
                f"{self._min_bytes / (1024 * 1024):g} MB minimum. Not copied. "
                f"It will be picked back up if it starts growing again."
            )
            watch.status = FAILED
            return

        watch.status = self._copy(name, size)

    def _copy(self, name: str, size: int) -> str:
        try:
            backup_copy = copy_verified(self._source / name, self._backup, size)
            copy_verified(backup_copy, self._processing, size)
        except DestinationConflict as e:
            logger.error(f"FAILED: {e}. Not overwriting; this needs a person to decide.")
            return FAILED
        except Exception as e:
            logger.error(f"Copy of {name} failed, will retry: {e}")
            return WATCHING
        logger.info(f"Done: {name} ({size:,} bytes) in backup and processing")
        return DONE


def main():
    """
    Main monitoring loop.
    """
    # Load configuration
    config = load_config("config.json")

    logger.info("Video File Monitor Started")
    logger.info(f"Source: {config['imageSource']}")
    logger.info(f"Processing: {config['processingLocation']}")
    logger.info(f"Backup: {config['backupLocation']}")
    logger.info(f"Pattern: {config['file_pattern']}")
    logger.info(f"Scan interval: {config['scan_interval_sec']} seconds")
    logger.info("-" * 60)

    # Ensure destination directories exist
    for dir_key in ["processingLocation", "backupLocation"]:
        Path(config[dir_key]).mkdir(parents=True, exist_ok=True)

    monitor = Monitor(config)
    while True:
        try:
            monitor.tick()
            # Poll faster while something is on its way to being finished.
            if monitor.pending:
                time.sleep(config["check_interval_sec"])
            else:
                time.sleep(config["scan_interval_sec"])

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            time.sleep(config["scan_interval_sec"])


if __name__ == "__main__":
    main()
