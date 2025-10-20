#!/usr/bin/env python3
"""
Video File Monitor and Copy Script

Monitors a network share for new video files being written by ffmpeg,
waits for them to complete, then copies them to processing and backup locations.
"""

import os
import sys
import time
import shutil
import json
from pathlib import Path
from typing import Set, Optional, Dict, Any
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


def find_new_files(source_dir: str, processing_dir: str, pattern: str) -> Set[str]:
    """
    Compare files in source directory to processing directory to find new files.

    Args:
        source_dir: Source directory to check for new files
        processing_dir: Processing directory to compare against
        pattern: File pattern to match

    Returns:
        Set of filenames that exist in source but not in processing
    """
    source_files = get_files_in_directory(source_dir, pattern)
    processing_files = get_files_in_directory(processing_dir, pattern)

    new_files = source_files - processing_files

    if new_files:
        logger.info(f"Found {len(new_files)} new file(s): {', '.join(new_files)}")

    return new_files


def wait_for_stable_file_size(
    file_path: str,
    stability_period_sec: int = 15,
    check_interval_sec: int = 5,
    min_file_size_mb: float = 10
) -> bool:
    """
    Waits until the file size has not changed for a certain period AND meets minimum size.

    Args:
        file_path: The full path to the file to monitor
        stability_period_sec: How long the file size must remain stable (seconds)
        check_interval_sec: How often to check the file size (seconds)
        min_file_size_mb: Minimum file size in MB (file must be larger than this)

    Returns:
        True if the file became stable and meets minimum size, False if file not found or error
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    min_file_size_bytes = min_file_size_mb * 1024 * 1024
    last_size = None
    first_stable_time = None

    logger.info(f"Monitoring file stability: {file_path}")
    logger.info(f"Requirements: stable for {stability_period_sec}s AND size > {min_file_size_mb} MB")

    while True:
        try:
            current_size = os.path.getsize(file_path)
        except OSError as e:
            logger.error(f"Error accessing file: {e}. Retrying in {check_interval_sec} seconds...")
            time.sleep(check_interval_sec)
            continue

        current_size_mb = current_size / (1024 * 1024)

        # Check if file meets minimum size requirement
        if current_size < min_file_size_bytes:
            if last_size is None or current_size != last_size:
                logger.info(f"File too small: {current_size_mb:.2f} MB (min: {min_file_size_mb} MB). Waiting...")
            first_stable_time = None  # Reset stability timer
            last_size = current_size
            time.sleep(check_interval_sec)
            continue

        # File meets minimum size, now check stability (comparing bytes, not MB)
        if last_size is not None and current_size == last_size:
            # Size hasn't changed
            if first_stable_time is None:
                # First time seeing this stable size
                first_stable_time = time.time()
                logger.info(f"File size stable at {current_size_mb:.2f} MB ({current_size:,} bytes). Starting stability timer...")
            else:
                # Calculate how long it's been stable
                elapsed = time.time() - first_stable_time
                logger.info(f"File stable for {elapsed:.0f}/{stability_period_sec}s at {current_size_mb:.2f} MB ({current_size:,} bytes)")

                # Both conditions met: stable for required period AND meets minimum size
                if elapsed >= stability_period_sec:
                    logger.info(f"File ready: stable for {stability_period_sec}s and size is {current_size_mb:.2f} MB ({current_size:,} bytes)")
                    return True
        else:
            # Size changed - reset stability tracking
            if last_size is not None:
                logger.info(f"File size changed: {last_size:,} -> {current_size:,} bytes ({current_size_mb:.2f} MB)")
            first_stable_time = None

        last_size = current_size
        time.sleep(check_interval_sec)


def copy_file_to_locations(
    source_file: str,
    destinations: list
) -> bool:
    """
    Copy a file to multiple destination directories.

    Args:
        source_file: Full path to the source file
        destinations: List of destination directory paths

    Returns:
        True if all copies succeeded, False otherwise
    """
    filename = os.path.basename(source_file)
    all_successful = True

    for dest_dir in destinations:
        try:
            # Ensure destination directory exists
            Path(dest_dir).mkdir(parents=True, exist_ok=True)

            dest_path = os.path.join(dest_dir, filename)
            logger.info(f"Copying to: {dest_path}")

            shutil.copy2(source_file, dest_path)
            logger.info(f"Successfully copied to: {dest_path}")

        except Exception as e:
            logger.error(f"Failed to copy to {dest_dir}: {e}")
            all_successful = False

    return all_successful


def process_file(file_name: str, config: dict) -> bool:
    """
    Process a single file: wait for completion and copy to destinations.

    Args:
        file_name: Name of the file to process
        config: Configuration dictionary

    Returns:
        True if processing succeeded, False otherwise
    """
    source_path = os.path.join(config["imageSource"], file_name)

    logger.info(f"=" * 60)
    logger.info(f"Processing file: {file_name}")
    logger.info(f"=" * 60)

    # Wait for file to be complete (stable and meets minimum size)
    if not wait_for_stable_file_size(
        source_path,
        config["stability_period_sec"],
        config["check_interval_sec"],
        config["min_file_size"]
    ):
        logger.error(f"Failed to verify file stability: {file_name}")
        return False

    # Copy to both locations
    destinations = [
        config["processingLocation"],
        config["backupLocation"]
    ]

    success = copy_file_to_locations(
        source_path,
        destinations
    )

    if success:
        logger.info(f"Successfully processed: {file_name}")
    else:
        logger.error(f"Processing completed with errors: {file_name}")

    return success


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

    while True:
        try:
            # Find new files that haven't been processed yet
            new_files = find_new_files(
                config["imageSource"],
                config["processingLocation"],
                config["file_pattern"]
            )

            # Process each new file
            for file_name in new_files:
                process_file(file_name, config)

            # Wait before next scan
            if not new_files:
                logger.debug(f"No new files found. Waiting {config['scan_interval_sec']} seconds...")

            time.sleep(config["scan_interval_sec"])

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            time.sleep(config["scan_interval_sec"])


if __name__ == "__main__":
    main()
