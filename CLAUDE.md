# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a video file monitoring system designed for a multi-server setup where:
- **Recording Server**: Runs ffmpeg to record streams and writes files to a network share
- **Processing Server**: Monitors the network share, detects when recordings are complete, and copies them to processing and backup locations

## Architecture

The system uses a file size stability detection approach to determine when ffmpeg has finished writing a video file. The core workflow is:

1. **File Discovery**: Compare files in `imageSource` (network share) against `processingLocation` to identify new recordings
2. **Stability Monitoring**: Monitor file size at regular intervals until it stops changing for a configured period
3. **Multi-destination Copy**: Once stable, copy the completed file to both `processingLocation` and `backupLocation`
4. **Continuous Loop**: Repeat the scan-detect-copy cycle indefinitely

## Running the Monitor

```bash
python video_monitor.py
```

The script runs continuously until stopped with Ctrl+C.

## Configuration

All configuration is in `config.json` (or the `DEFAULT_CONFIG` dictionary in `video_monitor.py`):

- **Paths**: Set `imageSource` (UNC path to network share), `processingLocation`, and `backupLocation`
- **File Pattern**: Use `file_pattern` to filter file types (supports glob patterns like `*.mp4`, `recording_*.mkv`)
- **File Size**: Set `min_file_size` (in MB) to specify the minimum file size required for processing (default: 10 MB)
- **Timing Parameters**:
  - `stability_period_sec`: How long file size must remain unchanged (default: 15 seconds)
  - `check_interval_sec`: Frequency of file size checks during monitoring (default: 5 seconds)
  - `scan_interval_sec`: How often to scan for new files (default: 30 seconds)

## Key Design Decisions

- **Dual Requirements for Processing**: Files must meet BOTH conditions before processing:
  1. File size must be greater than `min_file_size` (in MB)
  2. File size must remain stable (unchanged) for `stability_period_sec`

  This prevents processing of incomplete recordings or test files that are too small.

- **Byte-Level Stability Detection**: File stability is determined by comparing exact file sizes in bytes (not rounded MB). The file size must remain at exactly the same byte count for the entire `stability_period_sec` duration. Uses wall-clock time (`time.time()`) rather than counting check intervals to ensure accurate stability duration regardless of system load or timing variations.

- **Size-based Stability**: Files are considered "complete" when their size hasn't changed for `stability_period_sec`. This is more reliable than file locking mechanisms across network shares.
- **Comparison-based Discovery**: New files are detected by comparing source and processing directories, avoiding duplicate processing of already-copied files.
- **Automatic Directory Creation**: Destination directories are created automatically if they don't exist.
- **Error Resilience**: File access errors (common with network shares) are caught and retried rather than crashing the monitor.

## Reference Documentation

See `filecheck.md` for the original design requirements and explanation of the file stability monitoring approach.
