# Video File Monitor

A Python script that monitors a network share for video files being written by ffmpeg, detects when recording is complete, and automatically copies files to processing and backup locations.

## Use Case

This tool is designed for multi-server video recording workflows:

- **Recording Server**: Runs ffmpeg to capture video streams and writes files to a network share
- **Processing Server**: Runs this monitor script to detect completed recordings and distribute them

## Features

- **Automatic Detection**: Continuously scans for new video files on a network share
- **Smart Completion Detection**: Monitors file size stability to determine when ffmpeg has finished writing
- **Multi-destination Copy**: Automatically copies completed files to both processing and backup locations
- **Configurable**: Easy-to-modify settings for paths, timing, and file patterns
- **Robust**: Handles network share access issues gracefully with retry logic
- **Logging**: Comprehensive logging of all monitoring and copy operations

## Requirements

- Python 3.6 or higher
- Network access to the recording server's shared folder
- Write permissions to processing and backup directories

## Installation

1. Clone or download this repository
2. No additional dependencies required (uses Python standard library only)

## Configuration

The script uses a `config.json` file for all settings. When you first run the script, it will automatically create a `config.json` file with default values if one doesn't exist.

### Initial Setup

1. Run the script once to generate the default config file:
   ```bash
   python video_monitor.py
   ```

2. Edit the generated `config.json` file with your settings:
   ```json
   {
       "imageSource": "\\\\RECORDING-SERVER\\SharedFolder",
       "processingLocation": "C:\\path\\to\\processing",
       "backupLocation": "C:\\path\\to\\backup",
       "file_pattern": "*.mp4",
       "stability_period_sec": 15,
       "check_interval_sec": 5,
       "scan_interval_sec": 30,
       "min_file_size": 10
   }
   ```

3. Run the script again to start monitoring with your custom settings

**Note**: You can also copy `config.example.json` to `config.json` and edit it directly.

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `imageSource` | Network share path where ffmpeg writes files (UNC path format) | `\\RECORDING-SERVER\SharedFolder` |
| `processingLocation` | Local path where files are copied for processing | `C:\path\to\processing` |
| `backupLocation` | Local path where files are backed up | `C:\path\to\backup` |
| `file_pattern` | Glob pattern to match video files | `*.mp4` |
| `min_file_size` | Minimum file size in MB (files must exceed this size to be processed) | 10 MB |
| `stability_period_sec` | How long file size must remain unchanged to consider recording complete | 15 seconds |
| `check_interval_sec` | How often to check file size during monitoring | 5 seconds |
| `scan_interval_sec` | How often to scan source directory for new files | 30 seconds |

## Usage

### Basic Usage

After configuring your `config.json` file:

```bash
python video_monitor.py
```

The script will:
- Load settings from `config.json`
- Start monitoring the source directory
- Log all activity to the console
- Run continuously until stopped with `Ctrl+C`

### Running as a Background Service

#### Windows (using Task Scheduler)

1. Open Task Scheduler
2. Create a new task that runs at startup
3. Set the action to run: `python C:\path\to\video_monitor.py`
4. Configure to run whether user is logged in or not

#### Linux (using systemd)

Create `/etc/systemd/system/video-monitor.service`:

```ini
[Unit]
Description=Video File Monitor
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/vid-process
ExecStart=/usr/bin/python3 /path/to/vid-process/video_monitor.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Then enable and start:

```bash
sudo systemctl enable video-monitor
sudo systemctl start video-monitor
```

## How It Works

1. **Scan**: Compares files in `imageSource` to files already in `processingLocation`
2. **Detect**: Identifies new files that haven't been processed yet
3. **Monitor**: For each new file, monitors the file size at regular intervals
4. **Verify**: File must meet TWO requirements before being considered complete:
   - File size must be **greater than** `min_file_size` (in MB)
   - File size must remain **stable** (unchanged at the byte level) for `stability_period_sec`
5. **Copy**: Copies the completed file to both `processingLocation` and `backupLocation`
6. **Repeat**: Continues scanning for new files indefinitely

### Why File Size Monitoring?

Monitoring file size stability is the most reliable cross-platform method to detect when ffmpeg has finished writing a file, especially over network shares where file locking mechanisms may not work reliably.

The minimum file size requirement prevents processing of incomplete recordings, test files, or corrupted files that may have stopped growing but are too small to be valid recordings.

**Important**: Stability is determined by comparing exact file sizes in bytes, not rounded megabytes. This ensures that even small writes (like metadata updates) will be detected and the stability timer will reset. The file must remain at the exact same byte count for the entire stability period before it's considered ready for processing.

## Logging

The script outputs detailed logs to the console:

- `INFO`: Normal operations (file detection, copying, completion)
- `WARNING`: Non-critical issues (missing directories)
- `ERROR`: Failures (copy errors, access issues)
- `DEBUG`: Detailed monitoring information (change logging level in script)

Example output:

```
2025-10-17 14:32:10 - INFO - Video File Monitor Started
2025-10-17 14:32:10 - INFO - Source: \\RECORDING-SERVER\SharedFolder
2025-10-17 14:32:10 - INFO - Processing: C:\path\to\processing
2025-10-17 14:32:10 - INFO - Backup: C:\path\to\backup
2025-10-17 14:32:45 - INFO - Found 1 new file(s): recording_20251017_143200.mp4
2025-10-17 14:32:45 - INFO - Monitoring file stability: \\RECORDING-SERVER\SharedFolder\recording_20251017_143200.mp4
2025-10-17 14:32:45 - INFO - Requirements: stable for 15s AND size > 10 MB
2025-10-17 14:32:50 - INFO - File size changing... (Current size: 52.43 MB)
2025-10-17 14:33:05 - INFO - File ready: stable for 15s and size is 125.67 MB
2025-10-17 14:33:05 - INFO - Copying to: C:\path\to\processing\recording_20251017_143200.mp4
2025-10-17 14:33:12 - INFO - Successfully copied to: C:\path\to\processing\recording_20251017_143200.mp4
2025-10-17 14:33:12 - INFO - Copying to: C:\path\to\backup\recording_20251017_143200.mp4
2025-10-17 14:33:19 - INFO - Successfully copied to: C:\path\to\backup\recording_20251017_143200.mp4
2025-10-17 14:33:19 - INFO - Successfully processed: recording_20251017_143200.mp4
```

## Troubleshooting

### Script can't access network share

- Verify the UNC path is correct: `\\SERVER\Share`
- Ensure the user running the script has read permissions on the network share
- Test access manually: `dir \\SERVER\Share` (Windows) or `ls /mnt/share` (Linux)

### Files not being detected

- Check the `file_pattern` matches your video files
- Verify files exist in `imageSource` but not in `processingLocation`
- Increase logging level to `DEBUG` in the script for more details

### Copy operations failing

- Verify write permissions on `processingLocation` and `backupLocation`
- Ensure sufficient disk space is available
- Check that destination paths exist or can be created

### Script stops unexpectedly

- Check system logs for errors
- Consider running as a service with auto-restart enabled
- Monitor disk space and network connectivity

## License

This project is released into the public domain. Use it however you like.

## Support

For issues or questions, please check the documentation in `filecheck.md` for additional technical details about the monitoring approach.
