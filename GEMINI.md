# Project Overview

This project is a Python script that monitors a network share for video files being written by ffmpeg. It detects when a recording is complete by monitoring file size stability, and then automatically copies the completed files to both a processing and a backup location.

The script is configured using a `config.json` file, which is created with default values on the first run if it doesn't exist.

**Key Technologies:**
*   Python 3.6+ (uses only the standard library)

**Architecture:**
*   A single Python script (`video_monitor.py`) runs in a continuous loop.
*   It scans a source directory for new files based on a configurable file pattern.
*   For each new file, it monitors the file size. A file is considered "complete" when its size has not changed for a specified duration and it exceeds a minimum size.
*   Once a file is deemed complete, it is copied to two separate destination directories.
*   The script is designed to be resilient to network errors and can be run as a background service.

# Building and Running

**Prerequisites:**
*   Python 3.6 or higher

**Running the script:**

1.  **Initial Setup:**
    *   Run the script once to generate the default `config.json` file:
        ```bash
        python video_monitor.py
        ```
2.  **Configuration:**
    *   Edit the `config.json` file to specify the `imageSource` (network share), `processingLocation`, `backupLocation`, and other settings.
3.  **Start Monitoring:**
    *   Run the script again to start the monitoring process:
        ```bash
        python video_monitor.py
        ```

The script will run continuously and log its activities to the console.

**Testing:**
There are no automated tests in this project. To test, you would typically:
1.  Configure the `config.json` to point to test directories.
2.  Run the script.
3.  Copy a file into the `imageSource` directory to simulate a recording being written.
4.  Observe the console output to verify that the script detects the file, waits for it to be "stable", and then copies it to the configured destinations.

# Development Conventions

*   **Configuration:** All configuration is externalized to a `config.json` file. Default values are provided in the script itself.
*   **Logging:** The script uses the standard Python `logging` module to provide detailed information about its operations, including file detection, stability checks, and copy operations.
*   **Error Handling:** The script includes error handling for common issues like network share access problems and file I/O errors.
*   **Code Style:** The code is well-commented and follows standard Python conventions.

